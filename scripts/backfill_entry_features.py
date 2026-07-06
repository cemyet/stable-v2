"""Backfill the `entry_features` ML feature table.

Builds one point-in-time feature row per qualifying `entry` (Swedish trot,
race_date >= 2005-01-01). Every "as-of" feature is computed over ALL entries
globally (so a trainer's foreign starts count), using only data from races
STRICTLY BEFORE each entry's race_date — no future-information leakage.

The work is set-based: a handful of window-function passes over UNLOGGED
staging tables, then one INSERT...SELECT assembling the slice we keep.

Point-in-time trick: for any entity (trainer, driver, sire, post, ...) the
target entry itself guarantees a daily-aggregate row at (entity_key, race_date),
so the as-of value is just that entity's running aggregate over STRICTLY
EARLIER dates (window frame excluding the current date). That lets every
as-of join be a plain equi-join on (entity_key, race_date).

Usage:
    python3 -m scripts.backfill_entry_features --phase 1
    python3 -m scripts.backfill_entry_features --phase 2
    python3 -m scripts.backfill_entry_features --phase all        # 1 then 2
    python3 -m scripts.backfill_entry_features --phase 1 --keep-staging

Phase 1: schema-cheap core features (horse / trainer / driver / post / race
         relative + labels). Rebuilds the whole slice (idempotent).
Phase 2: expensive global partition passes (sire/dam galopp rates, post and
         post-by-track galopp rates). UPDATEs existing rows in place.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from core.db import get_connection

# Start of regular Swedish shoe data (2004 was the partial ramp-up year).
START_DATE = "2005-01-01"

# Qualifier (prov / premie) result codes — byte-for-byte the same predicate the
# app + career-stats matviews use, so "starts" here mean the same thing.
_QUALIFIER_RE = (
    r"^(gdk|ejg|ejp|gd|gk|egk|egdk|gkd|gdj|gdb|gdek|gdgk|gdl|ddk|frdk|erj|ejk|"
    r"ejgk|ejgd|ej|EJ|EJG|EJP|GDK|Gdk|g)[0-9]?$|^[12]?[pP][0-9]?$"
)

# Phase-2 staging (built by run_phase2). Listed separately so run_phase2 can
# drop just these before (re)building, staying idempotent.
_STAGING_P2 = [
    "_ef_sire_daily", "_ef_sire", "_ef_dam_daily", "_ef_dam",
    "_ef_post_auto_daily", "_ef_post_auto",
    "_ef_post_volt_daily", "_ef_post_volt",
    "_ef_post_auto_track_daily", "_ef_post_auto_track",
    "_ef_post_volt_track_daily", "_ef_post_volt_track",
    "_ef_horse_trainer_daily", "_ef_horse_trainer",
    "_ef_horse_track_daily", "_ef_horse_track",
]

# All staging tables. These are scratch tables for the backfill/refresh only and
# are hidden from the ML explorer by the '_ef_' blocklist prefix in web/app.py.
_STAGING = [
    "_ef_new", "_ef_base", "_ef_horse", "_ef_hp",
    "_ef_trainer_daily", "_ef_trainer", "_ef_trainer_track_daily", "_ef_trainer_track",
    "_ef_driver_daily", "_ef_driver",
    "_ef_quality", "_ef_race",
] + _STAGING_P2


def _run(cur, label, sql, params=None):
    t0 = time.time()
    cur.execute(sql, params or ())
    print(f"  [{time.time()-t0:6.1f}s] {label}", flush=True)


def _drop_tables(conn, names):
    with conn.cursor() as cur:
        for t in names:
            cur.execute(f"DROP TABLE IF EXISTS {t} CASCADE")
    conn.commit()


def _drop_staging(conn):
    _drop_tables(conn, _STAGING)


# ---------------------------------------------------------------------------
# Shared base: every global entry with derived flags, plus an is_target flag
# marking the Swedish-trot 2005+ rows we actually store.
# ---------------------------------------------------------------------------
def build_base(conn):
    print("Building _ef_base (global entries + derived flags)...", flush=True)
    with conn.cursor() as cur:
        _run(cur, "create _ef_base", f"""
            CREATE UNLOGGED TABLE _ef_base AS
            SELECT
                e.entry_id, e.race_id, e.horse_id, e.trainer_id, e.driver_id,
                r.track_id, e.race_date,
                (e.race_date - DATE '2000-01-01')        AS day_num,
                e.post, e.program_number,
                e.auto                                   AS is_auto,
                e.distance                               AS distance_m,
                e.tillagg                                AS distance_added_m,
                e.earnings_pre, e.shoe_code, e.age, e.sex,
                h.breed_code,
                h.sire_id, h.dam_id,
                NULLIF(e.time_seconds, 0)                AS time_seconds,
                CASE
                    WHEN e.distance IS NULL THEN NULL
                    WHEN e.distance <= 1700 THEN 's'
                    WHEN e.distance <= 2250 THEN 'm'
                    ELSE 'l'
                END                                      AS band,
                (e.shoe_code IN ('1','2','3'))           AS barefoot_any,
                (NOT COALESCE(e.withdrawn, false)
                  AND COALESCE(e.placement_text,'') !~ %s) AS start_ok,
                (e.placement = 1 AND NOT COALESCE(e.disqualified,false)) AS is_win,
                COALESCE(e.galopp, false)                AS is_gal,
                (e.odds IS NOT NULL AND e.odds > 1)      AS has_odds,
                CASE WHEN e.odds IS NOT NULL AND e.odds > 1
                     THEN LEAST(0.7::real, (1.0/e.odds)::real) ELSE 0 END AS expw,
                COALESCE(e.withdrawn,false)              AS withdrawn,
                e.placement, COALESCE(e.disqualified,false) AS disqualified,
                (t.country = 'SE' AND e.race_date >= DATE %s) AS is_target
            FROM entry e
            JOIN race r  ON r.race_id  = e.race_id
            LEFT JOIN track t ON t.track_id = r.track_id
            LEFT JOIN horse h ON h.horse_id = e.horse_id
            WHERE e.race_date IS NOT NULL
        """, (_QUALIFIER_RE, START_DATE))
        _run(cur, "index _ef_base(horse_id,race_date,entry_id)",
             "CREATE INDEX ON _ef_base (horse_id, race_date, entry_id)")
        _run(cur, "index _ef_base(trainer_id)",
             "CREATE INDEX ON _ef_base (trainer_id) WHERE trainer_id IS NOT NULL")
        _run(cur, "index _ef_base(driver_id)",
             "CREATE INDEX ON _ef_base (driver_id) WHERE driver_id IS NOT NULL")
        _run(cur, "index _ef_base(race_id)",
             "CREATE INDEX ON _ef_base (race_id)")
        # Full unique index on entry_id so incremental insert/update joins that
        # drive off the small _ef_new set are index lookups, not 7.6M scans.
        _run(cur, "unique index _ef_base(entry_id)",
             "CREATE UNIQUE INDEX ON _ef_base (entry_id)")
        cur.execute("SELECT COUNT(*), COUNT(*) FILTER (WHERE is_target) FROM _ef_base")
        tot, tgt = cur.fetchone()
        print(f"  _ef_base: {tot:,} global rows, {tgt:,} target rows", flush=True)
    conn.commit()


# ---------------------------------------------------------------------------
# Horse career as-of (entry-granular; previous-start semantics need entry order)
# ---------------------------------------------------------------------------
def build_horse(conn):
    print("Building _ef_horse (per-horse as-of)...", flush=True)
    with conn.cursor() as cur:
        _run(cur, "create _ef_horse", """
            CREATE UNLOGGED TABLE _ef_horse AS
            SELECT * FROM (
                SELECT
                    entry_id,
                    COUNT(*) FILTER (WHERE start_ok)                      OVER w AS h_starts,
                    COUNT(*) FILTER (WHERE is_win)                        OVER w AS h_wins,
                    COUNT(*) FILTER (WHERE is_gal AND start_ok)           OVER w AS h_gals,
                    -- gal-adjusted win-rate components (as-of, prior starts):
                    --  clean   = exclude galopp AND disqualified
                    --  non-dq  = exclude only disqualified (non-dq gals stay in)
                    COUNT(*) FILTER (WHERE start_ok AND NOT is_gal AND NOT disqualified) OVER w AS h_clean_starts,
                    COUNT(*) FILTER (WHERE is_win   AND NOT is_gal)                      OVER w AS h_clean_wins,
                    COUNT(*) FILTER (WHERE start_ok AND NOT disqualified)                OVER w AS h_nondq_starts,
                    COUNT(*) FILTER (WHERE is_win)                                       OVER w AS h_nondq_wins,
                    COUNT(*) FILTER (WHERE start_ok AND is_auto)          OVER w AS h_starts_auto,
                    COUNT(*) FILTER (WHERE is_gal AND start_ok AND is_auto) OVER w AS h_gals_auto,
                    COUNT(*) FILTER (WHERE start_ok AND is_auto = false)  OVER w AS h_starts_volt,
                    COUNT(*) FILTER (WHERE is_gal AND start_ok AND is_auto = false) OVER w AS h_gals_volt,
                    COUNT(*) FILTER (WHERE start_ok AND barefoot_any)     OVER w AS h_starts_bf,
                    COUNT(*) FILTER (WHERE is_gal AND start_ok AND barefoot_any) OVER w AS h_gals_bf,
                    LAG(is_gal)      OVER o AS gal_pre,
                    (race_date - LAG(race_date) OVER o) AS days_since_last,
                    MIN(time_seconds) FILTER (WHERE band='s') OVER w AS rec_s,
                    MIN(time_seconds) FILTER (WHERE band='m') OVER w AS rec_m,
                    MIN(time_seconds) FILTER (WHERE band='l') OVER w AS rec_l,
                    COUNT(*) FILTER (WHERE is_auto)         OVER w AS pa,
                    COUNT(*) FILTER (WHERE is_auto = false) OVER w AS pv,
                    COUNT(*) FILTER (WHERE band='s')        OVER w AS pbs,
                    COUNT(*) FILTER (WHERE band='m')        OVER w AS pbm,
                    COUNT(*) FILTER (WHERE band='l')        OVER w AS pbl,
                    COUNT(*) FILTER (WHERE barefoot_any)    OVER w AS pbf,
                    is_target
                FROM _ef_base
                WINDOW
                    o AS (PARTITION BY horse_id ORDER BY race_date, entry_id),
                    w AS (PARTITION BY horse_id ORDER BY race_date, entry_id
                          ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING)
            ) s
            WHERE is_target
        """)
        _run(cur, "pk _ef_horse", "ALTER TABLE _ef_horse ADD PRIMARY KEY (entry_id)")
    conn.commit()


# ---------------------------------------------------------------------------
# Per-horse prior starts with the SAME trainer / driver (first_* + counts)
# ---------------------------------------------------------------------------
def build_horse_person(conn):
    print("Building _ef_hp (prior starts of this horse w/ trainer & driver)...", flush=True)
    with conn.cursor() as cur:
        _run(cur, "create _ef_hp", """
            CREATE UNLOGGED TABLE _ef_hp AS
            SELECT * FROM (
                SELECT entry_id,
                    COUNT(*) OVER wt AS races_with_trainer,
                    COUNT(*) OVER wd AS races_with_driver,
                    is_target
                FROM _ef_base
                WINDOW
                    wt AS (PARTITION BY horse_id, trainer_id ORDER BY race_date, entry_id
                           ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING),
                    wd AS (PARTITION BY horse_id, driver_id ORDER BY race_date, entry_id
                           ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING)
            ) s
            WHERE is_target
        """)
        _run(cur, "pk _ef_hp", "ALTER TABLE _ef_hp ADD PRIMARY KEY (entry_id)")
    conn.commit()


# ---------------------------------------------------------------------------
# Trainer / driver: daily rollup, then career-as-of (ROWS) + 30d (RANGE days)
# ---------------------------------------------------------------------------
def _build_person_windows(conn, role, id_col):
    daily = f"_ef_{role}_daily"
    out = f"_ef_{role}"
    print(f"Building {out} (per-{role} as-of, career + 30d)...", flush=True)
    with conn.cursor() as cur:
        _run(cur, f"create {daily}", f"""
            CREATE UNLOGGED TABLE {daily} AS
            SELECT {id_col} AS pid, race_date,
                   MIN(day_num)                              AS day_num,
                   COUNT(*) FILTER (WHERE start_ok)          AS d_starts,
                   COUNT(*) FILTER (WHERE is_win)            AS d_wins,
                   COUNT(*) FILTER (WHERE is_gal AND start_ok) AS d_gals,
                   COUNT(*) FILTER (WHERE has_odds)          AS d_nodds,
                   COUNT(*) FILTER (WHERE is_win AND has_odds) AS d_actw,
                   COALESCE(SUM(expw),0)                     AS d_expw
            FROM _ef_base
            WHERE {id_col} IS NOT NULL
            GROUP BY {id_col}, race_date
        """)
        _run(cur, f"create {out}", f"""
            CREATE UNLOGGED TABLE {out} AS
            SELECT pid, race_date,
                SUM(d_starts) OVER wc AS p_starts,
                SUM(d_wins)   OVER wc AS p_wins,
                SUM(d_gals)   OVER wc AS p_gals,
                SUM(d_starts) OVER w30 AS p_starts30,
                SUM(d_wins)   OVER w30 AS p_wins30,
                SUM(d_nodds)  OVER w30 AS p_nodds30,
                SUM(d_actw)   OVER w30 AS p_actw30,
                SUM(d_expw)   OVER w30 AS p_expw30
            FROM {daily}
            WINDOW
                wc  AS (PARTITION BY pid ORDER BY race_date
                        ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING),
                w30 AS (PARTITION BY pid ORDER BY day_num
                        RANGE BETWEEN 30 PRECEDING AND 1 PRECEDING)
        """)
        _run(cur, f"pk {out}", f"ALTER TABLE {out} ADD PRIMARY KEY (pid, race_date)")
    conn.commit()


def build_trainer_track(conn):
    print("Building _ef_trainer_track (trainer win rate at track, as-of)...", flush=True)
    with conn.cursor() as cur:
        _run(cur, "create _ef_trainer_track_daily", """
            CREATE UNLOGGED TABLE _ef_trainer_track_daily AS
            SELECT trainer_id AS pid, track_id, race_date,
                   COUNT(*) FILTER (WHERE start_ok) AS d_starts,
                   COUNT(*) FILTER (WHERE is_win)   AS d_wins
            FROM _ef_base
            WHERE trainer_id IS NOT NULL AND track_id IS NOT NULL
            GROUP BY trainer_id, track_id, race_date
        """)
        _run(cur, "create _ef_trainer_track", """
            CREATE UNLOGGED TABLE _ef_trainer_track AS
            SELECT pid, track_id, race_date,
                SUM(d_starts) OVER wc AS tt_starts,
                SUM(d_wins)   OVER wc AS tt_wins
            FROM _ef_trainer_track_daily
            WINDOW wc AS (PARTITION BY pid, track_id ORDER BY race_date
                          ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING)
        """)
        _run(cur, "pk _ef_trainer_track",
             "ALTER TABLE _ef_trainer_track ADD PRIMARY KEY (pid, track_id, race_date)")
    conn.commit()


# ---------------------------------------------------------------------------
# Race-level: field size + completeness, earnings rank/pct, time z-score
# ---------------------------------------------------------------------------
def build_race(conn):
    print("Building _ef_quality + _ef_race (race-relative)...", flush=True)
    with conn.cursor() as cur:
        _run(cur, "create _ef_quality", """
            CREATE UNLOGGED TABLE _ef_quality AS
            SELECT race_id,
                   COUNT(*)                              AS n_entries,
                   COUNT(*) FILTER (WHERE NOT withdrawn) AS n_starters,
                   MAX(program_number)                   AS max_pgm
            FROM _ef_base
            WHERE race_id IN (SELECT race_id FROM _ef_base WHERE is_target)
            GROUP BY race_id
        """)
        _run(cur, "pk _ef_quality", "ALTER TABLE _ef_quality ADD PRIMARY KEY (race_id)")
        _run(cur, "create _ef_race", """
            CREATE UNLOGGED TABLE _ef_race AS
            SELECT * FROM (
                SELECT entry_id,
                    RANK() OVER (PARTITION BY race_id ORDER BY earnings_pre DESC NULLS LAST) AS earn_rank,
                    PERCENT_RANK() OVER (PARTITION BY race_id ORDER BY earnings_pre ASC) AS earn_pct,
                    CASE WHEN time_seconds > 0 THEN
                        (time_seconds - AVG(time_seconds) OVER pr)
                        / NULLIF(STDDEV_SAMP(time_seconds) OVER pr, 0)
                    END AS y_time_z,
                    is_target
                FROM _ef_base
                WHERE race_id IN (SELECT race_id FROM _ef_base WHERE is_target)
                WINDOW pr AS (PARTITION BY race_id)
            ) s
            WHERE is_target
        """)
        _run(cur, "pk _ef_race", "ALTER TABLE _ef_race ADD PRIMARY KEY (entry_id)")
    conn.commit()


# ---------------------------------------------------------------------------
# Phase 1 assembly
# ---------------------------------------------------------------------------
def assemble_phase1(conn, only_new=False):
    """Assemble the phase-1 rows. only_new=True appends just the entries that
    don't yet have a feature row (incremental update path) instead of a full
    TRUNCATE + rebuild."""
    print(f"Assembling entry_features (phase 1 INSERT, only_new={only_new})...", flush=True)
    # Incremental drives off the small _ef_new set (index lookups into _ef_base)
    # rather than scanning all 2.4M target rows then filtering.
    from_head = ("FROM _ef_new n JOIN _ef_base b ON b.entry_id = n.entry_id"
                 if only_new else "FROM _ef_base b")
    where_tail = "WHERE TRUE" if only_new else "WHERE b.is_target"
    with conn.cursor() as cur:
        if not only_new:
            _run(cur, "truncate entry_features", "TRUNCATE entry_features")
        _run(cur, "insert entry_features", """
            INSERT INTO entry_features (
                entry_id, race_id, horse_id, trainer_id, driver_id, track_id, race_date,
                is_auto, distance_m, distance_added_m, starters, race_complete,
                post, is_springspar, is_bakspar,
                age, sex, breed_code, shoe_code, is_barefoot_all, first_shoes_off,
                earnings_pre, earnings_rank_in_race, earnings_pct_in_race,
                horse_starts, horse_winrate, horse_galrate, horse_galrate_auto,
                horse_galrate_volt, horse_galrate_auto_volt_delta, horse_galrate_barefoot,
                horse_winrate_gal_adj, horse_winrate_gal_incl,
                gal_pre, days_since_last_start, season_debut,
                record_s_kmtime, record_m_kmtime, record_l_kmtime,
                trainer_starts, trainer_winrate, trainer_galrate, trainer_winrate_30d,
                trainer_winrate_delta, trainer_form_odds_30d, trainer_winrate_track,
                races_with_trainer, first_trainer,
                driver_starts, driver_winrate, driver_galrate, driver_winrate_30d,
                driver_winrate_delta, driver_form_odds_30d,                 races_with_driver, first_driver,
                first_method, first_distance_band, race_month,
                y_gal, y_win, y_top3, y_placement, y_disq, y_time_s, y_time_z,
                features_version, computed_at
            )
            SELECT
                b.entry_id, b.race_id, b.horse_id, b.trainer_id, b.driver_id, b.track_id, b.race_date,
                b.is_auto, b.distance_m, b.distance_added_m, q.n_starters,
                (q.n_starters >= 3 AND (q.max_pgm IS NULL OR q.max_pgm <= q.n_entries)),
                b.post,
                (b.is_auto = false AND b.post IN (6,7)),
                (b.is_auto = true  AND b.post >= 9),
                b.age, b.sex, b.breed_code, b.shoe_code,
                (b.shoe_code = '1'),
                (b.barefoot_any AND COALESCE(h.pbf,0) = 0),
                b.earnings_pre, r.earn_rank, r.earn_pct::real,
                h.h_starts,
                (h.h_wins::real / NULLIF(h.h_starts,0)),
                (h.h_gals::real / NULLIF(h.h_starts,0)),
                (h.h_gals_auto::real / NULLIF(h.h_starts_auto,0)),
                (h.h_gals_volt::real / NULLIF(h.h_starts_volt,0)),
                ((h.h_gals_auto::real / NULLIF(h.h_starts_auto,0))
                 - (h.h_gals_volt::real / NULLIF(h.h_starts_volt,0))),
                (h.h_gals_bf::real / NULLIF(h.h_starts_bf,0)),
                (h.h_clean_wins::real / NULLIF(h.h_clean_starts,0)),
                (h.h_nondq_wins::real / NULLIF(h.h_nondq_starts,0)),
                h.gal_pre, h.days_since_last, (h.days_since_last > 60),
                h.rec_s, h.rec_m, h.rec_l,
                COALESCE(t.p_starts, 0),
                (t.p_wins::real / NULLIF(t.p_starts,0)),
                (t.p_gals::real / NULLIF(t.p_starts,0)),
                (t.p_wins30::real / NULLIF(t.p_starts30,0)),
                ((t.p_wins30::real / NULLIF(t.p_starts30,0))
                 - (t.p_wins::real / NULLIF(t.p_starts,0))),
                ((t.p_actw30 - t.p_expw30) / NULLIF(t.p_nodds30,0)),
                (tt.tt_wins::real / NULLIF(tt.tt_starts,0)),
                COALESCE(hp.races_with_trainer,0), (COALESCE(hp.races_with_trainer,0) = 0),
                COALESCE(d.p_starts, 0),
                (d.p_wins::real / NULLIF(d.p_starts,0)),
                (d.p_gals::real / NULLIF(d.p_starts,0)),
                (d.p_wins30::real / NULLIF(d.p_starts30,0)),
                ((d.p_wins30::real / NULLIF(d.p_starts30,0))
                 - (d.p_wins::real / NULLIF(d.p_starts,0))),
                ((d.p_actw30 - d.p_expw30) / NULLIF(d.p_nodds30,0)),
                COALESCE(hp.races_with_driver,0), (COALESCE(hp.races_with_driver,0) = 0),
                CASE WHEN b.is_auto IS NULL THEN NULL
                     WHEN b.is_auto THEN (COALESCE(h.pa,0) = 0)
                     ELSE (COALESCE(h.pv,0) = 0) END,
                CASE b.band WHEN 's' THEN (COALESCE(h.pbs,0) = 0)
                            WHEN 'm' THEN (COALESCE(h.pbm,0) = 0)
                            WHEN 'l' THEN (COALESCE(h.pbl,0) = 0)
                            ELSE NULL END,
                EXTRACT(MONTH FROM b.race_date)::smallint,
                b.is_gal, b.is_win,
                (b.placement BETWEEN 1 AND 3 AND NOT b.disqualified),
                b.placement, b.disqualified,
                b.time_seconds, r.y_time_z::real,
                1, NOW()
            """ + from_head + """
            LEFT JOIN _ef_horse        h  ON h.entry_id  = b.entry_id
            LEFT JOIN _ef_hp           hp ON hp.entry_id = b.entry_id
            LEFT JOIN _ef_trainer      t  ON t.pid = b.trainer_id  AND t.race_date = b.race_date
            LEFT JOIN _ef_trainer_track tt ON tt.pid = b.trainer_id AND tt.track_id = b.track_id AND tt.race_date = b.race_date
            LEFT JOIN _ef_driver       d  ON d.pid = b.driver_id   AND d.race_date = b.race_date
            LEFT JOIN _ef_race         r  ON r.entry_id  = b.entry_id
            LEFT JOIN _ef_quality      q  ON q.race_id   = b.race_id
            """ + where_tail + """
        """)
        print(f"  inserted {cur.rowcount:,} rows", flush=True)
        cur.execute("SELECT COUNT(*) FROM entry_features")
        print(f"  entry_features total rows: {cur.fetchone()[0]:,}", flush=True)
    conn.commit()


def run_phase1(conn):
    t0 = time.time()
    _drop_staging(conn)
    build_base(conn)
    build_horse(conn)
    build_horse_person(conn)
    _build_person_windows(conn, "trainer", "trainer_id")
    build_trainer_track(conn)
    _build_person_windows(conn, "driver", "driver_id")
    build_race(conn)
    assemble_phase1(conn)
    print(f"Phase 1 complete in {time.time()-t0:.1f}s", flush=True)


# ---------------------------------------------------------------------------
# Phase 2: breeding + post galopp rates (expensive global partition passes).
# UPDATEs the rows produced by phase 1. Re-uses _ef_base.
# ---------------------------------------------------------------------------
def _build_galrate_asof(conn, name, key_cols, where, src_keys):
    """Generic as-of galopp-rate builder over _ef_base.

    Daily rollup by (key_cols, race_date), then running counts over strictly
    earlier dates. Produces table _ef_<name> keyed (key_cols, race_date) with
    g_starts, g_gals.
    """
    daily = f"_ef_{name}_daily"
    out = f"_ef_{name}"
    keys = ", ".join(key_cols)
    print(f"Building {out} (as-of galopp rate)...", flush=True)
    with conn.cursor() as cur:
        _run(cur, f"create {daily}", f"""
            CREATE UNLOGGED TABLE {daily} AS
            SELECT {src_keys}, race_date,
                   COUNT(*) FILTER (WHERE start_ok)            AS d_starts,
                   COUNT(*) FILTER (WHERE is_gal AND start_ok) AS d_gals
            FROM _ef_base
            WHERE {where}
            GROUP BY {src_keys}, race_date
        """)
        _run(cur, f"create {out}", f"""
            CREATE UNLOGGED TABLE {out} AS
            SELECT {keys}, race_date,
                SUM(d_starts) OVER w AS g_starts,
                SUM(d_gals)   OVER w AS g_gals
            FROM {daily}
            WINDOW w AS (PARTITION BY {keys} ORDER BY race_date
                         ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING)
        """)
        _run(cur, f"pk {out}", f"ALTER TABLE {out} ADD PRIMARY KEY ({keys}, race_date)")
    conn.commit()


def run_phase2(conn, only_new=False):
    """Compute phase-2 columns. only_new=True restricts the UPDATEs to the
    freshly-inserted rows in _ef_new (incremental path)."""
    t0 = time.time()
    # Incremental drives off _ef_new; full phase-2 touches every row.
    new_join = " JOIN _ef_new n ON n.entry_id = b.entry_id" if only_new else ""
    # Idempotent: clear any phase-2 staging left by an interrupted prior run.
    _drop_tables(conn, _STAGING_P2)
    with conn.cursor() as cur:
        cur.execute("SELECT to_regclass('_ef_base')")
        if cur.fetchone()[0] is None:
            build_base(conn)

    # Breeding: galopp rate over ALL offspring of the sire / dam, as-of.
    # The target entry's horse is itself an offspring racing on race_date, so a
    # (sire_id, race_date) daily row exists -> exact-key as-of join is valid.
    _build_galrate_asof(conn, "sire", ["sire_id"], "sire_id IS NOT NULL", "sire_id")
    _build_galrate_asof(conn, "dam",  ["dam_id"],  "dam_id IS NOT NULL",  "dam_id")
    # Post galopp rates, split by start method, global and per-track.
    _build_galrate_asof(conn, "post_auto", ["post"],
                        "post IS NOT NULL AND is_auto", "post")
    _build_galrate_asof(conn, "post_volt", ["post"],
                        "post IS NOT NULL AND is_auto = false", "post")
    _build_galrate_asof(conn, "post_auto_track", ["track_id", "post"],
                        "post IS NOT NULL AND track_id IS NOT NULL AND is_auto",
                        "track_id, post")
    _build_galrate_asof(conn, "post_volt_track", ["track_id", "post"],
                        "post IS NOT NULL AND track_id IS NOT NULL AND is_auto = false",
                        "track_id, post")
    # Horse-combination as-of gal rates: this horse's break rate with this
    # specific trainer, and at this specific track. The target entry itself
    # contributes a (key, race_date) daily row, so the strictly-earlier window
    # join is exact and leak-free (same pattern as sire/dam/post).
    _build_galrate_asof(conn, "horse_trainer", ["horse_id", "trainer_id"],
                        "horse_id IS NOT NULL AND trainer_id IS NOT NULL",
                        "horse_id, trainer_id")
    _build_galrate_asof(conn, "horse_track", ["horse_id", "track_id"],
                        "horse_id IS NOT NULL AND track_id IS NOT NULL",
                        "horse_id, track_id")

    print("Applying phase-2 UPDATEs to entry_features...", flush=True)
    with conn.cursor() as cur:
        _run(cur, "update sire/dam", """
            UPDATE entry_features ef SET
                sire_galrate = (s.g_gals::real / NULLIF(s.g_starts,0)),
                sire_starts  = COALESCE(s.g_starts, 0),
                dam_galrate  = (dm.g_gals::real / NULLIF(dm.g_starts,0)),
                dam_starts   = COALESCE(dm.g_starts, 0)
            FROM _ef_base b""" + new_join + """
            LEFT JOIN _ef_sire s  ON s.sire_id = b.sire_id AND s.race_date = b.race_date
            LEFT JOIN _ef_dam  dm ON dm.dam_id = b.dam_id  AND dm.race_date = b.race_date
            WHERE b.entry_id = ef.entry_id
        """)
        _run(cur, "update post rates", """
            UPDATE entry_features ef SET
                post_galrate_auto = (pa.g_gals::real / NULLIF(pa.g_starts,0)),
                post_galrate_volt = (pv.g_gals::real / NULLIF(pv.g_starts,0)),
                post_galrate_auto_track = (pat.g_gals::real / NULLIF(pat.g_starts,0)),
                post_galrate_volt_track = (pvt.g_gals::real / NULLIF(pvt.g_starts,0)),
                features_version = 2
            FROM _ef_base b""" + new_join + """
            LEFT JOIN _ef_post_auto pa  ON pa.post = b.post AND pa.race_date = b.race_date
            LEFT JOIN _ef_post_volt pv  ON pv.post = b.post AND pv.race_date = b.race_date
            LEFT JOIN _ef_post_auto_track pat ON pat.track_id = b.track_id AND pat.post = b.post AND pat.race_date = b.race_date
            LEFT JOIN _ef_post_volt_track pvt ON pvt.track_id = b.track_id AND pvt.post = b.post AND pvt.race_date = b.race_date
            WHERE b.entry_id = ef.entry_id
        """)
        # Horse×trainer / horse×track gal rates + reliability counts, and
        # race_month. race_month is also set in phase 1; setting it here lets a
        # phase-2-only run backfill rows created before the column existed.
        _run(cur, "update horse-combo rates + month", """
            UPDATE entry_features ef SET
                horse_trainer_galrate = (ht.g_gals::real / NULLIF(ht.g_starts,0)),
                horse_trainer_starts  = COALESCE(ht.g_starts, 0),
                horse_galrate_track   = (htk.g_gals::real / NULLIF(htk.g_starts,0)),
                horse_track_starts    = COALESCE(htk.g_starts, 0),
                race_month            = EXTRACT(MONTH FROM b.race_date)::smallint
            FROM _ef_base b""" + new_join + """
            LEFT JOIN _ef_horse_trainer ht  ON ht.horse_id = b.horse_id AND ht.trainer_id = b.trainer_id AND ht.race_date = b.race_date
            LEFT JOIN _ef_horse_track   htk ON htk.horse_id = b.horse_id AND htk.track_id = b.track_id AND htk.race_date = b.race_date
            WHERE b.entry_id = ef.entry_id
        """)
    conn.commit()
    print(f"Phase 2 complete in {time.time()-t0:.1f}s", flush=True)


def _run_phase3(conn, only_new=False):
    """Compute gal_recent_5 and field_avg_galrate for (optionally new-only) rows."""
    t0 = time.time()
    new_filter = " AND ef.entry_id IN (SELECT entry_id FROM _ef_new)" if only_new else ""
    with conn.cursor() as cur:
        cur.execute(f"""
            WITH horse_history AS (
                SELECT ef2.entry_id, ef2.horse_id, ef2.race_date,
                       e.galopp::int AS gal_int
                FROM entry_features ef2
                JOIN entry e ON e.entry_id = ef2.entry_id
                WHERE NOT e.withdrawn
            ),
            windowed AS (
                SELECT entry_id,
                       COALESCE(SUM(gal_int) OVER (
                           PARTITION BY horse_id ORDER BY race_date, entry_id
                           ROWS BETWEEN 5 PRECEDING AND 1 PRECEDING
                       ), 0)::smallint AS gal_recent_5
                FROM horse_history
            )
            UPDATE entry_features ef SET gal_recent_5 = w.gal_recent_5
            FROM windowed w
            WHERE w.entry_id = ef.entry_id{new_filter}
        """)
        cur.execute(f"""
            WITH race_stats AS (
                SELECT race_id,
                       SUM(horse_galrate) AS total_gr,
                       COUNT(horse_galrate) AS cnt_gr
                FROM entry_features WHERE horse_galrate IS NOT NULL GROUP BY race_id
            )
            UPDATE entry_features ef SET field_avg_galrate =
                CASE WHEN rs.cnt_gr > 1 THEN
                    (rs.total_gr - COALESCE(ef.horse_galrate, 0)) /
                    (rs.cnt_gr - CASE WHEN ef.horse_galrate IS NOT NULL THEN 1 ELSE 0 END)
                ELSE NULL END::real
            FROM race_stats rs
            WHERE rs.race_id = ef.race_id{new_filter}
        """)
    conn.commit()
    print(f"Phase 3 (recency + field) complete in {time.time()-t0:.1f}s", flush=True)


def _run_phase4(conn, only_new=False):
    """Compute streak / instability / rank features (optionally new-only)."""
    t0 = time.time()
    new_filter = " AND ef.entry_id IN (SELECT entry_id FROM _ef_new)" if only_new else ""
    with conn.cursor() as cur:
        cur.execute(f"""
            WITH hist AS (
                SELECT ef2.entry_id, ef2.horse_id, ef2.race_date,
                       ef2.is_auto, ef2.is_barefoot_all, ef2.distance_m, e.galopp,
                       ROW_NUMBER() OVER (PARTITION BY ef2.horse_id ORDER BY ef2.race_date, ef2.entry_id) AS rn
                FROM entry_features ef2
                JOIN entry e ON e.entry_id = ef2.entry_id
                WHERE NOT e.withdrawn
            ),
            calc AS (
                SELECT entry_id, rn,
                       MAX(CASE WHEN galopp THEN rn END) OVER w AS last_gal_rn,
                       MAX(CASE WHEN NOT galopp THEN rn END) OVER w AS last_clean_rn,
                       (is_auto IS DISTINCT FROM LAG(is_auto) OVER wo) AS method_switch,
                       (is_barefoot_all IS DISTINCT FROM LAG(is_barefoot_all) OVER wo) AS barefoot_change,
                       (distance_m - LAG(distance_m) OVER wo) AS distance_delta_vs_last,
                       (LAG(is_auto) OVER wo IS NULL) AS is_first
                FROM hist
                WINDOW w  AS (PARTITION BY horse_id ORDER BY race_date, entry_id
                              ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING),
                       wo AS (PARTITION BY horse_id ORDER BY race_date, entry_id)
            ),
            seq AS (
                SELECT entry_id,
                       ((rn - 1) - COALESCE(last_clean_rn, 0))::smallint AS gal_streak,
                       ((rn - 1) - COALESCE(last_gal_rn, 0))::smallint   AS clean_streak,
                       CASE WHEN is_first THEN NULL ELSE method_switch END   AS method_switch,
                       CASE WHEN is_first THEN NULL ELSE barefoot_change END AS barefoot_change,
                       distance_delta_vs_last
                FROM calc
            )
            UPDATE entry_features ef SET
                gal_streak             = seq.gal_streak,
                clean_streak           = seq.clean_streak,
                method_switch          = seq.method_switch,
                barefoot_change        = seq.barefoot_change,
                distance_delta_vs_last = seq.distance_delta_vs_last,
                galrate_method         = CASE WHEN ef.is_auto THEN ef.horse_galrate_auto
                                              ELSE ef.horse_galrate_volt END
            FROM seq
            WHERE seq.entry_id = ef.entry_id{new_filter}
        """)
        cur.execute(f"""
            WITH ranked AS (
                SELECT entry_id,
                       PERCENT_RANK() OVER (PARTITION BY race_id ORDER BY horse_galrate)::real AS gal_rank_in_field
                FROM entry_features WHERE horse_galrate IS NOT NULL
            )
            UPDATE entry_features ef SET gal_rank_in_field = ranked.gal_rank_in_field
            FROM ranked WHERE ranked.entry_id = ef.entry_id{new_filter}
        """)
    conn.commit()
    print(f"Phase 4 (streaks + rank) complete in {time.time()-t0:.1f}s", flush=True)


def _run_phase5(conn, only_new=False):
    """Compute recent-form rate / workload / driver-switch (optionally new-only)."""
    t0 = time.time()
    new_filter = " AND ef.entry_id IN (SELECT entry_id FROM _ef_new)" if only_new else ""
    with conn.cursor() as cur:
        cur.execute(f"""
            WITH hist AS (
                SELECT ef2.entry_id, ef2.horse_id, ef2.race_date, ef2.driver_id,
                       e.galopp::int AS gal_int
                FROM entry_features ef2
                JOIN entry e ON e.entry_id = ef2.entry_id
                WHERE NOT e.withdrawn
            ),
            calc AS (
                SELECT entry_id,
                       AVG(gal_int) OVER (
                           PARTITION BY horse_id ORDER BY race_date, entry_id
                           ROWS BETWEEN 10 PRECEDING AND 1 PRECEDING
                       )::real AS galrate_recent_10,
                       COUNT(*) OVER (
                           PARTITION BY horse_id ORDER BY race_date
                           RANGE BETWEEN INTERVAL '90 days' PRECEDING AND INTERVAL '1 day' PRECEDING
                       )::smallint AS starts_90d,
                       CASE WHEN LAG(driver_id) OVER (PARTITION BY horse_id ORDER BY race_date, entry_id) IS NULL
                            THEN NULL
                            ELSE driver_id IS DISTINCT FROM LAG(driver_id) OVER (PARTITION BY horse_id ORDER BY race_date, entry_id)
                       END AS driver_switch
                FROM hist
            )
            UPDATE entry_features ef SET
                galrate_recent_10 = calc.galrate_recent_10,
                starts_90d        = calc.starts_90d,
                driver_switch     = calc.driver_switch
            FROM calc WHERE calc.entry_id = ef.entry_id{new_filter}
        """)
    conn.commit()
    print(f"Phase 5 (recent rate + workload) complete in {time.time()-t0:.1f}s", flush=True)


def refresh_incremental(conn, log=print):
    """Add feature rows for newly-arrived Swedish-trot entries (>= START_DATE)
    that don't yet have a row. Their as-of features only depend on history
    strictly before their race date — which is already in `entry` — so a plain
    append is correct. Historical back-inserts / identity merges still warrant
    a periodic full rebuild (`--phase all`).

    Returns {"new": n}. Safe no-op when nothing is missing.
    """
    with conn.cursor() as cur:
        cur.execute("""
            SELECT COUNT(*)
            FROM entry e
            JOIN race r  ON r.race_id  = e.race_id
            JOIN track t ON t.track_id = r.track_id
            WHERE t.country = 'SE' AND e.race_date >= %s
              AND NOT EXISTS (SELECT 1 FROM entry_features ef WHERE ef.entry_id = e.entry_id)
        """, (START_DATE,))
        n_new = cur.fetchone()[0]
    if n_new == 0:
        log("entry_features: already up to date (0 new rows)")
        return {"new": 0}

    log(f"entry_features: {n_new:,} new entries — rebuilding staging and appending")
    _drop_staging(conn)
    try:
        # Snapshot the exact new entry_ids; every insert/update drives off this
        # small set so they're index lookups, not full-table scans.
        with conn.cursor() as cur:
            cur.execute("""
                CREATE UNLOGGED TABLE _ef_new AS
                SELECT e.entry_id
                FROM entry e
                JOIN race r  ON r.race_id  = e.race_id
                JOIN track t ON t.track_id = r.track_id
                WHERE t.country = 'SE' AND e.race_date >= %s
                  AND NOT EXISTS (SELECT 1 FROM entry_features ef WHERE ef.entry_id = e.entry_id)
            """, (START_DATE,))
            cur.execute("ALTER TABLE _ef_new ADD PRIMARY KEY (entry_id)")
        conn.commit()
        build_base(conn)
        build_horse(conn)
        build_horse_person(conn)
        _build_person_windows(conn, "trainer", "trainer_id")
        build_trainer_track(conn)
        _build_person_windows(conn, "driver", "driver_id")
        build_race(conn)
        assemble_phase1(conn, only_new=True)
        run_phase2(conn, only_new=True)
        _run_phase3(conn, only_new=True)
        _run_phase4(conn, only_new=True)
        _run_phase5(conn, only_new=True)
        log(f"entry_features: appended {n_new:,} rows")
        return {"new": n_new}
    finally:
        _drop_staging(conn)


def relabel_recent(conn, days: int | None = 90, log=print):
    """Re-derive the result-dependent columns for feature rows whose race ran in
    the last `days` days, writing back only the rows that actually changed.

    refresh_incremental() appends a feature row the moment an entry is scraped —
    for an upcoming race that is BEFORE any result exists, so its labels start
    out NULL/false. Nothing in the append path ever revisits that row once the
    race is run, so without this step the freshest (and most valuable) rows stay
    unlabeled forever. This recomputes every result-derived column straight from
    `entry`, using the exact same predicates as assemble_phase1 / build_race,
    guarded by IS DISTINCT FROM so unchanged rows are left untouched — that keeps
    both the write volume and the downstream publish delta (which rides on
    `computed_at`) minimal.

    X (as-of) features are intentionally NOT touched: they depend only on history
    strictly before the race and are already correct from the append. Pass
    days=None for a one-time full-table backfill.

    Returns {"changed": n}.
    """
    where_recent = "" if days is None else "AND ef.race_date >= CURRENT_DATE - %s::int"
    params: list = [] if days is None else [days]
    t0 = time.time()
    with conn.cursor() as cur:
        _run(cur, "relabel recent entry_features", f"""
            WITH tgt AS (
                SELECT ef.entry_id, ef.race_id
                FROM entry_features ef
                WHERE TRUE {where_recent}
            ),
            race_agg AS (
                SELECT e.race_id,
                       COUNT(*)                                               AS n_entries,
                       COUNT(*) FILTER (WHERE NOT COALESCE(e.withdrawn,false)) AS n_starters,
                       MAX(e.program_number)                                  AS max_pgm
                FROM entry e
                WHERE e.race_id IN (SELECT race_id FROM tgt)
                GROUP BY e.race_id
            ),
            tz AS (
                SELECT e.entry_id,
                       CASE WHEN NULLIF(e.time_seconds,0) > 0 THEN
                           (NULLIF(e.time_seconds,0) - AVG(NULLIF(e.time_seconds,0)) OVER pr)
                           / NULLIF(STDDEV_SAMP(NULLIF(e.time_seconds,0)) OVER pr, 0)
                       END AS y_time_z
                FROM entry e
                WHERE e.race_id IN (SELECT race_id FROM tgt)
                WINDOW pr AS (PARTITION BY e.race_id)
            )
            UPDATE entry_features ef SET
                starters      = ra.n_starters,
                race_complete = (ra.n_starters >= 3 AND (ra.max_pgm IS NULL OR ra.max_pgm <= ra.n_entries)),
                y_gal         = COALESCE(e.galopp, false),
                y_win         = (e.placement = 1 AND NOT COALESCE(e.disqualified,false)),
                y_top3        = (e.placement BETWEEN 1 AND 3 AND NOT COALESCE(e.disqualified,false)),
                y_placement   = e.placement,
                y_disq        = COALESCE(e.disqualified, false),
                y_time_s      = NULLIF(e.time_seconds, 0),
                y_time_z      = tz.y_time_z::real,
                computed_at   = NOW()
            FROM tgt
            JOIN entry e     ON e.entry_id = tgt.entry_id
            JOIN race_agg ra ON ra.race_id = e.race_id
            LEFT JOIN tz     ON tz.entry_id = e.entry_id
            WHERE ef.entry_id = tgt.entry_id
              AND (
                    ef.starters      IS DISTINCT FROM ra.n_starters
                 OR ef.race_complete IS DISTINCT FROM (ra.n_starters >= 3 AND (ra.max_pgm IS NULL OR ra.max_pgm <= ra.n_entries))
                 OR ef.y_gal         IS DISTINCT FROM COALESCE(e.galopp, false)
                 OR ef.y_win         IS DISTINCT FROM (e.placement = 1 AND NOT COALESCE(e.disqualified,false))
                 OR ef.y_top3        IS DISTINCT FROM (e.placement BETWEEN 1 AND 3 AND NOT COALESCE(e.disqualified,false))
                 OR ef.y_placement   IS DISTINCT FROM e.placement
                 OR ef.y_disq        IS DISTINCT FROM COALESCE(e.disqualified, false)
                 OR ef.y_time_s      IS DISTINCT FROM NULLIF(e.time_seconds, 0)
                 OR ef.y_time_z      IS DISTINCT FROM tz.y_time_z::real
              )
        """, params)
        changed = cur.rowcount
    conn.commit()
    log(f"entry_features: relabeled {changed:,} changed rows "
        f"(window={'all' if days is None else str(days)+'d'}) in {time.time()-t0:.1f}s")
    return {"changed": changed}


def backfill_galadj_winrate(conn):
    """One-time in-place backfill of horse_winrate_gal_adj / horse_winrate_gal_incl
    for EXISTING rows, without a full phase-1 rebuild (which would TRUNCATE the
    table and wipe phase 2-5 columns). New rows pick these up automatically via
    assemble_phase1's INSERT on the incremental path. Mirrors phase-2's
    in-place UPDATE off the _ef_horse staging table."""
    t0 = time.time()
    _drop_staging(conn)
    build_base(conn)
    build_horse(conn)
    print("Applying gal-adjusted win-rate UPDATE to entry_features...", flush=True)
    with conn.cursor() as cur:
        _run(cur, "update gal-adj winrates", """
            UPDATE entry_features ef SET
                horse_winrate_gal_adj  = (h.h_clean_wins::real / NULLIF(h.h_clean_starts,0)),
                horse_winrate_gal_incl = (h.h_nondq_wins::real / NULLIF(h.h_nondq_starts,0))
            FROM _ef_horse h
            WHERE h.entry_id = ef.entry_id
        """)
        print(f"  updated {cur.rowcount:,} rows", flush=True)
    conn.commit()
    print(f"Gal-adjusted win-rate backfill complete in {time.time()-t0:.1f}s", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--phase", choices=["1", "2", "all", "galadj", "relabel"], default="1")
    ap.add_argument("--days", type=int, default=None,
                    help="relabel window in days (relabel phase only; omit for "
                         "a full-table backfill)")
    ap.add_argument("--keep-staging", action="store_true",
                    help="don't drop the _ef_* staging tables on exit")
    args = ap.parse_args()

    conn = get_connection()
    try:
        if args.phase == "relabel":
            relabel_recent(conn, days=args.days)
        elif args.phase == "galadj":
            backfill_galadj_winrate(conn)
        else:
            if args.phase in ("1", "all"):
                run_phase1(conn)
            if args.phase in ("2", "all"):
                run_phase2(conn)
        if not args.keep_staging:
            _drop_staging(conn)
            print("Dropped staging tables.", flush=True)
    finally:
        conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
