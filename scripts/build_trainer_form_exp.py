"""
Build `trainer_form_exp`: candidate TRAINER-FORM features for the gold-standard
experiment, version 2. Nothing here touches `entry_features`.

Core idea (the user's "expected placement order" intuition)
-----------------------------------------------------------
Within each race we rank horses by their betting odds (1 = favorite). That rank
IS the market's expected finishing order. We then compare it to the ACTUAL
finishing order and measure how far each horse beat (or missed) its market
expectation:

    odds_rank      rank by odds (1 = shortest odds = favorite), among runners
                   that have odds + a finishing position
    fin_rank       rank by finish_order, on the SAME field (1 = winner)
    pos_beat       odds_rank - fin_rank      (>0 = finished ahead of market)
    pos_beat_norm  pos_beat / (n - 1)        (≈ -1..+1, comparable across fields)
    beat           fin_rank < odds_rank      (bool: outperformed the market)
    win_resid      is_win - implied_p        (implied_p = overround-removed 1/odds)

Why this kills the odds-noise problem
--------------------------------------
A 100/1 horse has odds_rank ≈ last, so finishing last gives pos_beat ≈ 0
(NEUTRAL — exactly as the market expected). It no longer drags the trainer's
form down. Outperformance is only credited when a horse finishes BETTER than
its market rank. This is the principled version of "odds-weighting the y".

Trainer-form variants (all strictly AS-OF, leak-free, SE, odds-gated)
---------------------------------------------------------------------
For several windows we aggregate the per-start outperformance of the trainer's
PRIOR starts:

  outperformance:  tf_posbeat_90 / _180 / _last50 / _last100
  beat rate:       tf_beatrate_90 / _180 / _last50
  win residual:    tf_winresid_90 / _180 / _last50
  plain (control): tf_top3_career, tf_top3_last50, tf_win_last50

Plus tf_n_180 / tf_n_last50 (recent start counts) for the min-starts gate.

Round 3 — PERFORMANCE family ("actual form", market-AGNOSTIC)
-------------------------------------------------------------
The pos_beat family is a market RESIDUAL: ~0 for a winning favourite and it
regresses to 0 as the market prices a hot stable in (a value/edge signal, not
"form"). This family answers the plain question "are the trainer's horses
finishing high lately?" over EVERY SE start (no odds gate → much more data),
counting a galopp/DQ/DNF as a bottom finish:

  finish percentile: tf_perf_30 / _90 / _180 / _last50 / _last100
                     tf_perf_decay120 / _decay365   (recency-weighted)
  win rate:          tf_winrate_30 / _90 / _last50
  top3 rate:         tf_top3rate_90

New experiment targets (stored per target entry)
------------------------------------------------
  y_beat_exp   bool   — did THIS horse finish ahead of its odds rank?
  y_outperf    real   — pos_beat_norm for THIS start (how far above expectation)
(plus y_win / y_top3 pulled from entry_features at eval time)

Population gates
----------------
  * SE tracks only (race location — NOT trainer nationality, which is buggy
    for merged trainers and unreliable).
  * Only starts with valid odds (odds > 1) AND a finish_order.
  * Only races with >= 4 such runners (ranking is meaningless in tiny fields).

Usage
-----
    python -m scripts.build_trainer_form_exp            # since 2012
    python -m scripts.build_trainer_form_exp --since 2010
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from core.db import get_connection  # noqa: E402

_QUALIFIER_RE = (
    r"^(gdk|ejg|ejp|gd|gk|egk|egdk|gkd|gdj|gdb|gdek|gdgk|gdl|ddk|frdk|erj|ejk|"
    r"ejgk|ejgd|ej|EJ|EJG|EJP|GDK|Gdk|g)[0-9]?$|^[12]?[pP][0-9]?$"
)

_MIN_FIELD = 4  # minimum runners-with-odds for the rank to be meaningful

_STAGING = ["_tfe_base", "_tfe_daily", "_tfe_win", "_tfe_lastn",
            "_tfp_base", "_tfp_daily", "_tfp_win", "_tfp_lastn"]


def _run(cur, label, sql, params=None):
    t = time.time()
    cur.execute(sql, params or ())
    print(f"  {label}: {time.time()-t:.1f}s", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser("build_trainer_form_exp")
    ap.add_argument("--since", type=int, default=2012,
                    help="first target year to materialise (default 2012)")
    args = ap.parse_args()

    conn = get_connection()
    cur = conn.cursor()

    print("[build_trainer_form_exp] dropping old staging + table…", flush=True)
    for t in _STAGING + ["trainer_form_exp"]:
        cur.execute(f"DROP TABLE IF EXISTS {t}")
    conn.commit()

    # ---- base: SE odds-starts with within-race ranks ---------------------
    print("Building _tfe_base (SE odds-starts, within-race ranks)…", flush=True)
    _run(cur, "_tfe_base", """
        CREATE UNLOGGED TABLE _tfe_base AS
        WITH ranked AS (
            SELECT e.entry_id,
                   e.trainer_id                          AS tid,
                   e.race_id,
                   e.race_date,
                   (e.race_date - DATE '2000-01-01')     AS day_num,
                   (e.placement = 1 AND NOT COALESCE(e.disqualified,false))::int       AS is_win,
                   (e.placement BETWEEN 1 AND 3 AND NOT COALESCE(e.disqualified,false))::int AS is_top3,
                   (1.0 / e.odds)                        AS imp_raw,
                   COUNT(*)    OVER w                     AS n_odds,
                   ROW_NUMBER() OVER (PARTITION BY e.race_id
                                      ORDER BY e.odds, e.program_number) AS odds_rank,
                   ROW_NUMBER() OVER (PARTITION BY e.race_id
                                      ORDER BY e.finish_order, e.program_number) AS fin_rank,
                   SUM(1.0 / e.odds) OVER w               AS imp_sum
            FROM entry e
            JOIN race r  ON r.race_id  = e.race_id
            JOIN track t ON t.track_id = r.track_id
            WHERE t.country = 'SE'
              AND e.trainer_id IS NOT NULL
              AND NOT COALESCE(e.withdrawn, false)
              AND e.odds IS NOT NULL AND e.odds > 1
              AND e.finish_order IS NOT NULL
              AND COALESCE(e.placement_text,'') !~ %s
            WINDOW w AS (PARTITION BY e.race_id)
        )
        SELECT entry_id, tid, race_date, day_num, is_win, is_top3, n_odds,
               odds_rank, fin_rank,
               ((odds_rank - fin_rank)::real / NULLIF(n_odds - 1, 0)) AS pos_beat_norm,
               (fin_rank < odds_rank)::int                            AS beat,
               (is_win - (imp_raw / NULLIF(imp_sum,0)))::real         AS win_resid
        FROM ranked
        WHERE n_odds >= %s
    """, (_QUALIFIER_RE, _MIN_FIELD))
    _run(cur, "idx base(tid,race_date)",
         "CREATE INDEX ON _tfe_base (tid, race_date)")
    cur.execute("SELECT count(*) FROM _tfe_base")
    print(f"  base rows: {cur.fetchone()[0]:,}", flush=True)
    conn.commit()

    # ---- daily rollup per trainer ----------------------------------------
    print("Building _tfe_daily…", flush=True)
    _run(cur, "_tfe_daily", """
        CREATE UNLOGGED TABLE _tfe_daily AS
        SELECT tid, race_date, MIN(day_num) AS day_num,
               COUNT(*)            AS d_n,
               SUM(pos_beat_norm)  AS d_posbeat,
               SUM(beat)           AS d_beat,
               SUM(win_resid)      AS d_winresid,
               SUM(is_top3)        AS d_top3,
               SUM(is_win)         AS d_win
        FROM _tfe_base
        GROUP BY tid, race_date
    """)
    conn.commit()

    # ---- day-based rolling windows (90d, 180d) + career ------------------
    print("Building _tfe_win (90d / 180d / career as-of)…", flush=True)
    _run(cur, "_tfe_win", """
        CREATE UNLOGGED TABLE _tfe_win AS
        SELECT tid, race_date,
            SUM(d_n)        OVER wc   AS c_n,
            SUM(d_top3)     OVER wc   AS c_top3,
            SUM(d_n)        OVER w90  AS n90,
            SUM(d_posbeat)  OVER w90  AS pb90,
            SUM(d_beat)     OVER w90  AS bt90,
            SUM(d_winresid) OVER w90  AS wr90,
            SUM(d_n)        OVER w180 AS n180,
            SUM(d_posbeat)  OVER w180 AS pb180,
            SUM(d_beat)     OVER w180 AS bt180,
            SUM(d_winresid) OVER w180 AS wr180
        FROM _tfe_daily
        WINDOW
            wc   AS (PARTITION BY tid ORDER BY race_date
                     ROWS  BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING),
            w90  AS (PARTITION BY tid ORDER BY day_num
                     RANGE BETWEEN 90  PRECEDING AND 1 PRECEDING),
            w180 AS (PARTITION BY tid ORDER BY day_num
                     RANGE BETWEEN 180 PRECEDING AND 1 PRECEDING)
    """)
    _run(cur, "pk win", "ALTER TABLE _tfe_win ADD PRIMARY KEY (tid, race_date)")
    conn.commit()

    # ---- last-N-starts rolling windows + round-2 creative variants -------
    # Round 2 additions (per analysis):
    #   * larger windows last150 / last200 (does more history help?)
    #   * exponential time-decay means (recency-weighted), half-lives 120/365d.
    #     weighted_mean = SUM(2^(day/H) * x) / SUM(2^(day/H)) over prior rows;
    #     the 2^(-D/H) reference factor cancels in the ratio, so cumulative
    #     absolute weights are numerically safe in double precision.
    print("Building _tfe_lastn (last 50/100/150/200 + decay)…", flush=True)
    _run(cur, "_tfe_lastn", """
        CREATE UNLOGGED TABLE _tfe_lastn AS
        SELECT entry_id,
            COUNT(*)           OVER w50  AS n50,
            SUM(pos_beat_norm) OVER w50  AS pb50,
            SUM(beat)          OVER w50  AS bt50,
            SUM(win_resid)     OVER w50  AS wr50,
            SUM(is_top3)       OVER w50  AS t3_50,
            SUM(is_win)        OVER w50  AS w_50,
            COUNT(*)           OVER w100 AS n100,
            SUM(pos_beat_norm) OVER w100 AS pb100,
            COUNT(*)           OVER w150 AS n150,
            SUM(pos_beat_norm) OVER w150 AS pb150,
            COUNT(*)           OVER w200 AS n200,
            SUM(pos_beat_norm) OVER w200 AS pb200,
            SUM(power(2.0, day_num::float8/120.0) * pos_beat_norm) OVER wall AS dA120,
            SUM(power(2.0, day_num::float8/120.0))                 OVER wall AS dB120,
            SUM(power(2.0, day_num::float8/365.0) * pos_beat_norm) OVER wall AS dA365,
            SUM(power(2.0, day_num::float8/365.0))                 OVER wall AS dB365
        FROM _tfe_base
        WINDOW
            w50  AS (PARTITION BY tid ORDER BY race_date, entry_id
                     ROWS BETWEEN 50  PRECEDING AND 1 PRECEDING),
            w100 AS (PARTITION BY tid ORDER BY race_date, entry_id
                     ROWS BETWEEN 100 PRECEDING AND 1 PRECEDING),
            w150 AS (PARTITION BY tid ORDER BY race_date, entry_id
                     ROWS BETWEEN 150 PRECEDING AND 1 PRECEDING),
            w200 AS (PARTITION BY tid ORDER BY race_date, entry_id
                     ROWS BETWEEN 200 PRECEDING AND 1 PRECEDING),
            wall AS (PARTITION BY tid ORDER BY race_date, entry_id
                     ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING)
    """)
    _run(cur, "pk lastn", "ALTER TABLE _tfe_lastn ADD PRIMARY KEY (entry_id)")
    conn.commit()

    # ======================================================================
    # PERFORMANCE FAMILY ("actual form") — market-AGNOSTIC, round 3
    # ----------------------------------------------------------------------
    # The pos_beat family is a market-RESIDUAL: it is structurally ~0 for a
    # winning favourite and regresses toward 0 as the market prices a hot
    # stable in. That makes it a value/edge signal, not "form". This family
    # answers the plain question "are this trainer's horses finishing high
    # lately?" using EVERY SE start (no odds gate → far more data), counting a
    # galopp / DQ / DNF as a bottom finish (poor form, not neutral).
    #
    #   perf  = (n_field - fin_rank) / (n_field - 1)   ∈ [0,1]; 1 = won
    #           non-finishers (galopp/DQ/no finish_order) → perf = 0
    #   is_win / is_top3 are placement-based (market-agnostic)
    # ======================================================================
    # NB: rank by `placement` (reliable classified finish), NOT finish_order,
    # which is often NULL even for clean finishers (esp. recent races).
    print("Building _tfp_base (ALL SE starts, finish percentile)…", flush=True)
    _run(cur, "_tfp_base", """
        CREATE UNLOGGED TABLE _tfp_base AS
        WITH ranked AS (
            SELECT e.entry_id,
                   e.trainer_id                      AS tid,
                   e.race_date,
                   (e.race_date - DATE '2000-01-01') AS day_num,
                   (e.placement = 1 AND NOT COALESCE(e.disqualified,false))::int       AS is_win,
                   (e.placement BETWEEN 1 AND 3 AND NOT COALESCE(e.disqualified,false))::int AS is_top3,
                   COUNT(*) OVER (PARTITION BY e.race_id)               AS n_field,
                   ROW_NUMBER() OVER (PARTITION BY e.race_id
                       ORDER BY e.placement, e.program_number) AS fin_rank
            FROM entry e
            JOIN race r  ON r.race_id  = e.race_id
            JOIN track t ON t.track_id = r.track_id
            WHERE t.country = 'SE'
              AND e.trainer_id IS NOT NULL
              AND NOT COALESCE(e.withdrawn, false)
              AND e.placement IS NOT NULL          -- classified finish
              AND COALESCE(e.placement_text,'') !~ %s
        )
        SELECT entry_id, tid, race_date, day_num, is_win, is_top3, n_field,
               (n_field - fin_rank)::real / (n_field - 1) AS perf
        FROM ranked
        WHERE n_field >= 4
    """, (_QUALIFIER_RE,))
    _run(cur, "idx tfp_base(tid,race_date)",
         "CREATE INDEX ON _tfp_base (tid, race_date)")
    cur.execute("SELECT count(*) FROM _tfp_base")
    print(f"  perf base rows: {cur.fetchone()[0]:,}", flush=True)
    conn.commit()

    print("Building _tfp_daily…", flush=True)
    _run(cur, "_tfp_daily", """
        CREATE UNLOGGED TABLE _tfp_daily AS
        SELECT tid, race_date, MIN(day_num) AS day_num,
               COUNT(*)     AS d_n,
               SUM(perf)    AS d_perf,
               SUM(is_win)  AS d_win,
               SUM(is_top3) AS d_top3
        FROM _tfp_base
        GROUP BY tid, race_date
    """)
    conn.commit()

    print("Building _tfp_win (30d / 90d / 180d as-of)…", flush=True)
    _run(cur, "_tfp_win", """
        CREATE UNLOGGED TABLE _tfp_win AS
        SELECT tid, race_date,
            SUM(d_n)    OVER w30  AS pn30,
            SUM(d_perf) OVER w30  AS pp30,
            SUM(d_win)  OVER w30  AS pw30,
            SUM(d_n)    OVER w90  AS pn90,
            SUM(d_perf) OVER w90  AS pp90,
            SUM(d_win)  OVER w90  AS pw90,
            SUM(d_top3) OVER w90  AS pt90,
            SUM(d_n)    OVER w180 AS pn180,
            SUM(d_perf) OVER w180 AS pp180
        FROM _tfp_daily
        WINDOW
            w30  AS (PARTITION BY tid ORDER BY day_num
                     RANGE BETWEEN 30  PRECEDING AND 1 PRECEDING),
            w90  AS (PARTITION BY tid ORDER BY day_num
                     RANGE BETWEEN 90  PRECEDING AND 1 PRECEDING),
            w180 AS (PARTITION BY tid ORDER BY day_num
                     RANGE BETWEEN 180 PRECEDING AND 1 PRECEDING)
    """)
    _run(cur, "pk tfp_win", "ALTER TABLE _tfp_win ADD PRIMARY KEY (tid, race_date)")
    conn.commit()

    print("Building _tfp_lastn (last 50/100 + decay 120/365)…", flush=True)
    _run(cur, "_tfp_lastn", """
        CREATE UNLOGGED TABLE _tfp_lastn AS
        SELECT entry_id,
            COUNT(*)    OVER w50  AS pn50,
            SUM(perf)   OVER w50  AS pp50,
            SUM(is_win) OVER w50  AS pw50,
            COUNT(*)    OVER w100 AS pn100,
            SUM(perf)   OVER w100 AS pp100,
            SUM(power(2.0, day_num::float8/120.0) * perf) OVER wall AS pdA120,
            SUM(power(2.0, day_num::float8/120.0))        OVER wall AS pdB120,
            SUM(power(2.0, day_num::float8/365.0) * perf) OVER wall AS pdA365,
            SUM(power(2.0, day_num::float8/365.0))        OVER wall AS pdB365
        FROM _tfp_base
        WINDOW
            w50  AS (PARTITION BY tid ORDER BY race_date, entry_id
                     ROWS BETWEEN 50  PRECEDING AND 1 PRECEDING),
            w100 AS (PARTITION BY tid ORDER BY race_date, entry_id
                     ROWS BETWEEN 100 PRECEDING AND 1 PRECEDING),
            wall AS (PARTITION BY tid ORDER BY race_date, entry_id
                     ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING)
    """)
    _run(cur, "pk tfp_lastn", "ALTER TABLE _tfp_lastn ADD PRIMARY KEY (entry_id)")
    conn.commit()

    # ---- final materialised table ----------------------------------------
    print(f"Materialising trainer_form_exp (targets since {args.since})…", flush=True)
    _run(cur, "trainer_form_exp", f"""
        CREATE TABLE trainer_form_exp AS
        SELECT b.entry_id,
            b.race_date,
            -- outperformance (odds-rank based)
            (w.pb90  / NULLIF(w.n90,0))    AS tf_posbeat_90,
            (w.pb180 / NULLIF(w.n180,0))   AS tf_posbeat_180,
            (ln.pb50  / NULLIF(ln.n50,0))  AS tf_posbeat_last50,
            (ln.pb100 / NULLIF(ln.n100,0)) AS tf_posbeat_last100,
            (ln.pb150 / NULLIF(ln.n150,0)) AS tf_posbeat_last150,
            (ln.pb200 / NULLIF(ln.n200,0)) AS tf_posbeat_last200,
            -- round-2: recency-weighted (exp decay) + sample-size shrinkage
            (ln.dA120 / NULLIF(ln.dB120,0)) AS tf_posbeat_decay120,
            (ln.dA365 / NULLIF(ln.dB365,0)) AS tf_posbeat_decay365,
            (ln.pb100 / NULLIF(ln.n100 + 30, 0)) AS tf_posbeat_last100_shrunk,
            -- beat rate
            (w.bt90::real  / NULLIF(w.n90,0))  AS tf_beatrate_90,
            (w.bt180::real / NULLIF(w.n180,0)) AS tf_beatrate_180,
            (ln.bt50::real / NULLIF(ln.n50,0)) AS tf_beatrate_last50,
            -- win residual (market-adjusted win)
            (w.wr90  / NULLIF(w.n90,0))    AS tf_winresid_90,
            (w.wr180 / NULLIF(w.n180,0))   AS tf_winresid_180,
            (ln.wr50 / NULLIF(ln.n50,0))   AS tf_winresid_last50,
            -- plain controls
            (w.c_top3::real / NULLIF(w.c_n,0))   AS tf_top3_career,
            (ln.t3_50::real / NULLIF(ln.n50,0))  AS tf_top3_last50,
            (ln.w_50::real  / NULLIF(ln.n50,0))  AS tf_win_last50,
            -- PERFORMANCE family ("actual form", market-agnostic, all starts)
            (pw.pp30  / NULLIF(pw.pn30,0))    AS tf_perf_30,
            (pw.pp90  / NULLIF(pw.pn90,0))    AS tf_perf_90,
            (pw.pp180 / NULLIF(pw.pn180,0))   AS tf_perf_180,
            (pl.pp50  / NULLIF(pl.pn50,0))    AS tf_perf_last50,
            (pl.pp100 / NULLIF(pl.pn100,0))   AS tf_perf_last100,
            (pl.pdA120 / NULLIF(pl.pdB120,0)) AS tf_perf_decay120,
            (pl.pdA365 / NULLIF(pl.pdB365,0)) AS tf_perf_decay365,
            (pw.pw30::real / NULLIF(pw.pn30,0))  AS tf_winrate_30,
            (pw.pw90::real / NULLIF(pw.pn90,0))  AS tf_winrate_90,
            (pl.pw50::real / NULLIF(pl.pn50,0))  AS tf_winrate_last50,
            (pw.pt90::real / NULLIF(pw.pn90,0))  AS tf_top3rate_90,
            -- gating counts
            COALESCE(w.n180,0)  AS tf_n_180,
            COALESCE(ln.n50,0)  AS tf_n_last50,
            COALESCE(pl.pn50,0) AS tf_perf_n_last50,
            -- new experiment targets (this start)
            b.beat::int        AS y_beat_exp,
            b.pos_beat_norm    AS y_outperf
        FROM _tfe_base b
        JOIN _tfe_win w  ON w.tid = b.tid AND w.race_date = b.race_date
        LEFT JOIN _tfe_lastn ln ON ln.entry_id = b.entry_id
        LEFT JOIN _tfp_win   pw ON pw.tid = b.tid AND pw.race_date = b.race_date
        LEFT JOIN _tfp_lastn pl ON pl.entry_id = b.entry_id
        WHERE b.race_date >= make_date(%s,1,1)
    """, (args.since,))
    _run(cur, "pk exp", "ALTER TABLE trainer_form_exp ADD PRIMARY KEY (entry_id)")
    conn.commit()

    cur.execute("SELECT count(*) FROM trainer_form_exp")
    print(f"trainer_form_exp rows: {cur.fetchone()[0]:,}", flush=True)

    print("Dropping staging…", flush=True)
    for t in _STAGING:
        cur.execute(f"DROP TABLE IF EXISTS {t}")
    conn.commit()
    conn.close()
    print("done.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
