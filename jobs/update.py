"""
stable-v2 update job.

  python3 -m jobs.update [--mode bridge|native] [--job-run-id N]

Modes
-----

bridge (default for now)
    Run v1's `python3 -m jobs.update` as a subprocess to keep v1 fresh
    (it still owns the live scrapers). Then call `etl.import_st`
    backfill helpers to mirror any changed v1 rows into stable_v2.

    This is the path the admin button calls today. It gives v2 the same
    update cadence as v1 with zero new scraping code.

native (placeholder)
    Will eventually run v2's own scrape→parse→UPSERT pipeline against
    each source. Currently logs "not implemented" and exits 0 so the
    admin button doesn't blow up if you flip the flag.

job_run logging
---------------

Same shape as v1: `job_run` row carries status, log, summary JSONB.
The v2 web/admin polls `job_run` for live updates. If `--job-run-id`
is given (web admin pre-creates the row), we attach to it; otherwise
a fresh row is created.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from core.db import get_connection, get_v1_connection  # noqa: E402
from etl import import_st  # noqa: E402


V1_PROJECT_ROOT = Path("/Users/jakob/Dev/stable")

# Cutover switch (Phase 3). When True, the bridge stops shelling out to v1's
# `--mode st` and instead owns ST ingestion natively via
# etl.import_st.run_native_st_recent: a recent-scoped, idempotent pipeline that
# (1) gap-fills blind-spot horses, (2) walks each track's nextRaceDayId chain
# forward from the latest known raceday to scrape only newly-published racedays
# (no historical drain), (3) ETLs those racedays, and (4) scrapes passports for
# genuinely-new starters. ATG stays on the v1 bridge regardless, so Swedish race
# *results* keep flowing even if ST scraping hiccups.
#
# This is fully reversible: set back to False to re-enable v1 `--mode st`
# (still runnable by hand too). Validated 2026-06: recent catch-up + idempotent
# re-runs (0 row deltas), steady-state run ~5s, minimal/cleanup-handled dup
# races. The v1 database is retained as a fallback (not deleted).
NATIVE_ST = True


# ---------------------------------------------------------------------------
# job_run helpers (mirrors v1)
# ---------------------------------------------------------------------------

def _start_run(conn, job_name: str = "update") -> int:
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO job_run (job_name, status, pid) VALUES (%s, 'running', %s) "
            "RETURNING job_run_id",
            (job_name, os.getpid()),
        )
        rid = cur.fetchone()[0]
    conn.commit()
    return rid


def _attach_run(conn, run_id: int) -> None:
    with conn.cursor() as cur:
        cur.execute(
            "UPDATE job_run SET status = 'running', pid = %s WHERE job_run_id = %s",
            (os.getpid(), run_id),
        )
    conn.commit()


def _log(conn, run_id: int, line: str) -> None:
    print(line, flush=True)
    try:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE job_run SET log = COALESCE(log, '') || %s WHERE job_run_id = %s",
                (line + "\n", run_id),
            )
        conn.commit()
    except Exception:
        pass


def _set_phase(conn, run_id: int, phase: str) -> None:
    """Record the current high-level phase so the admin UI can show a
    progress bar + phase label without parsing the free-text log. Also
    echoed into the log so the detail view keeps a timeline."""
    print(f"[phase] {phase}", flush=True)
    try:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE job_run SET phase = %s, "
                "log = COALESCE(log, '') || %s WHERE job_run_id = %s",
                (phase, f"[phase] {phase}\n", run_id),
            )
        conn.commit()
    except Exception:
        pass


def _merge_summary(conn, run_id: int, patch: dict) -> None:
    try:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE job_run "
                "SET summary = COALESCE(summary, '{}'::jsonb) || %s::jsonb "
                "WHERE job_run_id = %s",
                (json.dumps(patch, default=str), run_id),
            )
        conn.commit()
    except Exception:
        pass


def _finish(conn, run_id: int, status: str, summary_patch: dict | None = None) -> None:
    if summary_patch:
        _merge_summary(conn, run_id, summary_patch)
    with conn.cursor() as cur:
        cur.execute(
            "UPDATE job_run SET finished_at = NOW(), status = %s WHERE job_run_id = %s",
            (status, run_id),
        )
    conn.commit()


# ---------------------------------------------------------------------------
# Bridge mode
# ---------------------------------------------------------------------------

def run_bridge(conn, run_id: int) -> dict:
    """Run v1's update job, then mirror only the *recent* v1 changes to v2.

    The old approach re-imported ALL v1 rows every time (4M+ entries).
    Now we only pull entries whose race_date falls within the window v1
    just scraped, which is typically 3-10 days of data.
    """
    summary = {"mode": "bridge"}

    pre = _v2_counts(conn)
    summary["pre"] = pre

    _set_phase(conn, run_id, "Bridge — running v1 scrape (ATG + ST)")
    _log(conn, run_id, "[bridge] phase 1/2 — running v1 update (jakob db)")
    t0 = time.time()
    v1_summary = _run_v1_update(conn, run_id)
    summary["v1_update_seconds"] = round(time.time() - t0, 1)
    summary["v1_summary"] = v1_summary

    # Waterproofing: the v1 scrape is the *only* source of new ATG/ST races.
    # If it crashed (e.g. a schema-migration failure), there is nothing new to
    # sync and the rest of the bridge would happily reprocess stale raw cache,
    # making a broken run look successful. Fail loudly instead so the run is
    # marked failed and the breakage is visible immediately.
    v1_exit = v1_summary.get("exit_code")
    if v1_exit not in (0, None):
        modes = v1_summary.get("modes", {})
        failed_modes = [m for m, r in modes.items() if r.get("exit_code")]
        msg = (f"v1 update subprocess FAILED (exit_code={v1_exit}, "
               f"modes={failed_modes or '?'}); aborting bridge so the run is "
               f"not reported successful on stale data. Check the [v1:*] log "
               f"lines above for the traceback.")
        _log(conn, run_id, f"[bridge]   !! {msg}")
        raise RuntimeError(msg)

    _set_phase(conn, run_id, "Bridge — syncing recent v1 data into v2")
    _log(conn, run_id, "[bridge] phase 2/2 — incremental v1→v2 sync (recent races only)")
    t0 = time.time()
    v1_conn = get_v1_connection()
    try:
        counts = _incremental_sync(v1_conn, conn, run_id)
    finally:
        v1_conn.close()
    summary["refresh_seconds"] = round(time.time() - t0, 1)
    summary["refresh_counts"] = counts

    # Native ST. Closes v1's sequential-id-scan blind spot: SE-racing horses
    # with a numeric atg_id but no st_id (e.g. Giovaz) are scraped natively and
    # attached to their canonical row. When NATIVE_ST is off we only gap-fill
    # (v1 --mode st still owns racedays above); when on we run the full native
    # pipeline (racedays + starters) because v1 st no longer ran.
    _set_phase(conn, run_id, "Native ST — passports + racedays"
               if NATIVE_ST else "Native ST — gap-horse passports")
    _log(conn, run_id, f"[bridge] native ST (full_pipeline={NATIVE_ST})")
    t0 = time.time()
    _nlog = lambda m: _log(conn, run_id, f"[native-st]   {m}")
    try:
        if NATIVE_ST:
            # Recent-scoped pipeline: chain-walk forward from the latest known
            # raceday per track (no historical drain), ETL those racedays, and
            # scrape passports only for genuinely-new starters. See
            # import_st.run_native_st_recent.
            gap = import_st.run_native_st_recent(conn, log=_nlog)
        else:
            gap = import_st.gap_fill_st_horses(conn, log=_nlog)
    except Exception as exc:
        conn.rollback()
        gap = {"error": repr(exc)}
        _log(conn, run_id, f"[bridge]   ! native ST failed: {exc!r}\n{traceback.format_exc()}")
    summary["native_st"] = gap
    summary["native_st_seconds"] = round(time.time() - t0, 1)

    _set_phase(conn, run_id, "Bridge — refreshing career stats")
    _log(conn, run_id, "[bridge] refreshing career-stats materialized views (horse + person)...")
    t0 = time.time()
    _refresh_career_stats(conn)
    summary["career_stats_refresh_seconds"] = round(time.time() - t0, 1)

    post = _v2_counts(conn)
    summary["post"] = post
    summary["delta"] = {k: post[k] - pre[k] for k in pre}

    return summary


def _incremental_sync(v1_conn, v2_conn, run_id: int) -> dict:
    """Pull only recent v1 entries into v2 (last 14 days of race_date).

    This replaces the old full-table `backfill_from_v1` that re-imported
    everything every single time. We only need to sync data that v1's
    scraper just touched.
    """
    from datetime import date, timedelta
    from psycopg2.extras import execute_values

    SYNC_DAYS = 14
    cutoff = date.today() - timedelta(days=SYNC_DAYS)
    _log(v2_conn, run_id, f"[bridge]   sync window: race_date >= {cutoff}")

    with v2_conn.cursor() as cur:
        cur.execute("SELECT st_id, horse_id FROM horse WHERE st_id IS NOT NULL")
        horse_map = {r[0]: r[1] for r in cur.fetchall()}
        cur.execute("SELECT st_race_id, race_id FROM race WHERE st_race_id IS NOT NULL")
        race_st_map = {r[0]: r[1] for r in cur.fetchall()}
        cur.execute("SELECT atg_race_id, race_id FROM race WHERE atg_race_id IS NOT NULL")
        race_atg_map = {r[0]: r[1] for r in cur.fetchall()}
        cur.execute("SELECT st_id, person_id FROM person WHERE st_id IS NOT NULL")
        person_map = {r[0]: r[1] for r in cur.fetchall()}

    # --- Sync tracks (fast, small table) ---
    import_st.backfill_tracks(v1_conn, v2_conn)

    # --- Sync new horses/persons that appear in recent entries ---
    _log(v2_conn, run_id, "[bridge]   syncing new horses/persons from recent entries...")
    new_horses = 0
    new_persons = 0
    with v1_conn.cursor() as src, v2_conn.cursor() as dst:
        # Find horse IDs in recent entries
        src.execute("""
            SELECT DISTINCT e.horse_id
              FROM entry e
              JOIN race r ON r.race_id = e.race_id
              JOIN race_day rd ON rd.race_day_id = r.race_day_id
             WHERE rd.race_date >= %s
        """, (cutoff,))
        needed_horse_ids = [r[0] for r in src.fetchall() if r[0] not in horse_map]

        if needed_horse_ids:
            # Route every new horse through the central identity resolver,
            # which obeys the same protocol as the importers. This replaces
            # an older direct INSERT that referenced columns that no longer
            # exist in v2 (`birth_year`, `sex`, `breed`, `country`).
            from core.identity import resolve_horse  # local import: hot path only
            # Mirror the FULL ST passport field set (same columns as
            # import_st.backfill_horses_pass1) so recently-touched horses get
            # a rich record — reg/UELN, pedigree names + ids, scraped stats —
            # instead of the old 6-field shallow sync that left ST passports
            # half-empty and prone to duplicate creation.
            src.execute("""
                SELECT horse_id, name, date_of_birth, gender_code, color, breed_code,
                       registration_number, ueln_number, birth_country, bred_country,
                       registration_country, is_dead, is_guest_horse, has_offspring,
                       breed_index, inbreed_coefficient,
                       father_horse_id, father_name, mother_horse_id, mother_name,
                       scraped_starts, scraped_wins, scraped_prize_money_kr, scraped_record
                  FROM horse WHERE horse_id = ANY(%s)
            """, (needed_horse_ids,))
            for h in src.fetchall():
                (st_id, name, dob, gender, color, breed,
                 reg, ueln, bcountry, bred_c, reg_c,
                 is_dead, is_guest, has_off,
                 breed_idx, inbreed,
                 father_id, father_name, mother_id, mother_name,
                 sc_starts, sc_wins, sc_prize, sc_record) = h
                canonical = {
                    "name":          name,
                    "date_of_birth": dob,
                    "gender_code":   gender,
                    "color":         color,
                    "breed_code":    breed,
                    "birth_country": bcountry,
                    "bred_country":  bred_c,
                    "registration_country": reg_c,
                    "is_dead":       is_dead,
                    "is_guest_horse": is_guest,
                    "has_offspring": has_off,
                    "breed_index":   breed_idx,
                    "inbreed_coefficient": inbreed,
                    "sire_name":     father_name,
                    "dam_name":      mother_name,
                    "scraped_starts": sc_starts,
                    "scraped_wins":  sc_wins,
                    "scraped_prize_money_kr": sc_prize,
                    "scraped_record": sc_record,
                }
                try:
                    new_id = resolve_horse(
                        dst,
                        source="st",
                        source_id=st_id,
                        canonical_fields=canonical,
                        raw_payload={
                            "sync_via": "incremental_v1_bridge",
                            "father_horse_id": father_id,
                            "mother_horse_id": mother_id,
                        },
                        registration_number=reg,
                        ueln_number=ueln,
                    )
                except Exception as exc:
                    v2_conn.rollback()
                    _log(v2_conn, run_id,
                         f"[bridge]   ! resolve_horse st_id={st_id} failed: {exc!r}")
                    continue
                if new_id and st_id not in horse_map:
                    new_horses += 1
                horse_map[st_id] = new_id
            v2_conn.commit()

        # Find person IDs in recent entries
        src.execute("""
            SELECT DISTINCT pid FROM (
                SELECT driver_id AS pid FROM entry e
                  JOIN race r ON r.race_id = e.race_id
                  JOIN race_day rd ON rd.race_day_id = r.race_day_id
                 WHERE rd.race_date >= %s AND e.driver_id IS NOT NULL
                UNION
                SELECT trainer_id FROM entry e
                  JOIN race r ON r.race_id = e.race_id
                  JOIN race_day rd ON rd.race_day_id = r.race_day_id
                 WHERE rd.race_date >= %s AND e.trainer_id IS NOT NULL
            ) ep
        """, (cutoff, cutoff))
        needed_person_ids = [r[0] for r in src.fetchall() if r[0] not in person_map]

        if needed_person_ids:
            from core.identity import resolve_person  # local import: hot path only
            src.execute("""
                SELECT person_id, name, short_name
                  FROM person WHERE person_id = ANY(%s)
            """, (needed_person_ids,))
            for p in src.fetchall():
                st_id, display_name, short_name = p
                try:
                    new_id = resolve_person(
                        dst,
                        source="st",
                        source_id=st_id,
                        canonical_fields={"name": display_name, "short_name": short_name},
                        raw_payload={"sync_via": "incremental_v1_bridge"},
                    )
                except Exception as exc:
                    v2_conn.rollback()
                    _log(v2_conn, run_id,
                         f"[bridge]   ! resolve_person st_id={st_id} failed: {exc!r}")
                    continue
                if new_id and st_id not in person_map:
                    new_persons += 1
                person_map[st_id] = new_id
            v2_conn.commit()

    _log(v2_conn, run_id, f"[bridge]   new horses: {new_horses}, new persons: {new_persons}")

    # --- Sync recent races ---
    new_races = 0
    with v1_conn.cursor() as src, v2_conn.cursor() as dst:
        src.execute("""
            SELECT r.race_id, r.race_day_id, r.race_number, r.distance,
                   r.start_method, r.heading, r.proposition_text,
                   r.track_conditions, r.victory_margin, r.tempo_text,
                   r.total_prize_kr,
                   rd.race_date, rd.track_code
              FROM race r
              JOIN race_day rd ON rd.race_day_id = r.race_day_id
             WHERE rd.race_date >= %s
        """, (cutoff,))
        for row in src.fetchall():
            st_race_id = row[0]
            if st_race_id in race_st_map:
                continue
            race_date = row[11]
            track_code = row[12]
            dst.execute("SELECT track_id FROM track WHERE st_code = %s", (track_code,))
            trow = dst.fetchone()
            if not trow:
                continue
            track_id = trow[0]
            dst.execute("""
                INSERT INTO race (st_race_id, st_race_day_id, race_date, track_id,
                    race_number, distance, start_method, heading, proposition_text,
                    track_conditions, victory_margin, tempo_text, total_prize_kr,
                    primary_source)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,'st')
                ON CONFLICT (st_race_id) DO NOTHING
                RETURNING race_id
            """, (st_race_id, row[1], race_date, track_id,
                  row[2], row[3], row[4], row[5], row[6],
                  row[7], row[8], row[9], row[10]))
            ins = dst.fetchone()
            if ins:
                race_st_map[st_race_id] = ins[0]
                new_races += 1
        v2_conn.commit()
    _log(v2_conn, run_id, f"[bridge]   new races: {new_races}")

    # --- Sync recent entries ---
    new_entries = 0
    ENTRY_FIELDS = (
        "e.race_id, e.atg_id, e.horse_id, e.number, e.post, e.distance, "
        "e.placement, e.finish_order, e.placement_text, "
        "e.time_val, e.time_text, e.auto, e.gal, e.dq, e.withdrawn, "
        "e.odds, e.prize, "
        "e.driver_id, e.driver_changed, e.trainer_id, "
        "e.age, e.sex, e.earnings_pre, "
        "e.shoe_code, e.shoe_front_changed, e.shoe_back_changed, "
        "e.sulky, e.sulky_changed, e.source, e.tillagg"
    )
    INSERT_SQL = """
        INSERT INTO entry (
            race_id, horse_id, program_number, post, distance, tillagg,
            placement, finish_order, placement_text,
            withdrawn, galopp, disqualified,
            time_seconds, time_text, auto,
            odds, prize_kr,
            driver_id, driver_changed, trainer_id,
            age, sex, earnings_pre,
            shoe_code, shoe_front_changed, shoe_back_changed,
            sulky, sulky_changed,
            source_data, primary_source
        ) VALUES %s
        ON CONFLICT (race_id, horse_id) DO UPDATE SET
            placement = EXCLUDED.placement,
            finish_order = EXCLUDED.finish_order,
            placement_text = EXCLUDED.placement_text,
            withdrawn = EXCLUDED.withdrawn,
            galopp = EXCLUDED.galopp,
            disqualified = EXCLUDED.disqualified,
            time_seconds = EXCLUDED.time_seconds,
            time_text = EXCLUDED.time_text,
            odds = EXCLUDED.odds,
            prize_kr = EXCLUDED.prize_kr,
            -- Refresh people + sex so trainer switches (and the H/V/S sex
            -- normalization) propagate to already-ingested entries instead
            -- of sticking with whatever landed first.
            driver_id = COALESCE(EXCLUDED.driver_id, entry.driver_id),
            trainer_id = COALESCE(EXCLUDED.trainer_id, entry.trainer_id),
            sex = COALESCE(EXCLUDED.sex, entry.sex),
            last_updated_at = NOW()
    """

    with v1_conn.cursor(name="incr_entries") as src, v2_conn.cursor() as dst:
        src.execute(f"""
            SELECT {ENTRY_FIELDS}
              FROM entry e
              JOIN race r ON r.race_id = e.race_id
              JOIN race_day rd ON rd.race_day_id = r.race_day_id
             WHERE rd.race_date >= %s
        """, (cutoff,))
        batch = []
        for r in src:
            (st_race_id, atg_id, st_horse_id, number, post, distance,
             placement, finish_order, placement_text,
             time_val, time_text, auto, gal, dq, withdrawn,
             odds, prize,
             driver_st_id, driver_changed, trainer_st_id,
             age, sex, earnings_pre,
             shoe_code, shoe_front_changed, shoe_back_changed,
             sulky, sulky_changed, source, tillagg) = r

            # ST stores sex Swedish-lowercase (v/h/s); canonical entry.sex
            # is uppercase H/V/S.
            if isinstance(sex, str) and sex.lower() in ("v", "h", "s"):
                sex = sex.upper()

            race_id = race_st_map.get(st_race_id) if st_race_id else None
            if race_id is None and atg_id:
                race_id = race_atg_map.get(atg_id)
            if race_id is None:
                continue
            horse_id = horse_map.get(st_horse_id)
            if horse_id is None:
                continue

            driver_id = person_map.get(driver_st_id) if driver_st_id else None
            trainer_id = person_map.get(trainer_st_id) if trainer_st_id else None
            primary_source = source if source in ("st", "atg") else "st"
            source_data = '{"' + primary_source + '": {}}'

            batch.append((
                race_id, horse_id, number, post, distance, tillagg,
                placement, finish_order, placement_text,
                bool(withdrawn), bool(gal), bool(dq),
                time_val, time_text, auto,
                odds, prize or 0,
                driver_id, bool(driver_changed), trainer_id,
                age, sex, earnings_pre,
                shoe_code, shoe_front_changed, shoe_back_changed,
                sulky, sulky_changed,
                source_data, primary_source,
            ))
            if len(batch) >= 2000:
                execute_values(dst, INSERT_SQL, batch, page_size=1000)
                v2_conn.commit()
                new_entries += len(batch)
                batch = []
        if batch:
            execute_values(dst, INSERT_SQL, batch, page_size=1000)
            v2_conn.commit()
            new_entries += len(batch)
    _log(v2_conn, run_id, f"[bridge]   synced entries: {new_entries}")

    # --- Also ingest recent ATG raw data (covers races not yet in v1's race table) ---
    from etl import import_atg
    since_str = cutoff.isoformat()
    _log(v2_conn, run_id, f"[bridge]   importing ATG raw since {since_str}...")
    atg_counts = import_atg.backfill_from_v1_raw(
        v1_conn, v2_conn,
        since=since_str, only_foreign=False,
        batch_size=200, progress_every=9999,
    )
    atg_new = atg_counts.get("ingested", 0)
    _log(v2_conn, run_id, f"[bridge]   ATG raw: {atg_new} races ingested, "
         f"{atg_counts.get('skipped', 0)} skipped")

    # --- xLabs (kmtid) GPS sectionals. Pure enrichment; runs last so the
    # race + entry rows it attaches to are already up-to-date.
    #
    # We follow the homepage's hand-curated index (NOT the trailing 30-day
    # window) because:
    #   1. The index goes back ~17 months on V75 days
    #   2. Multi-day bundles (`elitloppet`) live ONLY on the index — their
    #      date-based URLs return 404
    #   3. Re-fetching all ~80 slugs is cheap (~45s, ~80 HTTP calls) and the
    #      ETL is fully idempotent
    from etl import import_kmtid
    _log(v2_conn, run_id, "[bridge]   importing xLabs (kmtid) GPS sectionals...")
    try:
        kmtid_counts = import_kmtid.import_from_index(v2_conn)
        _log(v2_conn, run_id,
             f"[bridge]   xLabs: {kmtid_counts['races_matched']} races, "
             f"{kmtid_counts['entries_matched']} entries enriched "
             f"({kmtid_counts['entries_skipped']} entry skips, "
             f"{kmtid_counts['slugs_with_data']}/{kmtid_counts['slugs_seen']} slugs)")
    except Exception as exc:
        v2_conn.rollback()
        kmtid_counts = {"error": repr(exc)}
        _log(v2_conn, run_id, f"[bridge]   ! xLabs import failed: {exc!r}")

    return {
        "new_races": new_races, "synced_entries": new_entries,
        "atg_raw_races": atg_new, "sync_days": SYNC_DAYS,
        "kmtid": kmtid_counts,
    }


def _refresh_career_stats(conn) -> None:
    """Rebuild the career-stats materialized views (horse + person), then
    append any new rows to the entry_features ML table.

    CONCURRENTLY allows reads while the refresh runs (requires the unique
    indexes which we create in the schema).
    """
    with conn.cursor() as cur:
        cur.execute("REFRESH MATERIALIZED VIEW CONCURRENTLY horse_career_stats")
        conn.commit()
        cur.execute("REFRESH MATERIALIZED VIEW CONCURRENTLY horse_year_stats")
        conn.commit()
        cur.execute("REFRESH MATERIALIZED VIEW CONCURRENTLY person_career_stats")
        conn.commit()
        try:
            cur.execute("REFRESH MATERIALIZED VIEW CONCURRENTLY track_stats")
            conn.commit()
        except Exception as exc:
            # track_stats is new; tolerate older DBs where it isn't created yet.
            conn.rollback()
            print(f"[track_stats] refresh skipped: {exc!r}", flush=True)
        # Browse-page stats views. horse_stats / person_stats build on
        # horse_career_stats (already refreshed above), so order matters.
        for mv in ("horse_stats", "person_stats", "track_post_stats"):
            try:
                cur.execute(f"REFRESH MATERIALIZED VIEW CONCURRENTLY {mv}")
                conn.commit()
            except Exception as exc:
                conn.rollback()
                print(f"[{mv}] refresh skipped: {exc!r}", flush=True)
    _refresh_entry_features(conn)


def _refresh_entry_features(conn) -> None:
    """Incrementally append new Swedish-trot entries to entry_features.

    Isolated so a failure here never aborts the surrounding update run (the
    feature table is derived/rebuildable). A periodic full rebuild
    (`python -m scripts.backfill_entry_features --phase all`) is recommended
    after historical back-fills or identity merges.
    """
    try:
        from scripts.backfill_entry_features import refresh_incremental
        refresh_incremental(conn)
    except Exception as exc:
        conn.rollback()
        print(f"[entry_features] incremental refresh failed: {exc!r}\n"
              f"{traceback.format_exc()}", flush=True)
    # s_form building block (per-entry market outperformance). Isolated for the
    # same reason — derived/rebuildable, must never abort the run.
    try:
        from scripts.refresh_entry_outperf import refresh_entry_outperf
        refresh_entry_outperf(conn)
    except Exception as exc:
        conn.rollback()
        print(f"[entry_outperf] incremental refresh failed: {exc!r}\n"
              f"{traceback.format_exc()}", flush=True)
    # `form` building block (per-entry finishing percentile). Same isolation:
    # derived/rebuildable, must never abort the run.
    try:
        from scripts.refresh_entry_perf import refresh_entry_perf
        refresh_entry_perf(conn)
    except Exception as exc:
        conn.rollback()
        print(f"[entry_perf] incremental refresh failed: {exc!r}\n"
              f"{traceback.format_exc()}", flush=True)
    _score_upcoming()


def _score_upcoming() -> None:
    """Refresh upcoming-race galopp-risk predictions (ml_prediction).

    Runs in the ML venv as a subprocess so this job's interpreter doesn't need
    the ML stack. Best-effort: a failure never aborts the update run."""
    try:
        import subprocess
        from pathlib import Path
        repo = Path(__file__).resolve().parent.parent
        venv_py = repo / '.venv-ml' / 'bin' / 'python'
        if not venv_py.exists():
            print('[score_upcoming] ML venv not found, skipping', flush=True)
            return
        subprocess.run([str(venv_py), '-m', 'scripts.score_upcoming'],
                       cwd=str(repo), timeout=600, check=False)
    except Exception as exc:
        print(f"[score_upcoming] failed: {exc!r}", flush=True)


def _v2_counts(conn) -> dict:
    """Snapshot row counts for the master tables."""
    out = {}
    with conn.cursor() as cur:
        for t in ("horse", "person", "track", "race", "entry"):
            cur.execute(f"SELECT COUNT(*) FROM {t}")
            out[t] = cur.fetchone()[0]
    return out


def _run_v1_update(conn, run_id: int) -> dict:
    """Subprocess-launch v1's update job for BOTH modes.

    v1 splits its work by `--mode`:
      * `atg` — scrapes ATG race results (the daily new races).
      * `st`  — re-scrapes `Ej färdigregistrerad` race rows (`refresh_stale`),
                imports new ST horse passports (`horses`) and new ST racedays
                (`racedays`), then runs the ST ETL.

    The v2 bridge used to call v1 with no `--mode`, which defaults to `atg`
    only — so ST horse passports + the "not yet fully registered" refresh
    silently stopped running, leaving shallow ST horse records. We now run
    BOTH modes so the ST passport pipeline is reconnected.

    stable-v2 only needs v1's scraper + normalized ETL side effects, so we
    keep `--skip-ml --skip-features` (ML/feature tables are v1-only).
    """
    if not (V1_PROJECT_ROOT / "jobs" / "update.py").exists():
        _log(conn, run_id, f"[bridge]   v1 update module not found at {V1_PROJECT_ROOT}; skipping")
        return {"skipped": True}

    # Phase 3 cutover: when NATIVE_ST is on, only ATG runs via the v1 bridge;
    # ST is owned by the native pipeline (invoked separately in run_bridge).
    v1_modes = ("atg",) if NATIVE_ST else ("atg", "st")

    results: dict = {}
    for mode in v1_modes:
        cmd = ["/usr/bin/python3", "-m", "jobs.update", "--mode", mode,
               "--skip-ml", "--skip-features"]
        _log(conn, run_id, f"[bridge]   $ cd {V1_PROJECT_ROOT} && {' '.join(cmd)}")
        proc = subprocess.run(
            cmd,
            cwd=str(V1_PROJECT_ROOT),
            capture_output=True,
            text=True,
        )
        last_lines = "\n".join(
            (proc.stdout + "\n" + proc.stderr).strip().splitlines()[-15:])
        for line in last_lines.splitlines():
            _log(conn, run_id, f"[v1:{mode}]   {line}")
        results[mode] = {
            "exit_code": proc.returncode,
            "tail": last_lines[-2000:],
        }
    return {
        "exit_code": max(r["exit_code"] for r in results.values()),
        "modes": results,
    }


# ---------------------------------------------------------------------------
# Native mode (placeholder)
# ---------------------------------------------------------------------------

def run_native(conn, run_id: int) -> dict:
    """Native ST pipeline standalone (scrape→parse→UPSERT, no v1 bridge).

    Owns ST only (ATG/letrot/kmtid native ingestion are separate future
    tasks). Bounded + idempotent; safe to run alongside the bridge.
    """
    summary = {"mode": "native"}
    pre = _v2_counts(conn)
    summary["pre"] = pre

    _set_phase(conn, run_id, "Native ST — full pipeline")
    _log(conn, run_id, "[native] running native ST pipeline (gap-fill + recent racedays + starters)")
    t0 = time.time()
    try:
        summary["native_st"] = import_st.run_native_st_recent(
            conn, log=lambda m: _log(conn, run_id, f"[native-st]   {m}"))
    except Exception as exc:
        conn.rollback()
        summary["native_st"] = {"error": repr(exc)}
        _log(conn, run_id, f"[native] FAILED: {exc!r}\n{traceback.format_exc()}")
    summary["native_seconds"] = round(time.time() - t0, 1)

    _set_phase(conn, run_id, "Native ST — refreshing career stats")
    _refresh_career_stats(conn)

    post = _v2_counts(conn)
    summary["post"] = post
    summary["delta"] = {k: post[k] - pre[k] for k in pre}
    return summary


# ---------------------------------------------------------------------------
# kmtid (xLabs GPS) — standalone mode used by the admin "update" button.
# Bridge mode also kicks off kmtid as part of its end-of-run enrichment;
# this mode is for "just refresh xLabs, leave ATG/ST alone".
# ---------------------------------------------------------------------------

def run_kmtid(conn, run_id: int) -> dict:
    from etl import import_kmtid

    _set_phase(conn, run_id, "xLabs — GPS sectionals")
    _log(conn, run_id, "[kmtid] walking kmtid.atgx.se homepage index...")
    t0 = time.time()
    try:
        counts = import_kmtid.import_from_index(conn)
    except Exception as exc:
        conn.rollback()
        _log(conn, run_id, f"[kmtid] FAILED: {exc!r}\n{traceback.format_exc()}")
        return {"mode": "kmtid", "error": repr(exc)}
    counts["seconds"] = round(time.time() - t0, 1)
    _log(
        conn, run_id,
        f"[kmtid] done in {counts['seconds']}s — "
        f"{counts['races_matched']} races, "
        f"{counts['entries_matched']} entries enriched "
        f"({counts['entries_skipped']} entry skips, "
        f"{counts['slugs_with_data']}/{counts['slugs_seen']} slugs)"
    )
    return {"mode": "kmtid", **counts}


# ---------------------------------------------------------------------------
# letrot (French trotting) — standalone mode used by the admin "update"
# button. Walks Le Trot's "yesterday" listing and imports every course it
# advertises. Daily volume is ~100-130 courses depending on the calendar.
# ---------------------------------------------------------------------------

def run_letrot(conn, run_id: int, when: str = "hier") -> dict:
    from etl import import_letrot

    _set_phase(conn, run_id, "LeTrot — French courses")
    _log(conn, run_id, f"[letrot] walking letrot.com /courses/{when} ...")
    t0 = time.time()
    try:
        counts = import_letrot.import_day(conn, when)
    except Exception as exc:
        conn.rollback()
        _log(conn, run_id, f"[letrot] FAILED: {exc!r}\n{traceback.format_exc()}")
        return {"mode": "letrot", "error": repr(exc)}
    counts["seconds"] = round(time.time() - t0, 1)
    _log(
        conn, run_id,
        f"[letrot] done in {counts['seconds']}s — "
        f"{counts['imported']}/{counts['scraped']} courses imported "
        f"({counts['skipped']} skipped), "
        f"{counts['horses_upserted']} horses, "
        f"{counts['entries_upserted']} entries"
    )

    # Inline pedigree enrichment for newly-created LeTrot horses: fetch their
    # identity pages so sire/dam (and gender_code) are populated. This lets
    # the next cleanup pass triangulate pedigree and merge fewer duplicates.
    counts["pedigree"] = _run_letrot_pedigree(conn, run_id)
    return {"mode": "letrot", **counts}


def _run_letrot_pedigree(conn, run_id: int, *, limit: int = 800) -> dict:
    """Spawn the LeTrot identity-page scraper for horses that still have no
    `identity_fetched_at` marker (i.e. newly-created LeTrot horses). Bounded
    by `limit` so a first-run backlog doesn't stall the daily pipeline — the
    next run picks up where this left off."""
    _set_phase(conn, run_id, "LeTrot — pedigree identity for new horses")
    cmd = [
        sys.executable, "-u", "-m", "scripts.scrape_letrot_pedigree",
        "--execute", "--limit", str(limit), "--workers", "5",
    ]
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONPATH"] = str(_ROOT) + (
        os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
    )
    _log(conn, run_id, f"[letrot] pedigree: $ {' '.join(cmd)}")
    t0 = time.time()
    try:
        proc = subprocess.Popen(
            cmd, cwd=str(_ROOT),
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            env=env, text=True, bufsize=1, close_fds=True,
        )
    except Exception as exc:
        _log(conn, run_id, f"[letrot] pedigree FAILED to spawn: {exc!r}")
        return {"error": repr(exc)}
    assert proc.stdout is not None
    for raw in proc.stdout:
        _log(conn, run_id, raw.rstrip("\n"))
    rc = proc.wait()
    secs = round(time.time() - t0, 1)
    _log(conn, run_id, f"[letrot] pedigree done in {secs}s — exit_code={rc}")
    return {"exit_code": rc, "seconds": secs}


# ---------------------------------------------------------------------------
# cleanup — runs all post-ingest dedup/merge scripts via
# scripts.cleanup_merges. Streams the child's stdout into our job_run.log
# so progress shows in real time in the admin UI.
# ---------------------------------------------------------------------------

def run_cleanup(conn, run_id: int, *, execute: bool = True,
                abort_on_error: bool = False) -> dict:
    cmd = [
        sys.executable, "-u", "-m", "scripts.cleanup_merges",
        "--job-run-id", str(run_id),
    ]
    if execute:
        cmd.append("--execute")
    if abort_on_error:
        cmd.append("--abort-on-error")

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    _set_phase(conn, run_id, "Cleanup — dedup + heal")
    _log(conn, run_id,
         f"[cleanup] running post-ingest dedup pipeline (execute={execute}, "
         f"abort_on_error={abort_on_error})")
    _log(conn, run_id, f"[cleanup] $ {' '.join(cmd)}")

    t0 = time.time()
    summary: dict = {"mode": "cleanup", "execute": execute}
    try:
        proc = subprocess.Popen(
            cmd, cwd=str(_ROOT),
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            env=env, text=True, bufsize=1, close_fds=True,
        )
    except Exception as exc:
        _log(conn, run_id, f"[cleanup] FAILED to spawn: {exc!r}")
        summary["error"] = repr(exc)
        summary["seconds"] = round(time.time() - t0, 1)
        return summary

    assert proc.stdout is not None
    for raw in proc.stdout:
        _log(conn, run_id, raw.rstrip("\n"))
    rc = proc.wait()
    summary["exit_code"] = rc
    summary["seconds"] = round(time.time() - t0, 1)
    _log(conn, run_id,
         f"[cleanup] done in {summary['seconds']}s — exit_code={rc}")
    return summary


# ---------------------------------------------------------------------------
# all — bridge (or st/atg) + letrot + cleanup, in one streamed run.
# This is what the "Update all" admin button triggers.
# ---------------------------------------------------------------------------

def run_all(conn, run_id: int, *, abort_on_error: bool = False) -> dict:
    summary = {"mode": "all", "phases": []}

    _set_phase(conn, run_id, "1/3 Bridge (v1 + ST/ATG + kmtid)")
    _log(conn, run_id, "\n[all] === phase 1/3 — bridge (v1 + ST/ATG + kmtid) ===")
    try:
        s_bridge = run_bridge(conn, run_id)
    except Exception as exc:
        _log(conn, run_id,
             f"[all] bridge FAILED: {exc!r}\n{traceback.format_exc()}")
        s_bridge = {"mode": "bridge", "error": repr(exc)}
    summary["phases"].append({"phase": "bridge", "result": s_bridge})
    if abort_on_error and s_bridge.get("error"):
        _log(conn, run_id, "[all] aborting — bridge phase failed")
        summary["aborted_at"] = "bridge"
        return summary

    _set_phase(conn, run_id, "2/3 LeTrot (yesterday's courses)")
    _log(conn, run_id, "\n[all] === phase 2/3 — letrot (yesterday's courses) ===")
    try:
        s_letrot = run_letrot(conn, run_id)
    except Exception as exc:
        _log(conn, run_id,
             f"[all] letrot FAILED: {exc!r}\n{traceback.format_exc()}")
        s_letrot = {"mode": "letrot", "error": repr(exc)}
    summary["phases"].append({"phase": "letrot", "result": s_letrot})
    if abort_on_error and s_letrot.get("error"):
        _log(conn, run_id, "[all] aborting — letrot phase failed")
        summary["aborted_at"] = "letrot"
        return summary

    _set_phase(conn, run_id, "3/3 Cleanup (dedup + heal)")
    _log(conn, run_id, "\n[all] === phase 3/3 — cleanup (dedup + heal) ===")
    s_cleanup = run_cleanup(conn, run_id, execute=True,
                            abort_on_error=abort_on_error)
    summary["phases"].append({"phase": "cleanup", "result": s_cleanup})

    # Waterproofing: surface any phase failure as a failed run. Without this
    # the run completes and main() marks it 'success' even though e.g. the
    # bridge (the source of new races) crashed — exactly the silent-stale-data
    # failure mode we want to make impossible.
    failed = [
        p["phase"] for p in summary["phases"]
        if isinstance(p.get("result"), dict) and p["result"].get("error")
    ]
    if failed:
        msg = f"update phase(s) failed: {', '.join(failed)}"
        _log(conn, run_id, f"[all] !! {msg}")
        raise RuntimeError(msg)

    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    # Accept v1's "atg" / "st" aliases too so the v2 admin UI works
    # without pre-coordination.
    ap.add_argument(
        "--mode",
        choices=("bridge", "native", "atg", "st", "kmtid", "letrot",
                 "cleanup", "all"),
        default="bridge",
    )
    ap.add_argument("--job-run-id", type=int, default=None,
                    help="Attach to a job_run row pre-created by the admin web ui")
    ap.add_argument("--abort-on-error", action="store_true",
                    help="Used by --mode all / --mode cleanup: stop the "
                         "pipeline on the first failed phase instead of "
                         "continuing.")
    args = ap.parse_args()

    # Standalone modes route directly; everything else falls back to bridge.
    if args.mode in ("kmtid", "letrot", "cleanup", "all"):
        mode = args.mode
    elif args.mode in ("atg", "st", "bridge"):
        mode = "bridge"
    else:
        mode = "native"

    conn = get_connection()
    if args.job_run_id is not None:
        run_id = args.job_run_id
        _attach_run(conn, run_id)
    else:
        job_name = (
            f"update_{mode}" if mode in ("kmtid", "letrot", "cleanup", "all")
            else "update"
        )
        run_id = _start_run(conn, job_name=job_name)
    _log(conn, run_id, f"[update] start  mode={mode} (cli={args.mode})  run_id={run_id}  pid={os.getpid()}  at={datetime.now().isoformat(timespec='seconds')}")

    try:
        if mode == "bridge":
            summary = run_bridge(conn, run_id)
        elif mode == "kmtid":
            summary = run_kmtid(conn, run_id)
        elif mode == "letrot":
            summary = run_letrot(conn, run_id)
        elif mode == "cleanup":
            summary = run_cleanup(conn, run_id, execute=True,
                                  abort_on_error=args.abort_on_error)
        elif mode == "all":
            summary = run_all(conn, run_id,
                              abort_on_error=args.abort_on_error)
        else:
            summary = run_native(conn, run_id)
    except Exception as exc:
        _log(conn, run_id, f"[update] FAILED: {exc!r}\n{traceback.format_exc()}")
        _set_phase(conn, run_id, "Failed")
        _finish(conn, run_id, "failed", {"error": repr(exc)})
        return 1

    _set_phase(conn, run_id, "Done")
    _log(conn, run_id, "[update] done.")
    _finish(conn, run_id, "success", summary)
    return 0


if __name__ == "__main__":
    sys.exit(main())
