"""
Repair ATG races that show up in v2 with incomplete results.

Background
----------
Two distinct failure modes are healed:

  A) MISSING TIMES (stale v2, complete cache).
     ATG flips a race's detail status to 'results' the moment the podium is
     graded — often within minutes of the race — before every starter's
     `finishOrder`/`kmTime` exists. A scrape landing in that window stores a
     podium-only snapshot (no times). v1 *later* re-fetches and its raw cache
     (`v2_atg_race_raw`) becomes complete, but v2's incremental sync only
     re-ingests the last ~14 days, so older races stay frozen with no times
     (e.g. Solvalla 2026-06-25_5_2).
     Fix: if the v1 cache is now COMPLETE, re-ingest it into v2 (no network).
     If the cache is itself still a snapshot, optionally re-fetch LIVE ATG
     (`--live`) and ingest that. The completeness gate guarantees we never
     overwrite good ST times with an incomplete snapshot.

  B) MISSING POSTS (dropped postPosition).
     v2's importer historically never read `postPosition`, so ATG-primary
     races have NULL `entry.post`. The importer now fills it; existing rows
     are backfilled here with a surgical, NULL-only UPDATE from the cached
     postPosition (program_number → post). We do NOT re-ingest these from
     cache, because that would force-overwrite result columns and could
     clobber good ST times if the cache is sparse.

Scope
-----
Defaults to SE tracks (the Swedish trot domain that drives the app and ML).
Foreign cards (DK/AU/FI/US/NO/...) frequently lack km-times in ATG by nature,
so they are not "failures"; widen with --countries if you really want them.

Single-threaded, no concurrency. Cache re-ingest needs no network; only
--live re-fetches hit ATG (rate-limited by --sleep).

Usage
-----
    python -m scripts.refetch_incomplete_atg                       # dry-run, all SE history
    python -m scripts.refetch_incomplete_atg --execute             # apply (cache only)
    python -m scripts.refetch_incomplete_atg --execute --live      # + live re-fetch snapshots
    python -m scripts.refetch_incomplete_atg --countries SE,NO --execute
    python -m scripts.refetch_incomplete_atg --only 2026-06-25_5_2 --execute --live
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from datetime import date, timedelta
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from core.config import ATG_RACE_URL, ATG_HEADERS  # noqa: E402
from core.db import get_connection, get_v1_connection  # noqa: E402
from etl import import_atg  # noqa: E402


_CANDIDATE_SQL = """
WITH agg AS (
  SELECT ra.race_id, ra.atg_race_id, ra.race_date,
         count(*) FILTER (WHERE NOT COALESCE(e.withdrawn,false))           AS n_run,
         count(e.placement)                                               AS n_place,
         count(e.post)                                                    AS n_post,
         count(*) FILTER (WHERE NOT COALESCE(e.withdrawn,false)
                            AND NOT COALESCE(e.galopp,false)
                            AND NOT COALESCE(e.disqualified,false))        AS n_clean,
         count(*) FILTER (WHERE e.time_seconds IS NOT NULL
                            AND NOT COALESCE(e.withdrawn,false)
                            AND NOT COALESCE(e.galopp,false)
                            AND NOT COALESCE(e.disqualified,false))        AS n_clean_time
  FROM race ra
  JOIN entry e ON e.race_id = ra.race_id
  JOIN track t ON t.track_id = ra.track_id
  WHERE ra.atg_race_id IS NOT NULL
    AND ra.race_date < CURRENT_DATE
    AND ra.race_date >= %(since)s
    AND t.country = ANY(%(countries)s)
  GROUP BY ra.race_id, ra.atg_race_id, ra.race_date
)
SELECT race_id, atg_race_id, race_date,
       (n_place > 0 AND n_clean >= 2 AND n_clean_time = 0) AS missing_times,
       (n_run  >= 2 AND n_post = 0)                        AS missing_posts
FROM agg
WHERE (n_place > 0 AND n_clean >= 2 AND n_clean_time = 0)
   OR (n_run  >= 2 AND n_post = 0)
ORDER BY race_date DESC, atg_race_id DESC
"""


def find_candidates(v2_conn, *, since, countries, only, limit):
    if only:
        with v2_conn.cursor() as cur:
            cur.execute(
                "SELECT race_id, atg_race_id, race_date FROM race WHERE atg_race_id = %s",
                (only,),
            )
            row = cur.fetchone()
        return [(row[0], row[1], row[2], True, True)] if row else []
    with v2_conn.cursor() as cur:
        cur.execute(_CANDIDATE_SQL, {"since": since, "countries": countries})
        rows = cur.fetchall()
    return rows[:limit] if limit else rows


def read_cache(v1_conn, atg_race_id: str) -> dict | None:
    with v1_conn.cursor() as cur:
        cur.execute(
            "SELECT raw_json FROM v2_atg_race_raw WHERE atg_race_id = %s",
            (atg_race_id,),
        )
        row = cur.fetchone()
    return row[0] if row else None


def fetch_live_atg(atg_race_id: str, timeout: float = 15.0) -> dict | None:
    url = ATG_RACE_URL.format(atg_race_id=atg_race_id)
    req = urllib.request.Request(url, headers=ATG_HEADERS)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception as exc:  # noqa: BLE001
        print(f"    ! live fetch failed for {atg_race_id}: {exc}", flush=True)
        return None


def payload_is_complete(raw: dict | None) -> bool:
    """A fully graded result has `finishOrder` AND a `kmTime` on >=1 starter."""
    if not raw:
        return False
    starts = raw.get("starts") or []
    if len(starts) < 2:
        return False
    has_fo = any((s.get("result") or {}).get("finishOrder") is not None for s in starts)
    has_km = any((s.get("result") or {}).get("kmTime") is not None for s in starts)
    return has_fo and has_km


def fill_posts_from_cache(v1_conn, v2_conn, race_id: int, atg_race_id: str) -> int:
    raw = read_cache(v1_conn, atg_race_id)
    starts = (raw or {}).get("starts") or []
    pairs = [(s.get("number"), s.get("postPosition")) for s in starts
             if s.get("number") is not None and s.get("postPosition") is not None]
    if not pairs:
        return 0
    filled = 0
    with v2_conn.cursor() as cur:
        for prog, post in pairs:
            cur.execute(
                "UPDATE entry SET post = %s "
                " WHERE race_id = %s AND program_number = %s AND post IS NULL",
                (post, race_id, prog),
            )
            filled += cur.rowcount
    return filled


def reingest(v2_conn, atg_race_id: str, raw: dict) -> None:
    with v2_conn.cursor() as cur:
        import_atg.ingest_race(cur, atg_race_id, raw)
    v2_conn.commit()


def main() -> int:
    ap = argparse.ArgumentParser("refetch_incomplete_atg")
    ap.add_argument("--execute", action="store_true", help="apply (default: dry-run)")
    ap.add_argument("--days", type=int, default=None, help="lookback window in days")
    ap.add_argument("--since", type=str, default=None, help="start date YYYY-MM-DD")
    ap.add_argument("--countries", type=str, default="SE",
                    help="comma list of track countries (default SE)")
    ap.add_argument("--only", type=str, default=None, help="single atg_race_id")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--live", action="store_true",
                    help="re-fetch live ATG when the cache is still incomplete")
    ap.add_argument("--sleep", type=float, default=0.25, help="delay between live fetches")
    args = ap.parse_args()

    if args.since:
        since = args.since
    elif args.days:
        since = (date.today() - timedelta(days=args.days)).isoformat()
    else:
        since = "1900-01-01"
    countries = [c.strip().upper() for c in args.countries.split(",") if c.strip()]

    v2_conn = get_connection()
    v1_conn = get_v1_connection()

    cands = find_candidates(v2_conn, since=since, countries=countries,
                            only=args.only, limit=args.limit)
    n_times = sum(1 for c in cands if c[3])
    n_posts = sum(1 for c in cands if c[4] and not c[3])
    print(f"[refetch_incomplete_atg] since={since} countries={countries} "
          f"execute={args.execute} live={args.live}")
    print(f"  candidates: {len(cands)} (missing-times: {n_times}, posts-only: {n_posts})")

    from_cache = from_live = posts_races = posts_entries = skipped = 0

    for race_id, atg_race_id, race_date, missing_times, missing_posts in cands:
        if missing_times:
            if not args.execute:
                continue
            cached = read_cache(v1_conn, atg_race_id)
            if payload_is_complete(cached):
                reingest(v2_conn, atg_race_id, cached)
                from_cache += 1
                continue
            if args.live:
                live = fetch_live_atg(atg_race_id)
                time.sleep(args.sleep)
                if payload_is_complete(live):
                    reingest(v2_conn, atg_race_id, live)
                    from_live += 1
                    continue
            skipped += 1
        elif missing_posts:
            if not args.execute:
                continue
            n = fill_posts_from_cache(v1_conn, v2_conn, race_id, atg_race_id)
            if n:
                v2_conn.commit()
                posts_races += 1
                posts_entries += n

    print("\nSummary:")
    print(f"  times fixed from cache: {from_cache}")
    print(f"  times fixed from live:  {from_live}")
    print(f"  posts backfilled:       {posts_races} races ({posts_entries} entries)")
    print(f"  still incomplete:       {skipped}")
    if not args.execute:
        print("\nDRY-RUN — no DB writes. Use --execute to apply.")

    v1_conn.close()
    v2_conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
