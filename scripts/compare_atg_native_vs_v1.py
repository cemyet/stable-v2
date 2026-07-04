#!/usr/bin/env python3
"""
Validation: compare v2-native ATG scraping against v1's archive for a date range.

For each day, this fetches the race set v2's native scraper would ingest and
diffs it against what v1 stored in `v2_atg_race_raw`, so you can confirm the
native path captures the same races (and the same starter counts) before
retiring v1.

Usage:
    python3 -m scripts.compare_atg_native_vs_v1 2026-07-03
    python3 -m scripts.compare_atg_native_vs_v1 2026-07-01 2026-07-03
    python3 -m scripts.compare_atg_native_vs_v1 --last 7
"""

from __future__ import annotations

import argparse
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from core.db import get_v1_connection
from scrapers import atg as atg_scraper


def _v1_races_for_day(v1, day: str) -> dict[str, int]:
    """{atg_race_id: n_starts} from v1's raw archive for a day."""
    with v1.cursor() as cur:
        # Trot-only, to match the native scraper (v2 excludes gallop entirely).
        cur.execute(
            "SELECT atg_race_id, jsonb_array_length(COALESCE(raw_json->'starts','[]'::jsonb)) "
            "FROM v2_atg_race_raw "
            "WHERE atg_race_id LIKE %s "
            "  AND COALESCE(raw_json->>'sport','trot') <> 'gallop'",
            (day + "_%",),
        )
        return {r[0]: r[1] for r in cur.fetchall()}


def _native_races_for_day(client, day: str) -> dict[str, int]:
    """{atg_race_id: n_starts} the native scraper would ingest for a day."""
    cstatus, cal = atg_scraper.fetch_calendar(client, day)
    if cstatus != 200 or not isinstance(cal, dict):
        return {}
    out: dict[str, int] = {}
    for rid, _rdate in atg_scraper.enumerate_race_ids(cal):
        status, payload = atg_scraper.fetch_race(client, rid)
        if status == 200 and isinstance(payload, dict):
            out[rid] = len(payload.get("starts") or [])
    return out


def compare_day(v1, client, day: str) -> dict:
    v1_races = _v1_races_for_day(v1, day)
    native = _native_races_for_day(client, day)
    v1_ids, native_ids = set(v1_races), set(native)

    only_v1 = sorted(v1_ids - native_ids)
    only_native = sorted(native_ids - v1_ids)
    both = v1_ids & native_ids
    start_mismatch = {rid: (v1_races[rid], native[rid])
                      for rid in both if v1_races[rid] != native[rid]}

    print(f"\n=== {day} ===")
    print(f"  v1 races:     {len(v1_ids)}")
    print(f"  native races: {len(native_ids)}")
    print(f"  common:       {len(both)}")
    if only_v1:
        print(f"  ONLY in v1 ({len(only_v1)}): {only_v1[:8]}{' ...' if len(only_v1) > 8 else ''}")
    if only_native:
        print(f"  ONLY native ({len(only_native)}): {only_native[:8]}{' ...' if len(only_native) > 8 else ''}")
    if start_mismatch:
        print(f"  start-count mismatches ({len(start_mismatch)}):")
        for rid, (a, b) in list(start_mismatch.items())[:10]:
            print(f"    {rid}: v1={a} native={b}")
    if not only_v1 and not only_native and not start_mismatch:
        print("  ✓ identical race set + starter counts")
    return {"day": day, "v1": len(v1_ids), "native": len(native_ids),
            "only_v1": len(only_v1), "only_native": len(only_native),
            "start_mismatch": len(start_mismatch)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dates", nargs="*", help="YYYY-MM-DD (one, or two for a range)")
    ap.add_argument("--last", type=int, help="compare the last N days up to today")
    args = ap.parse_args()

    if args.last:
        end = date.today()
        days = [(end - timedelta(days=i)).isoformat() for i in range(args.last)]
    elif len(args.dates) == 2:
        d0 = datetime.strptime(args.dates[0], "%Y-%m-%d").date()
        d1 = datetime.strptime(args.dates[1], "%Y-%m-%d").date()
        days = [(d0 + timedelta(days=i)).isoformat() for i in range((d1 - d0).days + 1)]
    elif len(args.dates) == 1:
        days = [args.dates[0]]
    else:
        ap.error("give a date, a start+end date, or --last N")

    v1 = get_v1_connection()
    client = atg_scraper.make_client()
    try:
        results = [compare_day(v1, client, d) for d in days]
    finally:
        client.close()
        v1.close()

    tot_mismatch = sum(r["only_v1"] + r["only_native"] + r["start_mismatch"] for r in results)
    print(f"\nTotal discrepancies across {len(results)} day(s): {tot_mismatch}")
    return 0 if tot_mismatch == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
