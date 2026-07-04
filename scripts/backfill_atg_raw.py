#!/usr/bin/env python3
"""
One-time ATG raw backfill.

Walks v1.v2_atg_race_raw and ingests each race into v2 master tables via
etl.import_atg. Designed for the "fill in foreign-race coverage" pass —
foreign tracks like Vincennes, Laval, Bjerke, Momarken end up properly
populated with all starters instead of just the 1-2 TravSport-licensed
horses.

Usage:
    # Quick dry-run on 100 most recent races:
    python3 -m scripts.backfill_atg_raw --since 2026 --limit 100

    # Foreign-only sweep for the last year (skips SE tracks):
    python3 -m scripts.backfill_atg_raw --since 2025 --foreign-only

    # Full backfill (~274k races, this is the big run):
    python3 -m scripts.backfill_atg_raw --execute

Flags:
    --since YYYY[-MM[-DD]]    only ingest races on/after this prefix
    --until YYYY[-MM[-DD]]    only ingest races strictly before this prefix
    --limit N                 cap on rows scanned (debug)
    --foreign-only            skip SE-track races (we already have these via ST)
    --batch N                 commit every N rows (default 200)
    --execute                 actually run (else: just print plan)
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from core.db import get_connection, get_v1_connection
from etl import import_atg


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--since", help="atg_race_id prefix lower bound (e.g. '2025-01')")
    ap.add_argument("--until", help="atg_race_id prefix upper bound (exclusive)")
    ap.add_argument("--limit", type=int, help="row cap for testing")
    ap.add_argument("--foreign-only", action="store_true",
                    help="skip races whose track country is SE")
    ap.add_argument("--batch", type=int, default=200,
                    help="rows per commit (default 200)")
    ap.add_argument("--progress-every", type=int, default=500,
                    help="stdout heartbeat cadence (default 500)")
    ap.add_argument("--execute", action="store_true",
                    help="actually run (else just preview)")
    args = ap.parse_args()

    print("=" * 70)
    print("ATG raw backfill")
    print("=" * 70)
    print(f"  since:         {args.since or '(none — start of table)'}")
    print(f"  until:         {args.until or '(none — end of table)'}")
    print(f"  limit:         {args.limit or '(no cap)'}")
    print(f"  foreign-only:  {args.foreign_only}")
    print(f"  batch:         {args.batch}")
    print(f"  EXECUTE:       {args.execute}")
    print()

    v1 = get_v1_connection()
    v2 = get_connection()

    try:
        # Preview the scope
        with v1.cursor() as c:
            where, params = [], []
            if args.since:
                where.append("atg_race_id >= %s"); params.append(args.since)
            if args.until:
                where.append("atg_race_id < %s"); params.append(args.until + "_zzz")
            wsql = " WHERE " + " AND ".join(where) if where else ""
            c.execute(f"SELECT COUNT(*) FROM v2_atg_race_raw{wsql}", params)
            total = c.fetchone()[0]
            scope = min(total, args.limit) if args.limit else total
            print(f"  scope:         {scope:,} of {total:,} races match")

        if not args.execute:
            print("\n(dry-run — re-run with --execute to ingest)")
            return 0

        t0 = time.time()
        totals = import_atg.backfill_from_v1_raw(
            v1, v2,
            since=args.since, until=args.until, limit=args.limit,
            only_foreign=args.foreign_only,
            batch_size=args.batch,
            progress_every=args.progress_every,
        )
        elapsed = time.time() - t0
        print()
        print("=" * 70)
        print(f"Done in {elapsed:.1f}s ({elapsed/60:.1f}min)")
        print("=" * 70)
        for k, v in totals.items():
            if isinstance(v, dict):
                print(f"  {k}:")
                for k2, v2_ in sorted(v.items(), key=lambda kv: -kv[1]):
                    print(f"    {k2:>20}: {v2_:>10,}")
            else:
                print(f"  {k:>12}: {v:>10,}")
    finally:
        v1.close()
        v2.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
