#!/usr/bin/env python3
"""
One-time migration: copy v1's historical ATG raw archive into v2.

Reads v1's `v2_atg_race_raw` (atg_race_id, raw_json) and inserts the rows into
v2's own `atg_race_raw` table (with a derived race_date). After this runs, v2
owns the full ATG raw history locally and the v1 project can be retired — the
raw archive is only ever needed for offline re-derivation, never at runtime and
never in the cloud.

This copies the JSON blobs verbatim; it does NOT ingest into master tables (the
derived data already lives in v2). Run `scripts.backfill_atg_raw` or
`etl.import_atg.load_atg_from_raw` separately if you want to re-derive.

Usage:
    python3 -m scripts.migrate_atg_raw_from_v1 --execute
    python3 -m scripts.migrate_atg_raw_from_v1 --since 2025-01-01 --execute
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from psycopg2.extras import Json

from core.db import get_connection, get_v1_connection


def _race_date(atg_race_id: str):
    try:
        return datetime.strptime(atg_race_id.split("_", 1)[0], "%Y-%m-%d").date()
    except (ValueError, AttributeError):
        return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--since", help="atg_race_id prefix lower bound (e.g. '2025-01')")
    ap.add_argument("--until", help="atg_race_id prefix upper bound (exclusive)")
    ap.add_argument("--batch", type=int, default=1000, help="rows per commit")
    ap.add_argument("--skip-existing", action="store_true",
                    help="don't overwrite rows already present in v2.atg_race_raw")
    ap.add_argument("--execute", action="store_true", help="actually run")
    args = ap.parse_args()

    where, params = ["raw_json IS NOT NULL"], []
    if args.since:
        where.append("atg_race_id >= %s"); params.append(args.since)
    if args.until:
        where.append("atg_race_id < %s"); params.append(args.until + "_zzz")
    wsql = " AND ".join(where)

    v1 = get_v1_connection()
    v2 = get_connection()
    try:
        with v1.cursor() as c:
            c.execute(f"SELECT COUNT(*) FROM v2_atg_race_raw WHERE {wsql}", params)
            total = c.fetchone()[0]
        print(f"v1 rows matching: {total:,}")
        if not args.execute:
            print("(dry-run — re-run with --execute)")
            return 0

        conflict = ("ON CONFLICT (atg_race_id) DO NOTHING" if args.skip_existing
                    else """ON CONFLICT (atg_race_id) DO UPDATE
                            SET raw_json = EXCLUDED.raw_json,
                                race_date = EXCLUDED.race_date""")
        insert_sql = (
            "INSERT INTO atg_race_raw (atg_race_id, raw_json, race_date, scraped_at) "
            "VALUES (%s, %s, %s, NOW()) " + conflict
        )

        t0 = time.time()
        copied = 0
        with v1.cursor(name="v1_atg_raw_stream") as src:
            src.itersize = args.batch
            src.execute(
                f"SELECT atg_race_id, raw_json FROM v2_atg_race_raw WHERE {wsql} "
                f"ORDER BY atg_race_id", params)
            with v2.cursor() as dst:
                batch = 0
                for atg_race_id, raw in src:
                    dst.execute(insert_sql,
                                (atg_race_id, Json(raw), _race_date(atg_race_id)))
                    copied += 1
                    batch += 1
                    if batch >= args.batch:
                        v2.commit(); batch = 0
                    if copied % 20000 == 0:
                        print(f"  copied {copied:,}/{total:,} "
                              f"({copied/max(total,1)*100:.0f}%)  "
                              f"{copied/(time.time()-t0):.0f} rows/s")
                v2.commit()
        print(f"Done: copied {copied:,} rows in {time.time()-t0:.1f}s")
    finally:
        v1.close()
        v2.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
