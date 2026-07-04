"""Create (or rebuild) the /stats browse materialized views.

One-time builder for horse_stats / person_stats / track_post_stats. Each does a
full pass over `entry` so this takes a few minutes; thereafter the update job
keeps them fresh with REFRESH MATERIALIZED VIEW CONCURRENTLY.

    python -m scripts.build_stats_views
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from core.db import get_connection
from core.schema import STATS_VIEWS_DDL


def main() -> int:
    conn = get_connection()
    try:
        t0 = time.time()
        print("building horse_stats / person_stats / track_post_stats ...", flush=True)
        with conn.cursor() as cur:
            cur.execute(STATS_VIEWS_DDL)
        conn.commit()
        with conn.cursor() as cur:
            for mv in ("horse_stats", "person_stats", "track_post_stats"):
                cur.execute(f"SELECT COUNT(*) FROM {mv}")
                print(f"  {mv}: {cur.fetchone()[0]:,} rows", flush=True)
        print(f"done in {time.time() - t0:.1f}s", flush=True)
    finally:
        conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
