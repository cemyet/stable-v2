"""Refresh every materialized view, in dependency order.

Needed after a large backfill or identity-merge campaign: the browse/stat
pages read these views, so until they are refreshed a healed horse still
shows its pre-heal numbers.

Order matters — horse_stats / person_stats / track_post_stats read from the
career views above them.

CONCURRENTLY keeps the views readable while the refresh runs, and cannot run
inside a transaction, hence autocommit. A refresh that is interrupted leaves
the previous contents intact, so this is safe to kill and re-run.

Usage:
    python -m scripts.refresh_all_matviews
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from core.db import get_connection  # noqa: E402

ORDER = [
    "horse_career_stats",
    "horse_year_stats",
    "person_career_stats",
    "track_stats",
    "horse_stats",
    "person_stats",
    "track_post_stats",
]


def main() -> int:
    conn = get_connection()
    conn.set_isolation_level(0)  # autocommit
    t0 = time.time()
    failed: list[str] = []
    for mv in ORDER:
        t = time.time()
        try:
            with conn.cursor() as cur:
                cur.execute(f"REFRESH MATERIALIZED VIEW CONCURRENTLY {mv}")
            print(f"[{time.strftime('%H:%M:%S')}] {mv:22} ok      "
                  f"{round(time.time() - t, 1)}s", flush=True)
        except Exception as exc:
            failed.append(mv)
            print(f"[{time.strftime('%H:%M:%S')}] {mv:22} FAILED  {exc!r}",
                  flush=True)
    print(f"ALL_MATVIEWS_DONE in {round(time.time() - t0, 1)}s "
          f"({len(ORDER) - len(failed)}/{len(ORDER)} ok"
          f"{', failed: ' + ','.join(failed) if failed else ''})", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
