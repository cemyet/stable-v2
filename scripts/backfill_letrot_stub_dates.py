"""
Fetch the French courses behind our "Frankrike" placeholder races.

ST records a Swedish horse's foreign start without naming the venue — the race
lands on a synthetic track literally called "Frankrike". Those rows are honest
but useless for anything track-level, and there are ~9.5k of them.

They are not, however, a gap in Le Trot's coverage. On a sampled date Le Trot
advertises 124 courses, we already hold 113, and there are exactly 11 stubs; the
same arithmetic holds within a course or two on every date checked. The stubs
*are* the courses we never fetched. So this walks the dates that carry stubs,
imports only the courses missing from each, and leaves the rest alone.

Repointing the stubs is deliberately not done here. Once the real course exists,
`scripts.match_stub_races` folds the stub into it using evidence this script has
no business second-guessing. Run that afterwards.

Pre-2010 stubs (~3.2k) are out of reach — Le Trot's listing does not go back
that far — so the floor is 2010-01-01.

    python -m scripts.backfill_letrot_stub_dates --dry-run
    python -m scripts.backfill_letrot_stub_dates --execute --limit-dates 50
    python -m scripts.backfill_letrot_stub_dates --execute
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import date as Date
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from core.db import get_connection  # noqa: E402
from etl.import_letrot import (  # noqa: E402
    BackfillAborted,
    BackfillWatchdog,
    import_date_range,
)

log = logging.getLogger("backfill_letrot_stub_dates")

#: Le Trot's per-date listing does not reach further back than this.
LETROT_FLOOR = Date(2010, 1, 1)


def stub_dates(conn, *, floor: Date = LETROT_FLOOR) -> list[tuple[Date, int]]:
    """Dates carrying "Frankrike" placeholder races, worst first is not useful
    here — chronological keeps the FX prefetch warm one year at a time."""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT r.race_date, COUNT(*) AS stubs
              FROM race r
              JOIN track t ON t.track_id = r.track_id
             WHERE t.name = 'Frankrike'
               AND r.race_date >= %s
               AND r.race_date < CURRENT_DATE
             GROUP BY 1
             ORDER BY 1
            """,
            (floor,),
        )
        return [(d, n) for d, n in cur.fetchall()]


def main() -> int:
    p = argparse.ArgumentParser(prog="backfill_letrot_stub_dates")
    p.add_argument("--execute", action="store_true",
                   help="actually scrape and import (default is dry-run)")
    p.add_argument("--dry-run", action="store_true", help="explicit dry-run")
    p.add_argument("--limit-dates", type=int, default=None,
                   help="process only the first N stub dates")
    p.add_argument("--since", type=str, default=None,
                   help="ignore stub dates before this ISO date")
    p.add_argument("--progress-every", type=int, default=10)
    p.add_argument("--heartbeat", type=str,
                   default="logs/letrot_stub_backfill_heartbeat.txt")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    conn = get_connection()
    floor = Date.fromisoformat(args.since) if args.since else LETROT_FLOOR
    dates = stub_dates(conn, floor=floor)
    if args.limit_dates:
        dates = dates[: args.limit_dates]
    if not dates:
        log.info("no stub dates to process")
        return 0

    total_stubs = sum(n for _, n in dates)
    log.info("%d stub dates, %d placeholder races, %s .. %s",
             len(dates), total_stubs, dates[0][0], dates[-1][0])

    if not args.execute:
        log.info("DRY-RUN — no HTTP, no writes. First 10 dates:")
        for d, n in dates[:10]:
            log.info("   %s  stubs=%d", d, n)
        log.info("estimate: ~%d listing requests + ~%d course fetches",
                 len(dates), total_stubs)
        return 0

    watchdog = BackfillWatchdog(log_fn=log.warning)
    try:
        summary = import_date_range(
            conn,
            dates[0][0],
            dates[-1][0],
            only_dates={d for d, _ in dates},
            skip_if_present=True,
            skip_scope="course",
            progress_every=args.progress_every,
            watchdog=watchdog,
            heartbeat_path=args.heartbeat,
        )
    except BackfillAborted as exc:
        log.error("aborted by watchdog: %s", exc)
        return 2

    log.info("done: %s", summary)
    if summary.get("aborted"):
        log.error("watchdog aborted: %s", summary["aborted"])
        return 2
    log.info("next: python -m scripts.match_stub_races --execute")
    return 0


if __name__ == "__main__":
    sys.exit(main())
