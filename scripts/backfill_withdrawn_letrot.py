"""One-shot backfix for entries wrongly marked `withdrawn=True` because of
LeTrot's overloaded `rang=0` representation.

Background
----------

LeTrot's results table (`ARRIVÉE`) uses `rang=0` for three different
real-world cases:

  1. Truly scratched (didn't start)
  2. Did-not-finish / non-placing finisher
  3. Post-race disqualification (`DI` / `DA` — *Distancé* / *Distancé
     arrivée* for irregular trotting)

The previous ETL collapsed all three into `entry.withdrawn = True`. The
column-level merge in `core.identity._ENTRY_OR_COLS` then OR-merged this
into the canonical row, overwriting ATG's correct `withdrawn = False`
whenever the same horse+race existed on both sources.

This script reverses that mistake for every entry where:
  - `withdrawn = True`
  - `time_seconds IS NOT NULL`  (you can't both be scratched and post a
    finish time — the time itself is proof the horse ran)
  - LeTrot is one of the contributing sources

`disqualified` is set to True only for the subset where the LeTrot block
explicitly carries a `DI` / `DA` km-time marker (post-race DQ).

The ETL change in `etl/import_letrot.py` prevents the bug from
re-occurring on future imports, so this script only needs to run once.

Audit / rollback
----------------

Every row that's updated is appended to a backup CSV with its previous
`withdrawn` and `disqualified` values, so the operation is fully
reversible. See `--backup-csv` (default
`logs/withdrawn_backfix_<ts>.csv`).

Usage
-----

    python -m scripts.backfill_withdrawn_letrot                  # dry-run
    python -m scripts.backfill_withdrawn_letrot --execute
    python -m scripts.backfill_withdrawn_letrot --execute \\
        --backup-csv logs/withdrawn_backfix.csv
    python -m scripts.backfill_withdrawn_letrot --rollback-csv logs/withdrawn_backfix.csv
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from core.db import get_connection  # noqa: E402

log = logging.getLogger("backfill_withdrawn_letrot")

_SELECT_SQL = """
    SELECT e.entry_id,
           e.withdrawn,
           e.disqualified,
           (e.source_data->'letrot'->>'km_time_text') AS letrot_km
      FROM entry e
     WHERE e.withdrawn
       AND e.time_seconds IS NOT NULL
       AND e.source_data ? 'letrot'
"""

_UPDATE_SQL = """
    UPDATE entry
       SET withdrawn = FALSE,
           disqualified = %s,
           last_updated_at = NOW()
     WHERE entry_id = %s
"""


def run(dry_run: bool, backup_csv: Path | None, batch: int) -> int:
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(_SELECT_SQL)
            rows = cur.fetchall()

        log.info("found %d candidate rows (withdrawn=TRUE AND time_seconds IS NOT NULL "
                 "AND LeTrot contributor)", len(rows))
        if not rows:
            return 0

        dq_count = sum(1 for _, _, _, km in rows if km in ("DI", "DA"))
        log.info("  of those, %d carry an explicit LeTrot DI/DA marker → "
                 "disqualified=TRUE", dq_count)

        if dry_run:
            log.info("dry-run: no DB writes. First 5 rows that would change:")
            for entry_id, w_old, dq_old, km in rows[:5]:
                new_dq = (dq_old or (km in ("DI", "DA")))
                log.info(
                    "  entry_id=%d  withdrawn %s→FALSE  disqualified %s→%s "
                    "(letrot km_text=%r)",
                    entry_id, w_old, dq_old, new_dq, km,
                )
            return 0

        if backup_csv:
            backup_csv.parent.mkdir(parents=True, exist_ok=True)
            new_file = not backup_csv.exists()
            with backup_csv.open("a", newline="") as f:
                w = csv.writer(f)
                if new_file:
                    w.writerow(["entry_id", "old_withdrawn",
                                "old_disqualified", "new_disqualified",
                                "letrot_km_text", "backed_up_at"])
                stamp = datetime.utcnow().isoformat()
                for entry_id, w_old, dq_old, km in rows:
                    new_dq = (dq_old or (km in ("DI", "DA")))
                    w.writerow([entry_id, w_old, dq_old, new_dq, km, stamp])
            log.info("backup written: %s (%d rows)", backup_csv, len(rows))

        with conn.cursor() as cur:
            t0 = time.time()
            for i in range(0, len(rows), batch):
                chunk = rows[i:i + batch]
                params = [
                    ((dq_old or (km in ("DI", "DA"))), entry_id)
                    for entry_id, _w_old, dq_old, km in chunk
                ]
                cur.executemany(_UPDATE_SQL, params)
                conn.commit()
                if (i // batch) % 10 == 0 or i + batch >= len(rows):
                    log.info("  …updated %d / %d (%.0f rows/s)",
                             min(i + batch, len(rows)), len(rows),
                             (i + batch) / max(1e-6, time.time() - t0))
        log.info("done — %d entries updated in %.1fs",
                 len(rows), time.time() - t0)
    finally:
        conn.close()
    return 0


def rollback(rollback_csv: Path, batch: int) -> int:
    """Reverse the UPDATE from a backup CSV (sets withdrawn/disqualified back)."""
    if not rollback_csv.exists():
        log.error("rollback CSV not found: %s", rollback_csv)
        return 2

    rows: list[tuple[int, bool, bool]] = []
    with rollback_csv.open() as f:
        rdr = csv.DictReader(f)
        for r in rdr:
            try:
                rows.append((
                    int(r["entry_id"]),
                    r["old_withdrawn"].lower() in ("true", "t"),
                    r["old_disqualified"].lower() in ("true", "t"),
                ))
            except (KeyError, ValueError):
                continue

    log.info("rollback: %d rows from %s", len(rows), rollback_csv)
    if not rows:
        return 0

    conn = get_connection()
    try:
        with conn.cursor() as cur:
            for i in range(0, len(rows), batch):
                chunk = rows[i:i + batch]
                cur.executemany(
                    "UPDATE entry SET withdrawn = %s, disqualified = %s, "
                    "last_updated_at = NOW() WHERE entry_id = %s",
                    [(w, dq, eid) for eid, w, dq in chunk],
                )
                conn.commit()
        log.info("rollback complete: %d rows", len(rows))
    finally:
        conn.close()
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--execute", action="store_true",
                   help="apply changes (default is dry-run)")
    p.add_argument("--backup-csv", default=None,
                   help="path to write a per-row backup before updating "
                        "(default: logs/withdrawn_backfix_<ts>.csv)")
    p.add_argument("--batch", type=int, default=500,
                   help="UPDATE batch size (default 500)")
    p.add_argument("--rollback-csv", default=None,
                   help="reverse a previous run using its backup CSV")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    if args.rollback_csv:
        return rollback(Path(args.rollback_csv), args.batch)

    backup = None
    if args.execute:
        backup = (Path(args.backup_csv) if args.backup_csv else
                  _ROOT / "logs" / f"withdrawn_backfix_{datetime.now():%Y%m%d_%H%M%S}.csv")
    return run(dry_run=not args.execute, backup_csv=backup, batch=args.batch)


if __name__ == "__main__":
    sys.exit(main())
