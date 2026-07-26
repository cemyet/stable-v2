"""Overnight catch-up loop: shallow passports, then foreign racedays.

Resumable / Ctrl-C safe. Writes progress to logs/catchup_status.json after
every batch. Stop auto if impossible_pairs jumps by >200 in a foreign batch.

Usage:
    python -u -m scripts.run_catchup_loop
    python -u -m scripts.run_catchup_loop --shallow-only
    python -u -m scripts.run_catchup_loop --foreign-only
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from core.db import get_connection  # noqa: E402
from etl.dq_sentinel import collect_metrics  # noqa: E402
from etl.import_st import (  # noqa: E402
    discover_raceday_ids_from_horse_raw,
    discover_shallow_st_horse_ids,
    run_foreign_raceday_drain,
    run_shallow_heal,
)

STATUS = _ROOT / "logs" / "catchup_status.json"
LOG_DIR = _ROOT / "logs"
# Touch this file to stop the loop at the next batch boundary. Checked before
# every batch and before entering the foreign phase, so a stop never lands
# mid-batch (which would leave horses scraped-but-not-ETLed).
STOP_FLAG = _ROOT / "logs" / "catchup_stop.flag"


def _log(msg: str) -> None:
    line = f"[{time.strftime('%Y-%m-%dT%H:%M:%S')}] {msg}"
    print(line, flush=True)


def _stop_requested() -> bool:
    return STOP_FLAG.exists()


def _write_status(conn, *, phase: str, batch: int) -> dict:
    m = collect_metrics(conn)
    out = {
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "phase": phase,
        "batch": batch,
        "shallow_backlog": len(
            discover_shallow_st_horse_ids(conn, lookback_days=730)
        ),
        "foreign_backlog": len(discover_raceday_ids_from_horse_raw(conn)),
        "impossible_horse_date_pairs": m.get("impossible_horse_date_pairs"),
        "dup_race_groups": m.get("dup_race_groups"),
        "placeholder_track_races": m.get("placeholder_track_races"),
        "alive": True,
    }
    STATUS.parent.mkdir(exist_ok=True)
    STATUS.write_text(json.dumps(out, indent=2) + "\n")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shallow-batch", type=int, default=800)
    ap.add_argument("--foreign-batch", type=int, default=600)
    ap.add_argument("--lookback-days", type=int, default=730)
    ap.add_argument("--shallow-only", action="store_true")
    ap.add_argument("--foreign-only", action="store_true")
    ap.add_argument("--max-shallow-batches", type=int, default=40)
    ap.add_argument("--max-foreign-batches", type=int, default=50)
    ap.add_argument("--stop-after-shallow", action="store_true",
                    help="exit cleanly once passports are drained, before "
                         "touching foreign racedays")
    args = ap.parse_args()

    conn = get_connection()
    _log("catchup loop starting")
    st = _write_status(conn, phase="start", batch=0)
    _log(f"status: {st}")

    if not args.foreign_only:
        for i in range(1, args.max_shallow_batches + 1):
            if _stop_requested():
                _log(f"stop flag present ({STOP_FLAG}) — halting before "
                     f"shallow batch {i}")
                break
            remaining = len(
                discover_shallow_st_horse_ids(
                    conn, lookback_days=args.lookback_days
                )
            )
            if remaining == 0:
                _log("shallow backlog drained")
                break
            _log(f"=== shallow batch {i} remaining={remaining} "
                 f"limit={args.shallow_batch} ===")
            try:
                res = run_shallow_heal(
                    conn,
                    limit=args.shallow_batch,
                    lookback_days=args.lookback_days,
                    log=_log,
                )
                _log(f"batch result: {res}")
            except Exception:
                _log("SHALLOW BATCH FAILED:\n" + traceback.format_exc())
                _write_status(conn, phase="shallow_failed", batch=i)
                return 1
            st = _write_status(conn, phase="shallow", batch=i)
            _log(f"status: shallow={st['shallow_backlog']} "
                 f"impossible={st['impossible_horse_date_pairs']}")

    if args.stop_after_shallow and not args.shallow_only:
        _log("--stop-after-shallow set — not entering foreign phase")
        args.shallow_only = True

    if not args.shallow_only:
        for i in range(1, args.max_foreign_batches + 1):
            if _stop_requested():
                _log(f"stop flag present ({STOP_FLAG}) — halting before "
                     f"foreign batch {i}")
                break
            remaining = len(discover_raceday_ids_from_horse_raw(conn))
            if remaining == 0:
                _log("foreign backlog drained")
                break
            before = collect_metrics(conn).get("impossible_horse_date_pairs", 0)
            _log(f"=== foreign batch {i} remaining={remaining} "
                 f"limit={args.foreign_batch} ===")
            try:
                res = run_foreign_raceday_drain(
                    conn, limit=args.foreign_batch, log=_log,
                )
                _log(f"batch result: {res}")
            except Exception:
                _log("FOREIGN BATCH FAILED:\n" + traceback.format_exc())
                _write_status(conn, phase="foreign_failed", batch=i)
                return 1
            after = collect_metrics(conn).get("impossible_horse_date_pairs", 0)
            delta = after - before
            _log(f"impossible_pairs delta={delta} (now {after})")
            st = _write_status(conn, phase="foreign", batch=i)
            if delta > 200:
                _log("STOP — impossible_pairs rose >200 this batch. "
                     "Resume after review with --foreign-only")
                st["alive"] = False
                st["stopped_reason"] = f"impossible_delta_{delta}"
                STATUS.write_text(json.dumps(st, indent=2) + "\n")
                return 2

    st = _write_status(conn, phase="done", batch=0)
    st["alive"] = False
    STATUS.write_text(json.dumps(st, indent=2) + "\n")
    _log(f"DONE catchup: {st}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
