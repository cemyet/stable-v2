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
    """Run v1's update job, then mirror any changed v1 rows to v2.

    Returns a summary dict shallow-merged into job_run.summary.
    """
    summary = {"mode": "bridge"}

    # Snapshot v2 row counts before
    pre = _v2_counts(conn)
    summary["pre"] = pre

    _log(conn, run_id, "[bridge] phase 1/2 — running v1 update (jakob db)")
    t0 = time.time()
    v1_summary = _run_v1_update(conn, run_id)
    summary["v1_update_seconds"] = round(time.time() - t0, 1)
    summary["v1_summary"] = v1_summary

    _log(conn, run_id, "[bridge] phase 2/2 — refreshing stable_v2 from v1")
    t0 = time.time()
    v1_conn = get_v1_connection()
    try:
        counts = import_st.backfill_from_v1(v1_conn, conn)
    finally:
        v1_conn.close()
    summary["refresh_seconds"] = round(time.time() - t0, 1)
    summary["refresh_counts"] = counts

    post = _v2_counts(conn)
    summary["post"] = post
    summary["delta"] = {k: post[k] - pre[k] for k in pre}

    return summary


def _v2_counts(conn) -> dict:
    """Snapshot row counts for the master tables."""
    out = {}
    with conn.cursor() as cur:
        for t in ("horse", "person", "track", "race", "entry"):
            cur.execute(f"SELECT COUNT(*) FROM {t}")
            out[t] = cur.fetchone()[0]
    return out


def _run_v1_update(conn, run_id: int) -> dict:
    """Subprocess-launch v1's update job. Captures last lines for summary."""
    if not (V1_PROJECT_ROOT / "jobs" / "update.py").exists():
        _log(conn, run_id, f"[bridge]   v1 update module not found at {V1_PROJECT_ROOT}; skipping")
        return {"skipped": True}
    cmd = ["python3", "-m", "jobs.update"]
    _log(conn, run_id, f"[bridge]   $ cd {V1_PROJECT_ROOT} && {' '.join(cmd)}")
    proc = subprocess.run(
        cmd,
        cwd=str(V1_PROJECT_ROOT),
        capture_output=True,
        text=True,
    )
    last_lines = "\n".join((proc.stdout + "\n" + proc.stderr).strip().splitlines()[-15:])
    for line in last_lines.splitlines():
        _log(conn, run_id, f"[v1]   {line}")
    return {
        "exit_code": proc.returncode,
        "tail": last_lines[-2000:],
    }


# ---------------------------------------------------------------------------
# Native mode (placeholder)
# ---------------------------------------------------------------------------

def run_native(conn, run_id: int) -> dict:
    _log(
        conn, run_id,
        "[native] not yet implemented — native scrape→parse→UPSERT pipeline "
        "is the v2 vision but lives in scrapers/* + etl/import_<source>.py "
        "stubs for now. Use --mode bridge until the native scrapers land."
    )
    return {"mode": "native", "skipped": True}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    # Accept v1's "atg" / "st" aliases too so the v2 admin UI works
    # without pre-coordination.
    ap.add_argument(
        "--mode",
        choices=("bridge", "native", "atg", "st"),
        default="bridge",
    )
    ap.add_argument("--job-run-id", type=int, default=None,
                    help="Attach to a job_run row pre-created by the admin web ui")
    args = ap.parse_args()

    # All v1-style modes route through bridge mode for now.
    mode = "bridge" if args.mode in ("atg", "st", "bridge") else "native"

    conn = get_connection()
    if args.job_run_id is not None:
        run_id = args.job_run_id
        _attach_run(conn, run_id)
    else:
        run_id = _start_run(conn)
    _log(conn, run_id, f"[update] start  mode={mode} (cli={args.mode})  run_id={run_id}  pid={os.getpid()}  at={datetime.now().isoformat(timespec='seconds')}")

    try:
        if mode == "bridge":
            summary = run_bridge(conn, run_id)
        else:
            summary = run_native(conn, run_id)
    except Exception as exc:
        _log(conn, run_id, f"[update] FAILED: {exc!r}\n{traceback.format_exc()}")
        _finish(conn, run_id, "failed", {"error": repr(exc)})
        return 1

    _log(conn, run_id, "[update] done.")
    _finish(conn, run_id, "success", summary)
    return 0


if __name__ == "__main__":
    sys.exit(main())
