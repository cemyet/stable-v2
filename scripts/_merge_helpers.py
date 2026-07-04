"""
Shared utilities for the Act 2 cleanup scripts.

Provides:
  - `build_argparser(name)` - identical CLI surface for every script:
      --dry-run (default), --execute, --limit, --job-run-id, --json,
      --rollback-job (rollback all merges from a previous job)
  - `JobLogger` - tee print() output to stdout AND the `job_run.log` column
    so the admin matching page's script runner can stream progress.
  - `script_runner(...)` - context manager that opens a connection and writes
    a `job_run` row (or honours an existing `--job-run-id` so the admin page
    can launch the same script).
  - `perform_merge(...)` - single merge with batched commits, used by
    `merge_horses` / `merge_persons` consumers.

Mass-merge runbook
------------------

Before invoking any of these scripts with `--execute`, follow
`scripts/MERGE_RUN_CHECKLIST.md`. The short version:

  1. `pg_dump --data-only` of (entry, horse, race, person, *_merge_log) to a
     timestamped folder. Restore recipe is in the checklist.
  2. Verify the EUR FX backfill landed — no `entry` rows should have
     `prize_currency='EUR'` AND `prize_fx_rate IS NULL`.
  3. Dry-run every script first. Check `summary.merged` matches your
     expectation before re-running with `--execute`.
  4. Rollback by `job_run_id` is supported per-script — see the checklist.
"""

from __future__ import annotations

import argparse
import json
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Callable

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def build_argparser(name: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog=name)
    p.add_argument("--execute", action="store_true",
                   help="apply merges (default is dry-run)")
    p.add_argument("--dry-run", dest="dry_run_explicit", action="store_true",
                   help="explicit dry-run (default behavior)")
    p.add_argument("--limit", type=int, default=None,
                   help="cap number of candidates processed")
    p.add_argument("--job-run-id", type=int, default=None,
                   help="stream progress into this job_run row")
    p.add_argument("--json", action="store_true",
                   help="dump full summary JSON")
    p.add_argument("--commit-every", type=int, default=200,
                   help="commit batch size when --execute")
    p.add_argument("--rollback-job", type=int, default=None, metavar="JOB_ID",
                   help="roll back every merge logged with this job_run_id "
                        "(walks *_merge_log in reverse chronological order)")
    return p


class JobLogger:
    """Tee print() output to stdout and the job_run.log column."""

    def __init__(self, conn, job_run_id: int | None):
        self.conn = conn
        self.rid = job_run_id
        self._buf: list[str] = []

    def __call__(self, msg: str = "") -> None:
        s = msg if msg.endswith("\n") else msg + "\n"
        sys.stdout.write(s)
        sys.stdout.flush()
        if self.rid is None:
            return
        self._buf.append(s)
        if sum(len(x) for x in self._buf) > 4096:
            self.flush()

    def flush(self) -> None:
        if not self._buf or self.rid is None:
            return
        chunk = "".join(self._buf)
        self._buf.clear()
        try:
            with self.conn.cursor() as cur:
                cur.execute(
                    "UPDATE job_run SET log = COALESCE(log,'') || %s WHERE job_run_id=%s",
                    (chunk, self.rid),
                )
            self.conn.commit()
        except Exception:
            pass

    def mark_done(self, status: str, summary: dict) -> None:
        self.flush()
        if self.rid is None:
            return
        try:
            with self.conn.cursor() as cur:
                cur.execute(
                    "UPDATE job_run SET status=%s, finished_at=NOW(), summary=%s "
                    "WHERE job_run_id=%s",
                    (status, json.dumps(summary, default=str), self.rid),
                )
            self.conn.commit()
        except Exception:
            pass


@contextmanager
def script_runner(name: str, args):
    """Wraps a script body with job_run bookkeeping and a single connection.

    Yields (conn, log, summary_dict). On exit, marks job_run success or
    failed and prints final JSON summary if --json.
    """
    from core.db import get_connection
    conn = get_connection()
    log = JobLogger(conn, args.job_run_id)
    summary: dict = {"script": name, "execute": bool(args.execute),
                     "candidates": 0, "merged": 0, "skipped": 0,
                     "errors": 0}
    try:
        yield conn, log, summary
        log.mark_done("success", summary)
        if args.json:
            print(json.dumps(summary, indent=2, default=str))
        else:
            log(f"\n[{name}] done. {summary}")
    except Exception as exc:
        summary["error"] = repr(exc)
        log(f"[{name}] FAILED: {exc!r}")
        log.mark_done("failed", summary)
        raise
    finally:
        conn.close()


def perform_merge(
    conn,
    log: JobLogger,
    summary: dict,
    *,
    from_id: int,
    to_id: int,
    reason: str,
    method: str,
    dry_run: bool,
    commit_every: int = 200,
    merge_fn: Callable | None = None,
    kind: str = "horse",
) -> dict:
    """Single merge with logging + commit batching.

    `merge_fn` lets a script swap in `merge_persons` instead of the default
    `merge_horses`. Returns the merge_horses summary dict.
    """
    from core.identity import merge_horses, merge_persons
    if merge_fn is None:
        merge_fn = merge_persons if kind == "person" else merge_horses

    with conn.cursor() as cur:
        try:
            res = merge_fn(cur, from_id, to_id,
                           reason=reason, method=method,
                           merged_by=method, dry_run=dry_run)
        except Exception as exc:
            conn.rollback()
            summary["errors"] += 1
            log(f"  ! error merging {from_id} -> {to_id}: {exc!r}")
            return {"error": str(exc)}

    if "error" in res:
        summary["skipped"] += 1
        log(f"  - skipped {from_id} -> {to_id}: {res['error']}")
        return res

    summary["merged"] += 1
    moved = res.get("entries_moved", 0)
    conflicts = res.get("conflicts_resolved", 0)
    log(f"  {'PREVIEW' if dry_run else 'merged '} #{from_id} -> #{to_id}  "
        f"moved={moved} conflicts={conflicts}  ({res.get('from_name')!r} -> {res.get('to_name')!r})")

    if not dry_run and summary["merged"] % commit_every == 0:
        conn.commit()
    return res


# ---------------------------------------------------------------------------
# Rollback helpers
# ---------------------------------------------------------------------------

def rollback_horse_merges_by_method(conn, log, method: str,
                                    *, job_run_id: int | None = None,
                                    since=None, until=None) -> dict:
    """Roll back every horse_merge_log entry written by `method` within the
    time window. When `job_run_id` is supplied, the window is taken from
    the matching job_run row.

    Returns {"reversed": N, "errors": M}.
    """
    from core.identity import rollback_horse_merge

    started_at, finished_at = since, until
    if job_run_id is not None:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT started_at, finished_at FROM job_run "
                " WHERE job_run_id = %s",
                (job_run_id,),
            )
            row = cur.fetchone()
            if row:
                started_at, finished_at = row

    with conn.cursor() as cur:
        cur.execute(
            "SELECT merge_id FROM horse_merge_log "
            " WHERE method = %s "
            "   AND (%s::timestamp IS NULL OR merged_at >= %s) "
            "   AND (%s::timestamp IS NULL OR merged_at <= %s) "
            "   AND NOT rolled_back "
            " ORDER BY merge_id DESC",
            (method, started_at, started_at, finished_at, finished_at),
        )
        merge_ids = [r[0] for r in cur.fetchall()]

    log(f"[rollback] {method}: {len(merge_ids)} merges to reverse")
    out = {"reversed": 0, "errors": 0}
    for mid in merge_ids:
        with conn.cursor() as cur:
            try:
                res = rollback_horse_merge(cur, mid)
            except Exception as exc:
                out["errors"] += 1
                conn.rollback()
                log(f"  ! rollback failed merge_id={mid}: {exc!r}")
                continue
        if "error" in res:
            out["errors"] += 1
            log(f"  ! merge_id={mid}: {res['error']}")
            continue
        out["reversed"] += 1
        log(f"  reversed merge_id={mid} restored horse={res.get('horse_restored')}")
        if out["reversed"] % 50 == 0:
            conn.commit()
    conn.commit()
    return out


def rollback_person_merges_by_method(conn, log, method: str,
                                     *, job_run_id: int | None = None,
                                     since=None, until=None) -> dict:
    """Same as `rollback_horse_merges_by_method` but for person_merge_log."""
    from core.identity import rollback_person_merge

    started_at, finished_at = since, until
    if job_run_id is not None:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT started_at, finished_at FROM job_run "
                " WHERE job_run_id = %s",
                (job_run_id,),
            )
            row = cur.fetchone()
            if row:
                started_at, finished_at = row

    with conn.cursor() as cur:
        cur.execute(
            "SELECT merge_id FROM person_merge_log "
            " WHERE method = %s "
            "   AND (%s::timestamp IS NULL OR merged_at >= %s) "
            "   AND (%s::timestamp IS NULL OR merged_at <= %s) "
            "   AND NOT rolled_back "
            " ORDER BY merge_id DESC",
            (method, started_at, started_at, finished_at, finished_at),
        )
        merge_ids = [r[0] for r in cur.fetchall()]

    log(f"[rollback] {method}: {len(merge_ids)} merges to reverse")
    out = {"reversed": 0, "errors": 0}
    for mid in merge_ids:
        with conn.cursor() as cur:
            try:
                res = rollback_person_merge(cur, mid)
            except Exception as exc:
                out["errors"] += 1
                conn.rollback()
                log(f"  ! rollback failed merge_id={mid}: {exc!r}")
                continue
        if "error" in res:
            out["errors"] += 1
            log(f"  ! merge_id={mid}: {res['error']}")
            continue
        out["reversed"] += 1
        if out["reversed"] % 50 == 0:
            conn.commit()
    conn.commit()
    return out
