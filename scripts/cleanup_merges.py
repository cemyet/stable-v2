"""
Unified post-ingest dedup / merge orchestrator.

After any fresh scrape (LeTrot daily / ATG bridge / etc.) new rows land in
the database without any cross-source linking — by design, `etl/matching.py`
NEVER auto-merges by name only. This script runs the full dedup pipeline
in the safe canonical order documented in `scripts/MERGE_RUN_CHECKLIST.md`,
so the new rows get folded into their canonical counterparts.

Pipeline (in order — each must finish before the next starts):

  1. `merge_duplicate_races`            — same (track, date, race#) dupes
  2. `match_french_races`               — cross-track French race variants
  3. `match_stub_races`                 — 1-entry foreign-country "stub"
                                          races (ST visitor scrape) folded
                                          into the canonical full race
  4. `match_french_horses`              — strict + no-DOB French horse dedup
  5. `merge_pedigree_duplicates`        — (name + year + sire + dam)
                                          triangulation across sources
  6. `merge_synth_pairs`                — ATG-synthetic horse rows folded
                                          into their ST-guest real row
  7. `match_persons_by_cooccurrence`    — drivers + trainers via horse
                                          co-occurrence
  8. `merge_synth_pairs_persons`        — ATG-synthetic person rows folded
                                          into their real ST/ATG row
  9. `normalize_french_sex`             — mare-axis sex heal (H/V/S)
 10. `link_pedigree_by_name`            — fill sire_id/dam_id FKs from names
  +final: `refresh_views`               — REFRESH MATERIALIZED VIEW
                                          horse_career_stats so leaderboards
                                          / horse pages reflect every merge

Each phase is run as a fire-and-forget subprocess so failures are
isolated; the orchestrator streams the child's stdout into its own
job_run.log so a single tail captures everything. Child scripts run
"headless" (no `--job-run-id`) — their per-script `script_runner`
becomes a no-op for DB-side job_run bookkeeping, and the orchestrator
owns the master job_run row.

CLI
---

    python -m scripts.cleanup_merges                       # dry-run
    python -m scripts.cleanup_merges --execute             # apply
    python -m scripts.cleanup_merges --execute --abort-on-error
    python -m scripts.cleanup_merges --skip merge_duplicate_races,normalize_french_sex
    python -m scripts.cleanup_merges --execute --job-run-id 1234   # admin path

`--job-run-id` (optional): attach to a job_run row pre-created by the
admin endpoint. Without it, a fresh row is created automatically so the
script is self-contained when invoked from the shell.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from core.db import get_connection  # noqa: E402


# Each phase: (admin_label, module_path, extra_args_when_execute)
# - module_path is what we pass to `python -m`
# - extra_args is appended only when --execute (so dry-runs don't get
#   --execute --commit-every etc.)
PHASES: list[tuple[str, str, list[str]]] = [
    ("merge_duplicate_races",
     "scripts.merge_duplicate_races",
     ["--execute", "--commit-every", "200"]),
    ("match_french_races",
     "scripts.match_french_races",
     ["--execute", "--commit-every", "100"]),
    ("match_stub_races",
     "scripts.match_stub_races",
     ["--execute", "--commit-every", "100"]),
    ("match_french_horses",
     "scripts.match_french_horses",
     ["--execute", "--commit-every", "200"]),
    ("merge_pedigree_duplicates",
     "scripts.merge_pedigree_duplicates",
     ["--execute", "--commit-every", "200"]),
    ("merge_synth_pairs",
     "scripts.merge_synth_pairs",
     ["--execute", "--commit-every", "200"]),
    ("match_persons_by_cooccurrence",
     "scripts.match_persons_by_cooccurrence",
     ["--execute", "--commit-every", "100"]),
    ("merge_synth_pairs_persons",
     "scripts.merge_synth_pairs_persons",
     ["--execute", "--commit-every", "200"]),
    ("normalize_french_sex",
     "scripts.normalize_french_sex",
     ["--execute"]),
    ("link_pedigree_by_name",
     "scripts.link_pedigree_by_name",
     ["--execute"]),
]

# Phases whose candidate discovery scans ALL history and therefore accept a
# `--since-days N` window. The nightly passes a small window (see jobs/update.py)
# so these never re-fingerprint years of already-resolved data every run.
_SUPPORTS_SINCE_DAYS = {"match_french_races"}


# ---------------------------------------------------------------------------
# job_run bookkeeping (mirrors jobs/update.py)
# ---------------------------------------------------------------------------

def _start_run(conn, job_name: str) -> int:
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO job_run (job_name, status, pid) "
            "VALUES (%s, 'running', %s) RETURNING job_run_id",
            (job_name, os.getpid()),
        )
        rid = cur.fetchone()[0]
    conn.commit()
    return rid


def _attach_run(conn, run_id: int) -> None:
    with conn.cursor() as cur:
        cur.execute(
            "UPDATE job_run SET status = 'running', pid = %s "
            "WHERE job_run_id = %s",
            (os.getpid(), run_id),
        )
    conn.commit()


def _log(conn, run_id: int, line: str, *, write_db: bool = True) -> None:
    """Print + append to job_run.log atomically.

    When invoked from a parent orchestrator (jobs.update --mode cleanup
    or --mode all) the parent captures our stdout and writes to the DB
    log itself, so we set `write_db=False` to avoid double-appending the
    same lines.
    """
    print(line, flush=True)
    if not write_db:
        return
    try:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE job_run SET log = COALESCE(log,'') || %s "
                "WHERE job_run_id = %s",
                (line + ("\n" if not line.endswith("\n") else ""), run_id),
            )
        conn.commit()
    except Exception:
        pass


def _merge_summary(conn, run_id: int, patch: dict) -> None:
    import json
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
            "UPDATE job_run SET finished_at = NOW(), status = %s "
            "WHERE job_run_id = %s",
            (status, run_id),
        )
    conn.commit()


# ---------------------------------------------------------------------------
# Phase runner — spawns one merge script and streams stdout into job_run.log
# ---------------------------------------------------------------------------

# Match the summary dict each script prints at the end, e.g.:
#   [merge_duplicate_races] done. {'script': 'merge_duplicate_races',
#       'execute': True, 'candidates': 2684, 'merged': 2684, 'errors': 0, ...}
_RX_SUMMARY = re.compile(r"\[\w+\] done\.\s*(\{.*\})\s*$")


def _run_phase(conn, run_id: int, label: str, module: str,
               args: list[str], *, write_db: bool = True) -> dict:
    """Run one phase as a subprocess. Returns a {label, exit_code,
    parsed_summary, lines_logged, seconds} dict.

    Stdout is read line-by-line and appended to job_run.log in real time
    so the admin UI sees progress as it happens. When `write_db=False`,
    we just print to stdout (the parent orchestrator captures it).
    """
    cmd = [sys.executable, "-u", "-m", module, *args]
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONPATH"] = str(_ROOT) + (
        os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
    )

    _log(conn, run_id, f"\n=== phase: {label} ===", write_db=write_db)
    _log(conn, run_id, f"$ {' '.join(cmd)}", write_db=write_db)

    t0 = time.time()
    parsed_summary: dict | None = None
    last_line: str = ""
    lines = 0

    try:
        proc = subprocess.Popen(
            cmd,
            cwd=str(_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=env,
            text=True,
            bufsize=1,
            close_fds=True,
        )
    except Exception as exc:
        _log(conn, run_id, f"[{label}] FAILED to spawn: {exc!r}",
             write_db=write_db)
        return {
            "label": label, "exit_code": -1,
            "summary": None, "lines": 0,
            "seconds": round(time.time() - t0, 1),
            "error": repr(exc),
        }

    assert proc.stdout is not None
    try:
        for raw in proc.stdout:
            line = raw.rstrip("\n")
            lines += 1
            last_line = line
            m = _RX_SUMMARY.search(line)
            if m:
                # Capture the script's final summary dict (best-effort).
                try:
                    import ast
                    parsed_summary = ast.literal_eval(m.group(1))
                except Exception:
                    parsed_summary = {"raw": m.group(1)}
            _log(conn, run_id, line, write_db=write_db)
    except Exception as exc:
        _log(conn, run_id, f"[{label}] log-stream error: {exc!r}",
             write_db=write_db)

    rc = proc.wait()
    dt = round(time.time() - t0, 1)
    _log(conn, run_id,
         f"[{label}] exit_code={rc}  seconds={dt}  lines={lines}",
         write_db=write_db)
    if rc != 0:
        _log(conn, run_id, f"[{label}] LAST LINE: {last_line[:300]}",
             write_db=write_db)
    return {
        "label": label, "exit_code": rc,
        "summary": parsed_summary, "lines": lines,
        "seconds": dt,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--execute", action="store_true",
                   help="apply merges. Without this every phase runs in "
                        "its own dry-run mode.")
    p.add_argument("--abort-on-error", action="store_true",
                   help="stop the pipeline if any phase returns non-zero "
                        "(default: continue and report at the end).")
    p.add_argument("--skip", default="",
                   help="comma-separated phase labels to skip "
                        "(labels are " + ",".join(l for l, _, _ in PHASES) + ")")
    p.add_argument("--job-run-id", type=int, default=None,
                   help="attach to a pre-created job_run row "
                        "(admin endpoint path); otherwise a new row is "
                        "created automatically.")
    p.add_argument("--since-days", type=int, default=None,
                   help="window (days) forwarded to history-scanning phases "
                        f"({', '.join(sorted(_SUPPORTS_SINCE_DAYS))}). Omit for "
                        "a full historical sweep (manual/occasional).")
    args = p.parse_args()

    skip = {s.strip() for s in args.skip.split(",") if s.strip()}
    valid_labels = {l for l, _, _ in PHASES} | {"refresh_views"}
    bad_skip = skip - valid_labels
    if bad_skip:
        print(f"ERROR: unknown --skip labels: {sorted(bad_skip)}",
              file=sys.stderr)
        return 2

    conn = get_connection()
    # When the orchestrator (e.g. jobs.update --mode cleanup or --mode all)
    # passes its own job_run_id, we attach to it for logging but do NOT
    # call _finish at the end — the parent owns the lifecycle of that row.
    # When called standalone, we create our own row and own the lifecycle.
    owns_run_row = (args.job_run_id is None)
    if owns_run_row:
        suffix = "" if args.execute else "_dryrun"
        run_id = _start_run(conn, job_name=f"cleanup_merges{suffix}")
    else:
        run_id = args.job_run_id
        # Do NOT call _attach_run here — that would overwrite the parent's
        # pid and could confuse the cancel/zombie-reap logic in the admin
        # web UI. The parent already attached this row.

    write_db = owns_run_row
    _log(conn, run_id,
         f"[cleanup_merges] start  execute={args.execute}  "
         f"abort_on_error={args.abort_on_error}  skip={sorted(skip) or '∅'}  "
         f"run_id={run_id}  pid={os.getpid()}  "
         f"at={datetime.now().isoformat(timespec='seconds')}",
         write_db=write_db)

    overall: dict = {
        "mode": "cleanup_merges",
        "execute": bool(args.execute),
        "phases": [],
        "aborted": False,
        "total_merged": 0,
        "total_errors": 0,
    }
    t_overall = time.time()

    final_status = "success"
    try:
        for label, module, extra in PHASES:
            if label in skip:
                _log(conn, run_id, f"\n=== phase: {label} — SKIPPED ===",
                     write_db=write_db)
                overall["phases"].append(
                    {"label": label, "skipped": True})
                continue

            phase_args = list(extra) if args.execute else []
            if args.since_days is not None and label in _SUPPORTS_SINCE_DAYS:
                phase_args += ["--since-days", str(args.since_days)]
            result = _run_phase(conn, run_id, label, module, phase_args,
                                write_db=write_db)
            overall["phases"].append(result)

            if result.get("summary"):
                overall["total_merged"] += int(
                    result["summary"].get("merged") or 0)
                overall["total_errors"] += int(
                    result["summary"].get("errors") or 0)

            if result["exit_code"] != 0:
                final_status = "failed"
                if args.abort_on_error:
                    _log(conn, run_id,
                         f"[cleanup_merges] aborting — phase {label!r} "
                         f"failed with exit_code={result['exit_code']}",
                         write_db=write_db)
                    overall["aborted"] = True
                    break
                else:
                    _log(conn, run_id,
                         f"[cleanup_merges] phase {label!r} failed "
                         f"(exit_code={result['exit_code']}) — continuing",
                         write_db=write_db)

        # Final phase: refresh materialized views that depend on entry rows.
        # Without this, horse-page stats / leaderboards keep showing pre-merge
        # snapshots until something else triggers a refresh.
        if "refresh_views" not in skip:
            mv_label = "refresh_views"
            t_mv = time.time()
            _log(conn, run_id,
                 f"\n=== phase: {mv_label} ===",
                 write_db=write_db)
            try:
                # Run outside a transaction (autocommit) because
                # REFRESH MATERIALIZED VIEW CONCURRENTLY cannot run inside one.
                old_iso = conn.isolation_level
                conn.set_isolation_level(0)  # AUTOCOMMIT
                with conn.cursor() as cur:
                    cur.execute(
                        "REFRESH MATERIALIZED VIEW CONCURRENTLY "
                        "horse_career_stats"
                    )
                    cur.execute(
                        "REFRESH MATERIALIZED VIEW CONCURRENTLY "
                        "horse_year_stats"
                    )
                    cur.execute(
                        "REFRESH MATERIALIZED VIEW CONCURRENTLY "
                        "person_career_stats"
                    )
                conn.set_isolation_level(old_iso)
                dt = round(time.time() - t_mv, 1)
                _log(conn, run_id,
                     f"[{mv_label}] refreshed horse_career_stats + "
                     f"horse_year_stats + person_career_stats in {dt}s",
                     write_db=write_db)
                overall["phases"].append({
                    "label": mv_label, "exit_code": 0,
                    "summary": {"refreshed": ["horse_career_stats",
                                              "horse_year_stats",
                                              "person_career_stats"]},
                    "seconds": dt,
                })
            except Exception as exc:
                dt = round(time.time() - t_mv, 1)
                _log(conn, run_id,
                     f"[{mv_label}] FAILED: {exc!r}",
                     write_db=write_db)
                overall["phases"].append({
                    "label": mv_label, "exit_code": 1,
                    "summary": None, "seconds": dt,
                    "error": repr(exc),
                })
                final_status = "failed"
        else:
            _log(conn, run_id,
                 "\n=== phase: refresh_views — SKIPPED ===",
                 write_db=write_db)
            overall["phases"].append({"label": "refresh_views", "skipped": True})

        overall["seconds"] = round(time.time() - t_overall, 1)

        _log(conn, run_id,
             f"\n[cleanup_merges] done in {overall['seconds']}s — "
             f"total merged={overall['total_merged']}, "
             f"errors={overall['total_errors']}, "
             f"phases ok={sum(1 for r in overall['phases'] if not r.get('skipped') and r.get('exit_code') == 0)}, "
             f"phases failed={sum(1 for r in overall['phases'] if not r.get('skipped') and r.get('exit_code') not in (0, None))}, "
             f"phases skipped={sum(1 for r in overall['phases'] if r.get('skipped'))}",
             write_db=write_db)
    except Exception as exc:
        _log(conn, run_id,
             f"[cleanup_merges] FAILED: {exc!r}\n{traceback.format_exc()}",
             write_db=write_db)
        final_status = "failed"
        overall["error"] = repr(exc)

    if owns_run_row:
        _finish(conn, run_id, final_status, overall)
    else:
        # Just patch the summary so the parent can roll it up; don't
        # touch finished_at / status — that's the parent's job.
        _merge_summary(conn, run_id, {"cleanup": overall})
    conn.close()
    return 0 if final_status == "success" else 1


if __name__ == "__main__":
    sys.exit(main())
