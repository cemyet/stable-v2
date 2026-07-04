"""
Populate `entry_perf` — the per-entry finishing percentile that powers the
`form` (actual recent form) metric shown on home / trainer / driver / race
pages and the stats/trainer table.

Market-AGNOSTIC counterpart to entry_outperf (which powers mkt±). For each race
we rank the runners by their actual finishing position and store

    perf = (n_field - fin_rank) / (n_field - 1)        ∈ [0, 1]

so 1.0 = won, 0.0 = finished last (or galopp/DQ/DNF → bottom). Averaging perf
over a person's recent starts = their `form`.

Why a second table next to entry_outperf? The trainer-form experiment
(scripts/eval_trainer_form) showed these two signals are near mirror images:
this finishing-percentile family PREDICTS WINNING (y_win AUC ≈ 0.67) and rewards
winning favourites, whereas the market-residual mkt± anti-correlates with
winning and is a value/edge signal. So `form` = "are the horses finishing high
lately", `mkt±` = "how much they beat the market".

Population: every start with a classified placement in a field of >= 4 (NOT
withdrawn, not a qualifier). No odds gate → materially more data than mkt±.

Usage
-----
    python -m scripts.refresh_entry_perf --full          # rebuild everything
    python -m scripts.refresh_entry_perf --since 2026-05-01
    refresh_entry_perf(conn, since='YYYY-MM-DD')          # from jobs/update.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from core.db import get_connection  # noqa: E402

_QUALIFIER_RE = (
    r"^(gdk|ejg|ejp|gd|gk|egk|egdk|gkd|gdj|gdb|gdek|gdgk|gdl|ddk|frdk|erj|ejk|"
    r"ejgk|ejgd|ej|EJ|EJG|EJP|GDK|Gdk|g)[0-9]?$|^[12]?[pP][0-9]?$"
)
_MIN_FIELD = 4

# Within-race finishing percentile for the chosen race set. {where} is an extra
# predicate on e.race_date; %(q)s = qualifier regex, %(mf)s = min field size.
#
# We rank by `placement` (official classified finishing position), NOT
# finish_order: finish_order is frequently NULL even for clean finishers
# (esp. recent races), whereas placement is the reliable field. n_field counts
# only classified runners (placement present); galopp/DQ (placement NULL) are
# excluded, matching entry_outperf's "rankable starts" notion.
_COMPUTE_SQL = """
    WITH ranked AS (
        SELECT e.entry_id, e.race_date,
               COUNT(*)     OVER (PARTITION BY e.race_id)             AS n_field,
               ROW_NUMBER() OVER (PARTITION BY e.race_id
                   ORDER BY e.placement, e.program_number)           AS fin_rank
        FROM entry e
        WHERE e.placement IS NOT NULL
          AND NOT COALESCE(e.withdrawn, false)
          AND COALESCE(e.placement_text,'') !~ %(q)s
          {where}
    )
    SELECT entry_id,
           (n_field - fin_rank)::real / (n_field - 1) AS perf,
           race_date
    FROM ranked
    WHERE n_field >= %(mf)s
"""


def refresh_entry_perf(conn, *, since: str | None = None,
                       full: bool = False, log=print) -> int:
    """(Re)compute perf. `full` truncates; otherwise upserts the window
    race_date >= `since` (defaults to last 45 days). Returns row count."""
    params = {"q": _QUALIFIER_RE, "mf": _MIN_FIELD}
    with conn.cursor() as cur:
        if full:
            log("[entry_perf] FULL rebuild…")
            cur.execute("TRUNCATE entry_perf")
            where = ""
        else:
            if since is None:
                cur.execute("SELECT (CURRENT_DATE - 45)::text")
                since = cur.fetchone()[0]
            log(f"[entry_perf] incremental since {since}…")
            where = "AND e.race_date >= %(since)s"
            params["since"] = since
        cur.execute(f"""
            INSERT INTO entry_perf (entry_id, perf, race_date)
            {_COMPUTE_SQL.format(where=where)}
            ON CONFLICT (entry_id) DO UPDATE
               SET perf      = EXCLUDED.perf,
                   race_date = EXCLUDED.race_date
        """, params)
        n = cur.rowcount
    conn.commit()
    log(f"[entry_perf] upserted {n:,} rows")
    return n


def main() -> int:
    ap = argparse.ArgumentParser("refresh_entry_perf")
    ap.add_argument("--full", action="store_true", help="full rebuild (truncate)")
    ap.add_argument("--since", type=str, default=None,
                    help="incremental start date YYYY-MM-DD (default last 45d)")
    args = ap.parse_args()
    conn = get_connection()
    refresh_entry_perf(conn, since=args.since, full=args.full)
    conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
