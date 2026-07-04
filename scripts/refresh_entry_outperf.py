"""
Populate `entry_outperf` — the per-entry market-outperformance that powers the
`s_form` (stallform) metric shown on home / trainer / driver / race pages.

For each race we rank runners by odds (the market's expected finishing order)
and by finish_order (the actual order), then store

    market_outperf = (odds_rank - fin_rank) / (n - 1)        ∈ [-1, +1]

so a positive value means the horse finished AHEAD of its market rank. Only
rankable starts get a row: valid odds (>1), a finish_order, not withdrawn, not
a qualifier, and a field of >= 4 such runners (rank is meaningless otherwise).

This is the deterministic gold-standard trainer-form signal found in the
trainer-form experiment (odds-rank outperformance beats win-rate / win-residual
form by a wide margin). Averaging market_outperf over a person's recent starts
= their s_form.

Usage
-----
    python -m scripts.refresh_entry_outperf --full          # rebuild everything
    python -m scripts.refresh_entry_outperf --since 2026-05-01
    refresh_entry_outperf(conn, since='YYYY-MM-DD')         # from jobs/update.py
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

# Within-race ranks → market_outperf, for the chosen race set. {where} is an
# extra predicate on race_date; %(q)s = qualifier regex, %(mf)s = min field.
_COMPUTE_SQL = """
    WITH ranked AS (
        SELECT e.entry_id, e.race_date,
               COUNT(*)     OVER w AS n_odds,
               ROW_NUMBER() OVER (PARTITION BY e.race_id
                                  ORDER BY e.odds, e.program_number) AS odds_rank,
               ROW_NUMBER() OVER (PARTITION BY e.race_id
                                  ORDER BY e.finish_order, e.program_number) AS fin_rank
        FROM entry e
        {where}
        WHERE e.odds IS NOT NULL AND e.odds > 1
          AND e.finish_order IS NOT NULL
          AND NOT COALESCE(e.withdrawn, false)
          AND COALESCE(e.placement_text,'') !~ %(q)s
        WINDOW w AS (PARTITION BY e.race_id)
    )
    SELECT entry_id,
           ((odds_rank - fin_rank)::real / NULLIF(n_odds - 1, 0)) AS market_outperf,
           race_date
    FROM ranked
    WHERE n_odds >= %(mf)s
"""


def refresh_entry_outperf(conn, *, since: str | None = None,
                          full: bool = False, log=print) -> int:
    """(Re)compute market_outperf. `full` truncates; otherwise upserts the
    window race_date >= `since` (defaults to last 45 days). Returns row count."""
    params = {"q": _QUALIFIER_RE, "mf": _MIN_FIELD}
    with conn.cursor() as cur:
        if full:
            log("[entry_outperf] FULL rebuild…")
            cur.execute("TRUNCATE entry_outperf")
            where = ""
            cur.execute(f"""
                INSERT INTO entry_outperf (entry_id, market_outperf, race_date)
                {_COMPUTE_SQL.format(where=where)}
                ON CONFLICT (entry_id) DO UPDATE
                   SET market_outperf = EXCLUDED.market_outperf,
                       race_date      = EXCLUDED.race_date
            """, params)
        else:
            if since is None:
                cur.execute("SELECT (CURRENT_DATE - 45)::text")
                since = cur.fetchone()[0]
            log(f"[entry_outperf] incremental since {since}…")
            where = "JOIN race r ON r.race_id = e.race_id AND r.race_date >= %(since)s"
            params["since"] = since
            cur.execute(f"""
                INSERT INTO entry_outperf (entry_id, market_outperf, race_date)
                {_COMPUTE_SQL.format(where=where)}
                ON CONFLICT (entry_id) DO UPDATE
                   SET market_outperf = EXCLUDED.market_outperf,
                       race_date      = EXCLUDED.race_date
            """, params)
        n = cur.rowcount
    conn.commit()
    log(f"[entry_outperf] upserted {n:,} rows")
    return n


def main() -> int:
    ap = argparse.ArgumentParser("refresh_entry_outperf")
    ap.add_argument("--full", action="store_true", help="full rebuild (truncate)")
    ap.add_argument("--since", type=str, default=None,
                    help="incremental start date YYYY-MM-DD (default last 45d)")
    args = ap.parse_args()
    conn = get_connection()
    refresh_entry_outperf(conn, since=args.since, full=args.full)
    conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
