"""One-time backfill: map French (LeTrot) result/position codes to the
canonical galopp / disqualified / withdrawn flags.

Background
----------
French trot has no "broke gait but kept its place" outcome the way Swedish
trot does. Breaking gait (allure irrégulière) is an automatic *distancement*
(disqualification), coded `DA` / `D` / `D0..D9` in the LeTrot ARRIVÉE table.
The historic LeTrot importer never set `entry.galopp` for these, so French
trainers showed a ~0% gallop rate. `NP` (non partant) was likewise not mapped
to `withdrawn`, inflating start counts.

This script reuses the single source of truth — `core.common.classify_letrot_placement`
— so the backfill can never drift from the importer. It is idempotent: rows
already carrying the correct flags are skipped, so it is safe to re-run.

Scope: French tracks only (`track.country = 'FR'`). French races that come via
ATG carry lowercase Scandinavian codes (`d`/`g`) and ATG's own `galloped`
flag, so they never match these uppercase French codes and are left untouched.

Usage:
    python -m scripts.fix_foreign_result_codes            # dry-run (no writes)
    python -m scripts.fix_foreign_result_codes --execute  # apply + commit
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict

from core.db import get_connection
from core.common import classify_letrot_placement


def _fetch_fr_codes(cur) -> dict[str, int]:
    """Distinct placement_text codes on French tracks, with row counts."""
    cur.execute(
        """
        SELECT e.placement_text, COUNT(*)
        FROM entry e
        JOIN race r  ON r.race_id = e.race_id
        JOIN track t ON t.track_id = r.track_id
        WHERE t.country = 'FR' AND e.placement_text IS NOT NULL
        GROUP BY e.placement_text
        """
    )
    return {pt: n for pt, n in cur.fetchall()}


def _gal_rate_snapshot(cur, label: str) -> None:
    cur.execute(
        """
        SELECT t.country,
               COUNT(*) FILTER (WHERE NOT e.withdrawn)              AS starts,
               COUNT(*) FILTER (WHERE NOT e.withdrawn AND e.galopp) AS gals
        FROM entry e
        JOIN race r  ON r.race_id = e.race_id
        JOIN track t ON t.track_id = r.track_id
        WHERE t.country IN ('FR','NO','DK','SE')
        GROUP BY t.country ORDER BY t.country
        """
    )
    print(f"\n  gal-rate by country [{label}]:")
    for country, starts, gals in cur.fetchall():
        rate = (100.0 * gals / starts) if starts else 0.0
        print(f"    {country}: starts={starts:>9,}  gals={gals:>9,}  rate={rate:5.1f}%")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--execute", action="store_true", help="apply and commit (default: dry-run)")
    args = ap.parse_args()

    conn = get_connection()
    cur = conn.cursor()

    codes = _fetch_fr_codes(cur)

    # Group codes by the exact flag-set the shared classifier assigns.
    distance_codes: list[str] = []   # galopp + disqualified
    withdrawn_codes: list[str] = []  # withdrawn
    dq_only_codes: list[str] = []    # disqualified (DNF: arrêté / tombé)
    skipped: list[tuple[str, int]] = []

    for code, n in sorted(codes.items(), key=lambda kv: -kv[1]):
        flags = classify_letrot_placement(code)
        if flags.get("galopp"):
            distance_codes.append(code)
        elif flags.get("withdrawn"):
            withdrawn_codes.append(code)
        elif flags.get("disqualified"):
            dq_only_codes.append(code)
        else:
            skipped.append((code, n))

    def _update(set_sql: str, where_extra: str, group_codes: list[str]) -> int:
        if not group_codes:
            return 0
        cur.execute(
            f"""
            UPDATE entry e
            SET {set_sql}
            FROM race r, track t
            WHERE e.race_id = r.race_id AND r.track_id = t.track_id
              AND t.country = 'FR'
              AND e.placement_text = ANY(%s)
              AND {where_extra}
            """,
            [group_codes],
        )
        return cur.rowcount

    print("=== French result-code backfill ===")
    print(f"distance (galopp+dq) codes: {distance_codes}")
    print(f"withdrawn (NP) codes:       {withdrawn_codes}")
    print(f"dq-only (arrêté/tombé):     {dq_only_codes}")
    print(f"skipped (ambiguous):        {[c for c, _ in skipped]}")

    _gal_rate_snapshot(cur, "before")

    n_dist = _update(
        "galopp = TRUE, disqualified = TRUE",
        "NOT (e.galopp AND e.disqualified)",
        distance_codes,
    )
    n_wd = _update("withdrawn = TRUE", "NOT e.withdrawn", withdrawn_codes)
    n_dq = _update("disqualified = TRUE", "NOT e.disqualified", dq_only_codes)

    print(f"\n  rows updated -> distance(galopp+dq)={n_dist:,}  withdrawn={n_wd:,}  dq-only={n_dq:,}")

    _gal_rate_snapshot(cur, "after (in-txn)")

    if args.execute:
        conn.commit()
        print("\n  COMMITTED.")
    else:
        conn.rollback()
        print("\n  DRY-RUN (rolled back). Re-run with --execute to apply.")

    conn.close()


if __name__ == "__main__":
    sys.exit(main())
