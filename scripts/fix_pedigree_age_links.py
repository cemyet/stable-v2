"""
Unlink pedigree FKs (`horse.sire_id` / `horse.dam_id`) that are provably
impossible because the linked parent is NOT older than the child.

Background
----------

`link_pedigree_by_name` historically resolved a textual `sire_name` /
`dam_name` to the unique same-name canonical horse, with no birth-year
sanity check. When a young foal shares a name with an older parent (very
common for French lines that recycle names), the matcher attached the
*foal* as the sire/dam of an *older* racing horse — producing horse pages
with "offspring older than the parent" (e.g. /horse/449588 Medina du Rib,
born 2022, linked as dam of horses born 2005-2015).

This script removes only the FK that is unambiguously wrong, keeping the
textual `sire_name` / `dam_name` intact (so the pedigree still displays —
it just isn't linked to the wrong horse). `link_pedigree_by_name` now
carries a matching age guard, so it will not re-create these links.

Safety filter
-------------

A link is unlinked only when BOTH parent and child have a *reliable* DOB:
  * year >= 1990 (modern racing data; pre-1990 DOBs are unreliable), and
  * not the `1910-11-12` placeholder some legacy imports used,
and the parent's DOB is on/after the child's DOB (parent not older).

Pre-1990 / placeholder rows are intentionally left untouched: their birth
dates are too noisy to make a confident call, and a blunt fix there would
remove legitimate links.

Usage
-----

    python -m scripts.fix_pedigree_age_links              # dry-run
    python -m scripts.fix_pedigree_age_links --execute    # apply
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts._merge_helpers import build_argparser, script_runner  # noqa: E402


# Both parent and child DOB must be "reliable" for us to act.
_RELIABLE = (
    "{a}.date_of_birth IS NOT NULL "
    "AND {a}.date_of_birth >= DATE '1990-01-01' "
    "AND {a}.date_of_birth <> DATE '1910-11-12'"
)


def _wrong_links_sql(role: str) -> str:
    fk = f"{role}_id"
    return f"""
        SELECT c.horse_id AS child_id, c.name AS child_name, c.date_of_birth AS child_dob,
               p.horse_id AS parent_id, p.name AS parent_name, p.date_of_birth AS parent_dob
          FROM horse c
          JOIN horse p ON p.horse_id = c.{fk}
         WHERE {_RELIABLE.format(a='c')}
           AND {_RELIABLE.format(a='p')}
           AND p.date_of_birth >= c.date_of_birth
         ORDER BY p.date_of_birth DESC
    """


def _fix_role(conn, log, summary, role: str, *, execute: bool,
              limit: int | None) -> None:
    fk = f"{role}_id"
    with conn.cursor() as cur:
        cur.execute(_wrong_links_sql(role))
        rows = cur.fetchall()

    if limit:
        rows = rows[:limit]

    log(f"[{role}] {len(rows)} impossible link(s) "
        f"(parent born on/after child, both DOB reliable)")
    for child_id, child_name, child_dob, parent_id, parent_name, parent_dob in rows[:10]:
        log(f"    /horse/{child_id} {child_name} ({child_dob}) "
            f"-{role}-> /horse/{parent_id} {parent_name} ({parent_dob})  [unlink]")
    if len(rows) > 10:
        log(f"    ... and {len(rows) - 10} more")

    if execute and rows:
        child_ids = [r[0] for r in rows]
        with conn.cursor() as cur:
            cur.execute(
                f"UPDATE horse SET {fk} = NULL, last_updated_at = NOW() "
                f" WHERE horse_id = ANY(%s)",
                (child_ids,),
            )
        conn.commit()
        log(f"[{role}] unlinked {len(child_ids)} row(s)")

    summary[f"{role}_wrong"] = len(rows)
    summary[f"{role}_unlinked"] = len(rows) if execute else 0


def main() -> int:
    args = build_argparser("fix_pedigree_age_links").parse_args()

    with script_runner("fix_pedigree_age_links", args) as (conn, log, summary):
        log(f"[fix_pedigree_age_links] execute={args.execute} limit={args.limit}")
        for role in ("sire", "dam"):
            log("")
            _fix_role(conn, log, summary, role,
                      execute=args.execute, limit=args.limit)

        summary["candidates"] = (
            summary.get("sire_wrong", 0) + summary.get("dam_wrong", 0)
        )
        summary["merged"] = (
            summary.get("sire_unlinked", 0) + summary.get("dam_unlinked", 0)
        )

        if not args.execute:
            log("\nDRY-RUN — no DB writes. Use --execute to apply.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
