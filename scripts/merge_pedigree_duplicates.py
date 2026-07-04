"""
Merge horses sharing the same (name, birth_year, sire_name, dam_name).

Identifies high-confidence duplicate groups (audit shows ~735) and merges
them into a single canonical row. Picks the canonical row per group:

  1. Most entries.
  2. Has an `st_id` (oldest registry).
  3. Lowest horse_id (stable tie-break).

All non-canonical rows in the group merge into the canonical via
`core.identity.merge_horses` (`method='pedigree'`).

CLI
---
    python -m scripts.merge_pedigree_duplicates              # dry-run
    python -m scripts.merge_pedigree_duplicates --execute    # apply
    python -m scripts.merge_pedigree_duplicates --limit 50   # process first 50 groups
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts._merge_helpers import build_argparser, script_runner, perform_merge  # noqa: E402


# Placeholder names that get reused across horses — never auto-merge.
_PLACEHOLDER_PATTERN = (
    "^(ej färdigreg|ej fardigreg|ej regist|"
    "unnamed|no name|sans nom|ohne name|"
    "namnlös|namnlos|nimeton)"
)


def _fetch_groups(cur, limit: int | None) -> list[dict]:
    sql = f"""
    WITH groups AS (
      SELECT v2_normalize_name(name)      AS nname,
             EXTRACT(year FROM date_of_birth)::int AS yr,
             v2_normalize_name(sire_name) AS sn,
             v2_normalize_name(dam_name)  AS dn,
             array_agg(horse_id ORDER BY
                 (SELECT COUNT(*) FROM entry e WHERE e.horse_id = h.horse_id) DESC,
                 (CASE WHEN st_id IS NULL THEN 1 ELSE 0 END),
                 horse_id ASC
             ) AS ranked_ids,
             COUNT(*) AS n,
             MAX(name) AS sample_name
        FROM horse h
       WHERE name IS NOT NULL AND date_of_birth IS NOT NULL
         AND sire_name IS NOT NULL AND dam_name IS NOT NULL
         -- Skip placeholder names that reuse across many horses.
         AND lower(name) !~ '{_PLACEHOLDER_PATTERN}'
       GROUP BY 1,2,3,4
      HAVING COUNT(*) > 1
    )
    SELECT sample_name, yr, n, ranked_ids
      FROM groups
     ORDER BY n DESC, sample_name
    """
    if limit:
        sql += f" LIMIT {int(limit)}"
    cur.execute(sql)
    rows = []
    for r in cur.fetchall():
        rows.append({"name": r[0], "year": r[1], "size": r[2], "ids": list(r[3] or [])})
    return rows


def main() -> int:
    args = build_argparser("merge_pedigree_duplicates").parse_args()
    with script_runner("merge_pedigree_duplicates", args) as (conn, log, summary):
        log(f"[merge_pedigree_duplicates] execute={args.execute}, limit={args.limit}")
        with conn.cursor() as cur:
            groups = _fetch_groups(cur, args.limit)
        summary["candidates"] = sum(max(0, g["size"] - 1) for g in groups)
        log(f"Found {len(groups)} pedigree groups, {summary['candidates']} merges to attempt.")

        for g in groups:
            ids = g["ids"]
            if len(ids) < 2:
                continue
            keeper = ids[0]
            for src_id in ids[1:]:
                reason = (f"pedigree merge: name={g['name']!r} year={g['year']} "
                          f"group_size={g['size']}")
                perform_merge(
                    conn, log, summary,
                    from_id=src_id, to_id=keeper,
                    reason=reason, method="pedigree",
                    dry_run=not args.execute,
                    commit_every=args.commit_every,
                )

        if args.execute:
            conn.commit()
    return 0


if __name__ == "__main__":
    sys.exit(main())
