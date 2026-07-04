"""
Category G: merge ATG-synthetic person rows into their ST real counterparts.

Mirrors `merge_synth_pairs.py` (horses) for drivers/trainers/owners.

For every person pair where:
  - `real` row has a numeric `atg_id` AND st_id set (or just numeric atg_id),
  - `synth` row has an `x:CC:NAME` atg_id with the same uppercased name,

…merge synth → real via `core.identity.merge_persons`. Audit shows
~2,114 such pairs.

CLI
---
    python -m scripts.merge_synth_pairs_persons              # dry-run
    python -m scripts.merge_synth_pairs_persons --execute    # apply
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts._merge_helpers import build_argparser, script_runner, perform_merge  # noqa: E402


def _fetch_pairs(cur, limit: int | None) -> list[dict]:
    sql = """
    SELECT real.person_id  AS real_id,
           real.name       AS real_name,
           real.atg_id     AS real_atg_id,
           synth.person_id AS synth_id,
           synth.name      AS synth_name,
           synth.atg_id    AS synth_atg_id
      FROM person synth
      JOIN person real
        ON real.atg_id ~ '^[0-9]+$'
       AND real.atg_id::int <> 0
       AND synth.atg_id LIKE 'x:%%'
       AND v2_normalize_name(real.name) = v2_normalize_name(synth.name)
       AND v2_normalize_name(real.name) <> ''
     WHERE synth.person_id <> real.person_id
     ORDER BY synth.person_id
    """
    if limit:
        sql += f" LIMIT {int(limit)}"
    cur.execute(sql)
    cols = [d.name for d in cur.description]
    return [dict(zip(cols, r)) for r in cur.fetchall()]


def main() -> int:
    args = build_argparser("merge_synth_pairs_persons").parse_args()
    with script_runner("merge_synth_pairs_persons", args) as (conn, log, summary):
        log(f"[merge_synth_pairs_persons] execute={args.execute}, limit={args.limit}")
        with conn.cursor() as cur:
            pairs = _fetch_pairs(cur, args.limit)
        summary["candidates"] = len(pairs)
        log(f"Found {len(pairs)} person synth↔real pairs.")

        merged_synth: set[int] = set()
        for p in pairs:
            if p["synth_id"] in merged_synth:
                continue
            reason = (f"category_g person synth pair: real_atg_id={p['real_atg_id']} "
                      f"synth_atg_id={p['synth_atg_id']}")
            res = perform_merge(
                conn, log, summary,
                from_id=p["synth_id"], to_id=p["real_id"],
                reason=reason, method="category_g",
                dry_run=not args.execute,
                commit_every=args.commit_every,
                kind="person",
            )
            if "error" not in res:
                merged_synth.add(p["synth_id"])

        if args.execute:
            conn.commit()
    return 0


if __name__ == "__main__":
    sys.exit(main())
