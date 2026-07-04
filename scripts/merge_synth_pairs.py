"""
Category B: merge ATG-synthetic horse rows into their ST-guest real counterparts.

For every horse pair where:
  - `real` row has a numeric `atg_id` (e.g. '749373') AND `st_id` set,
  - `synth` row has an `x:CC:NAME` atg_id with the same normalised name,
  - country matches (or one side has NULL country),

…merge synth → real. The real row keeps its st_id + numeric atg_id; the
synth row's foreign entries get moved over, and any same-race conflicts
are resolved (ATG-sourced entry wins on equipment, see core.identity).

Audit shows ~5,538 such pairs covering ~237k entries. Includes the
Go on Boy / Kobayashi class.

CLI
---
    python -m scripts.merge_synth_pairs              # dry-run
    python -m scripts.merge_synth_pairs --execute    # apply
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
    SELECT real.horse_id  AS real_id,
           real.name      AS real_name,
           real.atg_id    AS real_atg_id,
           synth.horse_id AS synth_id,
           synth.name     AS synth_name,
           synth.atg_id   AS synth_atg_id,
           (SELECT COUNT(*) FROM entry e WHERE e.horse_id = synth.horse_id) AS synth_entries
      FROM horse synth
      JOIN horse real
        ON real.atg_id ~ '^[0-9]+$'
       AND real.atg_id::int <> 0
       AND synth.atg_id LIKE 'x:%%'
       AND v2_normalize_name(real.name) = v2_normalize_name(synth.name)
       AND v2_normalize_name(real.name) <> ''
       AND (real.birth_country = synth.birth_country
            OR real.birth_country IS NULL
            OR synth.birth_country IS NULL)
     WHERE synth.horse_id <> real.horse_id
     ORDER BY (SELECT COUNT(*) FROM entry e WHERE e.horse_id = synth.horse_id) DESC
    """
    if limit:
        sql += f" LIMIT {int(limit)}"
    cur.execute(sql)
    cols = [d.name for d in cur.description]
    return [dict(zip(cols, r)) for r in cur.fetchall()]


def main() -> int:
    args = build_argparser("merge_synth_pairs").parse_args()
    with script_runner("merge_synth_pairs", args) as (conn, log, summary):
        log(f"[merge_synth_pairs] execute={args.execute}, limit={args.limit}")
        with conn.cursor() as cur:
            pairs = _fetch_pairs(cur, args.limit)
        summary["candidates"] = len(pairs)
        log(f"Found {len(pairs)} synth↔real pairs to merge.")

        # Track which synth ids we've already processed (multiple `real`
        # candidates per synth shouldn't double-merge after the first wins).
        merged_synth: set[int] = set()
        for p in pairs:
            if p["synth_id"] in merged_synth:
                continue
            reason = (f"category_b synth pair: real_atg_id={p['real_atg_id']} "
                      f"synth_atg_id={p['synth_atg_id']}")
            res = perform_merge(
                conn, log, summary,
                from_id=p["synth_id"], to_id=p["real_id"],
                reason=reason, method="category_b",
                dry_run=not args.execute,
                commit_every=args.commit_every,
            )
            if "error" not in res:
                merged_synth.add(p["synth_id"])

        if args.execute:
            conn.commit()
    return 0


if __name__ == "__main__":
    sys.exit(main())
