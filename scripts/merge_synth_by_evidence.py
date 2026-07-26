"""Merge ATG-synthetic horse rows into strong-id counterparts with evidence.

Nightly `merge_synth_pairs` (category B) already folds `x:CC:NAME` → numeric
`atg_id` when names + countries agree. This script covers the residual cases
where the keeper has `st_id` and/or `letrot_id` but no numeric ATG id, and
name alone is not enough — we require one of:

  * PEDIGREE: same normalised name + birth year + sire + dam
  * DOUBLE ENTRY: both horse rows have an entry in the SAME race_id
    (two ids for one starter — unique is (race_id, horse_id), so this
    slips through; same normalised name makes a true name-collision of
    two distinct horses in one race effectively impossible)
  * DUP RACE COPY: both appear in different race rows that share
    (track_id, race_date, race_number), with agreeing placement or km-time
  * YEAR+COUNTRY (ST keeper only): same name + birth year + compatible
    birth_country, and the synth has at least one entry (so we are not
    folding empty name collisions)

Ambiguous synths (matching >1 keeper under the same evidence tier) are
skipped. Method tag: `synth_evidence`.

CLI
---
    python -m scripts.merge_synth_by_evidence                 # dry-run
    python -m scripts.merge_synth_by_evidence --execute
    python -m scripts.merge_synth_by_evidence --execute --limit 100
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts._merge_helpers import build_argparser, script_runner, perform_merge  # noqa: E402


_PAIRS_SQL = """
WITH synth AS (
    SELECT horse_id, name, atg_id, birth_country, date_of_birth,
           sire_name, dam_name,
           EXTRACT(YEAR FROM date_of_birth) AS birth_year,
           (SELECT COUNT(*) FROM entry e WHERE e.horse_id = horse.horse_id) AS n_entries
      FROM horse
     WHERE atg_id LIKE 'x:%%'
       AND st_id IS NULL
       AND letrot_id IS NULL
),
keepers AS (
    SELECT horse_id, name, atg_id, st_id, letrot_id, birth_country,
           date_of_birth, sire_name, dam_name,
           EXTRACT(YEAR FROM date_of_birth) AS birth_year
      FROM horse
     WHERE (st_id IS NOT NULL OR letrot_id IS NOT NULL OR atg_id ~ '^[0-9]+$')
),
named AS (
    SELECT s.horse_id AS synth_id, k.horse_id AS keeper_id,
           s.name, s.n_entries,
           s.atg_id AS synth_atg, k.atg_id AS keeper_atg,
           k.st_id, k.letrot_id,
           (s.sire_name IS NOT NULL AND k.sire_name IS NOT NULL
            AND s.dam_name IS NOT NULL AND k.dam_name IS NOT NULL
            AND s.birth_year IS NOT NULL AND k.birth_year IS NOT NULL
            AND s.birth_year = k.birth_year
            AND v2_normalize_name(s.sire_name) = v2_normalize_name(k.sire_name)
            AND v2_normalize_name(s.dam_name) = v2_normalize_name(k.dam_name)
           ) AS pedigree_ok,
           (s.birth_year IS NOT NULL AND k.birth_year IS NOT NULL
            AND s.birth_year = k.birth_year
            AND (s.birth_country = k.birth_country
                 OR s.birth_country IS NULL OR k.birth_country IS NULL)
            AND k.st_id IS NOT NULL
            AND s.n_entries > 0
           ) AS year_country_ok,
           EXISTS (
               SELECT 1
                 FROM entry es
                 JOIN entry ek ON ek.horse_id = k.horse_id
                                AND ek.race_id = es.race_id
                WHERE es.horse_id = s.horse_id
           ) AS double_entry_ok,
           EXISTS (
               SELECT 1
                 FROM entry es
                 JOIN race rs ON rs.race_id = es.race_id
                 JOIN entry ek ON ek.horse_id = k.horse_id
                 JOIN race rk ON rk.race_id = ek.race_id
                WHERE es.horse_id = s.horse_id
                  AND es.race_id <> ek.race_id
                  AND rs.race_date = rk.race_date
                  AND rs.race_number IS NOT NULL
                  AND rs.race_number = rk.race_number
                  AND rs.track_id IS NOT DISTINCT FROM rk.track_id
                  AND (
                       (es.placement IS NOT NULL AND ek.placement IS NOT NULL
                        AND es.placement = ek.placement)
                    OR (es.time_seconds IS NOT NULL AND ek.time_seconds IS NOT NULL
                        AND abs(es.time_seconds - ek.time_seconds) <= 0.5)
                  )
           ) AS dup_race_ok
      FROM synth s
      JOIN keepers k
        ON v2_normalize_name(s.name) = v2_normalize_name(k.name)
       AND v2_normalize_name(s.name) <> ''
       AND s.horse_id <> k.horse_id
)
SELECT synth_id, keeper_id, name, n_entries, synth_atg, keeper_atg,
       st_id, letrot_id, pedigree_ok, year_country_ok,
       double_entry_ok, dup_race_ok,
       CASE
         WHEN pedigree_ok THEN 'pedigree'
         WHEN double_entry_ok THEN 'double_entry'
         WHEN dup_race_ok THEN 'dup_race'
         WHEN year_country_ok THEN 'year_country'
       END AS evidence
  FROM named
 WHERE pedigree_ok OR double_entry_ok OR dup_race_ok OR year_country_ok
 ORDER BY n_entries DESC, synth_id, keeper_id
"""


def _fetch_pairs(cur, limit: int | None) -> list[dict]:
    sql = _PAIRS_SQL
    if limit:
        # LIMIT applied after ambiguity filter in Python; fetch a cushion.
        sql += f" LIMIT {int(limit) * 5}"
    cur.execute(sql)
    cols = [d.name for d in cur.description]
    return [dict(zip(cols, r)) for r in cur.fetchall()]


def _unique_pairs(rows: list[dict], limit: int | None) -> list[dict]:
    """Keep only synths with exactly one keeper per evidence tier."""
    from collections import defaultdict

    by_synth: dict[int, list[dict]] = defaultdict(list)
    for r in rows:
        by_synth[r["synth_id"]].append(r)

    out: list[dict] = []
    for synth_id, group in by_synth.items():
        # Prefer strongest evidence; if multiple keepers share that tier, skip.
        for tier in ("pedigree", "double_entry", "dup_race", "year_country"):
            tier_hits = [g for g in group if g["evidence"] == tier]
            if not tier_hits:
                continue
            keepers = {g["keeper_id"] for g in tier_hits}
            if len(keepers) == 1:
                out.append(tier_hits[0])
            break  # only consider the strongest tier that fired
    out.sort(key=lambda r: (-int(r["n_entries"] or 0), r["synth_id"]))
    if limit is not None:
        out = out[: int(limit)]
    return out


def main() -> int:
    args = build_argparser("merge_synth_by_evidence").parse_args()
    with script_runner("merge_synth_by_evidence", args) as (conn, log, summary):
        log(f"[merge_synth_by_evidence] execute={args.execute} "
            f"limit={args.limit}")
        with conn.cursor() as cur:
            raw = _fetch_pairs(cur, args.limit)
        pairs = _unique_pairs(raw, args.limit)
        summary["raw_candidates"] = len(raw)
        summary["candidates"] = len(pairs)
        log(f"raw evidence rows={len(raw)}; unique unambiguous pairs={len(pairs)}")

        by_ev = {}
        for p in pairs:
            by_ev[p["evidence"]] = by_ev.get(p["evidence"], 0) + 1
        log(f"  by evidence: {by_ev}")

        for p in pairs:
            reason = (
                f"synth_evidence/{p['evidence']}: "
                f"synth={p['synth_atg']} -> keeper "
                f"st={p['st_id']} atg={p['keeper_atg']} letrot={p['letrot_id']}"
            )
            log(f"  merge {p['synth_id']} -> {p['keeper_id']} "
                f"({p['name']!r}, entries={p['n_entries']}, "
                f"evidence={p['evidence']})")
            perform_merge(
                conn, log, summary,
                from_id=p["synth_id"], to_id=p["keeper_id"],
                reason=reason, method="synth_evidence",
                dry_run=not args.execute,
                commit_every=args.commit_every,
            )

        if args.execute:
            conn.commit()
    return 0


if __name__ == "__main__":
    sys.exit(main())
