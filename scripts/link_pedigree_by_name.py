"""
Wire up `horse.sire_id` / `horse.dam_id` for rows that carry a textual
`sire_name` / `dam_name` but no FK to the parent's canonical horse row.

This is the structural fix for the "JOSH POWER → OFFSHORE DREAM not
clickable" symptom — it occurs across roughly 45k horses today, mostly
from older ATG/ST imports that recorded pedigree as plain text without
attempting any cross-row resolution.

Rule
----

For each unlinked parent reference:

  1. Normalise the parent name via `v2_normalize_name` (same function
     the strict French-horse matcher uses).
  2. Look up `horse` rows where `v2_normalize_name(name)` matches AND
     `gender_code` is consistent (sire ⇒ {'S', NULL, ''};
     dam ⇒ {'M', NULL, ''}).
  3. Unique match → set the FK.
  4. Zero match with gender filter → relax to name-only.
  5. Still zero or multiple → leave NULL, log to summary.

Legacy bad gender codes (Offshore Dream landed in ATG with
`gender_code='H'` — French hongre/gelding leftover that survived the
canonical translation) are handled by step 4's fallback. Risk of false
attachment is low when the normalised name uniquely matches: among 468k
rows, name collisions are rare. Spot-checks during the rollout will
catch the few exceptions.

Idempotent + fast
-----------------

Only touches rows where the FK is NULL. Once linked, never re-evaluated.
The functional index `idx_horse_v2_normalize_name` makes each lookup an
index probe (< 1ms), so this script can run as part of every cleanup
cycle.

Usage
-----

    python -m scripts.link_pedigree_by_name              # dry-run
    python -m scripts.link_pedigree_by_name --execute    # apply
    python -m scripts.link_pedigree_by_name --execute --limit 1000
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts._merge_helpers import build_argparser, script_runner  # noqa: E402


_CANDIDATES_SQL = {
    "sire": """
        SELECT horse_id, sire_name, date_of_birth
          FROM horse
         WHERE sire_id IS NULL
           AND sire_name IS NOT NULL
           AND v2_normalize_name(sire_name) <> ''
         {limit_clause}
    """,
    "dam": """
        SELECT horse_id, dam_name, date_of_birth
          FROM horse
         WHERE dam_id IS NULL
           AND dam_name IS NOT NULL
           AND v2_normalize_name(dam_name) <> ''
         {limit_clause}
    """,
}

# Gender constraint per role. NULL / '' kept in scope because many older
# rows have no gender code at all (and many wrong codes too — the
# fallback below handles those).
_GENDER_OK = {"sire": "S", "dam": "M"}


def _link_one_role(conn, log, summary, role: str, *, execute: bool,
                   limit: int | None) -> None:
    fk_col   = f"{role}_id"
    name_col = f"{role}_name"
    gender   = _GENDER_OK[role]

    limit_clause = f"LIMIT {int(limit)}" if limit else ""
    sql = _CANDIDATES_SQL[role].format(limit_clause=limit_clause)

    with conn.cursor() as cur:
        cur.execute(sql)
        rows = cur.fetchall()

    log(f"[{role}] {len(rows)} unlinked candidate(s) to evaluate")
    if not rows:
        summary[f"{role}_candidates"] = 0
        return

    stats = {"strict": 0, "relaxed": 0, "ambiguous": 0, "no_match": 0,
             "age_blocked": 0}

    with conn.cursor() as cur:
        for source_id, parent_name, child_dob in rows:
            # Phase 1: strict — name + consistent gender.
            cur.execute(
                """
                SELECT horse_id, date_of_birth FROM horse
                 WHERE v2_normalize_name(name) = v2_normalize_name(%s)
                   AND v2_normalize_name(name) <> ''
                   AND (gender_code = %s OR gender_code IS NULL OR gender_code = '')
                 LIMIT 5
                """,
                (parent_name, gender),
            )
            hits = [(r[0], r[1]) for r in cur.fetchall()]
            phase = "strict"

            # Phase 2: relax gender filter if strict found nothing.
            # Legacy ATG/ST rows commonly carry the French H/F codes for
            # foreign sires that nobody ever canonicalised.
            if not hits:
                cur.execute(
                    """
                    SELECT horse_id, date_of_birth FROM horse
                     WHERE v2_normalize_name(name) = v2_normalize_name(%s)
                       AND v2_normalize_name(name) <> ''
                     LIMIT 5
                    """,
                    (parent_name,),
                )
                hits = [(r[0], r[1]) for r in cur.fetchall()]
                phase = "relaxed"

            # Drop self-references (a row pointing at itself by name).
            hits = [h for h in hits if h[0] != source_id]

            # Parent-age guard: a parent must be born BEFORE its child. Drop
            # any candidate whose DOB is known and is on/after the child's DOB
            # (both known). This prevents the name-only matcher from attaching
            # a young same-named foal as the sire/dam of an older horse
            # (e.g. a 2025 foal linked as dam of a 2014 racer). When it leaves
            # exactly one survivor it also disambiguates same-name collisions.
            if child_dob is not None and hits:
                aged = [h for h in hits
                        if not (h[1] is not None and h[1] >= child_dob)]
                if not aged:
                    stats["age_blocked"] += 1
                    continue
                hits = aged

            if not hits:
                stats["no_match"] += 1
                continue
            if len(hits) > 1:
                stats["ambiguous"] += 1
                continue

            stats[phase] += 1
            if execute:
                cur.execute(
                    f"UPDATE horse SET {fk_col} = %s, last_updated_at = NOW() "
                    f" WHERE horse_id = %s AND {fk_col} IS NULL",
                    (hits[0][0], source_id),
                )

    if execute:
        conn.commit()

    log(f"[{role}]   strict matches:   {stats['strict']}")
    log(f"[{role}]   relaxed matches:  {stats['relaxed']}  "
        "(no consistent gender, name-only)")
    log(f"[{role}]   ambiguous (skip): {stats['ambiguous']}  "
        "(multiple normalised-name hits)")
    log(f"[{role}]   age-blocked(skip):{stats['age_blocked']}  "
        "(only same-name candidate(s) born on/after the child)")
    log(f"[{role}]   no match (skip):  {stats['no_match']}  "
        "(parent absent from horse table)")
    summary[f"{role}_candidates"]  = len(rows)
    summary[f"{role}_linked"]      = stats["strict"] + stats["relaxed"]
    summary[f"{role}_ambiguous"]   = stats["ambiguous"]
    summary[f"{role}_no_match"]    = stats["no_match"]
    summary[f"{role}_relaxed"]     = stats["relaxed"]


def main() -> int:
    args = build_argparser("link_pedigree_by_name").parse_args()

    with script_runner("link_pedigree_by_name", args) as (conn, log, summary):
        log(f"[link_pedigree_by_name] execute={args.execute} "
            f"limit={args.limit}")

        for role in ("sire", "dam"):
            log("")
            _link_one_role(conn, log, summary, role,
                           execute=args.execute, limit=args.limit)

        summary["candidates"] = (
            summary.get("sire_candidates", 0)
            + summary.get("dam_candidates", 0)
        )
        summary["merged"] = (
            summary.get("sire_linked", 0)
            + summary.get("dam_linked", 0)
        )

        if not args.execute:
            log("\nDRY-RUN — no DB writes. Use --execute to apply.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
