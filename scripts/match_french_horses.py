"""
Phase 4 — find LeTrot-only horse rows that should be merged into their
ATG / ST counterparts.

Three-pass:

1. STRICT — `(normalize_name(name), birth_year, birth_country)` exact
   match between LeTrot-side and ATG/ST-side horses that BOTH have
   `date_of_birth` populated. Where exactly one ATG-side row exists with
   this fingerprint, it's a definite duplicate.

2. CO-OCCURRENCE — for each strict candidate, verify that the LeTrot
   horse shares at least one race with the ATG candidate via merged
   `race_id` (only possible after Phase 2/3 have merged race rows). When
   co-occurrence present, confidence is "definite"; otherwise downgraded
   to "review" (CSV only, no auto-merge).

3. NO-DOB FALLBACK — many ATG/ST French horse rows were imported with no
   `date_of_birth` (Iguski Sautonne is the canonical example). For each
   LeTrot horse with full demographics, find ATG/ST candidates that share
   a normalized name AND have ≥ `--no-dob-min-shared` race entries with
   the LeTrot horse on a merged race_id. The shared-race count is the
   sole disambiguator since neither birth year nor country are usable.

Results:
  - Definite candidates → call `core.identity.merge_horses` (column-level).
  - Review candidates  → written to `out/match_french_horses_review.csv`.

Usage
-----

    python -m scripts.match_french_horses              # dry-run
    python -m scripts.match_french_horses --execute   # apply
    python -m scripts.match_french_horses --limit 100
    python -m scripts.match_french_horses --rollback-job <id>
    python -m scripts.match_french_horses --no-dob-min-shared 5  # stricter
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts._merge_helpers import (  # noqa: E402
    build_argparser,
    script_runner,
    perform_merge,
    rollback_horse_merges_by_method,
)


# ---------------------------------------------------------------------------
# Strict candidate discovery
# ---------------------------------------------------------------------------

_STRICT_SQL = """
WITH letrot_only AS (
  SELECT h.horse_id,
         h.name,
         h.date_of_birth,
         EXTRACT(year FROM h.date_of_birth)::int AS birth_year,
         COALESCE(h.birth_country, h.bred_country, h.registration_country) AS country
    FROM horse h
   WHERE h.letrot_id IS NOT NULL
     AND h.atg_id    IS NULL
     AND h.st_id     IS NULL
     AND h.date_of_birth IS NOT NULL
),
atg_side AS (
  SELECT h.horse_id,
         h.name,
         h.date_of_birth,
         EXTRACT(year FROM h.date_of_birth)::int AS birth_year,
         COALESCE(h.birth_country, h.bred_country, h.registration_country) AS country
    FROM horse h
   WHERE (h.atg_id IS NOT NULL OR h.st_id IS NOT NULL)
     AND h.letrot_id IS NULL
     AND h.date_of_birth IS NOT NULL
)
SELECT l.horse_id AS letrot_hid,
       l.name     AS letrot_name,
       l.birth_year,
       l.country,
       a.horse_id AS atg_hid,
       a.name     AS atg_name
  FROM letrot_only l
  JOIN atg_side a
    ON v2_normalize_name(a.name) = v2_normalize_name(l.name)
   AND a.birth_year = l.birth_year
   AND (a.country IS NULL OR l.country IS NULL OR a.country = l.country)
"""


def _fetch_strict_candidates(cur, limit: int | None) -> list[dict]:
    sql = _STRICT_SQL
    sql += " ORDER BY l.horse_id"
    if limit:
        sql += f" LIMIT {int(limit)}"
    cur.execute(sql)
    cols = [d.name for d in cur.description]
    rows = [dict(zip(cols, r)) for r in cur.fetchall()]

    # Resolve ambiguity: drop letrot rows that match >1 atg row (need
    # human review).
    by_letrot: dict[int, list[dict]] = {}
    for r in rows:
        by_letrot.setdefault(r["letrot_hid"], []).append(r)

    unique: list[dict] = []
    ambiguous: list[dict] = []
    for lhid, candidates in by_letrot.items():
        if len(candidates) == 1:
            unique.append(candidates[0])
        else:
            ambiguous.append({**candidates[0],
                              "n_atg_matches": len(candidates),
                              "atg_hids": [c["atg_hid"] for c in candidates]})
    return unique  # ambiguous returned via separate fetch below


def _fetch_ambiguous(cur, limit: int | None) -> list[dict]:
    """Re-fetch only the ambiguous side for the review CSV."""
    sql = _STRICT_SQL + " ORDER BY l.horse_id"
    if limit:
        sql += f" LIMIT {int(limit)}"
    cur.execute(sql)
    cols = [d.name for d in cur.description]
    rows = [dict(zip(cols, r)) for r in cur.fetchall()]
    by_letrot: dict[int, list[dict]] = {}
    for r in rows:
        by_letrot.setdefault(r["letrot_hid"], []).append(r)
    return [{"letrot_hid": k, "name": v[0]["letrot_name"],
             "birth_year": v[0]["birth_year"], "country": v[0]["country"],
             "n_atg_matches": len(v),
             "atg_candidates": [(c["atg_hid"], c["atg_name"]) for c in v]}
            for k, v in by_letrot.items() if len(v) > 1]


# ---------------------------------------------------------------------------
# Co-occurrence corroboration
# ---------------------------------------------------------------------------

def _co_occurrence(cur, letrot_hid: int, atg_hid: int) -> int:
    """Count races where both horses have entries AND the program numbers
    match (or both NULL). After Phase 2/3, identical races are already
    merged, so a shared race_id is strong evidence."""
    cur.execute(
        """
        SELECT COUNT(*)
          FROM entry e_l
          JOIN entry e_a
            ON e_l.race_id = e_a.race_id
           AND (e_l.program_number = e_a.program_number
                OR (e_l.program_number IS NULL AND e_a.program_number IS NULL))
         WHERE e_l.horse_id = %s
           AND e_a.horse_id = %s
        """,
        (letrot_hid, atg_hid),
    )
    return cur.fetchone()[0]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

_NO_DOB_SQL = """
WITH letrot_full AS (
  SELECT h.horse_id, h.name,
         v2_normalize_name(h.name) AS nn
    FROM horse h
   WHERE h.letrot_id IS NOT NULL
     AND h.atg_id IS NULL AND h.st_id IS NULL
), atg_nodob AS (
  SELECT h.horse_id, h.name,
         v2_normalize_name(h.name) AS nn
    FROM horse h
   WHERE (h.atg_id IS NOT NULL OR h.st_id IS NOT NULL)
     AND h.letrot_id IS NULL
     AND h.date_of_birth IS NULL
)
SELECT l.horse_id AS letrot_hid, l.name AS letrot_name,
       a.horse_id AS atg_hid,    a.name AS atg_name
  FROM letrot_full l JOIN atg_nodob a USING (nn)
 ORDER BY l.horse_id
"""


def _fetch_no_dob_candidates(cur, limit: int | None) -> tuple[list[dict], list[dict]]:
    """No-DOB pass: name match only. Returns (unique_per_letrot, ambiguous_groups).

    `unique_per_letrot` contains the LeTrot horses that map to exactly one
    ATG candidate (still subject to co-occurrence filtering downstream).
    Ambiguous groups (multiple ATG matches) go to the review CSV — too
    risky to auto-merge because the only disambiguator would be the
    shared-race count, and a single name collision can mislead.
    """
    sql = _NO_DOB_SQL
    if limit:
        sql += f" LIMIT {int(limit)}"
    cur.execute(sql)
    cols = [d.name for d in cur.description]
    rows = [dict(zip(cols, r)) for r in cur.fetchall()]

    by_letrot: dict[int, list[dict]] = {}
    for r in rows:
        by_letrot.setdefault(r["letrot_hid"], []).append(r)

    unique = [v[0] for v in by_letrot.values() if len(v) == 1]
    ambiguous = [{"letrot_hid": k, "name": v[0]["letrot_name"],
                  "n_atg_matches": len(v),
                  "atg_candidates": [(c["atg_hid"], c["atg_name"]) for c in v]}
                 for k, v in by_letrot.items() if len(v) > 1]
    return unique, ambiguous


def main() -> int:
    parser = build_argparser("match_french_horses")
    parser.add_argument("--no-dob-min-shared", type=int, default=3,
                        help="minimum shared races for the no-DOB fallback "
                             "pass to auto-merge (default 3)")
    args = parser.parse_args()

    if args.rollback_job:
        return _rollback(args)

    out_path = _ROOT / "out" / "match_french_horses_review.csv"

    with script_runner("match_french_horses", args) as (conn, log, summary):
        log(f"[match_french_horses] execute={args.execute} limit={args.limit}")
        with conn.cursor() as cur:
            unique = _fetch_strict_candidates(cur, args.limit)
            ambiguous = _fetch_ambiguous(cur, args.limit)

        log(f"Found {len(unique)} unique strict matches "
            f"(+ {len(ambiguous)} ambiguous → review CSV).")
        summary["candidates"] = len(unique)

        # Write ambiguous review CSV.
        if ambiguous:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(out_path, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["letrot_hid", "name", "birth_year", "country",
                            "n_atg_matches", "atg_candidates"])
                for a in ambiguous:
                    w.writerow([a["letrot_hid"], a["name"], a["birth_year"],
                                a["country"], a["n_atg_matches"],
                                "|".join(f"{h}:{n}" for h, n in a["atg_candidates"])])
            log(f"Wrote review CSV: {out_path}")

        definite = []
        review = []
        for r in unique:
            with conn.cursor() as cur:
                co = _co_occurrence(cur, r["letrot_hid"], r["atg_hid"])
            r["co_occurrence"] = co
            if co >= 1:
                r["confidence"] = "definite"
                definite.append(r)
            else:
                r["confidence"] = "review_no_cooccur"
                review.append(r)

        log(f"  {len(definite)} definite (≥1 shared race), "
            f"{len(review)} review (no co-occurrence found)")

        # Append no-cooccurrence to review CSV.
        if review:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(out_path, "a", newline="") as f:
                w = csv.writer(f)
                for r in review:
                    w.writerow([r["letrot_hid"], r["letrot_name"], r["birth_year"],
                                r["country"], "1_no_cooccur",
                                f"{r['atg_hid']}:{r['atg_name']}"])

        for r in definite[:50]:
            log(f"  {r['letrot_hid']} ({r['letrot_name']!r}, {r['birth_year']}, "
                f"{r['country']}) -> {r['atg_hid']} ({r['atg_name']!r}) "
                f"co_occur={r['co_occurrence']}")
        if len(definite) > 50:
            log(f"  ... and {len(definite) - 50} more definite candidates")

        # Third pass: no-DOB fallback (name match + strong co-occurrence).
        # Limited to LeTrot horses with no DOB-matching ATG counterpart
        # (those are already handled above). Skips any letrot_hid that
        # the strict pass already auto-merged.
        already_merged_lhids = {r["letrot_hid"] for r in definite}
        with conn.cursor() as cur:
            no_dob_unique, no_dob_amb = _fetch_no_dob_candidates(cur, args.limit)
        no_dob_unique = [r for r in no_dob_unique
                         if r["letrot_hid"] not in already_merged_lhids]
        no_dob_definite: list[dict] = []
        no_dob_review: list[dict] = []
        for r in no_dob_unique:
            with conn.cursor() as cur:
                co = _co_occurrence(cur, r["letrot_hid"], r["atg_hid"])
            r["co_occurrence"] = co
            if co >= args.no_dob_min_shared:
                no_dob_definite.append(r)
            elif co >= 1:
                no_dob_review.append(r)

        log(f"\nNo-DOB pass: {len(no_dob_unique)} name-only unique candidates → "
            f"{len(no_dob_definite)} definite (≥{args.no_dob_min_shared} "
            f"shared races), {len(no_dob_review)} review (1-{args.no_dob_min_shared - 1}), "
            f"{len(no_dob_amb)} ambiguous (multiple ATG matches).")

        # Write no-DOB ambiguous and weak co-occurrence to review CSV.
        if no_dob_amb or no_dob_review:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(out_path, "a", newline="") as f:
                w = csv.writer(f)
                for a in no_dob_amb:
                    w.writerow([
                        a["letrot_hid"], a["name"], "", "",
                        f"no_dob_ambig_{a['n_atg_matches']}",
                        "|".join(f"{h}:{n}" for h, n in a["atg_candidates"]),
                    ])
                for r in no_dob_review:
                    w.writerow([
                        r["letrot_hid"], r["letrot_name"], "", "",
                        f"no_dob_weak_cooccur_{r['co_occurrence']}",
                        f"{r['atg_hid']}:{r['atg_name']}",
                    ])

        for r in no_dob_definite[:20]:
            log(f"  [no-DOB] {r['letrot_hid']} ({r['letrot_name']!r}) -> "
                f"{r['atg_hid']} ({r['atg_name']!r}) co_occur={r['co_occurrence']}")
        if len(no_dob_definite) > 20:
            log(f"  ... and {len(no_dob_definite) - 20} more no-DOB definite")

        all_definite = definite + no_dob_definite
        summary["candidates"] = len(all_definite)

        if not args.execute:
            summary["merged"] = len(all_definite)
            log("\nDRY-RUN — no DB changes. Use --execute to apply.")
            return 0

        # Execute auto-merges via perform_merge.
        for r in definite:
            perform_merge(
                conn, log, summary,
                from_id=r["letrot_hid"],
                to_id=r["atg_hid"],
                reason=f"name+year+country match (co_occur={r['co_occurrence']})",
                method="match_french_horses",
                dry_run=False,
                commit_every=args.commit_every,
                kind="horse",
            )
        for r in no_dob_definite:
            perform_merge(
                conn, log, summary,
                from_id=r["letrot_hid"],
                to_id=r["atg_hid"],
                reason=f"name+co_occur={r['co_occurrence']} (no-DOB pass)",
                method="match_french_horses_no_dob",
                dry_run=False,
                commit_every=args.commit_every,
                kind="horse",
            )
        conn.commit()
    return 0


def _rollback(args) -> int:
    with script_runner("match_french_horses_rollback", args) as (conn, log, summary):
        # Roll back both the strict and no-DOB passes.
        for method in ("match_french_horses", "match_french_horses_no_dob"):
            res = rollback_horse_merges_by_method(
                conn, log, method,
                job_run_id=args.rollback_job,
            )
            for k, v in res.items():
                summary[k] = summary.get(k, 0) + (v if isinstance(v, int) else 0)
    return 0


if __name__ == "__main__":
    sys.exit(main())
