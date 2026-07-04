"""
Self-healing French sex-code normalizer for LeTrot-sourced entries.

The problem
-----------

LeTrot encodes sex as H (Hongre / gelding), F (Femelle / mare), M (Mâle /
stallion) which **collides** with the canonical convention used by ST,
ATG and the rest of the codebase (G / M / S, where 'M' means *Mare*).

Going forward `etl.import_letrot` calls `translate_french_sex` at ingest
time, so freshly-imported rows always land with canonical codes. This
script exists to:

  1. (One-time, already executed) backfill the historical pre-translation
     rows so legacy data uses canonical codes.
  2. (Every cleanup run, idempotent) heal any drift between
     `entry.sex` and `horse.gender_code` on LeTrot-primary French
     entries. Drift can appear after re-imports, after a manual merge
     fixes the horse but leaves old entries un-touched, or after a
     LeTrot scrape lands before a co-occurring ATG row pulls the
     `primary_source` flag away from letrot.

All sex data is now on the canonical Swedish H/V/S convention (see
`scripts.migrate_sex_hvs`). The self-heal pass corrects French entries
whose `entry.sex` disagrees with the horse's `gender_code` **only on the
mare axis** — i.e. exactly one of the two says mare ('S'). A mare never
changes sex, so a mare/non-mare disagreement is always a source mislabel
and the horse's `gender_code` wins.

Crucially we DO NOT touch the castration axis (stallion 'H' <-> gelding
'V'): a horse legitimately castrated between starts must keep its
race-time snapshot (stallion in old races, gelding later). Forcing those
entries to the current passport value would erase real history.

Scope is French races (`track.country = 'FR'`) where the H/F/M → H/V/S
collisions originate (LeTrot, and ATG's un-translated French starters).

This makes the script safe + idempotent to run as part of
`scripts/cleanup_merges` on every ingest cycle.

Usage
-----

    python -m scripts.normalize_french_sex                    # dry-run
    python -m scripts.normalize_french_sex --execute          # apply
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts._merge_helpers import build_argparser, script_runner  # noqa: E402


# ---------------------------------------------------------------------------
# Self-healing pass — entries whose sex disagrees with the horse.
# ---------------------------------------------------------------------------

# Mare-axis disagreement only: exactly one of (entry, passport) is mare.
# `(a = 'S') <> (b = 'S')` is TRUE only when one side is mare and the
# other isn't. Castration-axis disagreements (H <-> V) are excluded.
_HEAL_WHERE = """
       AND t.country  = 'FR'
       AND h.gender_code IN ('H', 'V', 'S')
       AND e.sex        IN ('H', 'V', 'S')
       AND e.sex <> h.gender_code
       AND ((h.gender_code = 'S') <> (e.sex = 'S'))
"""

_HEAL_SELECT = f"""
    SELECT e.sex AS entry_sex, h.gender_code AS horse_gc, COUNT(*) AS n
      FROM entry  e
      JOIN race   r ON r.race_id  = e.race_id
      JOIN track  t ON t.track_id = r.track_id
      JOIN horse  h ON h.horse_id = e.horse_id
     WHERE TRUE
       {_HEAL_WHERE}
     GROUP BY 1, 2
     ORDER BY 3 DESC
"""

_HEAL_UPDATE = f"""
    UPDATE entry e
       SET sex = h.gender_code,
           last_updated_at = NOW()
      FROM horse h, race r, track t
     WHERE e.horse_id = h.horse_id
       AND e.race_id  = r.race_id
       AND r.track_id = t.track_id
       {_HEAL_WHERE}
"""


def _run_heal(conn, log, summary, execute: bool) -> None:
    with conn.cursor() as cur:
        cur.execute(_HEAL_SELECT)
        rows = cur.fetchall()

    if not rows:
        log("self-heal pass: 0 mare-axis mismatches — nothing to do.")
        summary["heal_candidates"] = 0
        return

    total = sum(n for _, _, n in rows)
    log(f"self-heal pass: {total} French entries with a mare-axis mismatch")
    for esex, hgc, n in rows[:20]:
        log(f"  {esex!r:>5} → {hgc!r:>5}   ({n} rows)")
    if len(rows) > 20:
        log(f"  …and {len(rows) - 20} more drift combinations")
    summary["heal_candidates"] = total

    if not execute:
        log("(dry-run — pass --execute to apply)")
        return

    with conn.cursor() as cur:
        cur.execute(_HEAL_UPDATE)
        n = cur.rowcount
    log(f"self-heal pass: UPDATEd {n} entry rows")
    summary["heal_updated"] = n


def main() -> int:
    parser = build_argparser("normalize_french_sex")
    args = parser.parse_args()

    with script_runner("normalize_french_sex", args) as (conn, log, summary):
        log(f"[normalize_french_sex] execute={args.execute}")

        _run_heal(conn, log, summary, args.execute)

        # Roll up the canonical 'candidates' / 'merged' fields the
        # cleanup_merges orchestrator looks for.
        summary["candidates"] = summary.get("heal_candidates", 0)
        summary["merged"] = summary.get("heal_updated", 0)

        if args.execute:
            conn.commit()
            log("\nCommitted.")
        else:
            log("\nDRY-RUN — no DB writes.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
