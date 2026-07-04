"""
One-time migration: standardize ALL sex data onto the canonical Swedish
H/V/S convention (H=hingst/stallion, V=valack/gelding, S=sto/mare).

Background
----------

Historically stable-v2 carried THREE colliding sex conventions:

  * horse.gender_code   — meant to be H/V/S, but LeTrot ingest wrote the
                          old G/M/S codes (G=gelding, M=mare, S=stallion).
  * entry.sex           — a mix of:
                            - ST lowercase v/h/s  (valack/hingst/sto)
                            - ATG English first-letter S/G/M
                              (Stallion/Gelding/Mare)
                            - LeTrot-translated old G/M/S
                            - ATG French first-letter F/H (femelle/hongre)

This script rewrites both columns to a single H/V/S convention. It does
NOT touch the castration axis (a stallion that became a gelding keeps its
race-time snapshot) — the only cross-row "healing" lives in
`scripts.normalize_french_sex`, which runs every cleanup cycle.

horse.gender_code mapping
-------------------------

  'M' -> 'S'   (LeTrot mare;  'M' is only ever produced by LeTrot)
  'G' -> 'V'   (LeTrot gelding; 'G' is only ever produced by LeTrot)
  'S' -> 'H'   ONLY for letrot-only horses (primary_source='letrot' AND no
               atg_id/st_id). For those rows 'S' is LeTrot's old *stallion*
               code (mâle). Every other 'S' is a real H/V/S mare and is
               left untouched.
  'H' / 'V'    already canonical — untouched.

entry.sex mapping (case-sensitive, applied in a single CASE)
------------------------------------------------------------

  'v'->'V' 'h'->'H' 's'->'S'   (ST lowercase)
  'G'->'V'                     (gelding: LeTrot-old AND ATG English)
  'M'->'S'                     (mare:    LeTrot-old AND ATG English)
  'S'->'H'                     (stallion:LeTrot-old AND ATG English)
  'F'->'S'                     (ATG French femelle = mare)
  'H'->'V'                     (ATG French hongre  = gelding)

Then: any horse still missing a gender_code is seeded from its most
recent entry's (now H/V/S) sex.

Usage
-----

    python -m scripts.migrate_sex_hvs                 # dry-run
    python -m scripts.migrate_sex_hvs --execute       # apply
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts._merge_helpers import build_argparser, script_runner  # noqa: E402


# ---------------------------------------------------------------------------
# horse.gender_code
# ---------------------------------------------------------------------------

_HORSE_GLOBAL_UPDATE = """
    UPDATE horse
       SET gender_code = CASE gender_code WHEN 'M' THEN 'S' WHEN 'G' THEN 'V' END,
           last_updated_at = NOW()
     WHERE gender_code IN ('M', 'G')
"""

# letrot-only 'S' = LeTrot's old stallion (mâle) code -> H.
_HORSE_LETROT_S_UPDATE = """
    UPDATE horse
       SET gender_code = 'H',
           last_updated_at = NOW()
     WHERE gender_code = 'S'
       AND primary_source = 'letrot'
       AND atg_id IS NULL
       AND st_id  IS NULL
"""

# ---------------------------------------------------------------------------
# entry.sex (single case-sensitive CASE so we never double-map)
# ---------------------------------------------------------------------------

_ENTRY_UPDATE = """
    UPDATE entry
       SET sex = CASE sex
                     WHEN 'v' THEN 'V' WHEN 'h' THEN 'H' WHEN 's' THEN 'S'
                     WHEN 'G' THEN 'V' WHEN 'M' THEN 'S' WHEN 'S' THEN 'H'
                     WHEN 'F' THEN 'S' WHEN 'H' THEN 'V'
                     ELSE sex
                 END,
           last_updated_at = NOW()
     WHERE sex IN ('v', 'h', 's', 'G', 'M', 'S', 'F', 'H')
"""

# ---------------------------------------------------------------------------
# Seed null gender_code from the most recent entry sex.
# ---------------------------------------------------------------------------

_SEED_GENDER_UPDATE = """
    UPDATE horse h
       SET gender_code = sub.sex,
           last_updated_at = NOW()
      FROM (
            SELECT DISTINCT ON (e.horse_id) e.horse_id, e.sex
              FROM entry e
              JOIN race  r ON r.race_id = e.race_id
             WHERE e.sex IN ('H', 'V', 'S')
             ORDER BY e.horse_id, r.race_date DESC NULLS LAST
           ) sub
     WHERE h.horse_id = sub.horse_id
       AND (h.gender_code IS NULL OR h.gender_code = '' OR h.gender_code = 'ZZ')
"""


def _count(cur, table: str, where: str) -> int:
    cur.execute(f"SELECT COUNT(*) FROM {table} WHERE {where}")
    return cur.fetchone()[0]


def main() -> int:
    parser = build_argparser("migrate_sex_hvs")
    args = parser.parse_args()

    with script_runner("migrate_sex_hvs", args) as (conn, log, summary):
        log(f"[migrate_sex_hvs] execute={args.execute}")

        with conn.cursor() as cur:
            h_global = _count(cur, "horse", "gender_code IN ('M','G')")
            h_letrot_s = _count(
                cur, "horse",
                "gender_code = 'S' AND primary_source = 'letrot' "
                "AND atg_id IS NULL AND st_id IS NULL")
            e_legacy = _count(
                cur, "entry", "sex IN ('v','h','s','G','M','S','F','H')")
            h_null = _count(
                cur, "horse",
                "gender_code IS NULL OR gender_code = '' OR gender_code = 'ZZ'")

        log(f"  horse.gender_code M/G -> S/V : {h_global:,}")
        log(f"  horse.gender_code letrot-only 'S' -> H : {h_letrot_s:,}")
        log(f"  entry.sex legacy -> H/V/S    : {e_legacy:,}")
        log(f"  horse.gender_code NULL (seed from entry) : {h_null:,}")
        summary["candidates"] = h_global + h_letrot_s + e_legacy + h_null

        if not args.execute:
            log("\nDRY-RUN — no DB writes. Pass --execute to apply.")
            return 0

        with conn.cursor() as cur:
            # IMPORTANT: convert letrot-only 'S' (old stallion) -> 'H' FIRST.
            # If we ran the M->S step first it would mint new 'S' rows (former
            # mares) that this step would then wrongly flip to stallion.
            cur.execute(_HORSE_LETROT_S_UPDATE)
            n2 = cur.rowcount
            cur.execute(_HORSE_GLOBAL_UPDATE)
            n1 = cur.rowcount
            conn.commit()
            log(f"  horse.gender_code: {n2:,} letrot 'S' -> H, {n1:,} M/G rewritten")

            cur.execute(_ENTRY_UPDATE)
            n3 = cur.rowcount
            conn.commit()
            log(f"  entry.sex: {n3:,} rows rewritten to H/V/S")

            cur.execute(_SEED_GENDER_UPDATE)
            n4 = cur.rowcount
            conn.commit()
            log(f"  horse.gender_code: {n4:,} seeded from latest entry")

        summary["merged"] = n1 + n2 + n3 + n4
        summary["horse_mg"] = n1
        summary["horse_letrot_s"] = n2
        summary["entry"] = n3
        summary["horse_seeded"] = n4
        log("\nCommitted.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
