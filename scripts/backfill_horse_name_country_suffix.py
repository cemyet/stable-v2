"""
Backfill ` (XX)` country suffix into `horse.name` so the frontend's
`flagify()` helper can render a flag for all-caps unmatched horses.

Background
----------
The frontend `flagify(name)` JS helper extracts the country from a trailing
`(XX)` token in the name (e.g. "Aubrion du Gers (FR)" → 🇫🇷 flag). When a
horse name has NO suffix AND is ALL-UPPERCASE, the helper renders a "?"
indicating "source unknown" — even when `horse.birth_country` is populated.

ATG and LeTrot store horse names without a country suffix and in
uppercase, so ~20,000 ATG horses and ~44,000 LeTrot horses end up with
the "?" instead of a proper flag, despite us knowing their country.

This script appends the suffix in-place at the data layer (no frontend
changes), and is idempotent — only matches horses whose name is already
all-caps with no existing `(XX)` token.

Two passes
----------
  A. INFER COUNTRY: set birth_country='FR' on LeTrot-primary horses with
     NULL birth_country. LeTrot only covers French racing, so any horse
     whose canonical row was created from a LeTrot scrape is FR.
  B. APPEND SUFFIX: for every all-caps horse with a known
     birth_country and no existing (XX) suffix, set
        name := name || ' (' || birth_country || ')'

Rollback
--------
The original names are preserved in `horse.source_data._name_history`
before mutation, so this script is reversible:

    python -m scripts.backfill_horse_name_country_suffix --rollback

Usage
-----
    python -m scripts.backfill_horse_name_country_suffix           # dry-run
    python -m scripts.backfill_horse_name_country_suffix --execute # apply
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from psycopg2.extras import Json  # noqa: E402

from core.db import get_connection  # noqa: E402


_INFER_SQL = """
UPDATE horse
   SET birth_country = 'FR',
       last_updated_at = NOW()
 WHERE primary_source = 'letrot'
   AND birth_country IS NULL
"""

_SELECT_SUFFIX_CANDIDATES = """
SELECT horse_id, name, birth_country, source_data
  FROM horse
 WHERE name = upper(name)
   AND name !~ '\\([A-Z]{2,3}\\)'
   AND birth_country IS NOT NULL
   AND birth_country NOT IN ('SE', 'SWE')  -- SE is the frontend default
 ORDER BY horse_id
"""

_UPDATE_NAME_SQL = """
UPDATE horse
   SET name = %s,
       source_data = %s,
       last_updated_at = NOW()
 WHERE horse_id = %s
"""

_SELECT_ROLLBACK = """
SELECT horse_id, name, source_data
  FROM horse
 WHERE source_data ? '_name_history'
"""


def _build_suffix(name: str, country: str) -> str:
    return f"{name} ({country})"


def _record_history(source_data: dict | None, old_name: str,
                    new_name: str) -> dict:
    sd = dict(source_data or {})
    hist = list(sd.get("_name_history") or [])
    hist.append({"old": old_name, "new": new_name,
                 "by": "backfill_horse_name_country_suffix"})
    sd["_name_history"] = hist
    return sd


def run_forward(*, execute: bool) -> dict:
    conn = get_connection()
    cur = conn.cursor()

    summary = {
        "inferred_country_fr": 0,
        "suffixed":            0,
        "skipped":             0,
    }

    try:
        # Pass A — infer country
        cur.execute(
            "SELECT COUNT(*) FROM horse "
            "WHERE primary_source = 'letrot' AND birth_country IS NULL"
        )
        n = cur.fetchone()[0]
        summary["inferred_country_fr"] = n
        print(f"[infer-country] LeTrot horses with NULL country: {n}")
        if execute and n > 0:
            cur.execute(_INFER_SQL)
            conn.commit()
            print(f"[infer-country] set birth_country='FR' on {cur.rowcount} rows")

        # Pass B — append suffix
        cur.execute(_SELECT_SUFFIX_CANDIDATES)
        rows = cur.fetchall()
        print(f"[suffix] {len(rows)} candidate horse rows to suffix")

        # Print a few previews
        for hid, name, country, _sd in rows[:6]:
            print(f"  PREVIEW #{hid}  {name!r} ({country}) -> "
                  f"{_build_suffix(name, country)!r}")

        if not execute:
            print(f"\nDRY-RUN — no DB changes. Re-run with --execute to apply.")
            return summary

        commit_every = 1000
        for i, (hid, name, country, sd) in enumerate(rows, 1):
            new_name = _build_suffix(name, country)
            new_sd = _record_history(sd, name, new_name)
            cur.execute(_UPDATE_NAME_SQL, (new_name, Json(new_sd), hid))
            summary["suffixed"] += 1
            if i % commit_every == 0:
                conn.commit()
                print(f"  ... commit batch ({i}/{len(rows)})")
        conn.commit()

        print(f"\n[suffix] done. suffixed={summary['suffixed']} "
              f"inferred_fr={summary['inferred_country_fr']}")
        return summary
    finally:
        cur.close()
        conn.close()


def run_rollback() -> dict:
    """Restore original names from source_data._name_history."""
    conn = get_connection()
    cur = conn.cursor()
    try:
        cur.execute(_SELECT_ROLLBACK)
        rows = cur.fetchall()
        print(f"[rollback] {len(rows)} rows have _name_history")
        n = 0
        for hid, cur_name, sd in rows:
            hist = (sd or {}).get("_name_history") or []
            if not hist:
                continue
            last = hist[-1]
            # Only roll back if WE wrote the last entry AND the current name
            # matches what we stored as "new".
            if last.get("by") != "backfill_horse_name_country_suffix":
                continue
            if cur_name != last.get("new"):
                continue
            original = last["old"]
            sd2 = dict(sd)
            sd2["_name_history"] = hist[:-1]  # pop our entry
            if not sd2["_name_history"]:
                sd2.pop("_name_history")
            cur.execute(_UPDATE_NAME_SQL, (original, Json(sd2), hid))
            n += 1
            if n % 1000 == 0:
                conn.commit()
                print(f"  ... reverted {n}")
        conn.commit()
        print(f"[rollback] restored {n} horse names")
        return {"rolled_back": n}
    finally:
        cur.close()
        conn.close()


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--execute", action="store_true",
                   help="apply changes (default is dry-run)")
    p.add_argument("--rollback", action="store_true",
                   help="restore original names from _name_history")
    args = p.parse_args()

    if args.rollback:
        run_rollback()
        return 0
    run_forward(execute=args.execute)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
