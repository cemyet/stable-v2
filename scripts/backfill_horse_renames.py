"""Backfill horse renames our ingestion previously refused to apply.

Until Aug 2026 `etl.matching._pick_better_horse_name` could not tell a genuine
rename from a re-dressed spelling of the same name, so it kept the stored name
in both cases. Horses renamed before their first start (common) therefore kept
their original name forever, even when the registry already reported the new
one. This script replays the corrected decision over the raw we already have.

It asks the SAME functions ingestion uses — the name picker AND the source
priority gate — so a row is only touched when tonight's run would touch it too.

Names come from the ST passport (`st_horse_raw` basic-information), which owns
the registry name and carries both proper case and the trailing `*` ST puts on
a renamed horse. `source_data.atg.name` is deliberately NOT trusted by default:
an audit of the 73 rows where it disagrees found cross-horse pollution
("Dana Ek* (IT)" stored as "Italy") and dropped owner initials ("Keep Going
F.R." as "Keep Going"). Pass --include-atg to also replay ATG names, which is
only applied to rows ATG already owns (primary_source = atg).

Old names are preserved in `horse.source_data._name_history`.

Usage:
    python -m scripts.backfill_horse_renames                # dry-run (default)
    python -m scripts.backfill_horse_renames --execute
    python -m scripts.backfill_horse_renames --limit 50     # sample first
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
from etl.matching import (  # noqa: E402
    RENAME_SOURCES,
    _name_identity_key,
    _pick_better_horse_name,
    _record_name_history,
    _winning,
)

# Rows where a source's raw name differs verbatim from ours. The picker then
# decides which of those differences is a real rename; SQL only has to narrow
# 377k horses down to the candidates.
_CANDIDATES_SQL = """
SELECT h.horse_id,
       h.name,
       h.primary_source,
       h.source_data->'atg'->>'name'  AS atg_name,
       sr.raw_json->>'name'           AS st_name
  FROM horse h
  LEFT JOIN st_horse_raw sr
         ON sr.horse_id = h.st_id
        AND sr.data_type = 'horse-basic-information'
 WHERE h.name IS NOT NULL
   AND (
        (h.source_data->'atg'->>'name' IS NOT NULL
         AND h.source_data->'atg'->>'name' <> h.name)
     OR (sr.raw_json->>'name' IS NOT NULL
         AND sr.raw_json->>'name' <> h.name)
   )
 ORDER BY h.horse_id
"""


def _resolve(current: str, primary_source: str | None,
             names: list[tuple[str, str | None]]) -> tuple[str, str | None]:
    """Return (winning_name, deciding_source) after replaying each source."""
    name, by = current, None
    for source, candidate in names:
        if not candidate:
            continue
        # Same gates ingestion applies: a source may only rewrite a name it
        # outranks (or already owns), and only registry sources may replace
        # a name outright rather than re-dress it.
        if not _winning("horse", source, primary_source):
            continue
        picked = _pick_better_horse_name(
            name, candidate, allow_rename=source in RENAME_SOURCES)
        if picked and picked.strip() != name.strip():
            name, by = picked, source
    return name, by


def run(*, execute: bool, limit: int | None, include_atg: bool,
        include_decoration: bool) -> dict:
    conn = get_connection()
    cur = conn.cursor()
    summary = {"candidates": 0, "renames": 0, "decoration": 0, "applied": 0}
    renames: list[tuple[int, str, str, str]] = []
    decoration: list[tuple[int, str, str, str]] = []

    try:
        cur.execute(_CANDIDATES_SQL)
        rows = cur.fetchall()
        summary["candidates"] = len(rows)

        for horse_id, current, primary_source, atg_name, st_name in rows:
            sources: list[tuple[str, str | None]] = [("st", st_name)]
            if include_atg:
                sources.insert(0, ("atg", atg_name))
            new_name, by = _resolve(current, primary_source, sources)
            if by is None:
                continue  # picker kept our name
            # A different identity is a rename; same identity in a different
            # dress (ST's `*`, case, country suffix) is decoration.
            bucket = (renames if _name_identity_key(current) != _name_identity_key(new_name)
                      else decoration)
            bucket.append((horse_id, current, new_name, by))

        summary["renames"] = len(renames)
        summary["decoration"] = len(decoration)

        changes = renames + (decoration if include_decoration else [])
        if limit is not None:
            changes = changes[:limit]
        summary["applied"] = len(changes)

        if execute:
            wcur = conn.cursor()
            for horse_id, current, new_name, _by in changes:
                wcur.execute("SELECT source_data FROM horse WHERE horse_id = %s",
                             (horse_id,))
                sd = wcur.fetchone()[0]
                wcur.execute(
                    "UPDATE horse SET name = %s, source_data = %s, "
                    "last_updated_at = NOW() WHERE horse_id = %s",
                    (new_name,
                     Json(_record_name_history(sd, current, new_name,
                                               "backfill_horse_renames")),
                     horse_id),
                )
            conn.commit()
    finally:
        conn.close()

    def _dump(label, items, show):
        print(f"{label}: {len(items)}")
        for horse_id, old, new, by in items[:show]:
            print(f"  #{horse_id:<8} {old!r:34} -> {new!r:34} ({by})")
        if len(items) > show:
            print(f"  ... and {len(items) - show} more")
        print()

    _dump("RENAMES (identity changes)", renames, 40)
    _dump("DECORATION (star / case / country suffix)", decoration,
          40 if include_decoration else 5)

    print(f"candidates scanned : {summary['candidates']:,}")
    print(f"would change       : {summary['applied']:,}"
          + ("" if include_decoration else "  (decoration excluded; "
                                           "pass --include-decoration)"))
    print("applied" if execute else "DRY-RUN — nothing written (pass --execute)")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--execute", action="store_true",
                    help="apply the renames (default is a dry-run)")
    ap.add_argument("--limit", type=int, default=None,
                    help="stop after N renames (sampling)")
    ap.add_argument("--include-atg", action="store_true",
                    help="also replay source_data.atg.name (only affects rows "
                         "ATG owns; see module docstring for the caveats)")
    ap.add_argument("--include-decoration", action="store_true",
                    help="also apply same-name changes (ST's rename star, "
                         "case, country suffix), not just renames")
    args = ap.parse_args()
    run(execute=args.execute, limit=args.limit, include_atg=args.include_atg,
        include_decoration=args.include_decoration)


if __name__ == "__main__":
    main()
