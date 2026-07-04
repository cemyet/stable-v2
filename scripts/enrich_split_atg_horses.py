"""
Backfill ATG metadata onto synthetic horses created by split_polluted_atg_ids.

The split script intentionally moves polluted modern ATG entries off old ST
horses, but its newly-created rows may be thin: they often only have
`atg_id = x:CC:NAME` and a source_data marker. ATG race raw starts usually
contain enough metadata to render and match the synthetic row better:

  * nationality / country
  * age -> inferred DOB year
  * sex, color
  * sire_name / dam_name
  * selected raw start metadata under source_data.atg.enrichment

This script is conservative: it only targets rows created by
split_polluted_atg_ids (`source_data.atg.split_reason = category_a_polluted`)
and never touches ST rows.
"""

from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from psycopg2.extras import Json  # noqa: E402

from core.db import get_v1_connection  # noqa: E402
from core.identity import normalize_name  # noqa: E402
from scripts._merge_helpers import build_argparser, script_runner  # noqa: E402


def _safe_int(v: Any) -> int | None:
    try:
        if v is None:
            return None
        return int(v)
    except (TypeError, ValueError):
        return None


def _synth_country(atg_id: str | None) -> str | None:
    parts = (atg_id or "").split(":", 2)
    return parts[1] if len(parts) == 3 and parts[1] else None


def _gender_code(sex: str | None) -> str | None:
    if not sex:
        return None
    return {"stallion": "H", "gelding": "V", "mare": "S"}.get(sex.lower())


def _fetch_candidates(cur, limit: int | None) -> list[dict]:
    sql = """
    SELECT h.horse_id, h.name, h.atg_id, h.birth_country, h.date_of_birth,
           h.gender_code, h.color, h.sire_name, h.dam_name,
           h.source_data
      FROM horse h
     WHERE h.atg_id LIKE 'x:%'
       AND h.primary_source = 'atg'
       AND h.source_data->'atg'->>'split_reason' = 'category_a_polluted'
     ORDER BY h.horse_id
    """
    if limit:
        sql += f" LIMIT {int(limit)}"
    cur.execute(sql)
    cols = [d.name for d in cur.description]
    return [dict(zip(cols, row)) for row in cur.fetchall()]


def _entry_refs(cur, horse_id: int) -> list[tuple[str, date | None, int | None]]:
    cur.execute(
        """
        SELECT r.atg_race_id, r.race_date, e.program_number
          FROM entry e
          JOIN race r ON r.race_id = e.race_id
         WHERE e.horse_id = %s
           AND r.atg_race_id IS NOT NULL
         ORDER BY r.race_date DESC NULLS LAST
        """,
        (horse_id,),
    )
    return cur.fetchall()


def _find_best_start(v1_cur, candidate: dict, refs: list[tuple[str, date | None, int | None]]) -> tuple[dict, date | None] | tuple[None, None]:
    norm_name = normalize_name(candidate["name"])
    for atg_race_id, race_date, program_number in refs:
        v1_cur.execute(
            "SELECT raw_json FROM v2_atg_race_raw WHERE atg_race_id = %s",
            (atg_race_id,),
        )
        row = v1_cur.fetchone()
        if not row or not row[0]:
            continue
        raw = row[0]
        for start in raw.get("starts") or []:
            horse = start.get("horse") or {}
            start_number = _safe_int(start.get("number"))
            if program_number is not None and start_number != program_number:
                continue
            if normalize_name(horse.get("name")) == norm_name:
                return horse, race_date
    return None, None


def _fields_from_start(candidate: dict, horse: dict, race_date: date | None) -> tuple[dict, dict]:
    pedigree = horse.get("pedigree") or {}
    sire = pedigree.get("father") or {}
    dam = pedigree.get("mother") or {}
    nationality = horse.get("nationality") or _synth_country(candidate.get("atg_id"))
    age = _safe_int(horse.get("age"))

    fields: dict[str, Any] = {}
    raw_name = (horse.get("name") or "").strip()
    if raw_name:
        fields["name"] = raw_name
    if nationality:
        fields["birth_country"] = nationality
    if race_date and age:
        fields["date_of_birth"] = date(race_date.year - age, 1, 1)
    gc = _gender_code(horse.get("sex"))
    if gc:
        fields["gender_code"] = gc
    if horse.get("color"):
        fields["color"] = horse.get("color")
    if sire.get("name"):
        fields["sire_name"] = sire.get("name")
    if dam.get("name"):
        fields["dam_name"] = dam.get("name")

    enrichment = {
        "name": raw_name or horse.get("name"),
        "nationality": nationality,
        "age": age,
        "sex": horse.get("sex"),
        "color": horse.get("color"),
        "pedigree": {
            "father": sire,
            "mother": dam,
            "grandfather": pedigree.get("grandfather"),
        },
        "owner": horse.get("owner"),
        "trainer": horse.get("trainer"),
        "statistics": horse.get("statistics"),
    }
    return fields, enrichment


def _apply(cur, horse_id: int, fields: dict, enrichment: dict) -> None:
    if not fields:
        return
    set_cols = [f"{col} = %s" for col in fields]
    vals = list(fields.values())
    vals.extend([Json(enrichment), horse_id])
    cur.execute(
        f"""
        UPDATE horse
           SET {', '.join(set_cols)},
               source_data = jsonb_set(
                   COALESCE(source_data, '{{}}'::jsonb),
                   '{{atg,enrichment}}',
                   %s::jsonb,
                   true
               ),
               last_updated_at = NOW()
         WHERE horse_id = %s
        """,
        vals,
    )


def main() -> int:
    args = build_argparser("enrich_split_atg_horses").parse_args()
    dry_run = not args.execute
    v1_conn = get_v1_connection()
    try:
        with script_runner("enrich_split_atg_horses", args) as (conn, log, summary):
            with conn.cursor() as cur:
                candidates = _fetch_candidates(cur, args.limit)
            summary["candidates"] = len(candidates)
            log(f"[enrich_split_atg_horses] {len(candidates)} candidates ({'DRY-RUN' if dry_run else 'EXECUTE'})")

            with v1_conn.cursor() as v1_cur:
                for c in candidates:
                    with conn.cursor() as cur:
                        refs = _entry_refs(cur, c["horse_id"])
                    horse, race_date = _find_best_start(v1_cur, c, refs)
                    if not horse:
                        summary["skipped"] += 1
                        continue

                    fields, enrichment = _fields_from_start(c, horse, race_date)
                    if not fields:
                        summary["skipped"] += 1
                        continue

                    summary["merged"] += 1
                    log(
                        f"  {'PREVIEW' if dry_run else 'enrich '} horse_id={c['horse_id']} "
                        f"{c['name']!r} -> "
                        f"country={fields.get('birth_country')} dob={fields.get('date_of_birth')} "
                        f"sex={fields.get('gender_code')} sire={fields.get('sire_name')!r} "
                        f"dam={fields.get('dam_name')!r}"
                    )
                    if not dry_run:
                        with conn.cursor() as cur:
                            _apply(cur, c["horse_id"], fields, enrichment)
                        if summary["merged"] % args.commit_every == 0:
                            conn.commit()
                if not dry_run:
                    conn.commit()
    finally:
        v1_conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
