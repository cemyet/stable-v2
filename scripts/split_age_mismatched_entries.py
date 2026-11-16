"""
Move entries off an ST horse when the age the source stamped on the start
contradicts the horse's registered DOB.

The pattern (Kaiser 31453 class): a foreign namesake's starts end up on a
Swedish horse, e.g. via `split_polluted_atg_ids` picking the wrong target.
The Swedish Kaiser (born 2022) then carries French starts at "age 3" in
2023. ATG/LeTrot stamp every start with the horse's age, and on
ST-confirmed entries that age equals (race year - DOB year), or one less
when ST counted it from the birthday — so anything else is somebody else.

Detection: entry on a horse with st_id + DOB, entry not confirmed by ST,
`age - (race_year - dob_year)` not in (0, -1). Horses with no age-consistent
entry at all are skipped: there the DOB itself is the suspect.

Destination per entry (birth year = race year - entry age):
  1. Same race, a different starter whose name is the same horse name
     (or ST's initials variant, "Kaiser D.K.") and whose DOB fits →
     column-merge into that entry (it is the same start seen by another
     source) and delete the misplaced one.
  2. Exactly one other horse with the same normalised name and a fitting
     DOB year → move the entry there (merging if it already ran that race).
  3. Otherwise a new horse row per (source horse, birth year).

Every (source horse → destination) pair is audited in horse_merge_log with
method 'split_age_mismatch' (moved entry ids + per-entry merge blocks).
The source horse is never deleted, so `--rollback-job` does not apply.

Usage:
  python scripts/split_age_mismatched_entries.py --horse-id 31453
  python scripts/split_age_mismatched_entries.py --horse-id 31453 --execute
  python scripts/split_age_mismatched_entries.py            # all horses
  python scripts/split_age_mismatched_entries.py --no-new-rows --execute
"""

from __future__ import annotations

import sys
from collections import defaultdict
from datetime import date
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from psycopg2.extras import Json  # noqa: E402

from core.identity import _merge_entries_columnwise, normalize_name  # noqa: E402
from etl.matching import _is_name_variant, _name_identity_key  # noqa: E402
from scripts._merge_helpers import build_argparser, script_runner  # noqa: E402

METHOD = "split_age_mismatch"


def _fetch_mismatched(cur, horse_id: int | None) -> list[dict]:
    cur.execute(
        """
        WITH aged AS (
          SELECT e.entry_id, e.race_id, e.horse_id, e.age, r.race_date,
                 t.country AS track_country,
                 h.name AS horse_name, h.date_of_birth,
                 e.age - (EXTRACT(year FROM r.race_date)::int
                          - EXTRACT(year FROM h.date_of_birth)::int) AS delta,
                 (e.source_data ? 'st' OR e.primary_source = 'st') AS st_confirmed
            FROM entry e
            JOIN race  r ON r.race_id  = e.race_id
            JOIN horse h ON h.horse_id = e.horse_id
            LEFT JOIN track t ON t.track_id = r.track_id
           WHERE h.st_id IS NOT NULL
             AND h.date_of_birth IS NOT NULL
             AND e.age > 0
             AND (%(hid)s::int IS NULL OR h.horse_id = %(hid)s)
        ),
        anchored AS (
          SELECT DISTINCT horse_id FROM aged WHERE delta IN (0, -1)
        )
        SELECT a.*, EXTRACT(year FROM a.race_date)::int - a.age AS birth_year
          FROM aged a
          JOIN anchored USING (horse_id)
         WHERE a.delta NOT IN (0, -1)
           AND NOT a.st_confirmed
         ORDER BY a.horse_id, a.race_date
        """,
        {"hid": horse_id},
    )
    cols = [d.name for d in cur.description]
    return [dict(zip(cols, r)) for r in cur.fetchall()]


def _same_horse_name(a: str | None, b: str | None) -> bool:
    if not a or not b:
        return False
    return _name_identity_key(a) == _name_identity_key(b) or _is_name_variant(a, b)


def _dob_fits(dob: date | None, birth_year: int) -> bool:
    return dob is None or dob.year in (birth_year, birth_year - 1)


def _find_sibling_entry(cur, e: dict) -> tuple[int, int] | None:
    """(entry_id, horse_id) of the same start already recorded on another horse."""
    cur.execute(
        """
        SELECT x.entry_id, x.horse_id, h.name, h.date_of_birth
          FROM entry x JOIN horse h ON h.horse_id = x.horse_id
         WHERE x.race_id = %s AND x.horse_id <> %s
        """,
        (e["race_id"], e["horse_id"]),
    )
    hits = [(eid, hid) for eid, hid, name, dob in cur.fetchall()
            if _same_horse_name(name, e["horse_name"]) and _dob_fits(dob, e["birth_year"])]
    return hits[0] if len(hits) == 1 else None


def _find_namesake(cur, e: dict) -> int | None:
    cur.execute(
        """
        SELECT horse_id FROM horse
         WHERE v2_normalize_name(name) = v2_normalize_name(%s)
           AND horse_id <> %s
           AND EXTRACT(year FROM date_of_birth)::int IN (%s, %s)
         LIMIT 2
        """,
        (e["horse_name"], e["horse_id"], e["birth_year"], e["birth_year"] - 1),
    )
    rows = cur.fetchall()
    return rows[0][0] if len(rows) == 1 else None


def _create_split_row(cur, e: dict) -> int:
    cur.execute(
        """
        INSERT INTO horse (name, date_of_birth, primary_source,
                           source_data, last_updated_at)
        VALUES (%s, %s, 'atg', %s, NOW())
        RETURNING horse_id
        """,
        (normalize_name(e["horse_name"]).title(), date(e["birth_year"], 1, 1),
         Json({"atg": {"split_from_horse_id": e["horse_id"],
                       "split_reason": METHOD}})),
    )
    return cur.fetchone()[0]


def _fetch_entry(cur, entry_id: int) -> dict:
    cur.execute("SELECT * FROM entry WHERE entry_id = %s", (entry_id,))
    cols = [d.name for d in cur.description]
    return dict(zip(cols, cur.fetchone()))


def _merge_into(cur, loser_id: int, keeper_id: int, is_french: bool) -> dict:
    keeper = _fetch_entry(cur, keeper_id)
    loser = _fetch_entry(cur, loser_id)
    set_dict, audit = _merge_entries_columnwise(keeper, loser, is_french_race=is_french)
    if set_dict:
        cols_sql = ", ".join(f"{c} = %s" for c in set_dict)
        vals = [Json(v) if isinstance(v, dict) else v for v in set_dict.values()]
        cur.execute(
            f"UPDATE entry SET {cols_sql}, last_updated_at = NOW() WHERE entry_id = %s",
            [*vals, keeper_id],
        )
    cur.execute("DELETE FROM entry WHERE entry_id = %s", (loser_id,))
    return audit


def _move_entry(cur, e: dict, dst_horse_id: int) -> dict | None:
    """Repoint one entry; returns a merge audit block if it hit a same-race entry."""
    cur.execute(
        "SELECT entry_id FROM entry WHERE race_id = %s AND horse_id = %s",
        (e["race_id"], dst_horse_id),
    )
    row = cur.fetchone()
    if row:
        return _merge_into(cur, e["entry_id"], row[0], e["track_country"] == "FR")
    cur.execute(
        "UPDATE entry SET horse_id = %s, last_updated_at = NOW() WHERE entry_id = %s",
        (dst_horse_id, e["entry_id"]),
    )
    return None


def _log_split(cur, src_id: int, dst_id: int, plan: dict) -> None:
    cur.execute(
        """
        INSERT INTO horse_merge_log
            (from_horse_id, to_horse_id, reason, method,
             entries_moved, conflicts_resolved, from_snapshot, merged_by)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        """,
        (src_id, dst_id,
         f"age mismatch split: {len(plan['moved'])} entries whose start age "
         f"contradicts horse {src_id}'s DOB ({plan['how']})",
         METHOD, len(plan["moved"]) - len(plan["merges"]), len(plan["merges"]),
         Json({"moved_entry_ids": plan["moved"], "entry_merges": plan["merges"],
               "created_row": plan["how"] == "new_row"}),
         METHOD),
    )


def main() -> int:
    ap = build_argparser(METHOD)
    ap.add_argument("--horse-id", type=int, default=None,
                    help="only repair this horse")
    ap.add_argument("--no-new-rows", action="store_true",
                    help="leave entries in place when no existing horse claims them")
    args = ap.parse_args()
    with script_runner(METHOD, args) as (conn, log, summary):
        log(f"[{METHOD}] execute={args.execute} horse_id={args.horse_id} limit={args.limit}")
        with conn.cursor() as cur:
            rows = _fetch_mismatched(cur, args.horse_id)
        by_horse: dict[int, list[dict]] = defaultdict(list)
        for r in rows:
            by_horse[r["horse_id"]].append(r)
        horse_ids = list(by_horse)[: args.limit] if args.limit else list(by_horse)
        summary["candidates"] = len(horse_ids)
        summary["entries"] = sum(len(by_horse[h]) for h in horse_ids)
        log(f"{summary['entries']} mismatched entries on {len(horse_ids)} horses.")

        for n, hid in enumerate(horse_ids, 1):
            entries = by_horse[hid]
            new_rows: dict[int, int] = {}
            plans: dict[int, dict] = {}
            with conn.cursor() as cur:
                for e in entries:
                    sib = _find_sibling_entry(cur, e)
                    if sib:
                        dst, how = sib[1], "same_race_entry"
                    else:
                        dst = _find_namesake(cur, e)
                        how = "namesake"
                    if dst is None and args.no_new_rows:
                        summary["left_in_place"] = summary.get("left_in_place", 0) + 1
                        continue
                    if dst is None:
                        how = "new_row"
                        if not args.execute:
                            dst = -e["birth_year"]
                        elif e["birth_year"] not in new_rows:
                            new_rows[e["birth_year"]] = _create_split_row(cur, e)
                            dst = new_rows[e["birth_year"]]
                        else:
                            dst = new_rows[e["birth_year"]]
                    plan = plans.setdefault(dst, {"how": how, "moved": [], "merges": [],
                                                  "dates": []})
                    plan["moved"].append(e["entry_id"])
                    plan["dates"].append(e["race_date"])
                    if args.execute:
                        audit = _move_entry(cur, e, dst)
                        if audit:
                            plan["merges"].append(audit)

                if not plans:
                    continue
                log(f"  {hid} {entries[0]['horse_name']!r} DOB={entries[0]['date_of_birth']}  "
                    f"{len(entries)} entries:")
                for dst, plan in plans.items():
                    tgt = (f"new row (born {-dst})" if dst < 0 else f"horse {dst}")
                    log(f"    → {tgt} via {plan['how']}: {len(plan['moved'])} entries "
                        f"{min(plan['dates'])}..{max(plan['dates'])}")
                    if args.execute:
                        _log_split(cur, hid, dst, plan)

            summary["merged"] += 1
            if args.execute and n % args.commit_every == 0:
                conn.commit()

        if args.execute:
            conn.commit()
    return 0


if __name__ == "__main__":
    sys.exit(main())
