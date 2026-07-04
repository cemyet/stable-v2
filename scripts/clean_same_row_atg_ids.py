"""
Same-row pollution: horses with BOTH `st_id` AND `x:CC:NAME` atg_id
(Face Time Bourbon 41506 class — ~6,682 rows, ~146k entries).

The horse_id itself is fine — these are TravSport guest horses that an
old ATG ingest attached a synthetic key to. The synth key is a legitimate
identifier for ATG re-ingest (it lets us deduplicate the horse the next
time ATG sees it), so by default we don't touch it.

What this script does:

  1. Try to discover a numeric ATG id by probing one of the horse's
     existing entries' ATG raw JSON (v1.v2_atg_race_raw).
     - If a numeric `horse.id` is found, replace `x:CC:NAME` with the
       integer string. Future Category-B-style merges will then catch
       any sibling synth row automatically.
  2. Else stamp `source_data.atg.same_row_checked_at` with today's date
     so we don't re-probe on the next run.

This script does NOT delete or merge rows. It is purely an atg_id refresh.

CLI
---
    python -m scripts.clean_same_row_atg_ids                # dry-run
    python -m scripts.clean_same_row_atg_ids --execute      # apply
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from psycopg2.extras import Json  # noqa: E402

from scripts._merge_helpers import build_argparser, script_runner  # noqa: E402
from core.identity import normalize_name  # noqa: E402


def _fetch_candidates(cur, limit: int | None) -> list[dict]:
    sql = """
    SELECT horse_id, name, st_id, atg_id, source_data
      FROM horse
     WHERE st_id IS NOT NULL
       AND atg_id LIKE 'x:%%'
     ORDER BY (SELECT COUNT(*) FROM entry e WHERE e.horse_id = horse.horse_id) DESC
    """
    if limit:
        sql += f" LIMIT {int(limit)}"
    cur.execute(sql)
    cols = [d.name for d in cur.description]
    return [dict(zip(cols, r)) for r in cur.fetchall()]


def _find_recent_atg_races(cur, horse_id: int, n: int = 3) -> list[str]:
    cur.execute(
        """
        SELECT r.atg_race_id
          FROM entry e
          JOIN race  r ON r.race_id = e.race_id
         WHERE e.horse_id = %s
           AND r.atg_race_id IS NOT NULL
         ORDER BY r.race_date DESC
         LIMIT %s
        """,
        (horse_id, n),
    )
    return [r[0] for r in cur.fetchall()]


def _probe_atg_id(v1_cur, race_ids: list[str], horse_name: str) -> int | None:
    norm = normalize_name(horse_name)
    for rid in race_ids:
        v1_cur.execute(
            "SELECT raw_json FROM v2_atg_race_raw WHERE atg_race_id = %s",
            (rid,),
        )
        row = v1_cur.fetchone()
        if not row:
            continue
        raw = row[0]
        for s in (raw.get("starts") or []):
            h = s.get("horse") or {}
            if normalize_name(h.get("name")) == norm:
                hid = h.get("id")
                if isinstance(hid, int) and hid > 0:
                    return hid
    return None


def main() -> int:
    args = build_argparser("clean_same_row_atg_ids").parse_args()
    with script_runner("clean_same_row_atg_ids", args) as (conn, log, summary):
        from core.db import get_v1_connection
        v1 = get_v1_connection()
        log(f"[clean_same_row_atg_ids] execute={args.execute}, limit={args.limit}")
        try:
            with conn.cursor() as cur:
                cands = _fetch_candidates(cur, args.limit)
            summary["candidates"] = len(cands)
            log(f"Found {len(cands)} same-row horses to probe.")

            promoted = 0
            stamped = 0
            for c in cands:
                with conn.cursor() as cur:
                    race_ids = _find_recent_atg_races(cur, c["horse_id"])

                if not race_ids:
                    log(f"  {c['horse_id']:>7}  {c['name']!r:<40}  "
                        f"no atg races → mark checked")
                    if args.execute:
                        with conn.cursor() as cur:
                            sd = dict(c.get("source_data") or {})
                            atg_block = dict(sd.get("atg") or {})
                            atg_block["same_row_checked"] = True
                            sd["atg"] = atg_block
                            cur.execute(
                                "UPDATE horse SET source_data = %s WHERE horse_id = %s",
                                (Json(sd), c["horse_id"]),
                            )
                    stamped += 1
                    continue

                with v1.cursor() as v1c:
                    hid = _probe_atg_id(v1c, race_ids, c["name"])

                if hid is None:
                    log(f"  {c['horse_id']:>7}  {c['name']!r:<40}  "
                        f"no numeric atg.id in raw → mark checked")
                    if args.execute:
                        with conn.cursor() as cur:
                            sd = dict(c.get("source_data") or {})
                            atg_block = dict(sd.get("atg") or {})
                            atg_block["same_row_checked"] = True
                            sd["atg"] = atg_block
                            cur.execute(
                                "UPDATE horse SET source_data = %s WHERE horse_id = %s",
                                (Json(sd), c["horse_id"]),
                            )
                    stamped += 1
                else:
                    log(f"  {c['horse_id']:>7}  {c['name']!r:<40}  "
                        f"promoting x: atg_id → {hid}")
                    if args.execute:
                        try:
                            with conn.cursor() as cur:
                                cur.execute(
                                    "UPDATE horse SET atg_id = %s, last_updated_at = NOW() "
                                    "WHERE horse_id = %s",
                                    (str(hid), c["horse_id"]),
                                )
                        except Exception as exc:
                            conn.rollback()
                            summary["errors"] += 1
                            log(f"    ! could not promote (conflict with horse_atg_id_uk?): {exc}")
                            continue
                    promoted += 1

                if args.execute and (promoted + stamped) % args.commit_every == 0:
                    conn.commit()

            summary["promoted"] = promoted
            summary["stamped"]  = stamped
            summary["merged"]   = promoted + stamped
            if args.execute:
                conn.commit()
            log(f"Promoted={promoted}  stamped={stamped}")
        finally:
            v1.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
