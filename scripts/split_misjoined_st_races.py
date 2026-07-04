"""
Detach false st<->atg race joins.

Symptom:
  st aggregates foreign races (FRANKRIKE, NORGE, …) into loose race rows
  identified only by (date, FR-track-bucket, race_number). When upsert_race
  did the cross-source fallback on (track_id, race_date, race_number) it
  glued these onto the wrong atg race — producing races where:
    * st_entries and atg_entries have different winners (placement='1'),
    * st entries are typically a handful of horses with completely
      different names than the atg field, and
    * the st-implied distance differs from the atg distance.

Fix per affected race (default dry-run):
  1. Identify entries with primary_source='st' whose horse is NOT in the
     atg field for the same race. These are the "polluted" st entries.
  2. Delete those polluted entries.
  3. If 0 st entries remain, drop the st_race_id link and remove 'st'
     from the race's source_data.
  4. The next jobs/update.py run will re-import the st race; with the
     new distance-aware matching in upsert_race it will land in its
     own race row instead of being glued back here.

Conservative: never touches the atg-sourced data. Never deletes the
race row. Only removes confirmed orphan st entries.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from psycopg2.extras import Json  # noqa: E402

from scripts._merge_helpers import build_argparser, script_runner  # noqa: E402


def _fetch_misjoined(cur, limit: int | None) -> list[dict]:
    """Find races linked to both st + atg where st entries don't fit."""
    sql = """
    WITH joined AS (
      SELECT r.race_id, r.race_date, r.atg_race_id, r.st_race_id,
             r.distance, r.track_id,
             COUNT(DISTINCT CASE WHEN e.primary_source='st'  THEN e.entry_id END) AS st_n,
             COUNT(DISTINCT CASE WHEN e.primary_source='atg' THEN e.entry_id END) AS atg_n,
             COUNT(DISTINCT CASE WHEN e.placement_text='1' THEN e.primary_source END) AS winners
        FROM race r
        LEFT JOIN entry e ON e.race_id = r.race_id
       WHERE r.st_race_id IS NOT NULL AND r.atg_race_id IS NOT NULL
       GROUP BY r.race_id
    )
    SELECT j.race_id, j.race_date, j.atg_race_id, j.st_race_id,
           j.distance, j.st_n, j.atg_n, t.name AS track_name, t.country
      FROM joined j
      LEFT JOIN track t ON t.track_id = j.track_id
     WHERE j.st_n > 0
       AND j.atg_n > 0
       AND ( j.winners > 1
             OR NOT EXISTS (
               SELECT 1 FROM entry es
                JOIN entry ea ON ea.race_id = es.race_id AND ea.horse_id = es.horse_id
               WHERE es.race_id = j.race_id
                 AND es.primary_source = 'st'
                 AND ea.primary_source = 'atg'
             ))
     ORDER BY j.race_date DESC
    """
    if limit:
        sql += f" LIMIT {int(limit)}"
    cur.execute(sql)
    cols = [d.name for d in cur.description]
    return [dict(zip(cols, r)) for r in cur.fetchall()]


def _detach_one(cur, race_id: int, dry_run: bool) -> dict:
    """Delete orphan st entries on this race and detach st_race_id if empty."""
    # Polluted = st entries whose horse is NOT in the atg field for this race.
    cur.execute(
        """
        SELECT e.entry_id, h.name
          FROM entry e
          LEFT JOIN horse h ON h.horse_id = e.horse_id
         WHERE e.race_id = %s
           AND e.primary_source = 'st'
           AND NOT EXISTS (
             SELECT 1 FROM entry ea
              WHERE ea.race_id = e.race_id
                AND ea.horse_id = e.horse_id
                AND ea.primary_source = 'atg'
           )
        """,
        (race_id,),
    )
    polluted = cur.fetchall()
    polluted_ids = [r[0] for r in polluted]
    polluted_names = [r[1] for r in polluted]

    if not polluted_ids:
        return {"action": "noop", "polluted_count": 0}

    if not dry_run:
        cur.execute(
            "DELETE FROM entry WHERE entry_id = ANY(%s)",
            (polluted_ids,),
        )

    # If no st entries remain, detach the st_race_id and drop 'st' source_data.
    cur.execute(
        "SELECT COUNT(*) FROM entry WHERE race_id = %s AND primary_source = 'st'",
        (race_id,),
    )
    remaining = cur.fetchone()[0]
    detached = False
    if remaining == 0:
        if not dry_run:
            cur.execute(
                """
                UPDATE race
                   SET st_race_id = NULL,
                       source_data = COALESCE(source_data, '{}'::jsonb) - 'st',
                       last_updated_at = NOW()
                 WHERE race_id = %s
                """,
                (race_id,),
            )
        detached = True

    return {
        "action": "split",
        "polluted_count": len(polluted_ids),
        "polluted_names": polluted_names,
        "detached_st_link": detached,
    }


def main() -> None:
    parser = build_argparser("split_misjoined_st_races")
    args = parser.parse_args()

    with script_runner("split_misjoined_st_races", args) as (conn, log, summary):
        with conn.cursor() as cur:
            races = _fetch_misjoined(cur, args.limit)

        summary["candidates"] = len(races)
        log(f"[split_misjoined_st_races] {len(races)} candidate races "
            f"({'DRY-RUN' if not args.execute else 'EXECUTE'})")

        rows_by_country: dict[str, int] = {}
        total_polluted = 0
        for r in races:
            rows_by_country[r["country"] or "??"] = (
                rows_by_country.get(r["country"] or "??", 0) + 1)

        log(f"  by country: {rows_by_country}")

        for i, r in enumerate(races, 1):
            with conn.cursor() as cur:
                try:
                    res = _detach_one(cur, r["race_id"], dry_run=not args.execute)
                except Exception as exc:
                    conn.rollback()
                    summary["errors"] += 1
                    log(f"  ! error race_id={r['race_id']}: {exc!r}")
                    continue

            if res["action"] == "noop":
                summary["skipped"] += 1
                continue

            total_polluted += res["polluted_count"]
            summary["merged"] += 1
            tag = "PREVIEW" if not args.execute else "split  "
            log(f"  {tag} race_id={r['race_id']} {r['race_date']} "
                f"{r['track_name']}/{r['country']} #{r['atg_race_id']} "
                f"removed={res['polluted_count']} "
                f"detached={'y' if res['detached_st_link'] else 'n'} "
                f"horses={res['polluted_names']}")

            if args.execute and summary["merged"] % args.commit_every == 0:
                conn.commit()

        if args.execute:
            conn.commit()
        summary["total_polluted_entries"] = total_polluted


if __name__ == "__main__":
    main()
