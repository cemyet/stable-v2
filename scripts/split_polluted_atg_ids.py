"""
Category A: split polluted old ST horses whose `x:CC:NAME` atg_id has
attracted entries belonging to a modern foreign namesake.

The pattern (Indy Boy 148366 / Just A Gigolo 376512 class):

  * The ST row is an old horse (DOB ~1960s-90s, st_id set).
  * Some past ATG ingest attached an `x:CC:NAME` atg_id to this row.
  * Modern races for the namesake (born decades later) keep landing here
    because we lookup by atg_id.

Detection: ST horse with `x:` atg_id and at least one entry dated more
than ~18y (6,570 days) after `date_of_birth`.

Fix per horse:
  1. Detach the polluted `atg_id` (set to NULL).
  2. Find a candidate target: another horse with the same name + same
     country whose DOB plausibly matches the late entries' race years.
     - Match by name (normalised) + country + DOB-year within ±2 of
       (earliest polluted-race year) - typical age 3..10.
  3. If a single candidate found → move polluted entries to that horse
     (via horse_merge_log audit log, treating polluted ST row as `from`
     and candidate as `to` for the ENTRIES ONLY — implemented as a
     direct UPDATE since we don't want to delete the ST row).
  4. If no candidate → mint a new horse row with the synth atg_id and
     route entries to it, leaving the ST row untouched.

This script is conservative: it does NOT merge the ST row away. The
polluted ST row stays intact (it's a legitimate historical horse
record) — we only move misattributed entries off it.
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


def _fetch_polluted(cur, limit: int | None) -> list[dict]:
    sql = """
    WITH polluted AS (
      SELECT h.horse_id, h.name, h.atg_id, h.st_id, h.birth_country,
             h.date_of_birth,
             -- first race AMONG THE POLLUTED entries (not the horse's career)
             MIN(r.race_date) FILTER (WHERE (r.race_date - h.date_of_birth) > 6570)
                AS first_polluted_race,
             MAX(r.race_date) FILTER (WHERE (r.race_date - h.date_of_birth) > 6570)
                AS last_polluted_race,
             COUNT(*) FILTER (WHERE (r.race_date - h.date_of_birth) > 6570)
                AS polluted_entries
        FROM horse h
        JOIN entry e ON e.horse_id = h.horse_id
        JOIN race  r ON r.race_id  = e.race_id
       WHERE h.atg_id LIKE 'x:%%'
         AND h.st_id IS NOT NULL
         AND h.date_of_birth IS NOT NULL
       GROUP BY h.horse_id
      HAVING COUNT(*) FILTER (WHERE (r.race_date - h.date_of_birth) > 6570) > 0
    )
    SELECT * FROM polluted
     ORDER BY polluted_entries DESC
    """
    if limit:
        sql += f" LIMIT {int(limit)}"
    cur.execute(sql)
    cols = [d.name for d in cur.description]
    return [dict(zip(cols, r)) for r in cur.fetchall()]


def _find_target(cur, name: str, country: str | None, polluted_first_year: int) -> int | None:
    """Look for a likely correct foreign horse for the polluted entries.

    Candidate horse must:
      - share the normalised name (regardless of suffix marks)
      - share country OR have unknown country (relaxed)
      - have a DOB that could plausibly produce the first polluted race
        year (DOB year between polluted_first_year - 12 and
        polluted_first_year - 1)
      - NOT be the polluted horse itself

    Returns horse_id if exactly one candidate matches, else None
    (ambiguous → leave for manual review).
    """
    cur.execute(
        """
        SELECT h.horse_id, h.date_of_birth, h.st_id, h.atg_id, h.birth_country
          FROM horse h
         WHERE v2_normalize_name(h.name) = v2_normalize_name(%s)
           AND v2_normalize_name(h.name) <> ''
           AND (h.birth_country = %s OR h.birth_country IS NULL OR %s IS NULL)
           AND h.date_of_birth IS NOT NULL
           AND EXTRACT(year FROM h.date_of_birth)::int BETWEEN %s AND %s
        """,
        (name, country, country,
         polluted_first_year - 12, polluted_first_year - 1),
    )
    rows = cur.fetchall()
    if len(rows) == 1:
        return rows[0][0]
    return None


def _move_late_entries(cur, src_horse_id: int, dst_horse_id: int) -> int:
    """Repoint entries dated more than 18y after the SRC horse's DOB to DST.

    Returns count moved. Same-race conflicts: keep DST entry, delete SRC entry.
    """
    cur.execute(
        """
        WITH polluted AS (
          SELECT e.entry_id, e.race_id
            FROM entry e
            JOIN race  r ON r.race_id = e.race_id
            JOIN horse h ON h.horse_id = e.horse_id
           WHERE h.horse_id = %s
             AND (r.race_date - h.date_of_birth) > 6570
        ),
        conflict AS (
          DELETE FROM entry e
            USING polluted p
           WHERE e.entry_id = p.entry_id
             AND EXISTS (SELECT 1 FROM entry e2
                          WHERE e2.race_id = p.race_id AND e2.horse_id = %s)
          RETURNING e.entry_id
        )
        UPDATE entry SET horse_id = %s
         WHERE entry_id IN (
            SELECT entry_id FROM polluted
             WHERE entry_id NOT IN (SELECT entry_id FROM conflict)
         )
        """,
        (src_horse_id, dst_horse_id, dst_horse_id),
    )
    return cur.rowcount


def _create_split_row(cur, polluted: dict) -> int:
    """Insert a new foreign horse row to receive the misattributed entries."""
    # The polluted row is often an old Swedish/ST horse, so do not copy its
    # country onto the new synthetic ATG row. Prefer the synth key country:
    # x:FR:NEVER ON TIME -> FR.
    parts = (polluted.get("atg_id") or "").split(":", 2)
    split_country = parts[1] if len(parts) == 3 and parts[1] else polluted["birth_country"]
    cur.execute(
        """
        INSERT INTO horse (name, birth_country, atg_id, primary_source,
                           source_data, last_updated_at)
        VALUES (%s, %s, %s, 'atg', %s, NOW())
        RETURNING horse_id
        """,
        (polluted["name"], split_country, polluted["atg_id"],
         Json({"atg": {"split_from_horse_id": polluted["horse_id"],
                       "split_reason": "category_a_polluted"}})),
    )
    return cur.fetchone()[0]


def _log_split(cur, polluted: dict, dst_id: int, moved: int, method: str) -> None:
    """Audit a split into horse_merge_log even though src row stays."""
    snap = {k: (v.isoformat() if hasattr(v, "isoformat") else v)
            for k, v in polluted.items()}
    cur.execute(
        """
        INSERT INTO horse_merge_log
            (from_horse_id, to_horse_id, reason, method,
             entries_moved, conflicts_resolved, from_snapshot, merged_by)
        VALUES (%s, %s, %s, %s, %s, 0, %s, 'split_polluted_atg_ids')
        """,
        (polluted["horse_id"], dst_id,
         f"category_a split: detached atg_id={polluted['atg_id']!r} and moved "
         f"{moved} entries off old ST row (DOB {polluted['date_of_birth']})",
         method, moved, json.dumps(snap, default=str)),
    )


def main() -> int:
    args = build_argparser("split_polluted_atg_ids").parse_args()
    with script_runner("split_polluted_atg_ids", args) as (conn, log, summary):
        log(f"[split_polluted_atg_ids] execute={args.execute}, limit={args.limit}")
        with conn.cursor() as cur:
            polluted = _fetch_polluted(cur, args.limit)
        summary["candidates"] = len(polluted)
        log(f"Found {len(polluted)} polluted horses.")

        for p in polluted:
            first_year = p["first_polluted_race"].year
            target_id = None
            with conn.cursor() as cur:
                target_id = _find_target(cur, p["name"], p["birth_country"], first_year)
                # Never split into the polluted row itself.
                if target_id == p["horse_id"]:
                    target_id = None

            method = "category_a_existing" if target_id is not None else "category_a_split"
            tgt_name = f"horse {target_id}" if target_id else "NEW ROW"
            log(f"  {p['horse_id']}  {p['name']!r}  DOB={p['date_of_birth']}  "
                f"polluted={p['polluted_entries']}  → {method} ({tgt_name})")

            if not args.execute:
                summary["merged"] += 1
                continue

            with conn.cursor() as cur:
                # Detach atg_id from polluted row FIRST (prevents future
                # foreign re-pollution AND frees the synth key for the new row).
                cur.execute(
                    "UPDATE horse SET atg_id = NULL, last_updated_at = NOW() "
                    "WHERE horse_id = %s AND atg_id LIKE 'x:%%'",
                    (p["horse_id"],),
                )
                if target_id is None:
                    target_id = _create_split_row(cur, p)
                moved = _move_late_entries(cur, p["horse_id"], target_id)
                _log_split(cur, p, target_id, moved, method)
            summary["merged"] += 1
            if summary["merged"] % args.commit_every == 0:
                conn.commit()

        if args.execute:
            conn.commit()
    return 0


if __name__ == "__main__":
    sys.exit(main())
