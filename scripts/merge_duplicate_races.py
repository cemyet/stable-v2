"""
Category F: merge race rows that share (track_id, race_date, race_number).

Audit shows ~4,643 such groups, ~49k entries across them. They came from
incremental ATG ingest landing a row with a fresh atg_race_id when an ST
ingest had already created a row with no atg_race_id (or vice versa with
different source-id columns).

Resolution per group:
  1. Pick canonical race_id = the one with the highest source-priority
     primary_source (atg > st > letrot > ...), tie-break by most entries
     then lowest race_id.
  2. For each loser race, repoint its entries to the canonical race.
     When the same (race_id, horse_id) already exists on the keeper, run
     `core.identity._merge_entries_columnwise` to coalesce data into the
     keeper entry, then DELETE the loser entry.
  3. Merge per-source ID columns + source_data + canonical fields onto
     the keeper race row.
  4. DELETE the non-canonical race rows.

The column-level merge path is also re-used by `scripts.match_french_races`
via `merge_two_races_columnwise`.

CLI
---
    python -m scripts.merge_duplicate_races              # dry-run
    python -m scripts.merge_duplicate_races --execute    # apply
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from datetime import datetime  # noqa: E402

from psycopg2.extras import Json  # noqa: E402

from core.identity import _merge_entries_columnwise  # noqa: E402
from scripts._merge_helpers import build_argparser, script_runner  # noqa: E402

_RACE_SOURCE_PRIORITY = ("atg", "st", "letrot", "hvt", "kmtid", "usta")
_RACE_SOURCE_ID_COLS = (
    "st_race_id", "atg_race_id", "usta_race_id", "letrot_race_id",
    "kmtid_id", "hvt_race_id",
)
_RACE_CANONICAL_FILL_COLS = (
    "distance", "start_method", "heading", "proposition_text",
    "track_conditions", "victory_margin", "tempo_text", "total_prize_kr",
    "pool_types", "race_class", "age_requirement", "earnings_range",
    "driver_requirement", "start_time", "status",
    "atg_race_day_id", "st_race_day_id",
)


def _fetch_groups(cur, limit: int | None) -> list[dict]:
    sql = """
    SELECT track_id, race_date, race_number,
           array_agg(race_id ORDER BY race_id) AS ids,
           COUNT(*) AS n
      FROM race
     WHERE track_id IS NOT NULL AND race_number IS NOT NULL
     GROUP BY track_id, race_date, race_number
    HAVING COUNT(*) > 1
     ORDER BY n DESC, race_date DESC
    """
    if limit:
        sql += f" LIMIT {int(limit)}"
    cur.execute(sql)
    cols = [d.name for d in cur.description]
    return [dict(zip(cols, r)) for r in cur.fetchall()]


def _load_race(cur, race_id: int) -> dict:
    cur.execute("SELECT * FROM race WHERE race_id = %s", (race_id,))
    cols = [d.name for d in cur.description]
    return dict(zip(cols, cur.fetchone()))


def _entry_count(cur, race_id: int) -> int:
    cur.execute("SELECT COUNT(*) FROM entry WHERE race_id = %s", (race_id,))
    return cur.fetchone()[0]


def _pick_keeper(cur, race_ids: list[int]) -> int:
    """Highest source-priority primary_source → most entries → lowest race_id."""
    cur.execute(
        """
        SELECT r.race_id, r.primary_source,
               (SELECT COUNT(*) FROM entry e WHERE e.race_id = r.race_id) AS n
          FROM race r
         WHERE r.race_id = ANY(%s)
        """,
        (race_ids,),
    )
    rows = cur.fetchall()
    def rank(row):
        rid, src, n = row
        try:
            sp = _RACE_SOURCE_PRIORITY.index(src)
        except (TypeError, ValueError):
            sp = 999
        return (sp, -int(n or 0), rid)
    rows.sort(key=rank)
    return rows[0][0]


def _merge_entries(cur, src_race_id: int, dst_race_id: int,
                   *, is_french_race: bool = False) -> tuple[int, int, list[dict]]:
    """Move entries from src race to dst race; column-merge same-horse conflicts.

    Returns (moved, conflicts_resolved, entry_audit_blocks).
    """
    cur.execute("SELECT * FROM entry WHERE race_id = %s", (src_race_id,))
    cols = [d.name for d in cur.description]
    src_entries = [dict(zip(cols, r)) for r in cur.fetchall()]

    cur.execute("SELECT * FROM entry WHERE race_id = %s", (dst_race_id,))
    dst_by_horse = {dict(zip(cols, r))["horse_id"]: dict(zip(cols, r))
                    for r in cur.fetchall()}

    movable: list[int] = []
    conflicts = 0
    audits: list[dict] = []
    for src in src_entries:
        hid = src["horse_id"]
        if hid not in dst_by_horse:
            movable.append(src["entry_id"])
            continue

        keeper = dst_by_horse[hid]
        set_dict, audit_block = _merge_entries_columnwise(
            keeper, src, is_french_race=is_french_race,
        )
        audits.append(audit_block)

        if set_dict:
            cols_sql = ", ".join(f"{c} = %s" for c in set_dict)
            vals = [Json(v) if isinstance(v, dict) else v
                    for v in set_dict.values()]
            cur.execute(
                f"UPDATE entry SET {cols_sql}, last_updated_at = NOW() "
                f"WHERE entry_id = %s",
                [*vals, keeper["entry_id"]],
            )
        cur.execute("DELETE FROM entry WHERE entry_id = %s", (src["entry_id"],))
        conflicts += 1

    if movable:
        cur.execute(
            "UPDATE entry SET race_id = %s WHERE entry_id = ANY(%s)",
            (dst_race_id, movable),
        )
    return len(movable), conflicts, audits


def merge_two_races_columnwise(cur, *, keeper_id: int, loser_id: int,
                               method: str = "merge_duplicate_races") -> dict:
    """Public helper: merge `loser_id` into `keeper_id` using column-level
    entry merging, then dedupe the race row identity and DELETE the loser.

    Returns a summary dict (no errors) or {"error": ...} on failure.
    """
    dst = _load_race(cur, keeper_id)
    src = _load_race(cur, loser_id)
    if not dst or not src:
        return {"error": f"race row missing keeper={keeper_id} or loser={loser_id}"}

    # Detect French-ness once (used by the column-merge for sex/age pref).
    cur.execute(
        "SELECT t.country FROM race r LEFT JOIN track t ON r.track_id=t.track_id "
        "WHERE r.race_id = %s",
        (keeper_id,),
    )
    row = cur.fetchone()
    is_french = bool(row and row[0] == "FR")

    moved, conflicts, audits = _merge_entries(
        cur, loser_id, keeper_id, is_french_race=is_french,
    )
    _merge_race_identity(cur, dst, src)

    cur.execute("SELECT source_data FROM race WHERE race_id=%s", (keeper_id,))
    sd = cur.fetchone()[0] or {}
    merges = list(sd.get("_merges") or [])
    merges.append({
        "from_race_id":  loser_id,
        "method":        method,
        "moved":         moved,
        "conflicts":     conflicts,
        "merged_at":     datetime.utcnow().isoformat(),
        "from_snapshot": _race_row_jsonable(src),
        "entry_audits":  audits,
    })
    sd["_merges"] = merges
    cur.execute("UPDATE race SET source_data=%s WHERE race_id=%s",
                (Json(sd), keeper_id))
    cur.execute("DELETE FROM race WHERE race_id = %s", (loser_id,))
    return {"moved": moved, "conflicts": conflicts, "keeper": keeper_id,
            "loser": loser_id}


def _race_row_jsonable(row: dict) -> dict:
    """Best-effort JSON-serializable copy of a race row (dates → ISO strings)."""
    out: dict = {}
    for k, v in row.items():
        if v is None or isinstance(v, (str, int, float, bool, dict, list)):
            out[k] = v
        else:
            out[k] = str(v)
    return out


def _merge_race_identity(cur, dst: dict, src: dict) -> None:
    set_parts: list[str] = []
    params: list = []

    sd = dict(dst.get("source_data") or {})
    src_sd = dict(src.get("source_data") or {})

    for col in _RACE_SOURCE_ID_COLS:
        s = src.get(col)
        d = dst.get(col)
        if s is None:
            continue
        if d is None:
            # The loser still holds this source-id value; the column has a
            # UNIQUE constraint, so we must release it on the loser before
            # the keeper UPDATE — otherwise psycopg2 raises UniqueViolation
            # mid-merge and the entire batch aborts.
            cur.execute(
                f"UPDATE race SET {col} = NULL WHERE race_id = %s",
                (src["race_id"],),
            )
            set_parts.append(f"{col} = %s")
            params.append(s)
        elif d != s:
            try:
                source_name = col.removesuffix("_race_id").removesuffix("_id")
            except AttributeError:
                source_name = col.split("_")[0]
            block = dict(sd.get(source_name) or {})
            aliases = list(block.get("aliases") or [])
            if s not in aliases:
                aliases.append(s)
            block["aliases"] = aliases
            sd[source_name] = block

    for col in _RACE_CANONICAL_FILL_COLS:
        if dst.get(col) is None and src.get(col) is not None:
            set_parts.append(f"{col} = %s")
            params.append(src.get(col))

    for sk, sv in src_sd.items():
        if sk not in sd:
            sd[sk] = sv
        elif isinstance(sd[sk], dict) and isinstance(sv, dict):
            sd[sk] = {**sv, **sd[sk]}

    set_parts.append("source_data = %s")
    params.append(Json(sd))
    set_parts.append("last_updated_at = NOW()")
    params.append(dst["race_id"])
    cur.execute(
        f"UPDATE race SET {', '.join(set_parts)} WHERE race_id = %s",
        params,
    )


def main() -> int:
    args = build_argparser("merge_duplicate_races").parse_args()
    with script_runner("merge_duplicate_races", args) as (conn, log, summary):
        log(f"[merge_duplicate_races] execute={args.execute}, limit={args.limit}")
        with conn.cursor() as cur:
            groups = _fetch_groups(cur, args.limit)
        summary["candidates"] = sum(max(0, g["n"] - 1) for g in groups)
        log(f"Found {len(groups)} race groups, {summary['candidates']} merges to attempt.")

        total_moved = 0
        total_conflicts = 0

        for g in groups:
            with conn.cursor() as cur:
                keeper = _pick_keeper(cur, list(g["ids"]))
                losers = [rid for rid in g["ids"] if rid != keeper]
                for loser_id in losers:
                    src_n = _entry_count(cur, loser_id)
                    log(f"  race ({g['track_id']}, {g['race_date']}, #{g['race_number']}): "
                        f"merge {loser_id} → {keeper}  (loser_entries={src_n})")

                    if not args.execute:
                        summary["merged"] += 1
                        continue

                    res = merge_two_races_columnwise(
                        cur, keeper_id=keeper, loser_id=loser_id,
                        method="category_f",
                    )
                    if "error" in res:
                        summary["errors"] += 1
                        log(f"  ! error: {res['error']}")
                        continue
                    summary["merged"] += 1
                    total_moved += res["moved"]
                    total_conflicts += res["conflicts"]

                    if summary["merged"] % args.commit_every == 0:
                        conn.commit()

        if args.execute:
            conn.commit()
            summary["entries_moved"] = total_moved
            summary["conflicts_resolved"] = total_conflicts
            log(f"\nTotal entries moved={total_moved}, conflicts={total_conflicts}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
