"""
One-off fix for a specific class of "polluted horse" bug: a foreign
(synthetic-atg_id) horse's race history landed on the WRONG canonical
row because a same-named-but-different horse already held the
`x:CC:NAME` synth key.

This is NOT a merge — both horses are real, distinct animals. The fix:

  1. Detach the synthetic `atg_id` from `--src` (the row it wrongly
     attached to).
  2. Attach that same synthetic key onto `--dst` (the row it should
     have gone to), so future ingests land correctly. Refuses to run
     if `--dst` already has a *different* atg_id.
  3. Move every entry currently on `--src` onto `--dst`. Where `--dst`
     already has an entry for the same `race_id` (the common case —
     another source already had the correct history), the two entries
     are column-merged via `core.identity._merge_entries_columnwise`
     (keeper = dst's existing entry) and the src entry is deleted.
     Where there's no conflict, the entry is simply repointed.
  4. `--src` is NEVER deleted — it keeps its own identity (st_id /
     letrot_id / etc.) and whatever legitimate entries it has (usually
     none, if it hadn't raced yet).
  5. Logs an audit row to `horse_merge_log` (method='fix_wrong_synth_atg')
     with a full snapshot for traceability. `--src` is not deleted so
     the generic `rollback_horse_merge` does not apply; this is a
     record-only audit trail.

Example (Learn to Fly SE #368634 <- polluted by FR #442343's races):

    python -m scripts.fix_wrong_synth_atg_horse \\
        --src 368634 --dst 442343                  # dry-run
    python -m scripts.fix_wrong_synth_atg_horse \\
        --src 368634 --dst 442343 --execute
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from psycopg2.extras import Json  # noqa: E402

from core.identity import _merge_entries_columnwise  # noqa: E402
from scripts._merge_helpers import build_argparser, script_runner  # noqa: E402


def _fetch_horse(cur, horse_id: int) -> dict | None:
    cur.execute("SELECT * FROM horse WHERE horse_id = %s", (horse_id,))
    row = cur.fetchone()
    if not row:
        return None
    cols = [d.name for d in cur.description]
    return dict(zip(cols, row))


def _row_to_jsonable(row: dict) -> dict:
    out: dict = {}
    for k, v in row.items():
        if v is None or isinstance(v, (str, int, float, bool, dict, list)):
            out[k] = v
        else:
            out[k] = str(v)
    return out


def _plan(cur, src_id: int, dst_id: int) -> dict:
    """Fetch both rows + entry conflict/movable split. Raises on bad input."""
    src = _fetch_horse(cur, src_id)
    dst = _fetch_horse(cur, dst_id)
    if not src:
        raise SystemExit(f"--src {src_id} not found")
    if not dst:
        raise SystemExit(f"--dst {dst_id} not found")
    if not (src.get("atg_id") or "").startswith("x:"):
        raise SystemExit(
            f"--src {src_id} has atg_id={src.get('atg_id')!r}, "
            f"expected a synthetic 'x:CC:NAME' key — refusing to run"
        )
    if dst.get("atg_id") not in (None, src.get("atg_id")):
        raise SystemExit(
            f"--dst {dst_id} already has a different atg_id="
            f"{dst.get('atg_id')!r} — refusing to overwrite, resolve manually"
        )

    cur.execute(
        """
        SELECT e.entry_id AS src_entry, e.race_id, d.entry_id AS dst_entry,
               t.country
          FROM entry e
          LEFT JOIN entry d ON d.race_id = e.race_id AND d.horse_id = %s
          LEFT JOIN race  r ON r.race_id = e.race_id
          LEFT JOIN track t ON t.track_id = r.track_id
         WHERE e.horse_id = %s
         ORDER BY e.race_id
        """,
        (dst_id, src_id),
    )
    rows = cur.fetchall()
    conflicts = [(r[0], r[2], r[3]) for r in rows if r[2] is not None]
    movable = [r[0] for r in rows if r[2] is None]
    return {"src": src, "dst": dst, "conflicts": conflicts, "movable": movable}


def _apply(cur, src_id: int, dst_id: int, plan: dict) -> dict:
    src, dst = plan["src"], plan["dst"]
    synth_key = src["atg_id"]

    entry_audits: list[dict] = []
    conflicts_resolved = 0
    for src_eid, dst_eid, country in plan["conflicts"]:
        cur.execute("SELECT * FROM entry WHERE entry_id IN (%s, %s)", (src_eid, dst_eid))
        ecols = [d.name for d in cur.description]
        by_id = {r[ecols.index("entry_id")]: dict(zip(ecols, r)) for r in cur.fetchall()}
        keeper = by_id[dst_eid]
        loser = by_id[src_eid]
        set_dict, audit_block = _merge_entries_columnwise(
            keeper, loser, is_french_race=(country == "FR"),
        )
        if set_dict:
            cols_sql = ", ".join(f"{c} = %s" for c in set_dict)
            vals = [Json(v) if isinstance(v, dict) else v for v in set_dict.values()]
            cur.execute(
                f"UPDATE entry SET {cols_sql}, last_updated_at = NOW() WHERE entry_id = %s",
                [*vals, dst_eid],
            )
        cur.execute("DELETE FROM entry WHERE entry_id = %s", (src_eid,))
        entry_audits.append(audit_block)
        conflicts_resolved += 1

    moved = 0
    if plan["movable"]:
        cur.execute(
            "UPDATE entry SET horse_id = %s WHERE entry_id = ANY(%s)",
            (dst_id, plan["movable"]),
        )
        moved = cur.rowcount

    # Detach from src, attach to dst.
    cur.execute(
        "UPDATE horse SET atg_id = NULL, last_updated_at = NOW() WHERE horse_id = %s",
        (src_id,),
    )
    if dst.get("atg_id") is None:
        cur.execute(
            "UPDATE horse SET atg_id = %s, last_updated_at = NOW() WHERE horse_id = %s",
            (synth_key, dst_id),
        )

    snapshot = {
        "src_horse_row": _row_to_jsonable(src),
        "dst_horse_row_before": _row_to_jsonable(dst),
        "detached_synth_atg_id": synth_key,
        "reattached_to_dst": dst.get("atg_id") is None,
        "moved_entry_ids": list(plan["movable"]),
        "entry_merges": entry_audits,
    }
    cur.execute(
        """
        INSERT INTO horse_merge_log
            (from_horse_id, to_horse_id, reason, method,
             entries_moved, conflicts_resolved, from_snapshot, merged_by)
        VALUES (%s, %s, %s, 'fix_wrong_synth_atg', %s, %s, %s, 'fix_wrong_synth_atg')
        """,
        (
            src_id, dst_id,
            f"wrong synth atg_id={synth_key!r} was attached to horse {src_id} "
            f"({src.get('name')!r}, DOB {src.get('date_of_birth')}) instead of "
            f"horse {dst_id} ({dst.get('name')!r}, DOB {dst.get('date_of_birth')}); "
            f"moved/merged {conflicts_resolved + moved} entries, src row NOT deleted",
            moved, conflicts_resolved, Json(snapshot),
        ),
    )
    return {"moved": moved, "conflicts_resolved": conflicts_resolved}


def main() -> int:
    parser = build_argparser("fix_wrong_synth_atg_horse")
    parser.add_argument("--src", type=int, required=True,
                        help="horse_id currently (wrongly) holding the synth atg_id")
    parser.add_argument("--dst", type=int, required=True,
                        help="horse_id that should own the synth atg_id / entries")
    args = parser.parse_args()

    with script_runner("fix_wrong_synth_atg_horse", args) as (conn, log, summary):
        with conn.cursor() as cur:
            plan = _plan(cur, args.src, args.dst)

        src, dst = plan["src"], plan["dst"]
        log(f"[fix_wrong_synth_atg_horse] src={args.src} ({src['name']!r}, "
            f"DOB={src['date_of_birth']}, atg_id={src['atg_id']!r})")
        log(f"                            dst={args.dst} ({dst['name']!r}, "
            f"DOB={dst['date_of_birth']}, atg_id={dst['atg_id']!r})")
        log(f"  entries on src: {len(plan['conflicts']) + len(plan['movable'])} total, "
            f"{len(plan['conflicts'])} conflict (column-merge+delete), "
            f"{len(plan['movable'])} plain move")
        summary["candidates"] = 1

        if not args.execute:
            summary["merged"] = 1
            log("\nDRY-RUN — no DB changes. Use --execute to apply.")
            if args.json:
                print(json.dumps(summary, indent=2, default=str))
            return 0

        with conn.cursor() as cur:
            res = _apply(cur, args.src, args.dst, plan)
        summary["merged"] = 1
        summary["entries_moved"] = res["moved"]
        summary["entry_conflicts_dropped"] = res["conflicts_resolved"]
        conn.commit()
        log(f"  done: moved={res['moved']} conflicts_resolved={res['conflicts_resolved']}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
