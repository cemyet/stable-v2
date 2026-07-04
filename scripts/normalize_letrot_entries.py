"""
One-shot normaliser for existing LeTrot entries that were imported
BEFORE we converged the LeTrot ETL onto ATG/ST display conventions.

Two things to retro-fix on every entry whose race has a letrot_race_id:

  1. `placement_text` — strip the French 'e' ordinal suffix so the
     leaderboard / win-rate SQL (which keys off `IN ('1','2','3')`)
     actually counts LeTrot finishes. '2e' → '2', '15e' → '15'.
     Codes that aren't pure numerals (DA, NP, etc.) are left untouched.

  2. `time_text` — the old ETL stored the LeTrot TOTAL race time
     (`'2\\'53"6'`) here, but the ATG/ST convention is the per-km time
     (`'10,9'`). The km-time-in-seconds is already correct in
     `time_seconds`, so we just render it Swedish-style via the same
     helper the new ETL uses. The raw FR total time + km-time are
     preserved in `source_data.letrot.total_time_text` and
     `source_data.letrot.km_time_text` so we lose nothing.

The script is idempotent — running it again after a partial run picks up
where it left off without re-touching already-clean rows.
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from etl.import_letrot import km_time_to_swedish, placement_to_swedish  # noqa: E402
from scripts._merge_helpers import build_argparser, script_runner  # noqa: E402


def _fetch_dirty_entries(cur, limit: int | None) -> list[tuple]:
    """Return entries on letrot races whose placement_text/time_text
    aren't yet in Swedish convention."""
    sql = """
    SELECT e.entry_id, e.placement_text, e.time_text, e.time_seconds
      FROM entry e
      JOIN race  r ON r.race_id = e.race_id
     WHERE r.letrot_race_id IS NOT NULL
       AND (
            -- 'Ne' patterns: 2-digit-then-e, single-digit-then-e
            (e.placement_text ~ '^[0-9]+e$')
         OR -- French total-time format leaked into time_text: contains '
            (e.time_text LIKE '%''%')
         OR -- TNC or other non-numeric in time_text when we DO have seconds
            (e.time_text IS NULL AND e.time_seconds IS NOT NULL)
       )
    """
    if limit:
        sql += f"\n     LIMIT {int(limit)}"
    cur.execute(sql)
    return cur.fetchall()


def main() -> int:
    args = build_argparser("normalize_letrot_entries").parse_args()
    with script_runner("normalize_letrot_entries", args) as (conn, log, summary):
        summary.pop("merged", None)
        summary["updated"] = 0
        summary["placement_only"] = 0
        summary["time_only"] = 0
        summary["both"] = 0

        log(f"[normalize_letrot_entries] execute={args.execute} limit={args.limit}")

        with conn.cursor() as cur:
            rows = _fetch_dirty_entries(cur, args.limit)
        summary["candidates"] = len(rows)
        log(f"Found {len(rows):,} dirty LeTrot entries.")

        # Sample preview
        for r in rows[:6]:
            eid, pt, tt, ts = r
            new_pt = placement_to_swedish(pt)
            new_tt = km_time_to_swedish(ts) if ts is not None else tt
            log(f"  #{eid:<10} placement {pt!r:<8} → {new_pt!r:<8}    "
                f"time {tt!r:<14} (secs={ts}) → {new_tt!r}")
        if len(rows) > 6:
            log(f"  ... ({len(rows) - 6:,} more)")

        if not args.execute:
            log("\n[dry-run] no writes performed.")
            return 0

        with conn.cursor() as cur:
            n_done = 0
            for eid, pt, tt, ts in rows:
                new_pt = placement_to_swedish(pt)
                new_tt = km_time_to_swedish(ts) if ts is not None else None

                changes: dict[str, str] = {}
                if new_pt != pt:
                    changes["placement_text"] = new_pt
                if new_tt != tt and (new_tt is not None or tt is not None):
                    changes["time_text"] = new_tt

                if not changes:
                    continue

                cols = ", ".join(f"{k} = %s" for k in changes)
                cur.execute(
                    f"UPDATE entry SET {cols} WHERE entry_id = %s",
                    (*changes.values(), eid),
                )
                if "placement_text" in changes and "time_text" in changes:
                    summary["both"] += 1
                elif "placement_text" in changes:
                    summary["placement_only"] += 1
                else:
                    summary["time_only"] += 1
                summary["updated"] += 1
                n_done += 1

                if n_done % args.commit_every == 0:
                    conn.commit()
                    log(f"    [commit] {n_done:,}/{len(rows):,}")
            conn.commit()
            log(f"\n[done] updated {summary['updated']:,} entries "
                f"(placement_only={summary['placement_only']:,}  "
                f"time_only={summary['time_only']:,}  "
                f"both={summary['both']:,})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
