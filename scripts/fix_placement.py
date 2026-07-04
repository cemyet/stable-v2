#!/usr/bin/env python3
"""
Fix placement integers and time-rank unplaced entries.

Three passes:

1. Trust placement_text for entries where text is a clean digit 1-15
   but placement disagrees. (127k rows — the Vincennes/Chitchat bug.)

2. NULL out placement for entries where text='0' and placement=1 but
   the race has other entries with higher placements (provably wrong).
   Then recompute via time-ranking for autostart entries.

3. Report summary stats so we can verify the fix.

Usage:
    python3 -m scripts.fix_placement              # dry-run (default)
    python3 -m scripts.fix_placement --execute     # actually commit
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from core.db import get_connection


def _pass1_trust_text(cur, *, dry_run: bool) -> int:
    """Set placement = int(placement_text) where text is 1-15 and they disagree."""
    cur.execute("""
        SELECT COUNT(*) FROM entry
         WHERE placement_text ~ '^[1-9][0-9]?$'
           AND placement_text::int BETWEEN 1 AND 15
           AND placement IS DISTINCT FROM placement_text::int
    """)
    count = cur.fetchone()[0]
    print(f"\nPass 1: placement_text is clean digit 1-15 but placement disagrees")
    print(f"  rows to fix: {count:,}")

    if not dry_run and count > 0:
        cur.execute("""
            UPDATE entry
               SET placement = placement_text::int
             WHERE placement_text ~ '^[1-9][0-9]?$'
               AND placement_text::int BETWEEN 1 AND 15
               AND placement IS DISTINCT FROM placement_text::int
        """)
        print(f"  updated: {cur.rowcount:,}")
    return count


def _pass2_fix_zero_text(cur, *, dry_run: bool) -> int:
    """Fix text='0' entries that are wrongly marked placement=1.

    Step A: NULL out placement for provably-wrong rows.
    Step B: Recompute placement via time-ranking (autostart only).
    """
    cur.execute("""
        WITH race_max AS (
            SELECT race_id,
                   MAX(CASE WHEN placement_text ~ '^[1-9][0-9]?$'
                            THEN placement_text::int END) AS max_ranked
              FROM entry GROUP BY race_id
        )
        SELECT COUNT(*) FROM entry e
          JOIN race_max rm ON rm.race_id = e.race_id
         WHERE e.placement_text = '0' AND e.placement = 1
           AND rm.max_ranked IS NOT NULL AND rm.max_ranked > 1
    """)
    count = cur.fetchone()[0]
    print(f"\nPass 2A: text='0', placement=1, provably wrong")
    print(f"  rows to NULL: {count:,}")

    if not dry_run and count > 0:
        cur.execute("""
            WITH race_max AS (
                SELECT race_id,
                       MAX(CASE WHEN placement_text ~ '^[1-9][0-9]?$'
                                THEN placement_text::int END) AS max_ranked
                  FROM entry GROUP BY race_id
            )
            UPDATE entry e
               SET placement = NULL
              FROM race_max rm
             WHERE rm.race_id = e.race_id
               AND e.placement_text = '0' AND e.placement = 1
               AND rm.max_ranked IS NOT NULL AND rm.max_ranked > 1
        """)
        print(f"  nulled: {cur.rowcount:,}")

    cur.execute("""
        SELECT COUNT(*) FROM entry
         WHERE placement_text = '0' AND placement IS NULL
           AND auto = true
           AND NOT withdrawn AND NOT disqualified
           AND time_seconds IS NOT NULL AND time_seconds > 0
           AND race_id IS NOT NULL
    """)
    rankable = cur.fetchone()[0]
    print(f"\nPass 2B: time-rank NULLed autostart entries")
    print(f"  rankable rows: {rankable:,}")

    if not dry_run and rankable > 0:
        cur.execute("""
            WITH race_max_pl AS (
                SELECT race_id, MAX(placement) AS max_pl
                  FROM entry
                 WHERE race_id IS NOT NULL AND placement IS NOT NULL
                 GROUP BY race_id
            ),
            ranked AS (
                SELECT e.entry_id,
                       COALESCE(rm.max_pl, 0)
                         + ROW_NUMBER() OVER (
                               PARTITION BY e.race_id
                               ORDER BY e.time_seconds ASC
                           ) AS new_placement
                  FROM entry e
                  LEFT JOIN race_max_pl rm ON rm.race_id = e.race_id
                 WHERE e.placement_text = '0' AND e.placement IS NULL
                   AND e.auto = true
                   AND NOT e.withdrawn AND NOT e.disqualified
                   AND e.time_seconds IS NOT NULL AND e.time_seconds > 0
                   AND e.race_id IS NOT NULL
            )
            UPDATE entry e
               SET placement = r.new_placement::smallint
              FROM ranked r
             WHERE e.entry_id = r.entry_id
        """)
        print(f"  ranked: {cur.rowcount:,}")
    return count


def _pass3_verify(cur) -> None:
    """Print summary stats after the fix."""
    cur.execute("""
        SELECT
            COUNT(*) FILTER (WHERE placement_text = '1') AS wins_by_text,
            COUNT(*) FILTER (WHERE placement = 1) AS wins_by_int,
            COUNT(*) FILTER (WHERE placement_text = '1' AND placement = 1) AS agree,
            COUNT(*) FILTER (WHERE placement_text ~ '^[2-9]$' AND placement = 1) AS still_wrong,
            COUNT(*) FILTER (WHERE placement_text = '0' AND placement = 1) AS zero_still_p1,
            COUNT(*) FILTER (WHERE placement IS NULL AND NOT withdrawn) AS null_placement
          FROM entry
    """)
    r = cur.fetchone()
    print(f"\nVerification:")
    print(f"  wins by text='1':          {r[0]:>10,}")
    print(f"  wins by placement=1:       {r[1]:>10,}")
    print(f"  text='1' AND p=1 (agree):  {r[2]:>10,}")
    print(f"  text=2-9 AND p=1 (wrong):  {r[3]:>10,}")
    print(f"  text='0' AND p=1 (wrong):  {r[4]:>10,}")
    print(f"  NULL placement (non-wd):   {r[5]:>10,}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--execute', action='store_true',
                        help='Actually commit changes (default is dry-run)')
    args = parser.parse_args()
    dry_run = not args.execute

    if dry_run:
        print("=== DRY RUN (use --execute to commit) ===")
    else:
        print("=== EXECUTING — changes will be committed ===")

    conn = get_connection()
    try:
        with conn.cursor() as cur:
            _pass1_trust_text(cur, dry_run=dry_run)
            _pass2_fix_zero_text(cur, dry_run=dry_run)
            if not dry_run:
                conn.commit()
                print("\nCommitted.")
            _pass3_verify(cur)
    finally:
        conn.close()


if __name__ == '__main__':
    main()
