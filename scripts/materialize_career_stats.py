"""One-shot migration: VIEW -> MATERIALIZED VIEW for horse_career_stats.

Usage:
    python3 -m scripts.materialize_career_stats
"""
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from core.db import get_connection

MATVIEW_SQL = r"""
DROP VIEW IF EXISTS horse_career_stats CASCADE;
DROP MATERIALIZED VIEW IF EXISTS horse_career_stats CASCADE;

CREATE MATERIALIZED VIEW horse_career_stats AS
WITH computed AS (
    SELECT
        e.horse_id,
        COUNT(*) FILTER (
            WHERE NOT e.withdrawn
              AND COALESCE(e.placement_text, '') !~* '^(gdk|ejg|ejp)'
        ) AS starts,
        COUNT(*) FILTER (
            WHERE e.placement = 1
              AND COALESCE(e.placement_text, '') !~* '^(gdk|ejg|ejp)'
        ) AS wins,
        COUNT(*) FILTER (WHERE e.placement = 2) AS seconds,
        COUNT(*) FILTER (WHERE e.placement = 3) AS thirds,
        COUNT(*) FILTER (
            WHERE e.placement BETWEEN 1 AND 3
              AND COALESCE(e.placement_text, '') !~* '^(gdk|ejg|ejp)'
        ) AS placed,
        COUNT(*) FILTER (
            WHERE COALESCE(e.placement_text, '') ~* '^(gdk|ejg|ejp)'
        ) AS qualifiers,
        COALESCE(SUM(e.prize_kr), 0) AS prize_money_kr,
        MIN(r.race_date) AS first_start,
        MAX(r.race_date) AS last_start
    FROM entry e
    LEFT JOIN race r ON r.race_id = e.race_id
    GROUP BY e.horse_id
)
SELECT
    h.horse_id,
    GREATEST(COALESCE(c.starts, 0),         COALESCE(h.scraped_starts, 0))         AS starts,
    GREATEST(COALESCE(c.wins, 0),           COALESCE(h.scraped_wins, 0))           AS wins,
    COALESCE(c.seconds, 0)    AS seconds,
    COALESCE(c.thirds, 0)     AS thirds,
    COALESCE(c.placed, 0)     AS placed,
    COALESCE(c.qualifiers, 0) AS qualifiers,
    GREATEST(COALESCE(c.prize_money_kr, 0), COALESCE(h.scraped_prize_money_kr, 0)) AS prize_money_kr,
    c.first_start,
    c.last_start
FROM horse h
LEFT JOIN computed c ON c.horse_id = h.horse_id
WHERE c.horse_id IS NOT NULL OR h.scraped_starts IS NOT NULL;

CREATE UNIQUE INDEX ON horse_career_stats (horse_id);
"""


def main() -> int:
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            print("Dropping old view / matview...")
            cur.execute("DROP VIEW IF EXISTS horse_career_stats CASCADE")
            cur.execute("DROP MATERIALIZED VIEW IF EXISTS horse_career_stats CASCADE")
            conn.commit()

            print("Building materialized view (scanning entry table — may take ~30s)...")
            t0 = time.time()
            cur.execute(MATVIEW_SQL)
            conn.commit()
            elapsed = time.time() - t0
            print(f"Done in {elapsed:.1f}s")

            cur.execute("SELECT COUNT(*) FROM horse_career_stats")
            print(f"Rows: {cur.fetchone()[0]:,}")
    finally:
        conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
