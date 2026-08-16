"""Create the start-list rollups (horse_builder_stats, person_form_recent).

Both live in STATS_VIEWS_DDL, but that block drops and rebuilds every browse
matview; this applies just these two, for bootstrapping or after changing their
definition.

Local only, by design. The cloud gets its copies from `jobs.publish`, which
ships the finished rows — building them on the cloud instance means scanning the
4.3GB entry table it cannot cache, which took 10 and 40+ minutes.

Usage:
    python3 -m scripts.build_builder_stats
    python3 -m scripts.build_builder_stats --only person_form_recent
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import psycopg2

from core import config

_QUALIFIER = (
    r"^(gdk|ejg|ejp|gd|gk|egk|egdk|gkd|gdj|gdb|gdek|gdgk|gdl|ddk|frdk|erj|ejk"
    r"|ejgk|ejgd|ej|EJ|EJG|EJP|GDK|Gdk|g)[0-9]?$|^[12]?[pP][0-9]?$"
)

DDL = f"""
DROP MATERIALIZED VIEW IF EXISTS horse_builder_stats CASCADE;
CREATE MATERIALIZED VIEW horse_builder_stats AS
SELECT e.horse_id,
       COUNT(*) FILTER (WHERE q.started)                       AS starts,
       COUNT(*) FILTER (WHERE q.is_win)                        AS wins,
       COUNT(*) FILTER (WHERE q.started
                          AND NOT COALESCE(e.galopp, false)
                          AND NOT COALESCE(e.disqualified, false)) AS clean_starts,
       COUNT(*) FILTER (WHERE q.is_win
                          AND NOT COALESCE(e.galopp, false))    AS clean_wins,
       COALESCE(bool_or(q.ran AND e.shoe_code IN ('3','4')), false) AS ever_front_shod,
       COALESCE(bool_or(q.ran AND e.shoe_code IN ('1','2')), false) AS ever_front_bare,
       COALESCE(bool_or(q.ran AND e.shoe_code IN ('2','4')), false) AS ever_back_shod,
       COALESCE(bool_or(q.ran AND e.shoe_code IN ('1','3')), false) AS ever_back_bare,
       COALESCE(bool_or(q.ran AND upper(btrim(e.sulky)) = 'AM'), false) AS ever_am
FROM entry e
CROSS JOIN LATERAL (
    SELECT NOT e.withdrawn
             AND COALESCE(e.placement_text, '') !~ '{_QUALIFIER}' AS started,
           e.placement_text = '1'
             AND NOT COALESCE(e.disqualified, false)
             AND COALESCE(e.placement_text, '') !~ '{_QUALIFIER}' AS is_win,
           NOT COALESCE(e.withdrawn, false) AS ran
) q
WHERE e.horse_id IS NOT NULL
  AND e.race_date IS NOT NULL
  AND e.race_date < CURRENT_DATE
GROUP BY e.horse_id;

CREATE UNIQUE INDEX ON horse_builder_stats (horse_id);
"""

FORM_DDL = f"""
DROP MATERIALIZED VIEW IF EXISTS person_form_recent CASCADE;
CREATE MATERIALIZED VIEW person_form_recent AS
SELECT 'driver'::text AS role,
       e.driver_id    AS person_id,
       COUNT(*) FILTER (WHERE q.started)          AS starts,
       COUNT(*) FILTER (WHERE q.is_win)           AS wins,
       COUNT(eo.market_outperf)                   AS n_of,
       COALESCE(SUM(eo.market_outperf), 0)        AS sum_of,
       COUNT(ep.perf)                             AS n_pf,
       COALESCE(SUM(ep.perf), 0)                  AS sum_pf
FROM entry e
LEFT JOIN entry_outperf eo ON eo.entry_id = e.entry_id
LEFT JOIN entry_perf    ep ON ep.entry_id = e.entry_id
CROSS JOIN LATERAL (
    SELECT NOT COALESCE(e.withdrawn, false)
             AND COALESCE(e.placement_text, '') !~ '{_QUALIFIER}' AS started,
           e.placement_text = '1'
             AND NOT COALESCE(e.disqualified, false)
             AND COALESCE(e.placement_text, '') !~ '{_QUALIFIER}' AS is_win
) q
WHERE e.driver_id IS NOT NULL
  AND e.race_date >= CURRENT_DATE - INTERVAL '30 days'
  AND e.race_date <  CURRENT_DATE
GROUP BY e.driver_id
UNION ALL
SELECT 'trainer'::text,
       e.trainer_id,
       COUNT(*) FILTER (WHERE q.started),
       COUNT(*) FILTER (WHERE q.is_win),
       COUNT(eo.market_outperf),
       COALESCE(SUM(eo.market_outperf), 0),
       COUNT(ep.perf),
       COALESCE(SUM(ep.perf), 0)
FROM entry e
LEFT JOIN entry_outperf eo ON eo.entry_id = e.entry_id
LEFT JOIN entry_perf    ep ON ep.entry_id = e.entry_id
CROSS JOIN LATERAL (
    SELECT NOT COALESCE(e.withdrawn, false)
             AND COALESCE(e.placement_text, '') !~ '{_QUALIFIER}' AS started,
           e.placement_text = '1'
             AND NOT COALESCE(e.disqualified, false)
             AND COALESCE(e.placement_text, '') !~ '{_QUALIFIER}' AS is_win
) q
WHERE e.trainer_id IS NOT NULL
  AND e.race_date >= CURRENT_DATE - INTERVAL '30 days'
  AND e.race_date <  CURRENT_DATE
GROUP BY e.trainer_id;

CREATE UNIQUE INDEX ON person_form_recent (role, person_id);
"""


VIEWS = (("horse_builder_stats", DDL), ("person_form_recent", FORM_DDL))


def build(url: str, label: str, timeout: str, only: str | None = None) -> None:
    print(f"\n=== {label} ===", flush=True)
    conn = psycopg2.connect(url, connect_timeout=20, keepalives=1,
                            keepalives_idle=30, keepalives_interval=10,
                            keepalives_count=5)
    try:
        with conn.cursor() as cur:
            cur.execute(f"SET statement_timeout = '{timeout}'")
            for name, ddl in VIEWS:
                if only and name != only:
                    continue
                t0 = time.time()
                print(f"building {name} (scans entry)...", flush=True)
                cur.execute(ddl)
                conn.commit()
                cur.execute(f"SELECT COUNT(*) FROM {name}")
                n = cur.fetchone()[0]
                cur.execute(
                    f"SELECT pg_size_pretty(pg_total_relation_size('{name}'))")
                print(f"  {name}: {n:,} rows, {cur.fetchone()[0]}, "
                      f"{time.time() - t0:.1f}s", flush=True)
    finally:
        conn.close()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", help="build just this one view")
    args = ap.parse_args()
    build(config.DATABASE_URL, "local", "30min", args.only)
    print("\nRun `python3 -m jobs.publish --snapshots-only` to ship these "
          "to the cloud.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
