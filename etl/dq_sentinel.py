"""Data-quality sentinel: compute + record integrity metrics in dq_metric.

Two use-cases:

    snapshot(conn, measured_by="baseline")
        Full-database metrics. Takes ~20-30s (scans entry once). Run at
        milestones (before/after repair projects) and nightly after cleanup.

    incremental_check(conn, since)
        Cheap invariant check scoped to races touched since `since`
        (a timestamp) — returns violations suitable for the job summary,
        e.g. a horse imported tonight now standing at two tracks on one date.

Metrics recorded (one dq_metric row per metric per snapshot):

    impossible_horse_date_pairs   same horse entered at >1 track on one date
    dup_race_groups               >1 race row with same (track, date, number)
    synth_atg_horses              horse rows with atg_id LIKE 'x:%'
    weak_id_horses                horse rows with no strong source id at all
    synth_persons                 person rows with atg_id LIKE 'x:%'
    placeholder_track_races       races filed on country-placeholder tracks
    horses / persons / races / entries / tracks   raw row counts
"""

from __future__ import annotations

import json
import logging
from datetime import datetime

log = logging.getLogger(__name__)

PLACEHOLDER_TRACK_NAMES = (
    "italien", "belgien", "frankrike", "norge", "danmark", "finland",
    "tyskland", "holland", "usa", "österrike", "schweiz", "spanien",
    "malta", "utlandet",
)


def collect_metrics(conn) -> dict[str, int]:
    """Compute all sentinel metrics. Read-only."""
    cur = conn.cursor()
    out: dict[str, int] = {}

    for t in ("horse", "person", "race", "entry", "track"):
        cur.execute(f"SELECT COUNT(*) FROM {t}")
        out[f"{t}s" if t != "entry" else "entries"] = cur.fetchone()[0]

    cur.execute(
        """
        SELECT COUNT(*) FROM (
            SELECT e.horse_id, r.race_date
              FROM entry e JOIN race r ON r.race_id = e.race_id
             GROUP BY e.horse_id, r.race_date
            HAVING COUNT(DISTINCT r.track_id) > 1
        ) x
        """
    )
    out["impossible_horse_date_pairs"] = cur.fetchone()[0]

    cur.execute(
        """
        SELECT COUNT(*) FROM (
            SELECT track_id, race_date, race_number FROM race
             WHERE race_number IS NOT NULL AND track_id IS NOT NULL
             GROUP BY 1, 2, 3 HAVING COUNT(*) > 1
        ) x
        """
    )
    out["dup_race_groups"] = cur.fetchone()[0]

    cur.execute("SELECT COUNT(*) FROM horse WHERE atg_id LIKE 'x:%'")
    out["synth_atg_horses"] = cur.fetchone()[0]

    cur.execute(
        """
        SELECT COUNT(*) FROM horse
         WHERE st_id IS NULL AND letrot_id IS NULL AND hvt_id IS NULL
           AND usta_id IS NULL AND (atg_id IS NULL OR atg_id LIKE 'x:%')
        """
    )
    out["weak_id_horses"] = cur.fetchone()[0]

    cur.execute("SELECT COUNT(*) FROM person WHERE atg_id LIKE 'x:%'")
    out["synth_persons"] = cur.fetchone()[0]

    cur.execute(
        """
        SELECT COUNT(*) FROM race r JOIN track t ON t.track_id = r.track_id
         WHERE lower(t.name) = ANY(%s)
        """,
        (list(PLACEHOLDER_TRACK_NAMES),),
    )
    out["placeholder_track_races"] = cur.fetchone()[0]

    return out


def snapshot(conn, *, measured_by: str = "nightly", commit: bool = True) -> dict[str, int]:
    """Compute metrics and append one dq_metric row per metric."""
    metrics = collect_metrics(conn)
    cur = conn.cursor()
    for metric, value in metrics.items():
        cur.execute(
            "INSERT INTO dq_metric (metric, value, detail) VALUES (%s, %s, %s)",
            (metric, value, json.dumps({"measured_by": measured_by})),
        )
    if commit:
        conn.commit()
    log.info("dq snapshot (%s): %s", measured_by, metrics)
    return metrics


def compare_with_previous(conn, metrics: dict[str, int]) -> list[str]:
    """Return human-readable warnings for metrics that regressed vs the
    previous snapshot. 'Regressed' = integrity metric went UP."""
    watch_up = (
        "impossible_horse_date_pairs",
        "dup_race_groups",
    )
    warnings: list[str] = []
    cur = conn.cursor()
    for metric in watch_up:
        cur.execute(
            """
            SELECT value FROM dq_metric
             WHERE metric = %s
             ORDER BY measured_at DESC
             OFFSET 1 LIMIT 1
            """,
            (metric,),
        )
        row = cur.fetchone()
        if row is not None and metrics.get(metric, 0) > row[0]:
            warnings.append(
                f"DQ regression: {metric} rose {row[0]} -> {metrics[metric]}"
            )
    return warnings


def incremental_check(conn, since: datetime) -> list[dict]:
    """Invariant check scoped to recently-imported races: find horses that,
    counting only dates touched by races created/updated since `since`, now
    stand at more than one track on the same date. Cheap enough for the
    nightly summary."""
    cur = conn.cursor()
    cur.execute(
        """
        WITH recent AS (
            SELECT DISTINCT e.horse_id, r.race_date
              FROM race r JOIN entry e ON e.race_id = r.race_id
             WHERE r.last_updated_at >= %s
        )
        SELECT rec.horse_id, h.name, rec.race_date,
               array_agg(DISTINCT t.name ORDER BY t.name) AS tracks
          FROM recent rec
          JOIN entry e  ON e.horse_id = rec.horse_id
          JOIN race  r  ON r.race_id = e.race_id AND r.race_date = rec.race_date
          JOIN track t  ON t.track_id = r.track_id
          JOIN horse h  ON h.horse_id = rec.horse_id
         GROUP BY rec.horse_id, h.name, rec.race_date
        HAVING COUNT(DISTINCT r.track_id) > 1
         LIMIT 50
        """,
        (since,),
    )
    return [
        {"horse_id": r[0], "name": r[1], "date": str(r[2]), "tracks": r[3]}
        for r in cur.fetchall()
    ]


if __name__ == "__main__":
    import argparse

    from core.db import get_connection

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--by", default="manual", help="measured_by label")
    args = ap.parse_args()
    conn = get_connection()
    m = snapshot(conn, measured_by=args.by)
    for k, v in sorted(m.items()):
        print(f"{k:32s} {v:,}")
