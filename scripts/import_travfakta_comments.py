"""
Import the travmedia/travfakta bulk race-comment export into `entry_comment`.

The export is a flat CSV, one row per historic start:

    horse_id,horse_name,atg_id,start_id,race_id,race_date,track_name,
    race_number,distance,start_method,start_number,start_position,
    finishing_position,placement,km_speed,gallop,disqualified,comment

Matching is `etl.comment_match`, shared with the nightly feed
(`etl.import_tr_comments`) that keeps this data current — same source
database, same shape, so the same protocol applies to both. On the full export
1,606,936 of 1,613,377 rows had exactly one candidate entry and the remaining
6,441 were resolved to a single best candidate by corroboration, giving a 1:1
row→entry mapping with no entry claimed twice.

Rows scoring below `--min-score` are left unimported rather than guessed at.

Validation (fields deliberately excluded from matching) came out at 99.97%
agreement on km-time — noting the export drops the leading minute, so 14.2
means 1:14.2 = 74.2s — and 99.02% on the individual handicap distance
(`entry.distance`, not `race.distance`; the export carries the horse's own
distance including tillägg). Placement agrees wherever the two sources share
a convention; they differ on how a dead heat and a disqualified runner are
ranked, which is a result-data nuance and not a matching error.

Usage
-----
    python -m scripts.import_travfakta_comments --csv travfakta_comments_all.csv
    python -m scripts.import_travfakta_comments --csv FILE --execute
    python -m scripts.import_travfakta_comments --csv FILE --execute --report out.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from core.db import get_connection  # noqa: E402
from etl import comment_match  # noqa: E402

SOURCE = "travfakta"

_STAGING = "tf_comment_import_raw"

_CREATE_STAGING_SQL = f"""
DROP TABLE IF EXISTS {_STAGING};
CREATE UNLOGGED TABLE {_STAGING} (
  horse_id           integer,
  horse_name         text,
  atg_id             text,
  start_id           bigint,
  race_id            bigint,
  race_date          date,
  track_name         text,
  race_number        integer,
  distance           integer,
  start_method       text,
  start_number       integer,
  start_position     integer,
  finishing_position integer,
  placement          integer,
  km_speed           numeric,
  gallop             integer,
  disqualified       integer,
  comment            text
);
"""

def _load_csv(conn, csv_path: Path, log=print) -> int:
    with conn.cursor() as cur:
        cur.execute(comment_match.NORM_FN_SQL)
        cur.execute(_CREATE_STAGING_SQL)
        with csv_path.open("r", encoding="utf-8") as fh:
            cur.copy_expert(
                f"COPY {_STAGING} FROM STDIN WITH (FORMAT csv, HEADER true)", fh)
        cur.execute(f"SELECT COUNT(*) FROM {_STAGING}")
        n = cur.fetchone()[0]
    conn.commit()
    log(f"[load] staged {n:,} rows from {csv_path.name}")
    return n


def _build_matches(conn, log=print) -> dict:
    stats = comment_match.build_matches(conn, _STAGING, work_mem="512MB")
    commented, total = stats["rows_with_comment"], stats["matched"]
    log(f"[match] {total:,}/{commented:,} commented rows matched to an entry "
        f"({100.0*total/max(commented,1):.2f}%)")
    log(f"[match]   via atg_id: {stats['matched_by_atg_id']:,}, "
        f"via race context: {stats['matched_by_race_context']:,}")
    log(f"[match]   fully corroborated (3/3): {stats['fully_corroborated']:,}")
    log(f"[match]   distinct entries: {stats['distinct_entries']:,} "
        f"(collisions: {stats['collisions']:,})")
    log(f"[match]   unmatched: {stats['unmatched']:,}")
    return stats


def _write_report(conn, path: Path, log=print) -> None:
    """Dump every commented CSV row that did not reach an entry, so the
    residue can be eyeballed instead of silently disappearing."""
    with conn.cursor() as cur, path.open("w", encoding="utf-8") as fh:
        cur.copy_expert(
            f"""COPY (
                  SELECT c.horse_name, c.atg_id, c.race_date, c.track_name,
                         c.race_number, c.start_number, c.comment,
                         CASE WHEN c.atg_id !~ '^[0-9]+$' OR c.atg_id = '0'
                              THEN 'no atg_id in export'
                              WHEN NOT EXISTS (SELECT 1 FROM horse h
                                               WHERE h.st_id = c.atg_id::int)
                              THEN 'horse not in database'
                              ELSE 'no entry on that date' END AS reason
                    FROM {_STAGING} c
                    LEFT JOIN {comment_match.MATCH_TABLE} m
                           ON m.src_ctid = c.ctid
                   WHERE c.comment IS NOT NULL AND c.comment <> ''
                     AND m.src_ctid IS NULL
                ) TO STDOUT WITH (FORMAT csv, HEADER true)""", fh)
    log(f"[report] unmatched rows written to {path}")


def run_import(conn, csv_path: Path, *, execute: bool, min_score: int,
               report: Path | None = None, log=print) -> dict:
    _load_csv(conn, csv_path, log=log)
    stats = _build_matches(conn, log=log)
    if report:
        _write_report(conn, report, log=log)

    eligible = comment_match.eligible_count(conn, min_score)
    stats["eligible"] = eligible
    stats["below_min_score"] = stats["matched"] - eligible
    log(f"[import] eligible at score >= {min_score}: {eligible:,} "
        f"(withheld: {stats['below_min_score']:,})")

    if not execute:
        log("[import] DRY RUN — nothing written. Re-run with --execute.")
        conn.rollback()
        return stats

    stats["written"] = comment_match.upsert(conn, source=SOURCE,
                                            min_score=min_score)
    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM entry_comment")
        stats["table_total"] = cur.fetchone()[0]
    conn.commit()
    log(f"[import] wrote/updated {stats['written']:,} rows; "
        f"entry_comment now holds {stats['table_total']:,}")
    return stats


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", required=True, type=Path,
                    help="path to the travfakta comment export")
    ap.add_argument("--execute", action="store_true",
                    help="write to entry_comment (default is a dry run)")
    ap.add_argument("--min-score", type=int, default=3, choices=(1, 2, 3),
                    help="how many of race-number/track/start-number must "
                         "agree before a row is imported (default 3 = all)")
    ap.add_argument("--report", type=Path, default=None,
                    help="write unmatched commented rows to this CSV")
    ap.add_argument("--keep-staging", action="store_true",
                    help="leave the staging tables behind for inspection")
    args = ap.parse_args()

    if not args.csv.exists():
        print(f"csv not found: {args.csv}", file=sys.stderr)
        return 1

    conn = get_connection()
    try:
        run_import(conn, args.csv, execute=args.execute,
                   min_score=args.min_score, report=args.report)
        if not args.keep_staging:
            with conn.cursor() as cur:
                cur.execute(f"DROP TABLE IF EXISTS {_STAGING}")
            comment_match.drop_match_table(conn)
            conn.commit()
    finally:
        conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
