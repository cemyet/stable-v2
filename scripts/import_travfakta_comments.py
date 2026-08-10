"""
Import the travmedia/travfakta bulk race-comment export into `entry_comment`.

The export is a flat CSV, one row per historic start:

    horse_id,horse_name,atg_id,start_id,race_id,race_date,track_name,
    race_number,distance,start_method,start_number,start_position,
    finishing_position,placement,km_speed,gallop,disqualified,comment

Matching protocol
-----------------
The export's own `horse_id` and `race_id` are travfakta-internal surrogate
keys. They fall in the same numeric range as our `horse.st_id` /
`race.st_race_id` and so join happily — and produce *zero* name agreement.
Joining on them is the single most dangerous thing one could do here, so we
never touch them except to group rows by horse.

The column that does carry meaning is `atg_id`: for Swedish-registered horses
ATG's id and TravSport's id are the same number, which we store as
`horse.st_id`. Resolving through it agrees with our horse names on 99.3% of
63,481 horses, and the residual disagreements are all cosmetic (our names
carry a `*` breeding mark, a `(DK)`/`(NO)` country suffix, or Nordic
transliteration: Spøk/Spök, Kalypso Cæsar/Kalypso Caesar).

With the horse known, a row is matched to an entry by (horse, race_date) and
then corroborated against three fields the join did not use — race number,
track name and start number. On the full export 1,606,936 of 1,613,377 rows
had exactly one candidate entry and the remaining 6,441 were resolved to a
single best candidate by that corroboration, giving a 1:1 row→entry mapping
with no entry claimed twice.

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

SOURCE = "travfakta"

_STAGING = "tf_comment_import_raw"

# Our horse names carry decoration the export does not: a trailing `*`
# (Swedish-bred), a `(DK)`/`(NO)`/`(US)` country suffix, and case/spacing
# differences ("A Kind of Magic" vs "A Kind Of Magic"). Comparing on a
# stripped, alphanumeric-only form makes the two sides comparable.
_NORM_FN_SQL = """
CREATE OR REPLACE FUNCTION tf_norm_name(t text) RETURNS text
LANGUAGE sql IMMUTABLE AS $$
  SELECT regexp_replace(
           regexp_replace(
             regexp_replace(lower(coalesce(t,'')), '\\s*\\([a-z]{2,3}\\)\\s*$', '', 'g'),
             '\\*', '', 'g'),
           '[^[:alnum:]]', '', 'g');
$$;
"""

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

# One row per (csv row, candidate entry). The join is (resolved horse,
# race_date); `score` counts how many of the three withheld fields agree.
_MATCH_SQL = f"""
DROP TABLE IF EXISTS tf_comment_match;
CREATE UNLOGGED TABLE tf_comment_match AS
WITH horse_map AS (
    SELECT c.atg_id, h.horse_id
      FROM (SELECT DISTINCT atg_id FROM {_STAGING}
             WHERE atg_id ~ '^[0-9]+$' AND atg_id <> '0') c
      JOIN horse h ON h.st_id = c.atg_id::int
), cand AS (
    SELECT c.ctid AS csv_ctid, c.start_id, c.comment, c.race_date,
           e.entry_id,
           (c.race_number = r.race_number)::int
         + (tf_norm_name(c.track_name) = tf_norm_name(t.name))::int
         + (c.start_number IS NOT DISTINCT FROM e.program_number)::int AS score
      FROM {_STAGING} c
      JOIN horse_map m ON m.atg_id = c.atg_id
      JOIN entry e     ON e.horse_id = m.horse_id
      JOIN race  r     ON r.race_id = e.race_id AND r.race_date = c.race_date
      LEFT JOIN track t ON t.track_id = r.track_id
     WHERE c.comment IS NOT NULL AND c.comment <> ''
), ranked AS (
    SELECT *, ROW_NUMBER() OVER (PARTITION BY csv_ctid
                                 ORDER BY score DESC, entry_id) AS rn
      FROM cand
)
SELECT csv_ctid, start_id, comment, race_date, entry_id, score, 'atg_id'::text AS method
  FROM ranked WHERE rn = 1;
CREATE INDEX ON tf_comment_match (entry_id);
CREATE INDEX ON tf_comment_match (csv_ctid);
"""

# Second pass for the ~1% of the export that ships no atg_id at all (the
# column is 0). Those rows cannot be resolved by id, so they are placed by
# race context instead: normalised horse name plus date, track, race number
# and start number. Five agreeing fields is a tight enough key that on the
# full export it produced 16,349 matches with no ambiguity whatsoever — every
# row landed on its own distinct entry — and the two fields held back from the
# key, individual distance and km-time, agreed on 99.98% and 99.95% of them.
#
# Only rows pass 1 could not place are considered, so this can never override
# an id-based match.
_MATCH_BY_CONTEXT_SQL = f"""
INSERT INTO tf_comment_match (csv_ctid, start_id, comment, race_date, entry_id, score, method)
WITH miss AS (
    SELECT c.ctid AS csv_ctid, c.start_id, c.comment, c.race_date, c.horse_name,
           c.track_name, c.race_number, c.start_number
      FROM {_STAGING} c
      LEFT JOIN tf_comment_match m ON m.csv_ctid = c.ctid
     WHERE c.comment IS NOT NULL AND c.comment <> '' AND m.csv_ctid IS NULL
), cand AS (
    SELECT ms.csv_ctid, ms.start_id, ms.comment, ms.race_date, e.entry_id,
           COUNT(*) OVER (PARTITION BY ms.csv_ctid) AS n_cand
      FROM miss ms
      JOIN race  r ON r.race_date = ms.race_date AND r.race_number = ms.race_number
      JOIN track t ON t.track_id = r.track_id
                  AND tf_norm_name(t.name) = tf_norm_name(ms.track_name)
      JOIN entry e ON e.race_id = r.race_id AND e.program_number = ms.start_number
      JOIN horse h ON h.horse_id = e.horse_id
                  AND tf_norm_name(h.name) = tf_norm_name(ms.horse_name)
)
SELECT csv_ctid, start_id, comment, race_date, entry_id, 3, 'race_context'
  FROM cand WHERE n_cand = 1
"""

# An entry can only carry one comment, so a defensive dedup guards against a
# future export shipping the same start twice; keep the best-corroborated row.
_UPSERT_SQL = """
INSERT INTO entry_comment (entry_id, comment, source, source_ref, race_date, updated_at)
SELECT DISTINCT ON (entry_id)
       entry_id, comment, %(source)s, start_id::text, race_date, NOW()
  FROM tf_comment_match
 WHERE score >= %(min_score)s
 ORDER BY entry_id, score DESC
ON CONFLICT (entry_id) DO UPDATE SET
    comment    = EXCLUDED.comment,
    source     = EXCLUDED.source,
    source_ref = EXCLUDED.source_ref,
    race_date  = EXCLUDED.race_date,
    updated_at = NOW()
WHERE entry_comment.comment IS DISTINCT FROM EXCLUDED.comment
"""


def _load_csv(conn, csv_path: Path, log=print) -> int:
    with conn.cursor() as cur:
        cur.execute(_NORM_FN_SQL)
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
    with conn.cursor() as cur:
        cur.execute("SET work_mem = '512MB'")
        cur.execute(_MATCH_SQL)
        cur.execute(_MATCH_BY_CONTEXT_SQL)
        by_context = cur.rowcount
        cur.execute("""
            SELECT COUNT(*), COUNT(*) FILTER (WHERE score = 3),
                   COUNT(DISTINCT entry_id)
              FROM tf_comment_match
        """)
        total, full, distinct_entries = cur.fetchone()
        cur.execute(f"""
            SELECT COUNT(*) FROM {_STAGING}
             WHERE comment IS NOT NULL AND comment <> ''
        """)
        commented = cur.fetchone()[0]
    conn.commit()
    stats = {
        "csv_rows_with_comment": commented,
        "matched": total,
        "matched_by_atg_id": total - by_context,
        "matched_by_race_context": by_context,
        "fully_corroborated": full,
        "distinct_entries": distinct_entries,
        "unmatched": commented - total,
    }
    log(f"[match] {total:,}/{commented:,} commented rows matched to an entry "
        f"({100.0*total/max(commented,1):.2f}%)")
    log(f"[match]   via atg_id: {total - by_context:,}, "
        f"via race context: {by_context:,}")
    log(f"[match]   fully corroborated (3/3): {full:,}")
    log(f"[match]   distinct entries: {distinct_entries:,} "
        f"(collisions: {total - distinct_entries:,})")
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
                    LEFT JOIN tf_comment_match m ON m.csv_ctid = c.ctid
                   WHERE c.comment IS NOT NULL AND c.comment <> ''
                     AND m.csv_ctid IS NULL
                ) TO STDOUT WITH (FORMAT csv, HEADER true)""", fh)
    log(f"[report] unmatched rows written to {path}")


def run_import(conn, csv_path: Path, *, execute: bool, min_score: int,
               report: Path | None = None, log=print) -> dict:
    _load_csv(conn, csv_path, log=log)
    stats = _build_matches(conn, log=log)
    if report:
        _write_report(conn, report, log=log)

    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM tf_comment_match WHERE score >= %s",
                    (min_score,))
        eligible = cur.fetchone()[0]
    stats["eligible"] = eligible
    stats["below_min_score"] = stats["matched"] - eligible
    log(f"[import] eligible at score >= {min_score}: {eligible:,} "
        f"(withheld: {stats['below_min_score']:,})")

    if not execute:
        log("[import] DRY RUN — nothing written. Re-run with --execute.")
        conn.rollback()
        return stats

    with conn.cursor() as cur:
        cur.execute(_UPSERT_SQL, {"source": SOURCE, "min_score": min_score})
        stats["written"] = cur.rowcount
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
                cur.execute("DROP TABLE IF EXISTS tf_comment_match")
            conn.commit()
    finally:
        conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
