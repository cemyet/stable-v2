"""Resolve TR Media race-comment rows to our `entry` rows.

Shared by the one-shot historical import (`scripts.import_travfakta_comments`,
a 1.6M-row CSV) and the nightly feed (`etl.import_tr_comments`, a few hundred
rows a day). Both arrive in the same shape because they are the same database
upstream, so they get the same matching protocol and the same guarantees.

A caller stages its rows into a table carrying at least these columns:

    horse_name, atg_id, start_id, race_date, track_name, race_number,
    start_number, comment

and then calls `build_matches(conn, staging)` followed by `upsert(...)`.

Matching protocol
-----------------
TR Media's own `horse_id` and `race_id` are internal surrogate keys. They fall
in the same numeric range as our `horse.st_id` / `race.st_race_id` and so join
happily while meaning something entirely different — joining on them is the
single most dangerous thing one could do here, so we never touch them.

The column that does carry meaning is `atg_id`: for Swedish-registered horses
ATG's id and TravSport's id are the same number, which we store as
`horse.st_id`. Resolving through it agreed with our horse names on 99.3% of
63,481 horses in the historical export, and the residual disagreements were
all cosmetic (a `*` breeding mark, a `(DK)`/`(NO)` country suffix, or Nordic
transliteration: Spøk/Spök, Kalypso Cæsar/Kalypso Caesar).

With the horse known, a row is matched to an entry by (horse, race_date) and
then corroborated against three fields the join did not use — race number,
track name and start number. `score` counts how many of the three agree, and
callers refuse anything below their threshold rather than guess.

A second pass places rows carrying no `atg_id` at all by race context instead:
normalised horse name plus date, track, race number and start number. Five
agreeing fields is tight enough that on the historical export it produced
16,349 matches with no ambiguity whatsoever — every row landed on its own
distinct entry. It only ever considers rows the first pass could not place, so
it can never override an id-based match.
"""

from __future__ import annotations

MATCH_TABLE = "comment_match"

# Our horse names carry decoration TR Media's do not: a trailing `*`
# (Swedish-bred), a `(DK)`/`(NO)`/`(US)` country suffix, and case/spacing
# differences ("A Kind of Magic" vs "A Kind Of Magic"). Comparing on a
# stripped, alphanumeric-only form makes the two sides comparable.
NORM_FN_SQL = """
CREATE OR REPLACE FUNCTION tf_norm_name(t text) RETURNS text
LANGUAGE sql IMMUTABLE AS $$
  SELECT regexp_replace(
           regexp_replace(
             regexp_replace(lower(coalesce(t,'')), '\\s*\\([a-z]{2,3}\\)\\s*$', '', 'g'),
             '\\*', '', 'g'),
           '[^[:alnum:]]', '', 'g');
$$;
"""

# One row per (staged row, candidate entry). The join is (resolved horse,
# race_date); `score` counts how many of the three withheld fields agree.
_MATCH_SQL = """
DROP TABLE IF EXISTS {match};
CREATE UNLOGGED TABLE {match} AS
WITH horse_map AS (
    SELECT c.atg_id, h.horse_id
      FROM (SELECT DISTINCT atg_id FROM {staging}
             WHERE atg_id ~ '^[0-9]+$' AND atg_id <> '0') c
      JOIN horse h ON h.st_id = c.atg_id::int
), cand AS (
    SELECT c.ctid AS src_ctid, c.start_id, c.comment, c.race_date,
           e.entry_id,
           (c.race_number = r.race_number)::int
         + (tf_norm_name(c.track_name) = tf_norm_name(t.name))::int
         + (c.start_number IS NOT DISTINCT FROM e.program_number)::int AS score
      FROM {staging} c
      JOIN horse_map m ON m.atg_id = c.atg_id
      JOIN entry e     ON e.horse_id = m.horse_id
      JOIN race  r     ON r.race_id = e.race_id AND r.race_date = c.race_date
      LEFT JOIN track t ON t.track_id = r.track_id
     WHERE c.comment IS NOT NULL AND c.comment <> ''
), ranked AS (
    SELECT *, ROW_NUMBER() OVER (PARTITION BY src_ctid
                                 ORDER BY score DESC, entry_id) AS rn
      FROM cand
)
SELECT src_ctid, start_id, comment, race_date, entry_id, score,
       'atg_id'::text AS method
  FROM ranked WHERE rn = 1;
CREATE INDEX ON {match} (entry_id);
CREATE INDEX ON {match} (src_ctid);
"""

_MATCH_BY_CONTEXT_SQL = """
INSERT INTO {match} (src_ctid, start_id, comment, race_date, entry_id, score, method)
WITH miss AS (
    SELECT c.ctid AS src_ctid, c.start_id, c.comment, c.race_date, c.horse_name,
           c.track_name, c.race_number, c.start_number
      FROM {staging} c
      LEFT JOIN {match} m ON m.src_ctid = c.ctid
     WHERE c.comment IS NOT NULL AND c.comment <> '' AND m.src_ctid IS NULL
), cand AS (
    SELECT ms.src_ctid, ms.start_id, ms.comment, ms.race_date, e.entry_id,
           COUNT(*) OVER (PARTITION BY ms.src_ctid) AS n_cand
      FROM miss ms
      JOIN race  r ON r.race_date = ms.race_date AND r.race_number = ms.race_number
      JOIN track t ON t.track_id = r.track_id
                  AND tf_norm_name(t.name) = tf_norm_name(ms.track_name)
      JOIN entry e ON e.race_id = r.race_id AND e.program_number = ms.start_number
      JOIN horse h ON h.horse_id = e.horse_id
                  AND tf_norm_name(h.name) = tf_norm_name(ms.horse_name)
)
SELECT src_ctid, start_id, comment, race_date, entry_id, 3, 'race_context'
  FROM cand WHERE n_cand = 1
"""

# An entry can only carry one comment, so a defensive dedup guards against a
# feed shipping the same start twice; keep the best-corroborated row. The
# trailing WHERE makes a re-run of an unchanged window write nothing at all,
# which is what keeps the nightly cheap and leaves `updated_at` meaningful as
# a publish watermark.
_UPSERT_SQL = """
INSERT INTO entry_comment (entry_id, comment, source, source_ref, race_date, updated_at)
SELECT DISTINCT ON (entry_id)
       entry_id, comment, %(source)s, start_id::text, race_date, NOW()
  FROM {match}
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


def ensure_norm_fn(conn) -> None:
    with conn.cursor() as cur:
        cur.execute(NORM_FN_SQL)
    conn.commit()


def build_matches(conn, staging: str, *, work_mem: str | None = None) -> dict:
    """Resolve every commented row in `staging` to at most one entry.

    Leaves the result in `MATCH_TABLE`. Returns counts describing how well the
    staged rows placed, so a caller can refuse to write on a bad batch."""
    fmt = {"staging": staging, "match": MATCH_TABLE}
    with conn.cursor() as cur:
        cur.execute(NORM_FN_SQL)
        if work_mem:
            cur.execute(f"SET work_mem = '{work_mem}'")
        cur.execute(_MATCH_SQL.format(**fmt))
        cur.execute(_MATCH_BY_CONTEXT_SQL.format(**fmt))
        by_context = cur.rowcount
        cur.execute(f"""
            SELECT COUNT(*), COUNT(*) FILTER (WHERE score = 3),
                   COUNT(DISTINCT entry_id)
              FROM {MATCH_TABLE}
        """)
        total, full, distinct_entries = cur.fetchone()
        cur.execute(f"""
            SELECT COUNT(*) FROM {staging}
             WHERE comment IS NOT NULL AND comment <> ''
        """)
        commented = cur.fetchone()[0]
    conn.commit()
    return {
        "rows_with_comment": commented,
        "matched": total,
        "matched_by_atg_id": total - by_context,
        "matched_by_race_context": by_context,
        "fully_corroborated": full,
        "distinct_entries": distinct_entries,
        "collisions": total - distinct_entries,
        "unmatched": commented - total,
    }


def upsert(conn, *, source: str, min_score: int = 3) -> int:
    """Write matches scoring at least `min_score` into `entry_comment`.

    Returns the number of rows actually inserted or changed."""
    with conn.cursor() as cur:
        cur.execute(_UPSERT_SQL.format(match=MATCH_TABLE),
                    {"source": source, "min_score": min_score})
        written = cur.rowcount
    conn.commit()
    return written


def eligible_count(conn, min_score: int) -> int:
    with conn.cursor() as cur:
        cur.execute(f"SELECT COUNT(*) FROM {MATCH_TABLE} WHERE score >= %s",
                    (min_score,))
        return cur.fetchone()[0]


def drop_match_table(conn) -> None:
    with conn.cursor() as cur:
        cur.execute(f"DROP TABLE IF EXISTS {MATCH_TABLE}")
    conn.commit()
