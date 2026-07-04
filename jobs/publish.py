"""
Publish the local v2 "serving set" to Supabase (local → cloud sync).

Architecture: LOCAL is the writer/ML lab (scrapes, derives, trains models);
CLOUD (Supabase) is the read replica the Railway web app serves. This job runs
after the nightly update and incrementally upserts everything the cloud app
needs — master tables, ML feature/derived tables, and ML outputs — but NOT the
raw archives (atg_race_raw, st_*_raw, *_buffer) which stay local-only.

Sync strategy per table:
  * watermark tables: pull rows whose watermark column advanced since the last
    publish and upsert them (idempotent). Watermarks are stored in `publish_state`.
  * history tables (no per-row timestamp): pull rows for horses whose
    `last_updated_at` advanced.
  * materialized views: REFRESH on the cloud after base tables are synced.

Usage:
    # One-time, right after the initial full restore (Supabase already has the
    # data): set watermarks to the current local max WITHOUT copying anything.
    python3 -m jobs.publish --init

    # Nightly incremental (default):
    python3 -m jobs.publish

    # Force a full re-push of the serving set (ignores watermarks):
    python3 -m jobs.publish --full

    # Skip the (slower) materialized-view refresh on the cloud:
    python3 -m jobs.publish --no-matviews

Requires SUPABASE_DATABASE_URL in the environment (or .env).
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
from psycopg2.extras import execute_values, Json

from core import config


# FK-safe order. (table, watermark_column). watermark_column drives the
# incremental window; None means "sync via horse.last_updated_at" (history) or
# handled specially below.
SERVING_TABLES: list[tuple[str, str | None]] = [
    ("track",                 "last_updated_at"),
    ("person",                "last_updated_at"),
    ("horse",                 "last_updated_at"),
    ("race",                  "last_updated_at"),
    ("entry",                 "last_updated_at"),
    ("horse_owner_history",   None),   # via horse watermark
    ("horse_trainer_history", None),   # via horse watermark
    ("entry_features",        "race_date"),
    ("entry_outperf",         "race_date"),
    ("entry_perf",            "race_date"),
    ("trainer_form_exp",      "race_date"),
    ("identity_redirect",     "created_at"),
    ("watchlist",             "added_at"),
    ("ml_slice",              "created_at"),
    ("ml_model",              "created_at"),
    ("ml_prediction",         "updated_at"),
]

# Materialized views refreshed on the cloud after base tables land (order: the
# horse_*/person_* career views feed the browse-stat views).
MATVIEWS = [
    "horse_career_stats", "horse_year_stats", "person_career_stats",
    "track_stats", "horse_stats", "person_stats", "track_post_stats",
]

# Overlap re-pulled each run so boundary rows (equal timestamps / late same-day
# derived rows) are never missed. Upserts make the re-pull idempotent.
_TS_OVERLAP = "INTERVAL '1 hour'"
_DATE_OVERLAP = "INTERVAL '3 days'"

_BATCH = 5000


def _connect(url: str, *, readonly: bool = False):
    conn = psycopg2.connect(url)
    if readonly:
        conn.set_session(readonly=True)
    return conn


def _ensure_state(local) -> None:
    with local.cursor() as cur:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS publish_state (
                table_name  TEXT PRIMARY KEY,
                watermark   TEXT,
                updated_at  TIMESTAMP NOT NULL DEFAULT NOW()
            )
        """)
    local.commit()


def _get_watermark(local, table: str) -> str | None:
    with local.cursor() as cur:
        cur.execute("SELECT watermark FROM publish_state WHERE table_name=%s", (table,))
        row = cur.fetchone()
    return row[0] if row else None


def _set_watermark(local, table: str, value) -> None:
    with local.cursor() as cur:
        cur.execute("""
            INSERT INTO publish_state (table_name, watermark, updated_at)
            VALUES (%s, %s, NOW())
            ON CONFLICT (table_name)
            DO UPDATE SET watermark = EXCLUDED.watermark, updated_at = NOW()
        """, (table, str(value) if value is not None else None))
    local.commit()


def _columns(conn, table: str) -> list[str]:
    """Insertable columns, in ordinal order. Generated columns (attgenerated
    <> '') are excluded — they can't be written and are recomputed on the
    cloud from their base columns."""
    with conn.cursor() as cur:
        cur.execute("""
            SELECT a.attname
            FROM pg_attribute a
            JOIN pg_class c ON c.oid = a.attrelid
            JOIN pg_namespace n ON n.oid = c.relnamespace
            WHERE n.nspname='public' AND c.relname=%s
              AND a.attnum > 0 AND NOT a.attisdropped
              AND a.attgenerated = ''
            ORDER BY a.attnum
        """, (table,))
        return [r[0] for r in cur.fetchall()]


def _json_columns(conn, table: str) -> set[str]:
    """Column names of json/jsonb type — their values come back as Python
    dict/list and must be re-wrapped in Json() before re-inserting."""
    with conn.cursor() as cur:
        cur.execute("""
            SELECT column_name FROM information_schema.columns
            WHERE table_schema='public' AND table_name=%s
              AND data_type IN ('json', 'jsonb')
        """, (table,))
        return {r[0] for r in cur.fetchall()}


def _pk_columns(conn, table: str) -> list[str]:
    with conn.cursor() as cur:
        cur.execute("""
            SELECT a.attname
            FROM pg_index i
            JOIN pg_class c ON c.oid = i.indrelid
            JOIN pg_namespace n ON n.oid = c.relnamespace
            JOIN pg_attribute a ON a.attrelid = c.oid AND a.attnum = ANY(i.indkey)
            WHERE i.indisprimary AND n.nspname='public' AND c.relname=%s
            ORDER BY array_position(i.indkey, a.attnum)
        """, (table,))
        return [r[0] for r in cur.fetchall()]


def _upsert_sql(table: str, cols: list[str], pk: list[str]) -> str:
    col_list = ", ".join(cols)
    updates = [c for c in cols if c not in pk]
    set_clause = ", ".join(f"{c}=EXCLUDED.{c}" for c in updates)
    conflict = (f"ON CONFLICT ({', '.join(pk)}) DO UPDATE SET {set_clause}"
                if updates else f"ON CONFLICT ({', '.join(pk)}) DO NOTHING")
    return f"INSERT INTO {table} ({col_list}) VALUES %s {conflict}"


def _stream_upsert(local, cloud, table: str, where_sql: str, params: list,
                   cols: list[str], pk: list[str], *, log=print) -> int:
    """Stream rows matching where_sql from local and upsert into cloud."""
    sql = _upsert_sql(table, cols, pk)
    col_list = ", ".join(cols)
    json_cols = _json_columns(local, table)
    json_idx = [i for i, c in enumerate(cols) if c in json_cols]
    select = f"SELECT {col_list} FROM {table}"
    if where_sql:
        select += f" WHERE {where_sql}"

    def _prep(row):
        if not json_idx:
            return row
        r = list(row)
        for i in json_idx:
            if r[i] is not None:
                r[i] = Json(r[i])
        return r

    pushed = 0
    with local.cursor(name=f"pub_{table}") as src:
        src.itersize = _BATCH
        src.execute(select, params)
        with cloud.cursor() as dst:
            batch = []
            for row in src:
                batch.append(_prep(row))
                if len(batch) >= _BATCH:
                    execute_values(dst, sql, batch, page_size=1000)
                    cloud.commit()
                    pushed += len(batch)
                    batch = []
            if batch:
                execute_values(dst, sql, batch, page_size=1000)
                cloud.commit()
                pushed += len(batch)
    if pushed:
        log(f"  {table}: +{pushed:,} rows")
    return pushed


def _new_max(local, table: str, wm_col: str, where_sql: str, params: list):
    with local.cursor() as cur:
        q = f"SELECT MAX({wm_col}) FROM {table}"
        if where_sql:
            q += f" WHERE {where_sql}"
        cur.execute(q, params)
        return cur.fetchone()[0]


def sync_table(local_read, state, cloud, table: str, wm_col: str, *,
               full: bool, log=print) -> int:
    cols = _columns(local_read, table)
    pk = _pk_columns(local_read, table)
    if not pk:
        log(f"  {table}: no PK, skipping")
        return 0

    is_date = wm_col == "race_date"
    overlap = _DATE_OVERLAP if is_date else _TS_OVERLAP
    wm = None if full else _get_watermark(state, table)

    if wm is None:
        where_sql, params = "", []
    else:
        where_sql = f"{wm_col} >= %s::timestamptz - {overlap}" if not is_date \
            else f"{wm_col} >= %s::date - {overlap}"
        params = [wm]

    pushed = _stream_upsert(local_read, cloud, table, where_sql, params, cols, pk, log=log)
    new_max = _new_max(local_read, table, wm_col, where_sql, params)
    if new_max is not None:
        _set_watermark(state, table, new_max)
    return pushed


def sync_history(local_read, cloud, table: str, *, full: bool,
                 horse_wm: str | None, log=print) -> int:
    """History tables have no per-row timestamp; sync rows for horses whose
    last_updated_at advanced since the horse watermark (or all rows if full)."""
    cols = _columns(local_read, table)
    pk = _pk_columns(local_read, table)
    if full or horse_wm is None:
        where_sql, params = "", []
    else:
        where_sql = (f"horse_id IN (SELECT horse_id FROM horse "
                     f"WHERE last_updated_at >= %s::timestamptz - {_TS_OVERLAP})")
        params = [horse_wm]
    return _stream_upsert(local_read, cloud, table, where_sql, params, cols, pk, log=log)


def refresh_matviews(cloud, *, log=print) -> None:
    for mv in MATVIEWS:
        t0 = time.time()
        with cloud.cursor() as cur:
            try:
                cur.execute(f"REFRESH MATERIALIZED VIEW CONCURRENTLY {mv}")
                cloud.commit()
            except Exception:
                cloud.rollback()
                try:
                    cur.execute(f"REFRESH MATERIALIZED VIEW {mv}")
                    cloud.commit()
                except Exception as exc:
                    cloud.rollback()
                    log(f"  matview {mv}: FAILED {exc!r}")
                    continue
        log(f"  matview {mv}: refreshed in {time.time()-t0:.1f}s")


def init_watermarks(local, *, log=print) -> None:
    """Set each watermark table's watermark to its current local MAX without
    copying — used once after the initial full restore so the first incremental
    run only pushes genuinely new rows."""
    _ensure_state(local)
    for table, wm_col in SERVING_TABLES:
        if wm_col is None:
            continue
        with local.cursor() as cur:
            cur.execute(f"SELECT MAX({wm_col}) FROM {table}")
            mx = cur.fetchone()[0]
        _set_watermark(local, table, mx)
        log(f"  {table}: watermark = {mx}")
    # history tables ride the horse watermark, already set above.
    log("watermarks initialized.")


def run_publish(*, full: bool = False, do_matviews: bool = True, log=print) -> dict:
    if not config.SUPABASE_DATABASE_URL:
        raise SystemExit("SUPABASE_DATABASE_URL is not set (env or .env).")
    local_read = _connect(config.DATABASE_URL, readonly=True)   # streaming reads
    state = _connect(config.DATABASE_URL)                       # publish_state rw
    cloud = _connect(config.SUPABASE_DATABASE_URL)              # upsert target
    totals = {"tables": {}, "pushed": 0}
    t0 = time.time()
    try:
        _ensure_state(state)
        horse_wm = None if full else _get_watermark(state, "horse")
        for table, wm_col in SERVING_TABLES:
            try:
                if wm_col is None:
                    n = sync_history(local_read, cloud, table, full=full,
                                     horse_wm=horse_wm, log=log)
                else:
                    n = sync_table(local_read, state, cloud, table, wm_col,
                                   full=full, log=log)
                totals["tables"][table] = n
                totals["pushed"] += n
            except Exception as exc:
                cloud.rollback()
                log(f"  {table}: ERROR {exc!r}")
                totals["tables"][table] = f"error: {exc!r}"
        if do_matviews:
            log("refreshing materialized views on cloud...")
            refresh_matviews(cloud, log=log)
    finally:
        local_read.close()
        state.close()
        cloud.close()
    totals["seconds"] = round(time.time() - t0, 1)
    log(f"publish done in {totals['seconds']}s — pushed {totals['pushed']:,} rows")
    return totals


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--init", action="store_true",
                    help="set watermarks to current local max, copy nothing")
    ap.add_argument("--full", action="store_true",
                    help="ignore watermarks; push the entire serving set")
    ap.add_argument("--no-matviews", action="store_true",
                    help="skip refreshing materialized views on the cloud")
    args = ap.parse_args()

    if args.init:
        conn = _connect(config.DATABASE_URL)
        try:
            init_watermarks(conn)
        finally:
            conn.close()
        return 0

    run_publish(full=args.full, do_matviews=not args.no_matviews)
    return 0


if __name__ == "__main__":
    sys.exit(main())
