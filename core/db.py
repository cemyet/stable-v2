"""Tiny database helper for stable-v2.

v2 has no `_raw` save/load helpers — scrapers parse in-memory and call
upsert helpers in etl.import_<source> directly. The only persistence helpers
that live here are `get_connection()` and `buffer_insert()` for the rolling
per-source HTTP buffer used for retry/debug.
"""

from __future__ import annotations

import psycopg2
from psycopg2.extras import Json

from .config import (
    DATABASE_URL,
    BUFFER_RETENTION_DAYS,
    V1_DATABASE_HOST,
    V1_DATABASE_NAME,
    V1_DATABASE_PORT,
    V1_DATABASE_USER,
)


def get_connection():
    """Open a new connection to stable_v2."""
    return psycopg2.connect(DATABASE_URL)


def get_v1_connection():
    """Open a new READ-ONLY connection to v1's `jakob` database.

    Used by etl.import_<source>.backfill_from_v1() to read v1 rows
    directly. We chose this over postgres_fdw because Postgres.app
    blocks server-internal trust auth on macOS.
    """
    conn = psycopg2.connect(
        host=V1_DATABASE_HOST,
        port=V1_DATABASE_PORT,
        dbname=V1_DATABASE_NAME,
        user=V1_DATABASE_USER,
    )
    conn.set_session(readonly=True)
    return conn


def buffer_insert(
    conn,
    source: str,
    url: str,
    http_status: int,
    body: str | None,
    parsed_ok: bool,
) -> None:
    """Append a fetch attempt to the per-source rolling buffer.

    The buffer table is named `<source>_buffer` and follows the schema
    declared in core.schema. We keep `BUFFER_RETENTION_DAYS` days of rows
    for retry/debug; older rows are pruned by `buffer_prune` (called at
    the tail end of each scraper job).
    """
    table = f"{source}_buffer"
    with conn.cursor() as cur:
        cur.execute(
            f"INSERT INTO {table} (url, http_status, body, parsed_ok) "
            f"VALUES (%s, %s, %s, %s)",
            (url, http_status, body, parsed_ok),
        )
    conn.commit()


def buffer_prune(conn, source: str, days: int = BUFFER_RETENTION_DAYS) -> int:
    """Delete buffer rows older than `days`. Returns deleted row count."""
    table = f"{source}_buffer"
    with conn.cursor() as cur:
        cur.execute(
            f"DELETE FROM {table} WHERE fetched_at < NOW() - INTERVAL '{int(days)} days'"
        )
        n = cur.rowcount
    conn.commit()
    return n


def jsonb(payload) -> Json:
    """Wrap a Python dict/list for psycopg2 jsonb binding."""
    return Json(payload)
