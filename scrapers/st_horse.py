"""Native Svensk Travsport (ST) horse-passport scraper.

Fetches a horse's datasets from the api.travsport.se JSON API and persists them
durably to `st_horse_raw` (+ `st_horse_scrape_log` and the 7-day `st_buffer`).
ETL of the raw blobs into the `horse` master table lives in `etl.import_st`
(`upsert_horse_from_st` / `load_st_horses_from_raw`), so this module only does
fetch + raw persistence.

This is the v2-native replacement for v1's
/Users/jakob/Dev/stable/scrapers/horse_scraper.py. NOTE: unlike v1 (which parsed
SSR-embedded React Query blobs from the `/basic` HTML page), the public pages no
longer embed those datasets — only `horse-basic-information`. The React app now
pulls the rest from the JSON endpoints in `config.ST_HORSE_API_ENDPOINTS`, which
is what we call here. The dataset keys are kept identical to v1's data_type
names so the ETL can reuse the v1 field map.
"""

from __future__ import annotations

import json
import logging
import signal
import time

import httpx
from psycopg2.extras import Json

from core import config
from core.db import buffer_insert

log = logging.getLogger(__name__)

# The dataset that decides whether a horse exists at all. If basic info 404s we
# treat the horse as missing and skip the other endpoints.
_PRIMARY = "horse-basic-information"


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------

def make_client() -> httpx.Client:
    limits = httpx.Limits(max_keepalive_connections=0)
    return httpx.Client(
        headers=config.ST_RACE_HEADERS,  # Accept: application/json
        follow_redirects=True,
        timeout=httpx.Timeout(connect=15.0, read=30.0, write=15.0, pool=10.0),
        limits=limits,
    )


class _HardTimeout(Exception):
    """Raised by SIGALRM when an HTTP request exceeds the hard deadline."""


_HARD_TIMEOUT_S = 45  # must exceed httpx read timeout (30s) + SSL overhead


def _get_json(client: httpx.Client, url: str, *, retries: int | None = None
              ) -> tuple[int, object | None]:
    """GET `url` expecting JSON. Returns (http_status, parsed_json_or_None).

    http_status is 0 when the request never completed. Mirrors
    scrapers/letrot._get's SIGALRM hard-timeout guard for macOS/Python where the
    SSL layer can ignore the socket timeout, plus 429/5xx backoff.
    """
    if retries is None:
        retries = config.MAX_RETRIES

    def _alarm_handler(signum, frame):
        raise _HardTimeout(url)

    for attempt in range(retries + 1):
        t0 = time.monotonic()
        prev_handler = signal.signal(signal.SIGALRM, _alarm_handler)
        signal.alarm(_HARD_TIMEOUT_S)
        try:
            r = client.get(url)
        except _HardTimeout:
            log.warning("st %s -> hard timeout after %.1fs (retry %d/%d)",
                        url, time.monotonic() - t0, attempt + 1, retries)
            if attempt < retries:
                time.sleep(config.RETRY_BACKOFF ** attempt)
                continue
            return 0, None
        except (httpx.TimeoutException, httpx.ConnectError) as exc:
            signal.alarm(0)
            if attempt < retries:
                time.sleep(config.RETRY_BACKOFF ** attempt)
                continue
            log.warning("st %s -> %r (exhausted retries)", url, exc)
            return 0, None
        except Exception as exc:  # noqa: BLE001
            signal.alarm(0)
            log.warning("st %s -> exception %r", url, exc)
            return 0, None
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, prev_handler)

        status = r.status_code
        if status == 429 and attempt < retries:
            time.sleep(config.RETRY_BACKOFF ** (attempt + 2))
            continue
        if status >= 500 and attempt < retries:
            time.sleep(config.RETRY_BACKOFF ** (attempt + 1))
            continue
        if status != 200:
            return status, None
        try:
            return 200, json.loads(r.text)
        except json.JSONDecodeError:
            return 200, None
    return 0, None


# ---------------------------------------------------------------------------
# Fetch + persist one horse
# ---------------------------------------------------------------------------

def fetch_horse_datasets(client: httpx.Client, horse_id: int
                         ) -> tuple[int, dict[str, object]]:
    """Fetch all JSON datasets for a horse.

    Returns (primary_status, {data_type: payload}) where primary_status is the
    HTTP status of the basic-information endpoint (the existence signal). The
    secondary endpoints are only fetched when basic info is 200.
    """
    base = config.ST_API_BASE
    eps = config.ST_HORSE_API_ENDPOINTS
    primary_url = f"{base}/{eps[_PRIMARY].format(horse_id=horse_id)}"
    primary_status, primary_json = _get_json(client, primary_url)
    blobs: dict[str, object] = {}
    if primary_status == 200 and primary_json is not None:
        blobs[_PRIMARY] = primary_json
        for data_type, tmpl in eps.items():
            if data_type == _PRIMARY:
                continue
            status, payload = _get_json(client, f"{base}/{tmpl.format(horse_id=horse_id)}")
            if status == 200 and payload is not None:
                blobs[data_type] = payload
    return primary_status, blobs


def store_horse_raw(conn, horse_id: int, primary_status: int,
                    blobs: dict[str, object]) -> tuple[int, bool]:
    """Persist fetched datasets. Returns (data_type_rows_written, ok)."""
    ok = primary_status == 200 and _PRIMARY in blobs
    error_message = None if ok else (
        f"http_{primary_status}" if primary_status != 200 else "no_basic_info"
    )
    rows_written = 0
    with conn.cursor() as cur:
        for data_type, payload in blobs.items():
            cur.execute(
                """
                INSERT INTO st_horse_raw (horse_id, data_type, raw_json, scraped_at)
                VALUES (%s, %s, %s, NOW())
                ON CONFLICT (horse_id, data_type)
                DO UPDATE SET raw_json = EXCLUDED.raw_json,
                              scraped_at = EXCLUDED.scraped_at
                """,
                (horse_id, data_type, Json(payload)),
            )
            rows_written += 1
        cur.execute(
            """
            INSERT INTO st_horse_scrape_log (horse_id, http_status, scraped_at, error_message)
            VALUES (%s, %s, NOW(), %s)
            ON CONFLICT (horse_id)
            DO UPDATE SET http_status = EXCLUDED.http_status,
                          scraped_at = EXCLUDED.scraped_at,
                          error_message = EXCLUDED.error_message
            """,
            (horse_id, primary_status, error_message),
        )
    conn.commit()
    buffer_insert(conn, "st",
                  config.ST_HORSE_URL.format(horse_id=horse_id),
                  primary_status, None, ok)
    return rows_written, ok


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def scrape_horse_ids(conn, horse_ids, *, skip_done: bool = False,
                     delay: float | None = None,
                     client: httpx.Client | None = None) -> dict:
    """Fetch + persist raw for each horse id. Returns counts.

    `skip_done=True` skips ids that already have an HTTP 200 in
    `st_horse_scrape_log` (resumable bulk runs). The gap-fill / refresh paths
    pass `skip_done=False` because the point is to RE-fetch ids that may have
    gained data since they were last probed.
    """
    if delay is None:
        delay = config.REQUEST_DELAY
    ids = list(dict.fromkeys(int(h) for h in horse_ids))

    if skip_done and ids:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT horse_id FROM st_horse_scrape_log "
                "WHERE horse_id = ANY(%s) AND http_status = 200",
                (ids,),
            )
            done = {r[0] for r in cur.fetchall()}
        ids = [h for h in ids if h not in done]

    own_client = client is None
    client = client or make_client()
    counts = {"requested": len(ids), "ok": 0, "missing": 0, "failed": 0,
              "raw_rows": 0}
    try:
        for hid in ids:
            status, blobs = fetch_horse_datasets(client, hid)
            rows, ok = store_horse_raw(conn, hid, status, blobs)
            counts["raw_rows"] += rows
            if ok:
                counts["ok"] += 1
            elif status == 404:
                counts["missing"] += 1
            else:
                counts["failed"] += 1
            if delay:
                time.sleep(delay)
    finally:
        if own_client:
            client.close()
    return counts


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    import sys
    from core.db import get_connection

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    if len(sys.argv) < 2:
        print("usage: python -m scrapers.st_horse <horse_id> [horse_id ...]")
        raise SystemExit(1)
    ids = [int(a) for a in sys.argv[1:]]
    conn = get_connection()
    try:
        print(scrape_horse_ids(conn, ids, skip_done=False))
    finally:
        conn.close()


if __name__ == "__main__":
    main()
