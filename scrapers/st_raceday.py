"""Native Svensk Travsport (ST) raceday scraper.

Fetches the results API for a raceday (`config.ST_RACE_API_BASE`) and stores the
raw JSON durably in `st_raceday_raw` (+ `st_raceday_scrape_log` and the 7-day
`st_buffer`). ETL into v2 race/entry lives in `etl.import_st`
(`upsert_raceday_from_st` / `load_st_racedays_from_raw`).

v2-native replacement for v1's raceday scraping. The shape returned is the same
one v1 stored in `v2_raceday_raw`, so the ETL mirrors v1 etl/racedays.py.
"""

from __future__ import annotations

import json
import logging
import time

import httpx
from psycopg2.extras import Json

from core import config
from core.db import buffer_insert
from scrapers.st_horse import _get_json, make_client  # reuse the hardened client

log = logging.getLogger(__name__)


def fetch_raceday(client: httpx.Client, race_day_id: int) -> tuple[int, object | None]:
    """Fetch one raceday results blob. Returns (http_status, json_or_None)."""
    url = config.ST_RACE_API_BASE.format(race_day_id=race_day_id)
    return _get_json(client, url)


def store_raceday_raw(conn, race_day_id: int, http_status: int,
                      payload: object | None) -> bool:
    """Persist a raceday fetch. Returns ok (200 + parseable JSON object)."""
    ok = http_status == 200 and isinstance(payload, dict)
    error_message = None if ok else (
        f"http_{http_status}" if http_status != 200 else "no_json"
    )
    with conn.cursor() as cur:
        if ok:
            cur.execute(
                """
                INSERT INTO st_raceday_raw (race_day_id, raw_json, scraped_at)
                VALUES (%s, %s, NOW())
                ON CONFLICT (race_day_id)
                DO UPDATE SET raw_json = EXCLUDED.raw_json,
                              scraped_at = EXCLUDED.scraped_at
                """,
                (race_day_id, Json(payload)),
            )
        cur.execute(
            """
            INSERT INTO st_raceday_scrape_log (race_day_id, http_status, scraped_at, error_message)
            VALUES (%s, %s, NOW(), %s)
            ON CONFLICT (race_day_id)
            DO UPDATE SET http_status = EXCLUDED.http_status,
                          scraped_at = EXCLUDED.scraped_at,
                          error_message = EXCLUDED.error_message
            """,
            (race_day_id, http_status, error_message),
        )
    conn.commit()
    buffer_insert(conn, "st",
                  config.ST_RACE_API_BASE.format(race_day_id=race_day_id),
                  http_status, None, ok)
    return ok


def scrape_raceday_ids(conn, race_day_ids, *, skip_done: bool = False,
                       delay: float | None = None,
                       client: httpx.Client | None = None) -> dict:
    """Fetch + persist raw for each raceday id. Returns counts."""
    if delay is None:
        delay = config.REQUEST_DELAY
    ids = list(dict.fromkeys(int(r) for r in race_day_ids))

    if skip_done and ids:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT race_day_id FROM st_raceday_scrape_log "
                "WHERE race_day_id = ANY(%s) AND http_status = 200",
                (ids,),
            )
            done = {r[0] for r in cur.fetchall()}
        ids = [r for r in ids if r not in done]

    own_client = client is None
    client = client or make_client()
    counts = {"requested": len(ids), "ok": 0, "missing": 0, "failed": 0}
    try:
        for rdid in ids:
            status, payload = fetch_raceday(client, rdid)
            if store_raceday_raw(conn, rdid, status, payload):
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


def seed_raceday_ids(conn, *, lookback_days: int = 120) -> list[int]:
    """Latest known st raceday id per RECENTLY-ACTIVE track — the entry points
    for a forward chain walk.

    Recency matters: a stale seed (a track whose latest linked raceday is years
    old) would make the forward walk drain that track's entire history. By
    restricting to tracks with an st-linked race in the last `lookback_days`,
    every chain starts near the present and walks only a handful of steps up to
    today. Subsequent runs re-seed from where the last walk left off.
    """
    with conn.cursor() as cur:
        cur.execute(
            "SELECT r.track_id, MAX(r.st_race_day_id) "
            "FROM race r "
            "WHERE r.st_race_day_id IS NOT NULL "
            "  AND r.race_date >= CURRENT_DATE - %s "
            "GROUP BY r.track_id",
            (lookback_days,),
        )
        ids = {r[1] for r in cur.fetchall() if r[1] is not None}
        # Also seed from the latest raceday we natively scraped per its stored
        # date window, so chains continue across runs even if `race` lags.
        cur.execute(
            "SELECT MAX(race_day_id) FROM st_raceday_scrape_log WHERE http_status = 200"
        )
        row = cur.fetchone()
    if row and row[0]:
        ids.add(row[0])
    return sorted(ids)


def _next_raceday_id(conn, client: httpx.Client, rdid: int) -> tuple[object, object | None]:
    """Return (next_raceday_id, blob) for `rdid`, reading from st_raceday_raw
    when we already have a 200 blob (saves a request), else fetching live and
    persisting it."""
    with conn.cursor() as cur:
        cur.execute("SELECT raw_json FROM st_raceday_raw WHERE race_day_id = %s", (rdid,))
        row = cur.fetchone()
    if row and isinstance(row[0], dict):
        return row[0].get("nextRaceDayId"), row[0]
    status, payload = fetch_raceday(client, rdid)
    store_raceday_raw(conn, rdid, status, payload)
    if isinstance(payload, dict):
        return payload.get("nextRaceDayId"), payload
    return None, None


def walk_forward_racedays(conn, seed_ids, *, max_steps_per_chain: int = 80,
                          delay: float | None = None,
                          client: httpx.Client | None = None,
                          log=print) -> dict:
    """From each seed raceday, follow `nextRaceDayId` forward, scraping every
    new raceday until the chain ends (None/404) or `max_steps_per_chain`.

    This is the v2-native replacement for v1's prev/next discovery, scoped to
    *recent* data: each chain advances chronologically at a single track, so
    starting from the latest known raceday per track yields exactly the
    racedays published since the last run. Foreign racedays carry null links,
    so their seeds no-op after one read. Returns counts + the scraped id set.
    """
    if delay is None:
        delay = config.REQUEST_DELAY
    own_client = client is None
    client = client or make_client()
    with conn.cursor() as cur:
        cur.execute("SELECT race_day_id FROM st_raceday_scrape_log WHERE http_status = 200")
        done: set[int] = {r[0] for r in cur.fetchall()}
    scraped: set[int] = set()
    visited: set[int] = set()
    counts = {"seeds": 0, "scraped": 0, "steps": 0, "fetched": 0, "chains_capped": 0}
    try:
        for seed in dict.fromkeys(int(s) for s in seed_ids):
            counts["seeds"] += 1
            current = seed
            steps = 0
            while current is not None and current not in visited:
                visited.add(current)
                # Traverse the chain via stored nextRaceDayId when we already
                # have the blob (free) and only fetch racedays we haven't
                # scraped — so steady-state runs make almost no HTTP calls.
                nxt, _blob = _next_raceday_id(conn, client, current)
                if nxt is None or int(nxt) in visited:
                    break
                nxt = int(nxt)
                if nxt not in done:
                    status, payload = fetch_raceday(client, nxt)
                    if store_raceday_raw(conn, nxt, status, payload):
                        scraped.add(nxt)
                        done.add(nxt)
                    counts["fetched"] += 1
                    if delay:
                        time.sleep(delay)
                current = nxt
                steps += 1
                counts["steps"] += 1
                if steps >= max_steps_per_chain:
                    counts["chains_capped"] += 1
                    log(f"  chain from {seed} hit step cap ({max_steps_per_chain})")
                    break
    finally:
        if own_client:
            client.close()
    counts["scraped"] = len(scraped)
    counts["scraped_ids"] = sorted(scraped)
    return counts


def main() -> None:
    import sys
    from core.db import get_connection

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    if len(sys.argv) < 2:
        print("usage: python -m scrapers.st_raceday <race_day_id> [race_day_id ...]")
        raise SystemExit(1)
    ids = [int(a) for a in sys.argv[1:]]
    conn = get_connection()
    try:
        print(scrape_raceday_ids(conn, ids, skip_done=False))
    finally:
        conn.close()


if __name__ == "__main__":
    main()
