"""Native ATG (Aktiebolaget Trav och Galopp) scraper.

Fetches the ATG racing-info API directly and stores per-race raw JSON durably
in `atg_race_raw` (+ `atg_race_scrape_log` and the 7-day `atg_buffer`). ETL into
v2 master tables lives in `etl.import_atg` (`ingest_race` / `load_atg_from_raw`).

This is the v2-native replacement for v1's ATG scraping. The per-race payload we
store is exactly the shape v1 kept in `v2_atg_race_raw`, so `ingest_race` — which
already parses that shape — works unchanged. With this in place v2 no longer
shells out to the v1 project for ATG.

Enumeration strategy: the calendar/day endpoint returns `tracks[].races[]` with
every race's `id`, `number` and `status`. We take races whose status marks a
finished result (`FINAL_STATUSES`) on non-gallop tracks, then fetch each race's
full payload from `/races/{id}`.
"""

from __future__ import annotations

import logging
import time
from datetime import date, datetime, timedelta

import httpx
from psycopg2.extras import Json

from core import config
from core.db import buffer_insert
from scrapers.st_horse import _get_json  # reuse the hardened SIGALRM client

log = logging.getLogger(__name__)

# ATG race `status` values that mean "results are published". Anything else
# ('upcoming', 'ongoing', 'cancelled', ...) is skipped by the results scrape.
FINAL_STATUSES = frozenset({"results"})


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------

def make_client() -> httpx.Client:
    """httpx client configured for the ATG JSON API."""
    return httpx.Client(
        headers=config.ATG_HEADERS,  # Accept: application/json
        follow_redirects=True,
        timeout=httpx.Timeout(connect=15.0, read=30.0, write=15.0, pool=10.0),
        limits=httpx.Limits(max_keepalive_connections=0),
    )


def fetch_calendar(client: httpx.Client, date_str: str) -> tuple[int, object | None]:
    """Fetch the calendar for a YYYY-MM-DD date. Returns (http_status, json)."""
    url = config.ATG_CALENDAR_URL.format(date=date_str)
    return _get_json(client, url)


def fetch_race(client: httpx.Client, atg_race_id: str) -> tuple[int, object | None]:
    """Fetch one race's full payload. Returns (http_status, json)."""
    url = config.ATG_RACE_URL.format(atg_race_id=atg_race_id)
    return _get_json(client, url)


# ---------------------------------------------------------------------------
# Enumeration
# ---------------------------------------------------------------------------

def _race_date_from_id(atg_race_id: str) -> date | None:
    """The atg_race_id is `YYYY-MM-DD_track_number`; pull the date prefix."""
    try:
        return datetime.strptime(atg_race_id.split("_", 1)[0], "%Y-%m-%d").date()
    except (ValueError, AttributeError):
        return None


def enumerate_race_ids(calendar: dict,
                       statuses: frozenset[str] = FINAL_STATUSES) -> list[tuple[str, date | None]]:
    """From a calendar payload, list (atg_race_id, race_date) for finished trot
    races. Gallop tracks are excluded entirely (v2 is trot-only)."""
    out: list[tuple[str, date | None]] = []
    for track in (calendar or {}).get("tracks", []) or []:
        if (track.get("sport") or "trot") == "gallop":
            continue
        for r in track.get("races", []) or []:
            rid = r.get("id")
            if not rid:
                continue
            if statuses and (r.get("status") not in statuses):
                continue
            out.append((rid, _race_date_from_id(rid)))
    return out


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def store_race_raw(conn, atg_race_id: str, race_date: date | None,
                   http_status: int, payload: object | None,
                   *, status_label: str | None = None) -> bool:
    """Persist one race fetch. Returns ok (200 + JSON object with starts)."""
    ok = http_status == 200 and isinstance(payload, dict) and bool(payload.get("starts"))
    if race_date is None and isinstance(payload, dict):
        d = payload.get("date")
        if d:
            try:
                race_date = datetime.strptime(d, "%Y-%m-%d").date()
            except ValueError:
                race_date = None
    error_message = None if ok else (
        f"http_{http_status}" if http_status != 200 else "no_starts"
    )
    with conn.cursor() as cur:
        if ok:
            cur.execute(
                """
                INSERT INTO atg_race_raw (atg_race_id, raw_json, race_date, scraped_at)
                VALUES (%s, %s, %s, NOW())
                ON CONFLICT (atg_race_id)
                DO UPDATE SET raw_json = EXCLUDED.raw_json,
                              race_date = EXCLUDED.race_date,
                              scraped_at = EXCLUDED.scraped_at
                """,
                (atg_race_id, Json(payload), race_date),
            )
        cur.execute(
            """
            INSERT INTO atg_race_scrape_log (atg_race_id, http_status, status, scraped_at, error_message)
            VALUES (%s, %s, %s, NOW(), %s)
            ON CONFLICT (atg_race_id)
            DO UPDATE SET http_status = EXCLUDED.http_status,
                          status = EXCLUDED.status,
                          scraped_at = EXCLUDED.scraped_at,
                          error_message = EXCLUDED.error_message
            """,
            (atg_race_id, http_status, status_label, error_message),
        )
    conn.commit()
    buffer_insert(conn, "atg",
                  config.ATG_RACE_URL.format(atg_race_id=atg_race_id),
                  http_status, None, ok)
    return ok


# ---------------------------------------------------------------------------
# Day / window scrape
# ---------------------------------------------------------------------------

def scrape_day(conn, date_str: str, *,
               statuses: frozenset[str] = FINAL_STATUSES,
               skip_done: bool = False,
               delay: float | None = None,
               client: httpx.Client | None = None,
               log=print) -> dict:
    """Fetch the calendar for `date_str`, then fetch + store every finished trot
    race's raw JSON. Returns counts."""
    if delay is None:
        delay = config.REQUEST_DELAY
    own_client = client is None
    client = client or make_client()
    counts = {"date": date_str, "races_listed": 0, "ok": 0,
              "missing": 0, "failed": 0, "skipped": 0}
    try:
        cstatus, calendar = fetch_calendar(client, date_str)
        if cstatus != 200 or not isinstance(calendar, dict):
            counts["calendar_status"] = cstatus
            return counts
        listed = enumerate_race_ids(calendar, statuses)
        counts["races_listed"] = len(listed)

        done: set[str] = set()
        if skip_done and listed:
            ids = [rid for rid, _ in listed]
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT atg_race_id FROM atg_race_scrape_log "
                    "WHERE atg_race_id = ANY(%s) AND http_status = 200",
                    (ids,),
                )
                done = {r[0] for r in cur.fetchall()}

        for rid, rdate in listed:
            if rid in done:
                counts["skipped"] += 1
                continue
            status, payload = fetch_race(client, rid)
            if store_race_raw(conn, rid, rdate, status, payload, status_label="results"):
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


def scrape_window(conn, *, days_back: int = 4, end_date: date | None = None,
                  statuses: frozenset[str] = FINAL_STATUSES,
                  skip_done: bool = True,
                  delay: float | None = None,
                  log=print) -> dict:
    """Scrape the last `days_back` days (inclusive of `end_date`, default today).

    ATG publishes results shortly after each race, so a small trailing window
    picked up nightly keeps v2 current. `skip_done` avoids re-fetching races we
    already stored a 200 for."""
    end = end_date or date.today()
    client = make_client()
    totals = {"days": 0, "races_listed": 0, "ok": 0,
              "missing": 0, "failed": 0, "skipped": 0, "by_day": []}
    try:
        for i in range(days_back):
            d = end - timedelta(days=i)
            c = scrape_day(conn, d.isoformat(), statuses=statuses,
                           skip_done=skip_done, delay=delay, client=client, log=log)
            totals["days"] += 1
            for k in ("races_listed", "ok", "missing", "failed", "skipped"):
                totals[k] += c.get(k, 0)
            totals["by_day"].append(c)
            log(f"  [atg] {d.isoformat()}: listed={c['races_listed']} "
                f"ok={c['ok']} skip={c['skipped']} miss={c['missing']} fail={c['failed']}")
    finally:
        client.close()
    return totals


def main() -> None:
    import sys
    from core.db import get_connection

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    args = sys.argv[1:]
    conn = get_connection()
    try:
        if args and "-" in args[0] and len(args[0]) == 10:
            # explicit date(s): python -m scrapers.atg 2026-07-03 [2026-07-02 ...]
            for d in args:
                print(scrape_day(conn, d, skip_done=False))
        else:
            days = int(args[0]) if args else 4
            print(scrape_window(conn, days_back=days, skip_done=True))
    finally:
        conn.close()


if __name__ == "__main__":
    main()
