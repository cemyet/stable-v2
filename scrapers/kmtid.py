"""kmtid.atgx.se scraper — GPS-derived per-100m sectionals for Swedish trot.

The site publishes a JS bundle per race-day at
    https://kmtid.atgx.se/<YYMMDD>/js/races.js

The bundle is two top-level JS literals:
    const toplist = { ... };   -- best splits across the whole day
    const races   = [ ... ];   -- per-race per-horse sectional data (what we want)

Both are valid JSON inside the literal braces / brackets, so we extract them
with simple bracket-balancing rather than running JS.

Only racedays where ATG had GPS-equipped tracks publish data; the URL
returns 404 for days without coverage. The site keeps roughly a 30-day
rolling window — older days are pruned, so we run this scraper every day
and re-fetch the whole window each pass (cheap; one HTTP call per day).
"""

from __future__ import annotations

import json
import logging
import re
import time
from datetime import date, timedelta
from typing import Iterator

import httpx

from core.config import (
    KMTID_RACES_URL,
    KMTID_HEADERS,
    KMTID_BACKFILL_DAYS,
    REQUEST_DELAY,
)

log = logging.getLogger(__name__)

# We only care about `races` — the top list is denormalised summary data
# already covered by per-race entries.
_RE_RACES = re.compile(r"\bconst\s+races\s*=\s*", re.MULTILINE)

# Homepage URL — contains a hand-curated list of every available raceday.
# We use this for *historical* backfill because date-based probing only
# works for the trailing ~30 days; the homepage list goes back much further
# (~17 months as of 2026) and also includes multi-day event bundles like
# `elitloppet` whose race-days are NOT individually addressable.
KMTID_INDEX_URL = "https://kmtid.atgx.se/"

# Pattern matching `{name: "...", url: "..."}` entries in the homepage JS.
_RE_INDEX_ENTRY = re.compile(
    r'\{\s*name:\s*"([^"]*)"\s*,\s*url:\s*"([^"]*)"\s*\}'
)
# Block-comments hide RETIRED entries — those URLs 404 server-side.
_RE_BLOCK_COMMENT = re.compile(r"/\*.*?\*/", re.DOTALL)


# ---------------------------------------------------------------------------
# Bracket-balanced JSON slice
# ---------------------------------------------------------------------------

def _extract_json_array(src: str, start_offset: int) -> str | None:
    """Return the substring starting at the '[' at/after start_offset that
    forms a balanced JSON array."""
    i = src.find("[", start_offset)
    if i < 0:
        return None
    depth = 0
    in_str = False
    escape = False
    j = i
    while j < len(src):
        c = src[j]
        if in_str:
            if escape:
                escape = False
            elif c == "\\":
                escape = True
            elif c == '"':
                in_str = False
        else:
            if c == '"':
                in_str = True
            elif c == "[":
                depth += 1
            elif c == "]":
                depth -= 1
                if depth == 0:
                    return src[i:j + 1]
        j += 1
    return None


# ---------------------------------------------------------------------------
# HTTP fetch
# ---------------------------------------------------------------------------

def _fetch_day(client: httpx.Client, d: date) -> str | None:
    yymmdd = d.strftime("%y%m%d")
    url = KMTID_RACES_URL.format(yymmdd=yymmdd)
    r = client.get(url, headers=KMTID_HEADERS, timeout=30.0)
    if r.status_code == 404:
        return None
    if r.status_code != 200:
        log.warning("kmtid %s -> HTTP %s", url, r.status_code)
        return None
    return r.text


# ---------------------------------------------------------------------------
# Parse
# ---------------------------------------------------------------------------

def parse_races_js(text: str) -> list[dict]:
    """Extract the `races` array from a races.js bundle and return it as a
    list of plain Python dicts."""
    m = _RE_RACES.search(text)
    if not m:
        return []
    arr_str = _extract_json_array(text, m.end())
    if not arr_str:
        return []
    try:
        return json.loads(arr_str)
    except json.JSONDecodeError as e:
        log.warning("kmtid races json parse failed: %s", e)
        return []


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def scrape_day(d: date, client: httpx.Client | None = None) -> list[dict]:
    """Fetch + parse one race-day. Returns [] when the day has no coverage."""
    own_client = client is None
    if own_client:
        client = httpx.Client()
    try:
        text = _fetch_day(client, d)
        if text is None:
            return []
        return parse_races_js(text)
    finally:
        if own_client:
            client.close()


def scrape_window(
    days: int = KMTID_BACKFILL_DAYS,
    end_date: date | None = None,
) -> Iterator[tuple[date, list[dict]]]:
    """Yield (date, races) for every day in the trailing window.

    Days without coverage yield (date, [])."""
    end_date = end_date or date.today()
    with httpx.Client() as client:
        for off in range(days, -1, -1):
            d = end_date - timedelta(days=off)
            races = scrape_day(d, client=client)
            yield d, races
            if races:
                log.info("kmtid %s: %d races", d.isoformat(), len(races))
            time.sleep(REQUEST_DELAY)


# ---------------------------------------------------------------------------
# Index-driven historical backfill
# ---------------------------------------------------------------------------

def fetch_index_slugs(client: httpx.Client | None = None) -> list[tuple[str, str]]:
    """Return [(label, url_slug), ...] from the live entries on kmtid.atgx.se.

    Skips:
      * commented-out (retired) entries — their slugs 404 server-side
      * entries with empty `url` (the "inga tider" / no-data placeholders)
    """
    own = client is None
    if own:
        client = httpx.Client()
    try:
        r = client.get(KMTID_INDEX_URL, headers=KMTID_HEADERS, timeout=20.0)
        r.raise_for_status()
        html = r.text
    finally:
        if own:
            client.close()

    live = _RE_BLOCK_COMMENT.sub("", html)
    out: list[tuple[str, str]] = []
    for name, slug in _RE_INDEX_ENTRY.findall(live):
        slug = slug.strip()
        if not slug:
            continue
        out.append((name, slug))
    return out


def scrape_slug(slug: str, client: httpx.Client | None = None) -> list[dict]:
    """Fetch + parse one slug from kmtid (date string OR named bundle like
    'elitloppet'). Returns [] when the URL is missing or empty."""
    own = client is None
    if own:
        client = httpx.Client()
    try:
        url = KMTID_RACES_URL.format(yymmdd=slug)
        r = client.get(url, headers=KMTID_HEADERS, timeout=30.0)
        if r.status_code == 404:
            return []
        if r.status_code != 200:
            log.warning("kmtid slug %s -> HTTP %s", slug, r.status_code)
            return []
        return parse_races_js(r.text)
    finally:
        if own:
            client.close()


def scrape_all_listed() -> Iterator[tuple[str, str, list[dict]]]:
    """Yield (slug, label, races) for every live entry on the kmtid index.

    Use this for historical backfill — covers everything the homepage still
    publishes, which goes back ~17 months as of 2026 (vs. the ~30 days of
    date-based probing in `scrape_window`)."""
    with httpx.Client() as client:
        index = fetch_index_slugs(client=client)
        log.info("kmtid index: %d live slugs", len(index))
        for label, slug in index:
            races = scrape_slug(slug, client=client)
            yield slug, label, races
            if races:
                log.info("kmtid %-18s %3d races  (%s)", slug, len(races), label[:60])
            else:
                log.warning("kmtid %-18s 0 races (%s)", slug, label[:60])
            time.sleep(REQUEST_DELAY)


# ---------------------------------------------------------------------------
# CLI: `python -m scrapers.kmtid 260509` or `python -m scrapers.kmtid window`
# ---------------------------------------------------------------------------

def main() -> None:
    import sys
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print("usage: python -m scrapers.kmtid <YYMMDD|window>")
        return

    arg = sys.argv[1]
    if arg == "window":
        n = 0
        for d, races in scrape_window():
            n += len(races)
        print(f"kmtid scrape_window: {n} races across {KMTID_BACKFILL_DAYS+1} day(s)")
    else:
        # YYMMDD
        try:
            d = date(2000 + int(arg[0:2]), int(arg[2:4]), int(arg[4:6]))
        except (ValueError, IndexError):
            raise SystemExit(f"bad date arg: {arg!r} (expected YYMMDD)")
        races = scrape_day(d)
        print(f"kmtid scrape_day {d.isoformat()}: {len(races)} races")
        if races:
            r0 = races[0]
            print(f"  first race id={r0.get('id')!r} name={r0.get('name')!r} starts={len(r0.get('starts') or [])}")


if __name__ == "__main__":
    main()
