"""travrondenspel.se scraper — post-race comments, one line per start.

The comment is the trotting press's account of how a horse's race actually
unfolded ("Rygg led, 3-inv e 550, fram inv, höll bara farten över uppl."),
written after the fact. Neither ATG's nor TravSport's public APIs carry it,
and it is the one part of a result that cannot be derived from the numbers.

Two endpoints matter:

    meet/?start_date=&end_date=    racedays in a window, with track + slug
    race/{slug}-{number}/          one race; every start carries `comment`

Addressing a *finished* race directly is what makes this usable: the same
comments also appear under a horse in an upcoming race's start list, but
harvesting them that way means waiting for the horse to enter another race.

Comments are written by hand, so they trail the race by 1-3 days. Anything
reading this needs to re-visit recent days rather than assume yesterday is
complete. Foreign meets come back over the same endpoints but are never
commented, so callers filter to SE.
"""

from __future__ import annotations

import logging
import time
from datetime import date
from typing import Iterator

import httpx

from core.config import (
    TR_HEADERS,
    TR_MEET_LIST_URL,
    TR_RACE_URL,
    TR_REQUEST_DELAY,
    TR_TIMEOUT,
)

log = logging.getLogger(__name__)

# The list endpoint paginates; a window of racedays never approaches this.
_MEET_PAGE_LIMIT = 200


def _get(client: httpx.Client, url: str, params: dict | None = None) -> dict | None:
    """GET one JSON document. 404 means "no such raceday/race", which is an
    ordinary answer here (tracks race on different days), so it is not logged."""
    r = client.get(url, params=params, headers=TR_HEADERS, timeout=TR_TIMEOUT)
    if r.status_code == 404:
        return None
    if r.status_code != 200:
        log.warning("tr %s -> HTTP %s", url, r.status_code)
        return None
    try:
        return r.json()
    except ValueError as e:
        log.warning("tr %s -> bad json: %s", url, e)
        return None


# ---------------------------------------------------------------------------
# Racedays
# ---------------------------------------------------------------------------

def fetch_meets(start_date: date, end_date: date, *,
                country: str | None = "SE",
                client: httpx.Client | None = None) -> list[dict]:
    """Return the racedays held between the two dates inclusive.

    Each meet carries `slug` (what `fetch_race` addresses races by), `race_date`
    and a nested `track`. Passing country=None keeps foreign meets too."""
    own = client is None
    if own:
        client = httpx.Client()
    try:
        payload = _get(client, TR_MEET_LIST_URL, {
            "ordering": "start_time",
            "limit": _MEET_PAGE_LIMIT,
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
        })
    finally:
        if own:
            client.close()

    meets = (payload or {}).get("results") or []
    total = (payload or {}).get("count")
    if total and total > len(meets):
        # Silently returning a short window would look like "those days had no
        # racing" and quietly skip them, so say so instead.
        log.warning("tr meet list truncated: %s of %s meets for %s..%s; "
                    "narrow the window", len(meets), total, start_date, end_date)
    if country:
        meets = [m for m in meets if meet_track(m).get("country") == country]
    return meets


# ---------------------------------------------------------------------------
# One race
# ---------------------------------------------------------------------------

def fetch_race(track_slug: str, race_date: date | str, race_number: int,
               client: httpx.Client | None = None) -> dict | None:
    """Return one race, or None when that number does not exist on the day."""
    own = client is None
    if own:
        client = httpx.Client()
    try:
        d = race_date if isinstance(race_date, str) else race_date.isoformat()
        return _get(client, TR_RACE_URL.format(
            track_slug=track_slug, race_date=d, race_number=race_number))
    finally:
        if own:
            client.close()


def meet_track(meet: dict) -> dict:
    """Track name/slug/country for a meet, whichever shape it arrived in.

    The list endpoint nests it (`track: {name, slug, country}`) while a race
    payload flattens it (`track_name`, `track_slug`, `country`)."""
    nested = meet.get("track")
    if isinstance(nested, dict):
        return {"name": nested.get("name"), "slug": nested.get("slug"),
                "country": nested.get("country")}
    return {"name": meet.get("track_name"), "slug": meet.get("track_slug"),
            "country": meet.get("country")}


def comments_from_race(race: dict) -> list[dict]:
    """Flatten one race payload into a row per commented start.

    The field names mirror the historical travfakta export because they are
    the same database underneath — `start_id` here is that export's `start_id`,
    and `atg_id` is the horse's TravSport id (our `horse.st_id`). Uncommented
    starts are dropped; a race that has not been written up yet yields nothing.
    """
    meet = race.get("meet") or {}
    track = meet_track(meet)
    race_date = meet.get("race_date")
    race_number = race.get("race_number")

    out: list[dict] = []
    for s in race.get("starts") or []:
        comment = (s.get("comment") or "").strip()
        if not comment:
            continue
        horse = s.get("horse") or {}
        out.append({
            "start_id":     s.get("id"),
            "horse_name":   horse.get("name"),
            "atg_id":       horse.get("atg_id"),
            "race_date":    race_date,
            "track_name":   track.get("name"),
            "race_number":  race_number,
            "start_number": s.get("start_number"),
            "comment":      comment,
        })
    return out


def scrape_races(targets: list[tuple[str, str, int]],
                 client: httpx.Client | None = None,
                 log_fn=None) -> Iterator[dict]:
    """Yield comment rows for each (track_slug, race_date, race_number).

    Serial by design — see TR_REQUEST_DELAY. Races that 404 or carry no
    comments simply contribute nothing."""
    own = client is None
    if own:
        client = httpx.Client()
    try:
        for i, (slug, d, n) in enumerate(targets):
            race = fetch_race(slug, d, n, client=client)
            if race:
                rows = comments_from_race(race)
                for row in rows:
                    yield row
                if log_fn and rows:
                    log_fn(f"{slug}-{d}-{n}: {len(rows)} comments")
            time.sleep(TR_REQUEST_DELAY)
            if log_fn and i and i % 100 == 0:
                log_fn(f"...{i}/{len(targets)} races fetched")
    finally:
        if own:
            client.close()


# ---------------------------------------------------------------------------
# CLI: `python -m scrapers.tr bollnas 2026-07-25 3`
# ---------------------------------------------------------------------------

def main() -> None:
    import sys
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    args = sys.argv[1:]
    if len(args) == 2:
        meets = fetch_meets(date.fromisoformat(args[0]), date.fromisoformat(args[1]))
        print(f"{len(meets)} SE meets")
        for m in meets:
            print(f"  {m.get('race_date')}  {m.get('slug'):<28} "
                  f"races={len(m.get('races') or [])}")
        return

    if len(args) != 3:
        print("usage: python -m scrapers.tr <track-slug> <YYYY-MM-DD> <race-number>")
        print("       python -m scrapers.tr <start-date> <end-date>   # list meets")
        return

    race = fetch_race(args[0], args[1], int(args[2]))
    if not race:
        print("no such race")
        return
    rows = comments_from_race(race)
    print(f"status={race.get('status')} starts={len(race.get('starts') or [])} "
          f"commented={len(rows)}")
    for r in rows:
        print(f"  nr {str(r['start_number']):<3} {(r['horse_name'] or '')[:24]:<24} "
              f"{r['comment']}")


if __name__ == "__main__":
    main()
