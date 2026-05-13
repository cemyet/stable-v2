"""
ATG calendar / race / game scraper (PLACEHOLDER).

Will fetch https://www.atg.se/services/racinginfo/v1/api/calendar/day/<date>
and per-race endpoints, parse, then UPSERT into stable_v2.race + .entry
attaching the atg_race_id alongside any st_race_id we already have.

For now use jobs.update --mode bridge.
"""

from __future__ import annotations


def main() -> None:
    raise SystemExit(
        "scrapers.atg not implemented yet — use jobs.update --mode bridge."
    )


if __name__ == "__main__":
    main()
