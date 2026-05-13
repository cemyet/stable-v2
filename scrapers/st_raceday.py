"""
TravSport raceday API scraper (PLACEHOLDER).

Will fetch the raceday JSON, parse race + entry rows, and UPSERT into
stable_v2.race + stable_v2.entry via etl.import_st live-mode helpers.

For now use jobs.update --mode bridge.
"""

from __future__ import annotations


def main() -> None:
    raise SystemExit(
        "scrapers.st_raceday not implemented yet — use jobs.update --mode bridge."
    )


if __name__ == "__main__":
    main()
