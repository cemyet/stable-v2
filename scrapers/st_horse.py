"""
TravSport horse passport scraper (PLACEHOLDER).

Will fetch https://sportapp.travsport.se/sportinfo/horse/ts<id>/basic,
parse with core.parser.parse_horse_page, and UPSERT into stable_v2.horse
via etl.import_st live-mode helpers.

For now the v1 scraper at /Users/jakob/Dev/stable/scrapers/horse_scraper.py
keeps doing this job; bridge-mode (jobs.update --mode bridge) will mirror
its results into stable_v2.
"""

from __future__ import annotations


def main() -> None:
    raise SystemExit(
        "scrapers.st_horse not implemented yet — use jobs.update --mode bridge "
        "to keep stable_v2 fresh via v1's scrapers."
    )


if __name__ == "__main__":
    main()
