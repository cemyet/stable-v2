"""
Native v2 scrapers (scrape → parse → UPSERT into master tables).

Each scraper follows the same shape:

    1. Fetch the source URL with httpx.
    2. Append the raw response to <source>_buffer for retry/debug.
    3. Parse in-memory.
    4. Call etl.import_<source>.upsert_*() helpers to write into the
       master tables (horse, person, race, entry, track) using the
       cross-source matching rules in etl.matching.

For now (v2 launch) the live update path is the "bridge" mode in
jobs.update — it runs v1's update job + mirrors changed v1 rows into
stable_v2. Native scrapers will be added incrementally per source
(`st_horse.py`, `st_raceday.py`, `atg.py`, `usta.py`, `letrot.py`,
`kmtid.py`).
"""
