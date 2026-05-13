# stable-v2

Second iteration of the stable trotting database. Runs alongside v1 (`/Users/jakob/Dev/stable/`) on port **5002** against the **`stable_v2`** PostgreSQL database.

## Why v2?

v1 worked, but it grew organically into "store raw → transform later" with two-layer identity / per-source-passport tables and a 22 GB raw-game table. v2 collapses everything to:

- **5 flat global master tables**: `horse`, `person`, `race`, `entry`, `track`. Each row has our own canonical SERIAL id, plus nullable per-source id columns (`st_id`, `atg_id`, `usta_id`, `letrot_id`, ...) and a `source_data JSONB` for source-specific extras.
- **scrape → parse → UPSERT**: scrapers parse in-memory and write directly into master tables. A tiny rolling 7-day per-source buffer holds raw HTTP for retry/debug only.
- **postgres_fdw link to v1**: historical data is read from v1's `jakob` database via a foreign-data-wrapper schema (`v1_raw.*`) for one-time backfill. v2 itself never accumulates bulk raw.

## Layout

```
core/        config, db, parser, common
etl/         per-source import_<source>.py + matching.py
scrapers/    one .py per source
jobs/        update.py CLI + admin entrypoint
web/         Flask app on port 5002 + templates/ + static/
logs/        per-job logs
```

## Quick start

```sh
createdb stable_v2
psql -d stable_v2 -c "CREATE EXTENSION postgres_fdw;"
pip install -r requirements.txt
python -m core.schema apply        # create tables
python -m jobs.fdw                 # wire up v1_raw schema
python -m web.app                  # start frontend on :5002
```

## Status

Under construction. v1 stays running on port 5001 against `jakob` until v2 reaches parity.
