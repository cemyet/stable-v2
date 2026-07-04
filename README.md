# stable-v2

Second iteration of the stable trotting database. Runs alongside v1 (`/Users/jakob/Dev/stable/`) on port **5002** against the **`stable_v2`** PostgreSQL database.

## Why v2?

v1 worked, but it grew organically into "store raw → transform later" with two-layer identity / per-source-passport tables and a 22 GB raw-game table. v2 collapses everything to:

- **5 flat global master tables**: `horse`, `person`, `race`, `entry`, `track`. Each row has our own canonical SERIAL id, plus nullable per-source id columns (`st_id`, `atg_id`, `usta_id`, `letrot_id`, ...) and a `source_data JSONB` for source-specific extras.
- **scrape → parse → UPSERT**: scrapers parse in-memory and write directly into master tables. A tiny rolling 7-day per-source buffer holds raw HTTP for retry/debug only.
- **postgres_fdw link to v1**: historical data is read from v1's `jakob` database via a foreign-data-wrapper schema (`v1_raw.*`) for one-time backfill. v2 itself never accumulates bulk raw.

## Layout

```
core/        config, db, parser, common, identity (cross-source resolver)
etl/         per-source import_<source>.py + matching.py
scrapers/    one .py per source
jobs/        update.py CLI + admin entrypoint
web/         Flask app on port 5002 + templates/ + static/
scripts/     one-off + recurring CLI tools (audit_matching, merge_*)
logs/        per-job logs
```

## Identity matching

Cross-source horse and person identity is resolved by `core/identity.py`.
Every importer goes through `core.identity.resolve_horse` / `resolve_person`
(never raw `INSERT INTO horse`). The resolver applies a strict layered
protocol — name-only matches are **never** auto-applied:

  1. **Strong ID** — same `source_id` already seen for this source.
  2. **Cross-source ID** — `registration_number`, `ueln_number`, or the
     SE id-equivalence quirk (ATG's `horse.id` == TravSport `st_id`).
  3. **Race-context** — when a race already has an entry with the same
     `program_number`, that horse is reused (covers cases like LeTrot
     ingesting a race ST already wrote).
  4. **Pedigree triangulation** — name + birth-year + sire-name +
     dam-name (all four required, case-insensitive) — auto-merge eligible.
  5. **Synthetic key** — `x:CC:NORMALIZED_NAME` for foreign horses
     without a stable id. Reused across re-ingests.
  6. **INSERT** — never matched, mint a new canonical row.

### Audit + admin tooling

- `python -m scripts.audit_matching` (or `python -m scripts.audit_matching --json`)
  emits a per-category health snapshot of duplicate counts. Same queries
  power the web dashboard at **/admin/matching**.

- The /admin/matching page has four stacked panels:
    1. **Health** — live counts per category, click a card to drill in.
    2. **Browse** — top-50 affected rows for the selected category.
    3. **Manual merge** — side-by-side horse picker with dry-run preview
       and execute (every merge is logged to `horse_merge_log`).
    4. **Cleanup scripts** — one-click dry-run or execute for each Act 2
       script via the `job_run` subprocess runner.

### Cleanup scripts (Act 2)

All scripts default to `--dry-run` and require `--execute` to mutate.
Every merge writes a snapshot row to `horse_merge_log` / `person_merge_log`
for rollback.

```
python -m scripts.merge_pedigree_duplicates       # name+year+sire+dam groups
python -m scripts.merge_synth_pairs               # Cat B (ST-guest ↔ ATG-synth)
python -m scripts.split_polluted_atg_ids          # Cat A (old ST + foreign entries)
python -m scripts.clean_same_row_atg_ids          # both st_id AND x: atg_id
python -m scripts.merge_duplicate_races           # races sharing track+date+#
python -m scripts.merge_synth_pairs_persons       # person Cat G equivalent
```

### Adding a new source

When wiring a new scraper into `etl/`, never lookup horses by name. Always go
through `core.identity.resolve_horse(cur, source=..., source_id=..., ...)`
and pass everything available: `registration_number`, `ueln_number`,
`race_id` + `program_number` (for race-context match), and `sire_name` +
`dam_name` (for pedigree triangulation).

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
