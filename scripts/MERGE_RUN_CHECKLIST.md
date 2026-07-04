# Mass merge run checklist

This checklist must be completed end-to-end **before** running any of the
`--execute` merges. Every step is idempotent unless noted.

The corresponding implementation plan: `column-merge-and-french-cleanup`.

---

## 0. Pre-flight (do this first, every time)

1. **Confirm FX backfill is complete and clean.**

   ```bash
   psql -h localhost -p 5432 -d stable_v2 <<'SQL'
   SELECT prize_currency,
          COUNT(*)                                  AS n_entries,
          COUNT(*) FILTER (WHERE prize_kr = 0
                          AND prize_original > 0)   AS unconverted,
          COUNT(*) FILTER (WHERE prize_fx_rate IS NULL
                          AND prize_currency <> 'SEK') AS missing_rate
     FROM entry
    WHERE prize_currency IS NOT NULL
    GROUP BY 1
    ORDER BY 1;
   SQL
   ```

   `unconverted` and `missing_rate` should be **0 for EUR** before merging.

2. **Take a timestamped DB snapshot** of the tables we will mutate.

   ```bash
   TS=$(date +%Y%m%d_%H%M%S)
   DUMP_DIR="/tmp/stable_v2_merge_${TS}"
   mkdir -p "$DUMP_DIR"

   # Schema-only (small, full restore reference).
   pg_dump -h localhost -p 5432 -d stable_v2 --schema-only \
       -f "$DUMP_DIR/schema.sql"

   # Data-only for the tables we mutate. These are the only things merges touch.
   for tbl in entry horse race person \
              horse_merge_log person_merge_log \
              horse_owner_history horse_trainer_history; do
       pg_dump -h localhost -p 5432 -d stable_v2 \
           --data-only --table="public.${tbl}" \
           --file="$DUMP_DIR/${tbl}.sql"
       echo "  dumped ${tbl}"
   done

   echo "snapshot at: $DUMP_DIR"
   du -sh "$DUMP_DIR"
   ```

   Keep this folder until you are confident the merge run is final. Restore
   recipe: `psql -d stable_v2 -f $DUMP_DIR/<table>.sql` (after TRUNCATE).

3. **Note current row counts** so you can spot anything weird mid-run.

   ```bash
   psql -h localhost -p 5432 -d stable_v2 <<'SQL'
   SELECT 'entry'  tbl, COUNT(*) FROM entry
   UNION ALL SELECT 'horse',    COUNT(*) FROM horse
   UNION ALL SELECT 'race',     COUNT(*) FROM race
   UNION ALL SELECT 'person',   COUNT(*) FROM person;
   SQL
   ```

---

## 1. Smoke tests (dry-run only — safe to run any time)

Run the canary cases first so you can compare apples to apples after execute.

```bash
cd /Users/jakob/Dev/stable-v2

# Phase 2: French race matcher — show what the Iguski 2022-04-29 row would do.
PYTHONPATH=. python3 -m scripts.match_french_races --limit 50

# Phase 4: Horse fingerprint — strict pass dry-run.
PYTHONPATH=. python3 -m scripts.match_french_horses --limit 50

# Phase 5: Person co-occurrence — dry-run with low threshold (so we see edges).
PYTHONPATH=. python3 -m scripts.match_persons_by_cooccurrence --min-shared 3 --limit 50
```

Expectation: each prints PREVIEW lines for ~50 candidates, no errors, no DB mutations.

---

## 2. Live run order

Run **in this order**. Each phase commits in batches via `--commit-every`,
so a partial run is OK to resume.

```bash
# Phase 3 first — race-row dedupe so horse + person matchers have correct race graph.
PYTHONPATH=. python3 -m scripts.merge_duplicate_races --execute --commit-every 200

# Phase 2 — French race matcher (different-track variants).
PYTHONPATH=. python3 -m scripts.match_french_races --execute --commit-every 100

# Phase 4 — strict-pass French horse matcher.
PYTHONPATH=. python3 -m scripts.match_french_horses --execute --commit-every 200

# Phase 5 — person co-occurrence (drivers + trainers).
PYTHONPATH=. python3 -m scripts.match_persons_by_cooccurrence \
    --execute --min-shared 5 --commit-every 200

# Phase 6 — one-time French sex code backfill.
PYTHONPATH=. python3 -m scripts.normalize_french_sex --execute
```

After each script, eyeball:

* `summary.merged` matches your dry-run prediction (±5%).
* `summary.errors == 0`.
* A handful of horses via the frontend — pick a few from the script's last
  log lines and load `/horse/<id>` to confirm the result table looks right.

---

## 3. Rollback

If a phase produced clearly wrong results, roll back by `job_run_id`:

```bash
# Find the job_run_id from the script run (printed in summary, or query job_run).
psql -d stable_v2 -c "SELECT job_run_id, script, started_at, summary->>'merged' AS merged
                        FROM job_run ORDER BY job_run_id DESC LIMIT 10"

# Roll back all merges from that job (reverses in reverse-chronological order).
PYTHONPATH=. python3 -m scripts.<the_script> --rollback-job <id>
```

If the rollback can't restore (e.g. for race-track switches that have since
been re-merged), restore from the dump:

```bash
psql -d stable_v2 -c "TRUNCATE entry, horse, race, person,
                       horse_merge_log, person_merge_log RESTART IDENTITY CASCADE"
for tbl in race horse person entry horse_merge_log person_merge_log; do
    psql -d stable_v2 -f $DUMP_DIR/${tbl}.sql
done
```

---

## 4. Out-of-scope deferrals

These do NOT run as part of this checklist:

* **LeTrot pedigree scrape** — multi-day operation; run separately after the
  merge work has settled. Pedigree FK migration during `merge_horses`
  already handles re-pointing, so doing it later is safe.
* **USTA / kmtid cross-source merges** — low volume right now; defer.
* **Career-stat recomputation** — handled by the existing nightly aggregation
  job; no manual step.
