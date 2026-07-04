-- One-time migration for EXISTING databases: backfill entry.race_date.
--
-- core.schema (create_schema) adds the entry.race_date column, the sync
-- triggers, the composite (entity, race_date) indexes and the
-- person_career_stats materialized view. On a brand-new DB that is enough.
-- On an EXISTING DB the freshly-added column is NULL for every historical
-- row until it is backfilled — run this script ONCE after applying the schema
-- (and ideally build the big composite indexes CONCURRENTLY afterwards so the
-- backfill doesn't fight the index build):
--
--   psql -d stable_v2 -f scripts/migrate_entry_race_date.sql
--
-- It is idempotent (the IS DISTINCT FROM guard makes re-runs cheap) and works
-- in race_id batches so it never holds a long lock or bloats one transaction.
DO $$
DECLARE
    lo integer := 0;
    hi integer;
    step integer := 50000;
    maxid integer;
    updated bigint := 0;
BEGIN
    SELECT max(race_id) INTO maxid FROM race;
    WHILE lo <= maxid LOOP
        hi := lo + step;
        UPDATE entry e
           SET race_date = r.race_date
          FROM race r
         WHERE r.race_id = e.race_id
           AND e.race_id >= lo
           AND e.race_id < hi
           AND e.race_date IS DISTINCT FROM r.race_date;
        GET DIAGNOSTICS updated = ROW_COUNT;
        RAISE NOTICE 'race_id [%, %): % rows', lo, hi, updated;
        lo := hi;
    END LOOP;
END $$;
