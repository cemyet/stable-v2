"""
stable-v2 schema.

Five flat global master tables — one row per canonical concept:
    horse, person, race, entry, track

Each master row carries:
    - our own canonical SERIAL id (e.g. horse_id)
    - per-source nullable id columns (st_id, atg_id, usta_id, letrot_id, ...)
    - canonical fields (name, dates, etc.) — populated from the highest-priority
      source available, with explicit priority rules in etl.matching
    - a `source_data` JSONB with source-specific extras we don't promote
    - `primary_source` (which source we trust for canonical fields)
    - `last_updated_at`

Plus per-source rolling buffers (`<source>_buffer`) used for retry/debug only.
Buffers are pruned to BUFFER_RETENTION_DAYS by the scraper at job tail.

NO bulk raw tables. Historical raw data lives in v1 (`jakob`) and is reachable
from v2 via the `v1_raw.*` foreign-data-wrapper schema (set up by core.fdw).
"""

from __future__ import annotations

from .config import KNOWN_SOURCES


SCHEMA_DDL = r"""
-- =====================================================================
-- HORSE — one row per canonical horse
-- =====================================================================
CREATE TABLE IF NOT EXISTS horse (
    horse_id            SERIAL PRIMARY KEY,

    -- Per-source ids (nullable, unique).
    st_id               INTEGER,                   -- TravSport (was v1's horse.horse_id)
    atg_id              TEXT,                      -- ATG horse identifier
    usta_id             TEXT,                      -- USTA Pathway id
    letrot_id           TEXT,                      -- Le Trot id (France)
    kmtid_id            TEXT,                      -- kmtid.atgx.se id

    -- Cross-source identifiers when present
    registration_number TEXT,
    ueln_number         TEXT,                      -- Universal Equine Life Number

    -- Canonical horse fields (priority promotion: see etl.matching)
    name                VARCHAR(200),
    date_of_birth       DATE,
    gender_code         VARCHAR(2),                -- 'H' / 'V' / 'S'
    color               VARCHAR(60),
    breed_code          VARCHAR(2),                -- 'V' (varmblod) / 'K' (kallblod)
    birth_country       CHAR(2),
    bred_country        CHAR(2),
    registration_country CHAR(2),
    is_dead             BOOLEAN,
    is_guest_horse      BOOLEAN,
    has_offspring       BOOLEAN,
    breed_index         VARCHAR(20),               -- raw text, semantics vary
    inbreed_coefficient VARCHAR(20),               -- raw text

    -- Pedigree as canonical horse_ids in this same table.
    sire_id             INTEGER REFERENCES horse(horse_id),
    dam_id              INTEGER REFERENCES horse(horse_id),
    -- Snapshot of the parent's name when we don't (yet) have a canonical row
    -- for the parent. This is enough for the frontend to display the pedigree.
    sire_name           VARCHAR(200),
    dam_name            VARCHAR(200),

    -- Lifetime stats per source (best-effort snapshots; not authoritative).
    -- We keep them to render horse pages without recomputing from `entry`.
    scraped_starts          INTEGER,
    scraped_wins            INTEGER,
    scraped_prize_money_kr  BIGINT,
    scraped_record          VARCHAR(120),

    -- Source-specific extras: { "st": {...}, "usta": {...}, ... }
    source_data         JSONB,

    primary_source      VARCHAR(20),
    last_updated_at     TIMESTAMP DEFAULT NOW()
);

-- Per-source uniqueness — only when the column is populated.
CREATE UNIQUE INDEX IF NOT EXISTS horse_st_id_uk     ON horse (st_id)     WHERE st_id     IS NOT NULL;
CREATE UNIQUE INDEX IF NOT EXISTS horse_atg_id_uk    ON horse (atg_id)    WHERE atg_id    IS NOT NULL;
CREATE UNIQUE INDEX IF NOT EXISTS horse_usta_id_uk   ON horse (usta_id)   WHERE usta_id   IS NOT NULL;
CREATE UNIQUE INDEX IF NOT EXISTS horse_letrot_id_uk ON horse (letrot_id) WHERE letrot_id IS NOT NULL;
CREATE UNIQUE INDEX IF NOT EXISTS horse_kmtid_id_uk  ON horse (kmtid_id)  WHERE kmtid_id  IS NOT NULL;

-- New foreign sources (added late — use ALTER for idempotency).
ALTER TABLE horse ADD COLUMN IF NOT EXISTS hvt_id     TEXT;
ALTER TABLE horse ADD COLUMN IF NOT EXISTS breedly_id TEXT;
CREATE UNIQUE INDEX IF NOT EXISTS horse_hvt_id_uk     ON horse (hvt_id)     WHERE hvt_id     IS NOT NULL;
CREATE UNIQUE INDEX IF NOT EXISTS horse_breedly_id_uk ON horse (breedly_id) WHERE breedly_id IS NOT NULL;

-- Cross-source matching helpers
CREATE UNIQUE INDEX IF NOT EXISTS horse_reg_uk  ON horse (registration_number) WHERE registration_number IS NOT NULL;
CREATE UNIQUE INDEX IF NOT EXISTS horse_ueln_uk ON horse (ueln_number)         WHERE ueln_number         IS NOT NULL;

-- Pedigree lookup
CREATE INDEX IF NOT EXISTS horse_sire_idx ON horse (sire_id) WHERE sire_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS horse_dam_idx  ON horse (dam_id)  WHERE dam_id  IS NOT NULL;

-- Search
CREATE INDEX IF NOT EXISTS horse_name_trgm ON horse USING gin (name gin_trgm_ops);
CREATE INDEX IF NOT EXISTS horse_dob_idx   ON horse (date_of_birth);

-- =====================================================================
-- PERSON — drivers, trainers, owners, breeders
-- =====================================================================
CREATE TABLE IF NOT EXISTS person (
    person_id           SERIAL PRIMARY KEY,

    st_id               INTEGER,                   -- TravSport license id
    atg_id              TEXT,                      -- ATG person id
    usta_id             TEXT,
    letrot_id           TEXT,

    name                VARCHAR(200),
    short_name          VARCHAR(40),               -- ATG abbrev, e.g. "Goo Bj"
    person_type         VARCHAR(20),               -- 'NATURAL' / 'LEGAL'
    license_country     CHAR(2),

    -- Roles — a person can hold multiple at once.
    is_driver           BOOLEAN NOT NULL DEFAULT FALSE,
    is_trainer          BOOLEAN NOT NULL DEFAULT FALSE,
    is_owner            BOOLEAN NOT NULL DEFAULT FALSE,
    is_breeder          BOOLEAN NOT NULL DEFAULT FALSE,

    source_data         JSONB,

    primary_source      VARCHAR(20),
    last_updated_at     TIMESTAMP DEFAULT NOW()
);

CREATE UNIQUE INDEX IF NOT EXISTS person_st_id_uk     ON person (st_id)     WHERE st_id     IS NOT NULL;
CREATE UNIQUE INDEX IF NOT EXISTS person_atg_id_uk    ON person (atg_id)    WHERE atg_id    IS NOT NULL;
CREATE UNIQUE INDEX IF NOT EXISTS person_usta_id_uk   ON person (usta_id)   WHERE usta_id   IS NOT NULL;
CREATE UNIQUE INDEX IF NOT EXISTS person_letrot_id_uk ON person (letrot_id) WHERE letrot_id IS NOT NULL;

ALTER TABLE person ADD COLUMN IF NOT EXISTS hvt_id TEXT;
CREATE UNIQUE INDEX IF NOT EXISTS person_hvt_id_uk ON person (hvt_id) WHERE hvt_id IS NOT NULL;

CREATE INDEX IF NOT EXISTS person_name_trgm ON person USING gin (name gin_trgm_ops);

-- =====================================================================
-- TRACK — one row per canonical track worldwide
-- =====================================================================
CREATE TABLE IF NOT EXISTS track (
    track_id            SERIAL PRIMARY KEY,

    st_code             VARCHAR(8),                -- 'S' (Solvalla), 'F' (Färjestad), ...
    atg_track_id        INTEGER,                   -- ATG numeric id (e.g. 32)
    usta_id             TEXT,
    letrot_id           TEXT,

    name                VARCHAR(120),
    country             CHAR(2),
    sport               VARCHAR(20),               -- 'trot' / 'gallop'

    first_seen_at       DATE,
    last_seen_at        DATE,

    source_data         JSONB,
    primary_source      VARCHAR(20),
    last_updated_at     TIMESTAMP DEFAULT NOW()
);

CREATE UNIQUE INDEX IF NOT EXISTS track_st_code_uk    ON track (st_code)      WHERE st_code      IS NOT NULL;
CREATE UNIQUE INDEX IF NOT EXISTS track_atg_id_uk     ON track (atg_track_id) WHERE atg_track_id IS NOT NULL;
CREATE UNIQUE INDEX IF NOT EXISTS track_usta_id_uk    ON track (usta_id)      WHERE usta_id      IS NOT NULL;
CREATE UNIQUE INDEX IF NOT EXISTS track_letrot_id_uk  ON track (letrot_id)    WHERE letrot_id    IS NOT NULL;
CREATE INDEX IF NOT EXISTS track_country_idx ON track (country);

ALTER TABLE track ADD COLUMN IF NOT EXISTS hvt_id TEXT;
CREATE UNIQUE INDEX IF NOT EXISTS track_hvt_id_uk ON track (hvt_id) WHERE hvt_id IS NOT NULL;

-- Static physical attributes of the track (oval geometry + start equipment).
-- Populated from cross-checked public sources (track operators, Svensk Travsport,
-- Wikipedia). Mostly meaningful for oval trotting tracks; foreign / non-oval
-- courses (e.g. Vincennes) may leave these NULL. physical_source carries the
-- per-attribute provenance + notes so the figures stay auditable.
ALTER TABLE track ADD COLUMN IF NOT EXISTS track_length_m     INTEGER;  -- lap length (varvets längd), metres
ALTER TABLE track ADD COLUMN IF NOT EXISTS home_stretch_m     INTEGER;  -- home/final stretch (upploppets längd), metres
ALTER TABLE track ADD COLUMN IF NOT EXISTS num_open_stretches SMALLINT; -- number of open stretches (öppna linjer / sprinterspår)
ALTER TABLE track ADD COLUMN IF NOT EXISTS track_width_m      INTEGER;  -- racing-surface width, metres
ALTER TABLE track ADD COLUMN IF NOT EXISTS auto_car_wings     TEXT;     -- mobile-start (auto) car wing type, e.g. 'standard' / 'wide'
ALTER TABLE track ADD COLUMN IF NOT EXISTS surface            TEXT;     -- 'sand' / 'dirt' / 'allweather' / 'grass'
ALTER TABLE track ADD COLUMN IF NOT EXISTS shape              TEXT;     -- 'oval' / 'tri-oval' / 'irregular'
ALTER TABLE track ADD COLUMN IF NOT EXISTS opened_year        SMALLINT; -- year the track opened
ALTER TABLE track ADD COLUMN IF NOT EXISTS physical_source    JSONB;    -- per-attribute source urls + notes

-- =====================================================================
-- RACE — one row per canonical race
-- =====================================================================
CREATE TABLE IF NOT EXISTS race (
    race_id             SERIAL PRIMARY KEY,

    -- Per-source ids
    st_race_id          INTEGER,                   -- v1 race.race_id
    st_race_day_id      INTEGER,                   -- v1 race_day.race_day_id (TravSport raceday)
    atg_race_id         TEXT,                      -- e.g. '2025-04-26_32_5'
    atg_race_day_id     TEXT,                      -- e.g. '2025-04-26_32'
    usta_race_id        TEXT,
    letrot_race_id      TEXT,

    -- Canonical race fields
    race_date           DATE,
    track_id            INTEGER REFERENCES track(track_id),
    race_number         INTEGER,
    start_time          TIMESTAMPTZ,
    distance            INTEGER,                   -- meters; majority distance for the race
    start_method        CHAR(1),                   -- 'V' (volt) / 'A' (auto)
    status              VARCHAR(20),               -- 'upcoming' / 'paused' / 'results' / 'cancelled'
    heading             VARCHAR(400),
    proposition_text    TEXT,
    track_conditions    VARCHAR(120),
    victory_margin      VARCHAR(60),
    tempo_text          TEXT,
    total_prize_kr      BIGINT,

    -- Pool tags (game types this race is part of)
    pool_types          TEXT[],                    -- e.g. {V75,V5,DD}

    -- Race-class metadata parsed from terms / proposition
    race_class          VARCHAR(120),              -- 'Stodivisionen', 'Klass I', ...
    age_requirement     VARCHAR(120),
    earnings_range      VARCHAR(120),
    driver_requirement  VARCHAR(120),

    source_data         JSONB,
    primary_source      VARCHAR(20),
    last_updated_at     TIMESTAMP DEFAULT NOW()
);

CREATE UNIQUE INDEX IF NOT EXISTS race_st_id_uk      ON race (st_race_id)     WHERE st_race_id     IS NOT NULL;
CREATE UNIQUE INDEX IF NOT EXISTS race_atg_id_uk     ON race (atg_race_id)    WHERE atg_race_id    IS NOT NULL;
CREATE UNIQUE INDEX IF NOT EXISTS race_usta_id_uk    ON race (usta_race_id)   WHERE usta_race_id   IS NOT NULL;
CREATE UNIQUE INDEX IF NOT EXISTS race_letrot_id_uk  ON race (letrot_race_id) WHERE letrot_race_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS race_date_idx       ON race (race_date);
CREATE INDEX IF NOT EXISTS race_track_date_idx ON race (track_id, race_date);
CREATE INDEX IF NOT EXISTS race_pools_gin      ON race USING gin (pool_types);
CREATE INDEX IF NOT EXISTS race_st_day_idx     ON race (st_race_day_id) WHERE st_race_day_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS race_atg_day_idx    ON race (atg_race_day_id) WHERE atg_race_day_id IS NOT NULL;

-- kmtid race id (e.g. '2026-05-09_6_1' = date_atgTrackId_raceNumber).
ALTER TABLE race ADD COLUMN IF NOT EXISTS kmtid_id TEXT;
CREATE UNIQUE INDEX IF NOT EXISTS race_kmtid_id_uk ON race (kmtid_id) WHERE kmtid_id IS NOT NULL;

-- HVT raceday id, when known (foreign source — German raceday).
ALTER TABLE race ADD COLUMN IF NOT EXISTS hvt_race_id TEXT;
CREATE UNIQUE INDEX IF NOT EXISTS race_hvt_id_uk ON race (hvt_race_id) WHERE hvt_race_id IS NOT NULL;

-- =====================================================================
-- ENTRY — one row per (race, horse). The workhorse table.
-- =====================================================================
CREATE TABLE IF NOT EXISTS entry (
    entry_id            BIGSERIAL PRIMARY KEY,
    race_id             INTEGER NOT NULL REFERENCES race(race_id) ON DELETE CASCADE,
    horse_id            INTEGER NOT NULL REFERENCES horse(horse_id),

    -- Position / program
    program_number      SMALLINT,                  -- 1..15 typically; NULL / sentinel allowed
    post                SMALLINT,                  -- post position (volt order or auto rank)
    distance            INTEGER,                   -- per-horse distance
    tillagg             INTEGER,                   -- distance - MIN(distance) for the race

    -- Result
    placement           SMALLINT,                  -- 1..N; NULL when DNF/DQ/withdrawn
    finish_order        SMALLINT,                  -- raw ATG finishOrder
    placement_text      VARCHAR(40),               -- '1','d','gdk','utg','0'
    is_winner           BOOLEAN GENERATED ALWAYS AS (placement = 1) STORED,
    is_placed           BOOLEAN GENERATED ALWAYS AS (placement BETWEEN 1 AND 3) STORED,
    withdrawn           BOOLEAN NOT NULL DEFAULT FALSE,
    galopp              BOOLEAN NOT NULL DEFAULT FALSE,    -- gait break
    disqualified        BOOLEAN NOT NULL DEFAULT FALSE,

    -- Time
    time_seconds        REAL,                      -- numeric seconds per km (NULL when 0/parse fail)
    time_text           VARCHAR(16),               -- raw TravSport string '14,4a' / '13,2g'
    auto                BOOLEAN,                   -- TRUE = autostart, FALSE = volt, NULL = unknown

    -- Odds + money
    odds                NUMERIC(8,2),              -- final winner odds (NULL when 0)
    odds_plats_text     VARCHAR(20),               -- TravSport place odds, raw text
    prize_kr            BIGINT NOT NULL DEFAULT 0,

    -- People (canonical refs)
    driver_id           INTEGER REFERENCES person(person_id),
    driver_changed      BOOLEAN NOT NULL DEFAULT FALSE,
    trainer_id          INTEGER REFERENCES person(person_id),
    trainer_changed     BOOLEAN,                   -- true when trainer differs from horse's prior race

    -- Horse snapshot at race time
    age                 SMALLINT,
    sex                 CHAR(1),
    earnings_pre        BIGINT,                    -- lifetime earnings before this race

    -- Equipment
    shoe_code           VARCHAR(8),                -- 1=none / 2=back / 3=front / 4=all
    shoe_front_changed  BOOLEAN,
    shoe_back_changed   BOOLEAN,
    sulky               VARCHAR(8),                -- 'VA' / 'AM' / ...
    sulky_changed       BOOLEAN,

    source_data         JSONB,
    primary_source      VARCHAR(20),
    last_updated_at     TIMESTAMP DEFAULT NOW(),

    -- Denormalised copy of race.race_date. race_date physically lives on
    -- `race`, but nearly every per-person / per-horse query needs to filter
    -- or sort an entity's entries by date. Without the date on `entry` those
    -- queries must read ALL of an entity's entries and join `race` (or scan
    -- `race` by date and probe entries) — seconds per page. Kept perfectly in
    -- sync by the triggers below, so query *results* are unchanged.
    race_date           DATE,

    UNIQUE (race_id, horse_id)
);
-- Existing DBs created `entry` before race_date existed.
ALTER TABLE entry ADD COLUMN IF NOT EXISTS race_date DATE;

-- Keep entry.race_date in lock-step with race.race_date on every write path
-- (ETL upserts, identity merges, race re-parents, manual fixes) so we never
-- have to remember to set it. BEFORE trigger fills it on insert / re-parent;
-- the race-side trigger propagates rare race_date corrections downward.
CREATE OR REPLACE FUNCTION entry_set_race_date() RETURNS trigger AS $$
BEGIN
    SELECT r.race_date INTO NEW.race_date FROM race r WHERE r.race_id = NEW.race_id;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS entry_race_date_sync ON entry;
CREATE TRIGGER entry_race_date_sync
    BEFORE INSERT OR UPDATE OF race_id ON entry
    FOR EACH ROW EXECUTE FUNCTION entry_set_race_date();

CREATE OR REPLACE FUNCTION race_propagate_race_date() RETURNS trigger AS $$
BEGIN
    IF NEW.race_date IS DISTINCT FROM OLD.race_date THEN
        UPDATE entry SET race_date = NEW.race_date WHERE race_id = NEW.race_id;
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS race_race_date_propagate ON race;
CREATE TRIGGER race_race_date_propagate
    AFTER UPDATE OF race_date ON race
    FOR EACH ROW EXECUTE FUNCTION race_propagate_race_date();

-- Composite (entity, race_date DESC, entry_id DESC) indexes. These supersede
-- the old single-column id indexes (a leading-column lookup still works), so
-- write cost is unchanged while date-ordered / date-bounded history queries
-- become tight index range scans instead of full-history scan + sort.
CREATE INDEX IF NOT EXISTS entry_driver_date_idx
    ON entry (driver_id, race_date DESC NULLS LAST, entry_id DESC)  WHERE driver_id  IS NOT NULL;
CREATE INDEX IF NOT EXISTS entry_trainer_date_idx
    ON entry (trainer_id, race_date DESC NULLS LAST, entry_id DESC) WHERE trainer_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS entry_horse_date_idx
    ON entry (horse_id, race_date DESC NULLS LAST, entry_id DESC);
-- Covering indexes for the trainer/driver history aggregates. The composite
-- indexes above locate a person's rows quickly but the aggregate columns still
-- live in the heap, so the monthly-series (3 yr), top-horses (career) and
-- 30-day rolling-form scans each did thousands of random heap fetches per page
-- load (≈500 ms cold on Supabase). INCLUDE-ing those columns makes all three
-- index-only, cutting the cold cost 5-7×. entry_id is included so the
-- outperf/perf joins stay index-only on the entry side too.
CREATE INDEX IF NOT EXISTS entry_trainer_cover_idx
    ON entry (trainer_id, race_date)
    INCLUDE (entry_id, withdrawn, placement_text, disqualified, prize_kr, horse_id)
    WHERE trainer_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS entry_driver_cover_idx
    ON entry (driver_id, race_date)
    INCLUDE (entry_id, withdrawn, placement_text, disqualified)
    WHERE driver_id IS NOT NULL;

-- Recent-window leaderboard slice (all entries in the last N days).
CREATE INDEX IF NOT EXISTS entry_race_date_idx   ON entry (race_date);
CREATE INDEX IF NOT EXISTS entry_horse_winner_idx ON entry (horse_id)   WHERE placement = 1;
CREATE INDEX IF NOT EXISTS entry_horse_placed_idx ON entry (horse_id)   WHERE placement BETWEEN 1 AND 3;

-- Superseded by the composite indexes above (kept the DROP so existing DBs
-- shed the redundant single-column copies and don't pay double write cost).
DROP INDEX IF EXISTS entry_horse_idx;
DROP INDEX IF EXISTS entry_driver_idx;
DROP INDEX IF EXISTS entry_trainer_idx;

-- ---------------------------------------------------------------------------
-- Foreign-currency support: every prize is stored in SEK (`prize_kr`) for
-- consistency. For non-SEK sources we *also* keep the original amount + ccy
-- in the columns below, so we can recompute later if FX assumptions change.
-- ---------------------------------------------------------------------------
ALTER TABLE entry ADD COLUMN IF NOT EXISTS prize_currency CHAR(3) DEFAULT 'SEK';
ALTER TABLE entry ADD COLUMN IF NOT EXISTS prize_original NUMERIC(14,2);
ALTER TABLE entry ADD COLUMN IF NOT EXISTS prize_fx_rate  NUMERIC(12,6);
ALTER TABLE entry ADD COLUMN IF NOT EXISTS prize_fx_date  DATE;

-- ---------------------------------------------------------------------------
-- kmtid (atgx GPS) sectional data — promoted "hot" splits as columns,
-- full per-100m intervals retained in `kmtid_intervals` JSONB.
-- All times are in milliseconds (kmtid native unit).
-- ---------------------------------------------------------------------------
ALTER TABLE entry ADD COLUMN IF NOT EXISTS kmtid_first_200ms          REAL;
ALTER TABLE entry ADD COLUMN IF NOT EXISTS kmtid_last_200ms           REAL;
ALTER TABLE entry ADD COLUMN IF NOT EXISTS kmtid_best_100ms           REAL;
ALTER TABLE entry ADD COLUMN IF NOT EXISTS kmtid_best_100_start_m     INTEGER;
ALTER TABLE entry ADD COLUMN IF NOT EXISTS kmtid_actual_distance_m    INTEGER;
ALTER TABLE entry ADD COLUMN IF NOT EXISTS kmtid_actual_km_time_ms    REAL;
ALTER TABLE entry ADD COLUMN IF NOT EXISTS kmtid_slipstream_distance_m INTEGER;
ALTER TABLE entry ADD COLUMN IF NOT EXISTS kmtid_intervals             JSONB;

CREATE INDEX IF NOT EXISTS entry_kmtid_present_idx
    ON entry (race_id) WHERE kmtid_actual_km_time_ms IS NOT NULL;

-- =====================================================================
-- ROLE-HISTORY tables — time-ranged owner / trainer / driver assignments
-- on a horse. Optional but useful for the frontend's history sections.
-- =====================================================================

CREATE TABLE IF NOT EXISTS horse_owner_history (
    horse_id        INTEGER NOT NULL REFERENCES horse(horse_id) ON DELETE CASCADE,
    owner_id        INTEGER REFERENCES person(person_id),
    owner_name      VARCHAR(200),
    ownership_form  VARCHAR(8),                    -- 'ÄG' / 'LE' / ...
    from_date       DATE NOT NULL,
    to_date         DATE,
    source          VARCHAR(20),
    PRIMARY KEY (horse_id, from_date)
);
CREATE INDEX IF NOT EXISTS horse_owner_history_owner_idx ON horse_owner_history (owner_id);
CREATE INDEX IF NOT EXISTS horse_owner_history_curr_idx  ON horse_owner_history (horse_id) WHERE to_date IS NULL;

CREATE TABLE IF NOT EXISTS horse_trainer_history (
    horse_id        INTEGER NOT NULL REFERENCES horse(horse_id) ON DELETE CASCADE,
    trainer_id      INTEGER REFERENCES person(person_id),
    trainer_name    VARCHAR(200),
    license_text    VARCHAR(200),
    from_date       DATE NOT NULL,
    to_date         DATE,
    source          VARCHAR(20),
    PRIMARY KEY (horse_id, from_date)
);
CREATE INDEX IF NOT EXISTS horse_trainer_history_trainer_idx ON horse_trainer_history (trainer_id);
CREATE INDEX IF NOT EXISTS horse_trainer_history_curr_idx    ON horse_trainer_history (horse_id) WHERE to_date IS NULL;

-- =====================================================================
-- JOB-RUN bookkeeping (admin update job tracking — same shape as v1)
-- =====================================================================

CREATE TABLE IF NOT EXISTS job_run (
    job_run_id    SERIAL PRIMARY KEY,
    job_name      VARCHAR(40) NOT NULL DEFAULT 'update',
    started_at    TIMESTAMP NOT NULL DEFAULT NOW(),
    finished_at   TIMESTAMP,
    status        VARCHAR(20) NOT NULL DEFAULT 'running',  -- running / success / failed
    summary       JSONB DEFAULT '{}'::jsonb,
    log           TEXT DEFAULT '',
    phase         VARCHAR(120),                            -- human label of the current phase
    pid           INTEGER
);
CREATE INDEX IF NOT EXISTS job_run_started_idx ON job_run (started_at DESC);
ALTER TABLE job_run ADD COLUMN IF NOT EXISTS phase VARCHAR(120);

-- =====================================================================
-- MERGE LOGS — audit trail + rollback support for horse/person merges.
-- Every merge (scripted or manual via admin UI) writes one row here.
-- `from_snapshot` is the complete horse/person row at merge time so we
-- can rollback by re-inserting it and pointing entries back.
-- =====================================================================

CREATE TABLE IF NOT EXISTS horse_merge_log (
    merge_id      SERIAL PRIMARY KEY,
    from_horse_id INTEGER NOT NULL,
    to_horse_id   INTEGER NOT NULL,
    reason        TEXT    NOT NULL,
    method        VARCHAR(40) NOT NULL,           -- 'category_b' / 'pedigree' / 'manual' / 'category_a' / ...
    entries_moved INTEGER NOT NULL DEFAULT 0,
    conflicts_resolved INTEGER NOT NULL DEFAULT 0,
    from_snapshot JSONB   NOT NULL,
    merged_at     TIMESTAMP NOT NULL DEFAULT NOW(),
    merged_by     VARCHAR(80),                    -- 'system' / 'admin' / script name
    rolled_back   BOOLEAN NOT NULL DEFAULT FALSE,
    rolled_back_at TIMESTAMP
);
CREATE INDEX IF NOT EXISTS horse_merge_log_to_idx     ON horse_merge_log (to_horse_id);
CREATE INDEX IF NOT EXISTS horse_merge_log_from_idx   ON horse_merge_log (from_horse_id);
CREATE INDEX IF NOT EXISTS horse_merge_log_merged_idx ON horse_merge_log (merged_at DESC);

CREATE TABLE IF NOT EXISTS person_merge_log (
    merge_id       SERIAL PRIMARY KEY,
    from_person_id INTEGER NOT NULL,
    to_person_id   INTEGER NOT NULL,
    reason         TEXT    NOT NULL,
    method         VARCHAR(40) NOT NULL,
    entries_moved  INTEGER NOT NULL DEFAULT 0,
    from_snapshot  JSONB   NOT NULL,
    merged_at      TIMESTAMP NOT NULL DEFAULT NOW(),
    merged_by      VARCHAR(80),
    rolled_back    BOOLEAN NOT NULL DEFAULT FALSE,
    rolled_back_at TIMESTAMP
);
CREATE INDEX IF NOT EXISTS person_merge_log_to_idx   ON person_merge_log (to_person_id);
CREATE INDEX IF NOT EXISTS person_merge_log_from_idx ON person_merge_log (from_person_id);

-- =====================================================================
-- IDENTITY REDIRECTS — make manual/scripted merges PERMANENT.
-- A merge moves FKs and deletes the losing row, but its source ids and
-- synthetic name keys would otherwise be re-minted into a fresh duplicate
-- on the next import. We register every losing key here pointing at the
-- keeper; resolve_*/upsert_* consult this table BEFORE inserting so future
-- imports of the same horse/person land on the canonical row instead.
--   entity     : 'horse' | 'person'
--   source     : the source the key belongs to ('st','atg','letrot',...)
--                or 'synth' for normalized-name synthetic keys.
--   source_key : the source_id value (numeric-as-text) or 'x:CC:NAME'.
-- =====================================================================

CREATE TABLE IF NOT EXISTS identity_redirect (
    entity      VARCHAR(10) NOT NULL,
    source      VARCHAR(20) NOT NULL,
    source_key  TEXT        NOT NULL,
    to_id       INTEGER     NOT NULL,
    created_at  TIMESTAMP   NOT NULL DEFAULT NOW(),
    PRIMARY KEY (entity, source, source_key)
);
CREATE INDEX IF NOT EXISTS identity_redirect_to_idx ON identity_redirect (entity, to_id);

-- =====================================================================
-- DERIVED VIEWS — compute on read, do NOT materialise
-- =====================================================================

-- MATERIALIZED VIEW so offspring / search / leaderboard queries don't
-- re-aggregate the 4M-row entry table on every page load.
-- Refresh after each update run via: REFRESH MATERIALIZED VIEW CONCURRENTLY horse_career_stats;
DROP MATERIALIZED VIEW IF EXISTS horse_career_stats CASCADE;
CREATE MATERIALIZED VIEW horse_career_stats AS
WITH computed AS (
    SELECT
        e.horse_id,
        COUNT(*) FILTER (
            WHERE NOT e.withdrawn
              AND COALESCE(e.placement_text, '') !~ '^(gdk|ejg|ejp|gd|gk|egk|egdk|gkd|gdj|gdb|gdek|gdgk|gdl|ddk|frdk|erj|ejk|ejgk|ejgd|ej|EJ|EJG|EJP|GDK|Gdk|g)[0-9]?$|^[12]?[pP][0-9]?$'
        )                                                   AS starts,
        COUNT(*) FILTER (
            WHERE e.placement_text = '1'
              AND NOT COALESCE(e.disqualified, false)
        )                                                   AS wins,
        COUNT(*) FILTER (WHERE e.placement_text = '2'
              AND NOT COALESCE(e.disqualified, false))      AS seconds,
        COUNT(*) FILTER (WHERE e.placement_text = '3'
              AND NOT COALESCE(e.disqualified, false))      AS thirds,
        COUNT(*) FILTER (
            WHERE e.placement_text IN ('1','2','3')
              AND NOT COALESCE(e.disqualified, false)
        )                                                   AS placed,
        COUNT(*) FILTER (
            WHERE COALESCE(e.placement_text, '') ~ '^(gdk|ejg|ejp|gd|gk|egk|egdk|gkd|gdj|gdb|gdek|gdgk|gdl|ddk|frdk|erj|ejk|ejgk|ejgd|ej|EJ|EJG|EJP|GDK|Gdk|g)[0-9]?$|^[12]?[pP][0-9]?$'
        )                                                   AS qualifiers,
        COALESCE(SUM(e.prize_kr), 0)                        AS prize_money_kr,
        MIN(r.race_date)                                    AS first_start,
        MAX(r.race_date)                                    AS last_start
    FROM entry e
    LEFT JOIN race r ON r.race_id = e.race_id
    GROUP BY e.horse_id
)
SELECT
    h.horse_id,
    GREATEST(COALESCE(c.starts, 0),         COALESCE(h.scraped_starts, 0))         AS starts,
    GREATEST(COALESCE(c.wins, 0),           COALESCE(h.scraped_wins, 0))           AS wins,
    COALESCE(c.seconds, 0)                  AS seconds,
    COALESCE(c.thirds, 0)                   AS thirds,
    COALESCE(c.placed, 0)                   AS placed,
    COALESCE(c.qualifiers, 0)               AS qualifiers,
    GREATEST(COALESCE(c.prize_money_kr, 0), COALESCE(h.scraped_prize_money_kr, 0)) AS prize_money_kr,
    c.first_start,
    c.last_start
FROM horse h
LEFT JOIN computed c ON c.horse_id = h.horse_id
WHERE c.horse_id IS NOT NULL OR h.scraped_starts IS NOT NULL;

CREATE UNIQUE INDEX ON horse_career_stats (horse_id);

-- Per-year horse stats for YTD leaderboards. This keeps offspring panels from
-- scanning all entry rows on every cache miss.
DROP MATERIALIZED VIEW IF EXISTS horse_year_stats CASCADE;
CREATE MATERIALIZED VIEW horse_year_stats AS
SELECT
    e.horse_id,
    EXTRACT(YEAR FROM r.race_date)::integer AS race_year,
    COUNT(*) FILTER (
        WHERE NOT e.withdrawn
          AND COALESCE(e.placement_text, '') !~ '^(gdk|ejg|ejp|gd|gk|egk|egdk|gkd|gdj|gdb|gdek|gdgk|gdl|ddk|frdk|erj|ejk|ejgk|ejgd|ej|EJ|EJG|EJP|GDK|Gdk|g)[0-9]?$|^[12]?[pP][0-9]?$'
    ) AS starts,
    COUNT(*) FILTER (
        WHERE e.placement_text = '1'
          AND NOT COALESCE(e.disqualified, false)
    ) AS wins,
    COALESCE(SUM(e.prize_kr), 0) AS prize_money_kr
FROM entry e
JOIN race r ON r.race_id = e.race_id
WHERE e.horse_id IS NOT NULL
GROUP BY e.horse_id, EXTRACT(YEAR FROM r.race_date)::integer;

CREATE UNIQUE INDEX ON horse_year_stats (race_year, horse_id);

-- Career starts/wins per person, per role. The home form leaderboard and the
-- driver/trainer pages previously recomputed these by seq-scanning the whole
-- 7.5M-row entry table on every cache miss (~5s per role). Precompute once per
-- update run instead; refresh via
--   REFRESH MATERIALIZED VIEW CONCURRENTLY person_career_stats;
-- The win/qualifier predicates below are byte-for-byte the same as the app's
-- _IS_WIN / _NOT_QUALIFIER, so displayed win rates are identical.
DROP MATERIALIZED VIEW IF EXISTS person_career_stats CASCADE;
CREATE MATERIALIZED VIEW person_career_stats AS
SELECT 'driver'::text AS role,
       e.driver_id     AS person_id,
       COUNT(*) FILTER (
           WHERE NOT e.withdrawn
             AND COALESCE(e.placement_text, '') !~ '^(gdk|ejg|ejp|gd|gk|egk|egdk|gkd|gdj|gdb|gdek|gdgk|gdl|ddk|frdk|erj|ejk|ejgk|ejgd|ej|EJ|EJG|EJP|GDK|Gdk|g)[0-9]?$|^[12]?[pP][0-9]?$'
       ) AS starts,
       COUNT(*) FILTER (
           WHERE e.placement_text = '1'
             AND NOT COALESCE(e.disqualified, false)
             AND COALESCE(e.placement_text, '') !~ '^(gdk|ejg|ejp|gd|gk|egk|egdk|gkd|gdj|gdb|gdek|gdgk|gdl|ddk|frdk|erj|ejk|ejgk|ejgd|ej|EJ|EJG|EJP|GDK|Gdk|g)[0-9]?$|^[12]?[pP][0-9]?$'
       ) AS wins
FROM entry e
WHERE e.driver_id IS NOT NULL
GROUP BY e.driver_id
UNION ALL
SELECT 'trainer'::text AS role,
       e.trainer_id     AS person_id,
       COUNT(*) FILTER (
           WHERE NOT e.withdrawn
             AND COALESCE(e.placement_text, '') !~ '^(gdk|ejg|ejp|gd|gk|egk|egdk|gkd|gdj|gdb|gdek|gdgk|gdl|ddk|frdk|erj|ejk|ejgk|ejgd|ej|EJ|EJG|EJP|GDK|Gdk|g)[0-9]?$|^[12]?[pP][0-9]?$'
       ) AS starts,
       COUNT(*) FILTER (
           WHERE e.placement_text = '1'
             AND NOT COALESCE(e.disqualified, false)
             AND COALESCE(e.placement_text, '') !~ '^(gdk|ejg|ejp|gd|gk|egk|egdk|gkd|gdj|gdb|gdek|gdgk|gdl|ddk|frdk|erj|ejk|ejgk|ejgd|ej|EJ|EJG|EJP|GDK|Gdk|g)[0-9]?$|^[12]?[pP][0-9]?$'
       ) AS wins
FROM entry e
WHERE e.trainer_id IS NOT NULL
GROUP BY e.trainer_id;

CREATE UNIQUE INDEX ON person_career_stats (role, person_id);

-- =====================================================================
-- TRACK_STATS — per-track aggregates for the track pages (galopp rates by
-- start method, win/field stats). Avoids scanning a track's entire entry
-- slice (Solvalla alone is ~300k rows) on every page load. The `started`
-- predicate is the same NOT-withdrawn / NOT-qualifier filter used by
-- person_career_stats, so counts are consistent across the app.
-- Refresh via: REFRESH MATERIALIZED VIEW CONCURRENTLY track_stats;
-- =====================================================================
DROP MATERIALIZED VIEW IF EXISTS track_stats CASCADE;
CREATE MATERIALIZED VIEW track_stats AS
SELECT r.track_id,
       COUNT(DISTINCT e.race_id)                                 AS races,
       MIN(r.race_date)                                          AS first_date,
       MAX(r.race_date)                                          AS last_date,
       COUNT(*) FILTER (WHERE q.started)                         AS starts,
       COUNT(*) FILTER (WHERE q.started AND e.galopp)            AS gals,
       COUNT(*) FILTER (WHERE q.started AND e.auto)              AS starts_auto,
       COUNT(*) FILTER (WHERE q.started AND e.auto AND e.galopp) AS gals_auto,
       COUNT(*) FILTER (WHERE q.started AND e.auto = false)              AS starts_volt,
       COUNT(*) FILTER (WHERE q.started AND e.auto = false AND e.galopp) AS gals_volt
FROM race r
JOIN entry e ON e.race_id = r.race_id
CROSS JOIN LATERAL (SELECT (
        NOT e.withdrawn
        AND COALESCE(e.placement_text, '') !~ '^(gdk|ejg|ejp|gd|gk|egk|egdk|gkd|gdj|gdb|gdek|gdgk|gdl|ddk|frdk|erj|ejk|ejgk|ejgd|ej|EJ|EJG|EJP|GDK|Gdk|g)[0-9]?$|^[12]?[pP][0-9]?$'
    ) AS started) q
WHERE r.track_id IS NOT NULL
GROUP BY r.track_id;

CREATE UNIQUE INDEX ON track_stats (track_id);

-- =====================================================================
-- ENTRY_FEATURES — point-in-time ML feature snapshot, one row per
-- qualifying `entry` (Swedish trot, race_date >= 2005-01-01).
--
-- This is a COMPLEMENT to `entry`, not a copy: it stores derived,
-- as-of-race-day features for model training. Every feature is computed
-- over ALL entries globally (so a trainer's form reflects their foreign
-- starts too), but we only store rows for the Swedish-trot 2005+ slice.
--
-- Point-in-time rule: every "as-of" feature uses only data from races
-- STRICTLY BEFORE this entry's race_date (same-day races excluded), so
-- there is no future-information leakage into training data.
--
-- Storage policy: raw values only (rates that are naturally 0..1 stay
-- as-is; absolute magnitudes like earnings/time stay raw). Encoding
-- (one-hot / target) and scaling (standardisation) are TRAINING-TIME
-- concerns handled by the slice/training pipeline, not baked in here.
--
-- Populated by scripts/backfill_entry_features.py (one-time, phased) and
-- maintained incrementally by jobs/update.py for newly-added races.
-- =====================================================================
CREATE TABLE IF NOT EXISTS entry_features (
    -- Keys (NOT trainable features — used to join back to `entry`).
    entry_id            BIGINT PRIMARY KEY REFERENCES entry(entry_id) ON DELETE CASCADE,
    race_id             INTEGER NOT NULL,
    horse_id            INTEGER NOT NULL,
    trainer_id          INTEGER,
    driver_id           INTEGER,
    track_id            INTEGER,
    race_date           DATE    NOT NULL,

    -- Race context.
    is_auto             BOOLEAN,        -- TRUE autostart / FALSE volt
    distance_m          INTEGER,        -- per-horse distance
    distance_added_m    INTEGER,        -- tillagg (distance - race min)
    starters            SMALLINT,       -- non-withdrawn entries in the race
    race_complete       BOOLEAN,        -- field-completeness quality flag

    -- Post position.
    post                SMALLINT,
    is_springspar       BOOLEAN,        -- volt & post in 6..7 (favourable)
    is_bakspar          BOOLEAN,        -- auto & post >= 9 (second row)

    -- Horse snapshot at race time.
    age                 SMALLINT,
    sex                 CHAR(1),
    breed_code          VARCHAR(2),     -- 'V' varmblod / 'K' kallblod
    shoe_code           VARCHAR(8),     -- 1 barefoot-all / 2 bf-back / 3 bf-front / 4 shod
    is_barefoot_all     BOOLEAN,        -- shoe_code = '1'
    first_shoes_off     BOOLEAN,        -- first ever barefoot race (prev all shod)
    earnings_pre        BIGINT,         -- lifetime earnings before this race
    earnings_rank_in_race SMALLINT,     -- 1 = richest in field
    earnings_pct_in_race  REAL,         -- 0..1 percentile within field

    -- Horse career, as-of (strictly before race_date).
    horse_starts                INTEGER,
    horse_winrate               REAL,
    horse_galrate               REAL,
    horse_galrate_auto          REAL,
    horse_galrate_volt          REAL,
    horse_galrate_auto_volt_delta REAL, -- auto rate - volt rate
    horse_galrate_barefoot      REAL,   -- gal rate in barefoot (1/2/3) races
    gal_pre                     BOOLEAN, -- galopped in previous start
    days_since_last_start       INTEGER,
    season_debut                BOOLEAN, -- gap since last start > 60 days
    record_s_kmtime             REAL,    -- best km-time, short  (<= 1700 m)
    record_m_kmtime             REAL,    -- best km-time, medium (1701..2250 m)
    record_l_kmtime             REAL,    -- best km-time, long   (> 2250 m)

    -- Trainer, as-of (over ALL of the trainer's entries).
    trainer_starts          INTEGER,
    trainer_winrate         REAL,
    trainer_galrate         REAL,
    trainer_winrate_30d     REAL,        -- candidate form metric
    trainer_winrate_delta   REAL,        -- candidate form: 30d - career
    trainer_form_odds_30d   REAL,        -- candidate form: market-adjusted
    trainer_winrate_track   REAL,        -- win rate at this track
    races_with_trainer      INTEGER,     -- prior starts of THIS horse w/ trainer
    first_trainer           BOOLEAN,     -- debut with this trainer

    -- Driver, as-of (over ALL of the driver's entries).
    driver_starts           INTEGER,
    driver_winrate          REAL,
    driver_galrate          REAL,
    driver_winrate_30d      REAL,
    driver_winrate_delta    REAL,
    driver_form_odds_30d    REAL,
    races_with_driver       INTEGER,
    first_driver            BOOLEAN,

    -- First-time-experience flags for this horse.
    first_method            BOOLEAN,     -- first race in this start method
    first_distance_band     BOOLEAN,     -- first race in this distance band

    -- Labels (Y). Kept alongside features; pick X/Y at training time.
    y_gal                   BOOLEAN,     -- galopped in THIS race
    y_win                   BOOLEAN,
    y_top3                  BOOLEAN,
    y_placement             SMALLINT,    -- NULL when DNF/DQ/withdrawn
    y_disq                  BOOLEAN,
    y_time_s                REAL,        -- km-time this race (finishers only)
    y_time_z                REAL,        -- within-race standardised km-time

    -- Bookkeeping.
    features_version        SMALLINT NOT NULL DEFAULT 1,
    computed_at             TIMESTAMP NOT NULL DEFAULT NOW()
);

-- Phase 2 columns (expensive global partition passes). Added via ALTER so
-- the table can exist + be Phase-1-populated before these are computed, and
-- so existing DBs pick them up idempotently.
ALTER TABLE entry_features ADD COLUMN IF NOT EXISTS sire_galrate            REAL;
ALTER TABLE entry_features ADD COLUMN IF NOT EXISTS sire_starts             INTEGER;
ALTER TABLE entry_features ADD COLUMN IF NOT EXISTS dam_galrate             REAL;
ALTER TABLE entry_features ADD COLUMN IF NOT EXISTS dam_starts              INTEGER;
ALTER TABLE entry_features ADD COLUMN IF NOT EXISTS post_galrate_auto       REAL;
ALTER TABLE entry_features ADD COLUMN IF NOT EXISTS post_galrate_volt       REAL;
ALTER TABLE entry_features ADD COLUMN IF NOT EXISTS post_galrate_auto_track REAL;
ALTER TABLE entry_features ADD COLUMN IF NOT EXISTS post_galrate_volt_track REAL;

-- Horse-combination as-of galopp rates + reliability counts, and seasonality.
-- horse_trainer_galrate: this horse's gal rate while trained by THIS trainer
--   (as-of, strictly prior starts) — captures "this pairing keeps it together".
-- horse_galrate_track:   this horse's gal rate at THIS track (as-of) — track
--   shape / final-stretch / start method suit some horses more than others.
-- *_starts are the sample sizes so the tree can discount thin histories.
-- Computed in phase 2 (same as-of machinery as sire/dam/post rates).
ALTER TABLE entry_features ADD COLUMN IF NOT EXISTS horse_trainer_galrate   REAL;
ALTER TABLE entry_features ADD COLUMN IF NOT EXISTS horse_trainer_starts    INTEGER;
ALTER TABLE entry_features ADD COLUMN IF NOT EXISTS horse_galrate_track     REAL;
ALTER TABLE entry_features ADD COLUMN IF NOT EXISTS horse_track_starts      INTEGER;
-- race_month (1-12): seasonality. Set in phase 1 (deterministic from race_date)
-- and also backfilled in phase 2 so a phase-2-only run fills pre-existing rows.
ALTER TABLE entry_features ADD COLUMN IF NOT EXISTS race_month              SMALLINT;

-- Phase 3: recency + field-level features (added for gal-risk model tuning).
ALTER TABLE entry_features ADD COLUMN IF NOT EXISTS gal_recent_5           SMALLINT;  -- gals in last 5 qualifying starts
ALTER TABLE entry_features ADD COLUMN IF NOT EXISTS field_avg_galrate      REAL;      -- avg career gal rate of opponents in race

-- Phase 4: sequential streak + instability + relative-rank features.
ALTER TABLE entry_features ADD COLUMN IF NOT EXISTS gal_streak             SMALLINT;  -- consecutive breaks immediately before this start
ALTER TABLE entry_features ADD COLUMN IF NOT EXISTS clean_streak           SMALLINT;  -- consecutive clean starts immediately before
ALTER TABLE entry_features ADD COLUMN IF NOT EXISTS method_switch          BOOLEAN;   -- start method differs from last start
ALTER TABLE entry_features ADD COLUMN IF NOT EXISTS barefoot_change        BOOLEAN;   -- barefoot-all status differs from last start
ALTER TABLE entry_features ADD COLUMN IF NOT EXISTS distance_delta_vs_last INTEGER;   -- distance_m minus last start's distance_m
ALTER TABLE entry_features ADD COLUMN IF NOT EXISTS galrate_method         REAL;      -- horse gal rate for THIS race's method
ALTER TABLE entry_features ADD COLUMN IF NOT EXISTS gal_rank_in_field      REAL;      -- percent-rank of horse_galrate within the race

-- Phase 5: recent-form rate + workload + driver instability.
ALTER TABLE entry_features ADD COLUMN IF NOT EXISTS galrate_recent_10      REAL;      -- mean gal over last 10 prior starts (recent form rate)
ALTER TABLE entry_features ADD COLUMN IF NOT EXISTS starts_90d             SMALLINT;  -- starts in the prior 90 days (workload / freshness)
ALTER TABLE entry_features ADD COLUMN IF NOT EXISTS driver_switch          BOOLEAN;   -- driver differs from last start

-- Gal-adjusted win rates (as-of, strictly prior starts), companions to
-- horse_winrate. These answer "how often does this horse win when it trots
-- cleanly?" by removing gait-breaks/DQs from the denominator.
--   horse_winrate_gal_adj:  wins / starts EXCLUDING galopp AND disqualified races.
--   horse_winrate_gal_incl: wins / starts excluding ONLY disqualified races
--                           (non-disqualifying galopp results stay in the rate).
-- Computed in phase 1 alongside horse_winrate / horse_galrate.
ALTER TABLE entry_features ADD COLUMN IF NOT EXISTS horse_winrate_gal_adj  REAL;
ALTER TABLE entry_features ADD COLUMN IF NOT EXISTS horse_winrate_gal_incl REAL;

CREATE INDEX IF NOT EXISTS entry_features_race_idx     ON entry_features (race_id);
CREATE INDEX IF NOT EXISTS entry_features_date_idx     ON entry_features (race_date);
CREATE INDEX IF NOT EXISTS entry_features_horse_idx    ON entry_features (horse_id);
CREATE INDEX IF NOT EXISTS entry_features_complete_idx ON entry_features (race_date) WHERE race_complete;
CREATE INDEX IF NOT EXISTS entry_features_track_idx    ON entry_features (track_id);

-- =====================================================================
-- ML_MODEL — registry of trained models (metadata + evaluation metrics).
-- The serialized estimator lives on disk (artifact_path); everything the
-- models page needs to render is stored here as JSONB.
-- =====================================================================
CREATE TABLE IF NOT EXISTS ml_model (
    model_id      SERIAL PRIMARY KEY,
    name          TEXT NOT NULL,
    scope         TEXT NOT NULL DEFAULT 'general',  -- 'general' | 'track'
    track_id      INTEGER,
    track_name    TEXT,
    target        TEXT NOT NULL,                     -- e.g. 'y_gal'
    algo          TEXT NOT NULL,                     -- e.g. 'HistGradientBoosting'
    slice_def     JSONB NOT NULL DEFAULT '{}'::jsonb,-- filters + feature list + counts
    metrics       JSONB NOT NULL DEFAULT '{}'::jsonb,-- auc, logloss, importances, ...
    artifact_path TEXT,
    created_at    TIMESTAMP NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS ml_model_scope_idx ON ml_model (scope, track_id);

-- =====================================================================
-- ML_SLICE — saved training slices (filters + feature/target selection).
-- Lets the training page launch a run without re-specifying the slice.
-- =====================================================================
CREATE TABLE IF NOT EXISTS ml_slice (
    slice_id    SERIAL PRIMARY KEY,
    name        TEXT NOT NULL,
    table_name  TEXT NOT NULL DEFAULT 'entry_features',
    filters     JSONB NOT NULL DEFAULT '{}'::jsonb,  -- track_ids, years, method, complete, notnull, starters_only
    target      TEXT,                                 -- default Y column
    features    JSONB NOT NULL DEFAULT '[]'::jsonb,   -- default X columns
    created_at  TIMESTAMP NOT NULL DEFAULT NOW()
);

-- =====================================================================
-- ML_PREDICTION — model scores for entries (e.g. upcoming-race xgal, the
-- expected/predicted galopp probability).
-- Precomputed by scripts/score_upcoming.py so the web app can read them
-- without loading the ML stack. One row per (entry, target).
-- Targets: y_gal_general (best any-method model), y_gal_track (track model or general).
-- =====================================================================
CREATE TABLE IF NOT EXISTS ml_prediction (
    entry_id    BIGINT NOT NULL,
    target      TEXT NOT NULL,
    model_id    INTEGER,
    prob        REAL,
    updated_at  TIMESTAMP NOT NULL DEFAULT NOW(),
    PRIMARY KEY (entry_id, target)
);

-- =====================================================================
-- ENTRY_OUTPERF — per-entry market outperformance ("s_form" building block).
-- Within each race we rank runners by odds (market's expected order) and by
-- finish_order (actual order); market_outperf = (odds_rank - fin_rank)/(n-1),
-- so >0 means the horse beat its market rank. Averaging this over a person's
-- recent starts gives s_form (stallform) — see web/app.py form helpers.
-- Populated by scripts/refresh_entry_outperf.py (full + incremental); only
-- rankable starts (valid odds + finish_order, field >= 4) get a row.
-- =====================================================================
CREATE TABLE IF NOT EXISTS entry_outperf (
    entry_id        BIGINT NOT NULL PRIMARY KEY,
    market_outperf  REAL NOT NULL,
    race_date       DATE
);
CREATE INDEX IF NOT EXISTS idx_entry_outperf_date ON entry_outperf (race_date);

-- =====================================================================
-- ENTRY_PERF — per-entry finishing percentile ("form" building block).
-- Market-AGNOSTIC counterpart to entry_outperf: within each race
-- perf = (n_field - fin_rank)/(n_field - 1) ∈ [0,1], 1 = won, and a
-- galopp/DQ/DNF (no finish_order) counts as a bottom finish (perf = 0).
-- Averaging this over a person's recent starts gives "form" (actual recent
-- performance), which — unlike s_form/mkt± — rewards winning favourites and
-- predicts winning. Populated by scripts/refresh_entry_perf.py. Uses every
-- SE start with a placement in a field of >= 4 (no odds gate → more data).
-- =====================================================================
CREATE TABLE IF NOT EXISTS entry_perf (
    entry_id    BIGINT NOT NULL PRIMARY KEY,
    perf        REAL NOT NULL,
    race_date   DATE
);
CREATE INDEX IF NOT EXISTS idx_entry_perf_date ON entry_perf (race_date);

-- =====================================================================
-- WATCHLIST — user's tracked horses (single-user app, no user_id needed).
-- =====================================================================
CREATE TABLE IF NOT EXISTS watchlist (
    horse_id    INTEGER NOT NULL REFERENCES horse(horse_id) PRIMARY KEY,
    added_at    TIMESTAMP NOT NULL DEFAULT NOW(),
    note        TEXT
);
"""

# Per-source rolling buffer DDL is generated from KNOWN_SOURCES so we don't
# have to repeat ourselves for each new source.
def _buffer_ddl(source: str) -> str:
    return f"""
CREATE TABLE IF NOT EXISTS {source}_buffer (
    buffer_id   BIGSERIAL PRIMARY KEY,
    fetched_at  TIMESTAMP NOT NULL DEFAULT NOW(),
    url         TEXT      NOT NULL,
    http_status SMALLINT,
    body        TEXT,
    parsed_ok   BOOLEAN
);
CREATE INDEX IF NOT EXISTS {source}_buffer_fetched_idx ON {source}_buffer (fetched_at);
"""


BUFFER_DDL = "\n".join(_buffer_ddl(s) for s in KNOWN_SOURCES)


# Durable raw store for the NATIVE Svensk Travsport (ST) scraper.
#
# This intentionally deviates from the "no bulk raw tables" policy stated at
# the top of this file: ST is the one source v2 now scrapes itself (rather than
# inheriting via the v1 bridge), so we keep its raw responses durably to be able
# to re-run the ETL after a parser change WITHOUT re-scraping ~800k horse pages.
# Shapes mirror v1's v2_horse_raw / v2_raceday_raw (/Users/jakob/Dev/stable).
# The 7-day {st_buffer} table is a debug/retry log only and is NOT a substitute.
ST_RAW_DDL = r"""
-- One row per (horse_id, data_type); data_type is a parse_horse_page blob key
-- e.g. 'horse-basic-information', 'horse-history', 'lineage-small',
-- 'horse-statistics', 'race-results'.
CREATE TABLE IF NOT EXISTS st_horse_raw (
    horse_id    INTEGER     NOT NULL,
    data_type   VARCHAR(50) NOT NULL,
    raw_json    JSONB,
    scraped_at  TIMESTAMP   NOT NULL DEFAULT NOW(),
    PRIMARY KEY (horse_id, data_type)
);
CREATE INDEX IF NOT EXISTS st_horse_raw_scraped_idx ON st_horse_raw (scraped_at);

-- One row per horse_id: latest fetch attempt (200 = passport present).
CREATE TABLE IF NOT EXISTS st_horse_scrape_log (
    horse_id      INTEGER   PRIMARY KEY,
    http_status   INTEGER,
    scraped_at    TIMESTAMP NOT NULL DEFAULT NOW(),
    error_message TEXT
);

-- One JSON blob per raceday (the full results API response).
CREATE TABLE IF NOT EXISTS st_raceday_raw (
    race_day_id INTEGER   PRIMARY KEY,
    raw_json    JSONB,
    scraped_at  TIMESTAMP NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS st_raceday_raw_scraped_idx ON st_raceday_raw (scraped_at);

CREATE TABLE IF NOT EXISTS st_raceday_scrape_log (
    race_day_id   INTEGER   PRIMARY KEY,
    http_status   INTEGER,
    scraped_at    TIMESTAMP NOT NULL DEFAULT NOW(),
    error_message TEXT
);
"""


# Durable raw store for the NATIVE ATG scraper (scrapers/atg.py).
#
# This replaces v1's `v2_atg_race_raw` table. Once v2 scrapes ATG itself we own
# the per-race raw JSON here so etl.import_atg.ingest_race can be re-run after a
# parser change WITHOUT re-scraping. The historical v1 archive can be copied in
# once (scripts/migrate_atg_raw_from_v1.py) so v1 can be fully retired.
# This lives LOCAL-only — the cloud/Supabase serving set never includes raw.
ATG_RAW_DDL = r"""
CREATE TABLE IF NOT EXISTS atg_race_raw (
    atg_race_id VARCHAR(64) PRIMARY KEY,
    raw_json    JSONB,
    race_date   DATE,
    scraped_at  TIMESTAMP NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS atg_race_raw_date_idx    ON atg_race_raw (race_date);
CREATE INDEX IF NOT EXISTS atg_race_raw_scraped_idx ON atg_race_raw (scraped_at);

CREATE TABLE IF NOT EXISTS atg_race_scrape_log (
    atg_race_id   VARCHAR(64) PRIMARY KEY,
    http_status   INTEGER,
    status        VARCHAR(20),
    scraped_at    TIMESTAMP NOT NULL DEFAULT NOW(),
    error_message TEXT
);
"""


NORMALIZE_NAME_FUNC_DDL = r"""
-- Mirror of core.identity.normalize_name(). Strip diacritics, * / ·,
-- trailing (XX), trailing ?, collapse whitespace, uppercase.
-- Does NOT strip trailing `X.Y.` patterns: those overlap with genuine
-- owner-initial disambiguators (`Anna K.J.`, `Kasper T.T.`).
CREATE OR REPLACE FUNCTION v2_normalize_name(input text) RETURNS text AS $$
    SELECT CASE
        WHEN input IS NULL OR input = '' THEN ''
        ELSE upper(
            regexp_replace(
                regexp_replace(
                    translate(public.unaccent(input), '*·', ''),
                    '\s*\([A-Z]{1,3}\)\s*$|\s*\?\s*$',
                    '', 'g'),
                '\s+', ' ', 'g'
            )
        )
    END
$$ LANGUAGE SQL IMMUTABLE;

-- Functional index so pedigree / synth-key / French-horse matching
-- queries that filter on v2_normalize_name(name) are index-served
-- instead of seqscanning 468k rows. (Without `public.` qualification on
-- unaccent above, CREATE INDEX fails because the extension is not
-- visible to the index expression compilation context.)
CREATE INDEX IF NOT EXISTS idx_horse_v2_normalize_name
  ON horse (v2_normalize_name(name))
  WHERE v2_normalize_name(name) <> '';
"""


# ---------------------------------------------------------------------------
# STATS VIEWS — wide, denormalised aggregates backing the /stats/* browse
# pages. Each is a single self-sufficient row-per-entity table so the list
# endpoints can filter/sort/paginate against ONE indexed relation instead of
# joining + re-aggregating the 7.5M-row `entry` table on every request.
# The `started` / win / top3 predicates are byte-for-byte the same as
# horse_career_stats so displayed numbers are consistent across the app.
# Refresh via REFRESH MATERIALIZED VIEW CONCURRENTLY after each update run.
# ---------------------------------------------------------------------------
STATS_VIEWS_DDL = r"""
-- ===== horse_stats — one row per horse, everything the horse browse needs ===
DROP MATERIALIZED VIEW IF EXISTS horse_stats CASCADE;
CREATE MATERIALIZED VIEW horse_stats AS
WITH ent AS (
    SELECT e.horse_id,
        COUNT(*) FILTER (WHERE q.started)                                       AS starts,
        COUNT(*) FILTER (WHERE e.placement_text = '1' AND NOT COALESCE(e.disqualified,false)) AS wins,
        COUNT(*) FILTER (WHERE e.placement_text IN ('1','2','3') AND NOT COALESCE(e.disqualified,false)) AS placed,
        COUNT(*) FILTER (WHERE q.started AND e.galopp)                           AS gals,
        MIN(e.time_seconds) FILTER (WHERE NOT e.withdrawn AND e.time_seconds > 0) AS best_time_s,
        MAX(e.race_date)                                                         AS last_start,
        COALESCE(SUM(e.prize_kr), 0)                                             AS prize_computed
    FROM entry e
    CROSS JOIN LATERAL (SELECT (
            NOT e.withdrawn
            AND COALESCE(e.placement_text, '') !~ '^(gdk|ejg|ejp|gd|gk|egk|egdk|gkd|gdj|gdb|gdek|gdgk|gdl|ddk|frdk|erj|ejk|ejgk|ejgd|ej|EJ|EJG|EJP|GDK|Gdk|g)[0-9]?$|^[12]?[pP][0-9]?$'
        ) AS started) q
    GROUP BY e.horse_id
),
last_tr AS (
    SELECT DISTINCT ON (e.horse_id) e.horse_id, e.trainer_id
    FROM entry e
    WHERE e.trainer_id IS NOT NULL
    ORDER BY e.horse_id, e.race_date DESC NULLS LAST, e.entry_id DESC
),
off AS (
    SELECT parent_id,
           COUNT(*)                          AS offspring_count,
           COALESCE(SUM(child_prize), 0)      AS offspring_prize,
           MAX(child_dob_year)                AS last_offspring_year
    FROM (
        SELECT h.sire_id AS parent_id,
               GREATEST(COALESCE(cs.prize_money_kr,0), COALESCE(h.scraped_prize_money_kr,0)) AS child_prize,
               EXTRACT(YEAR FROM h.date_of_birth)::int AS child_dob_year
        FROM horse h LEFT JOIN horse_career_stats cs ON cs.horse_id = h.horse_id
        WHERE h.sire_id IS NOT NULL
        UNION ALL
        SELECT h.dam_id,
               GREATEST(COALESCE(cs.prize_money_kr,0), COALESCE(h.scraped_prize_money_kr,0)),
               EXTRACT(YEAR FROM h.date_of_birth)::int
        FROM horse h LEFT JOIN horse_career_stats cs ON cs.horse_id = h.horse_id
        WHERE h.dam_id IS NOT NULL
    ) u
    GROUP BY parent_id
)
SELECT h.horse_id,
       h.name,
       h.breed_code,
       h.gender_code,
       EXTRACT(YEAR FROM h.date_of_birth)::int                                   AS dob_year,
       COALESCE(h.is_dead, false)                                                AS is_dead,
       h.sire_id,
       COALESCE((SELECT sh.name FROM horse sh WHERE sh.horse_id = h.sire_id), h.sire_name) AS sire_name,
       h.dam_id,
       COALESCE((SELECT dh.name FROM horse dh WHERE dh.horse_id = h.dam_id), h.dam_name)   AS dam_name,
       dam.sire_id                                                               AS mgs_id,
       COALESCE((SELECT gh.name FROM horse gh WHERE gh.horse_id = dam.sire_id), dam.sire_name) AS mgs_name,
       lt.trainer_id                                                             AS trainer_id,
       tp.name                                                                   AS trainer_name,
       GREATEST(COALESCE(ent.starts,0), COALESCE(h.scraped_starts,0))            AS starts,
       GREATEST(COALESCE(ent.wins,0),   COALESCE(h.scraped_wins,0))              AS wins,
       COALESCE(ent.placed,0)                                                    AS placed,
       COALESCE(ent.gals,0)                                                      AS gals,
       ent.best_time_s,
       ent.last_start,
       GREATEST(COALESCE(ent.prize_computed,0), COALESCE(h.scraped_prize_money_kr,0)) AS prize_money_kr,
       COALESCE(off.offspring_count,0)                                           AS offspring_count,
       COALESCE(off.offspring_prize,0)                                           AS offspring_prize,
       off.last_offspring_year
FROM horse h
LEFT JOIN ent     ON ent.horse_id = h.horse_id
LEFT JOIN last_tr lt ON lt.horse_id = h.horse_id
LEFT JOIN person  tp ON tp.person_id = lt.trainer_id
LEFT JOIN horse   dam ON dam.horse_id = h.dam_id
LEFT JOIN off     ON off.parent_id = h.horse_id;

CREATE UNIQUE INDEX ON horse_stats (horse_id);
CREATE INDEX ON horse_stats (breed_code);
CREATE INDEX ON horse_stats (prize_money_kr DESC);
CREATE INDEX ON horse_stats (starts DESC);
CREATE INDEX ON horse_stats (wins DESC);
CREATE INDEX ON horse_stats (offspring_count DESC);
CREATE INDEX ON horse_stats (offspring_prize DESC);
CREATE INDEX ON horse_stats (best_time_s);

-- ===== person_stats — one row per (role, person) for driver/trainer pages ===
DROP MATERIALIZED VIEW IF EXISTS person_stats CASCADE;
CREATE MATERIALIZED VIEW person_stats AS
WITH agg AS (
    SELECT 'driver'::text AS role, e.driver_id AS pid,
           COUNT(*) FILTER (WHERE q.started)                                       AS starts,
           COUNT(*) FILTER (WHERE e.placement_text = '1' AND NOT COALESCE(e.disqualified,false)) AS wins,
           COUNT(*) FILTER (WHERE e.placement_text IN ('1','2','3') AND NOT COALESCE(e.disqualified,false)) AS placed,
           COUNT(*) FILTER (WHERE q.started AND e.galopp)                           AS gals,
           COALESCE(SUM(e.prize_kr),0)                                              AS prize,
           MAX(e.race_date)                                                         AS last_start
    FROM entry e
    CROSS JOIN LATERAL (SELECT (
            NOT e.withdrawn
            AND COALESCE(e.placement_text, '') !~ '^(gdk|ejg|ejp|gd|gk|egk|egdk|gkd|gdj|gdb|gdek|gdgk|gdl|ddk|frdk|erj|ejk|ejgk|ejgd|ej|EJ|EJG|EJP|GDK|Gdk|g)[0-9]?$|^[12]?[pP][0-9]?$'
        ) AS started) q
    WHERE e.driver_id IS NOT NULL
    GROUP BY e.driver_id
    UNION ALL
    SELECT 'trainer'::text AS role, e.trainer_id AS pid,
           COUNT(*) FILTER (WHERE q.started),
           COUNT(*) FILTER (WHERE e.placement_text = '1' AND NOT COALESCE(e.disqualified,false)),
           COUNT(*) FILTER (WHERE e.placement_text IN ('1','2','3') AND NOT COALESCE(e.disqualified,false)),
           COUNT(*) FILTER (WHERE q.started AND e.galopp),
           COALESCE(SUM(e.prize_kr),0),
           MAX(e.race_date)
    FROM entry e
    CROSS JOIN LATERAL (SELECT (
            NOT e.withdrawn
            AND COALESCE(e.placement_text, '') !~ '^(gdk|ejg|ejp|gd|gk|egk|egdk|gkd|gdj|gdb|gdek|gdgk|gdl|ddk|frdk|erj|ejk|ejgk|ejgd|ej|EJ|EJG|EJP|GDK|Gdk|g)[0-9]?$|^[12]?[pP][0-9]?$'
        ) AS started) q
    WHERE e.trainer_id IS NOT NULL
    GROUP BY e.trainer_id
)
SELECT a.role, a.pid AS person_id,
       COALESCE(p.name, p.short_name)  AS name,
       p.short_name,
       p.license_country,
       a.starts, a.wins, a.placed, a.gals, a.prize, a.last_start
FROM agg a
JOIN person p ON p.person_id = a.pid;

CREATE UNIQUE INDEX ON person_stats (role, person_id);
CREATE INDEX ON person_stats (role, prize DESC);
CREATE INDEX ON person_stats (role, starts DESC);
CREATE INDEX ON person_stats (role, wins DESC);

-- ===== track_post_stats — per (track, method, post) aggregates ==============
-- Powers the track browse + post-position visualisation. Tiny (≈ tracks×2×15),
-- so method totals / top-post / cross-track relative rank are all derived in
-- the query layer from this one MV.
DROP MATERIALIZED VIEW IF EXISTS track_post_stats CASCADE;
CREATE MATERIALIZED VIEW track_post_stats AS
SELECT r.track_id,
       e.auto,
       e.program_number AS post,
       COUNT(*) FILTER (WHERE q.started)                                         AS starts,
       COUNT(*) FILTER (WHERE e.placement_text = '1' AND NOT COALESCE(e.disqualified,false)) AS wins,
       COUNT(*) FILTER (WHERE e.placement_text IN ('1','2','3') AND NOT COALESCE(e.disqualified,false)) AS placed,
       COUNT(*) FILTER (WHERE q.started AND e.galopp)                            AS gals,
       SUM(e.odds) FILTER (WHERE e.placement_text = '1' AND NOT COALESCE(e.disqualified,false) AND e.odds > 0) AS winner_odds_sum,
       COUNT(*)    FILTER (WHERE e.placement_text = '1' AND NOT COALESCE(e.disqualified,false) AND e.odds > 0) AS winner_cnt
FROM race r
JOIN entry e ON e.race_id = r.race_id
CROSS JOIN LATERAL (SELECT (
        NOT e.withdrawn
        AND COALESCE(e.placement_text, '') !~ '^(gdk|ejg|ejp|gd|gk|egk|egdk|gkd|gdj|gdb|gdek|gdgk|gdl|ddk|frdk|erj|ejk|ejgk|ejgd|ej|EJ|EJG|EJP|GDK|Gdk|g)[0-9]?$|^[12]?[pP][0-9]?$'
    ) AS started) q
WHERE r.track_id IS NOT NULL
  AND e.auto IS NOT NULL
  AND e.program_number BETWEEN 1 AND 15
GROUP BY r.track_id, e.auto, e.program_number;

-- (track_id, auto, post) is the natural key (see GROUP BY + the NOT NULL /
-- BETWEEN guards above) and all three are non-null, so a UNIQUE index is valid.
-- It is REQUIRED for REFRESH MATERIALIZED VIEW CONCURRENTLY, and its leading
-- track_id column still serves the track-browse filter (no separate index
-- needed).
CREATE UNIQUE INDEX ON track_post_stats (track_id, auto, post);
"""


def create_schema(conn) -> None:
    """Apply the v2 schema. Idempotent."""
    with conn.cursor() as cur:
        cur.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm")
        cur.execute("CREATE EXTENSION IF NOT EXISTS unaccent")
        cur.execute(SCHEMA_DDL)
        cur.execute(BUFFER_DDL)
        cur.execute(ST_RAW_DDL)
        cur.execute(ATG_RAW_DDL)
        cur.execute(NORMALIZE_NAME_FUNC_DDL)
        # Stats browse matviews. MUST run after SCHEMA_DDL: horse_stats reads
        # horse_career_stats, which SCHEMA_DDL drops+recreates with CASCADE on
        # every apply — that cascade also drops horse_stats, so it has to be
        # rebuilt here or `core.schema apply` would silently leave it missing.
        cur.execute(STATS_VIEWS_DDL)
    conn.commit()


def drop_schema(conn) -> None:
    """Drop everything in the v2 schema. Use with caution."""
    drop_buffers = "\n".join(
        f"DROP TABLE IF EXISTS {s}_buffer CASCADE;" for s in KNOWN_SOURCES
    )
    with conn.cursor() as cur:
        cur.execute(f"""
            {drop_buffers}
            DROP TABLE IF EXISTS st_horse_raw           CASCADE;
            DROP TABLE IF EXISTS st_horse_scrape_log    CASCADE;
            DROP TABLE IF EXISTS st_raceday_raw         CASCADE;
            DROP TABLE IF EXISTS st_raceday_scrape_log  CASCADE;
            DROP TABLE IF EXISTS atg_race_raw           CASCADE;
            DROP TABLE IF EXISTS atg_race_scrape_log    CASCADE;
            DROP MATERIALIZED VIEW IF EXISTS horse_stats CASCADE;
            DROP MATERIALIZED VIEW IF EXISTS person_stats CASCADE;
            DROP MATERIALIZED VIEW IF EXISTS track_post_stats CASCADE;
            DROP MATERIALIZED VIEW IF EXISTS horse_career_stats CASCADE;
            DROP MATERIALIZED VIEW IF EXISTS horse_year_stats CASCADE;
            DROP MATERIALIZED VIEW IF EXISTS person_career_stats CASCADE;
            DROP MATERIALIZED VIEW IF EXISTS track_stats CASCADE;
            DROP TABLE IF EXISTS entry_features         CASCADE;
            DROP TABLE IF EXISTS ml_model               CASCADE;
            DROP TABLE IF EXISTS ml_slice               CASCADE;
            DROP TABLE IF EXISTS ml_prediction          CASCADE;
            DROP TABLE IF EXISTS person_merge_log       CASCADE;
            DROP TABLE IF EXISTS horse_merge_log        CASCADE;
            DROP TABLE IF EXISTS job_run                CASCADE;
            DROP TABLE IF EXISTS horse_trainer_history  CASCADE;
            DROP TABLE IF EXISTS horse_owner_history    CASCADE;
            DROP TABLE IF EXISTS entry                  CASCADE;
            DROP TABLE IF EXISTS race                   CASCADE;
            DROP TABLE IF EXISTS track                  CASCADE;
            DROP TABLE IF EXISTS person                 CASCADE;
            DROP TABLE IF EXISTS horse                  CASCADE;
        """)
    conn.commit()


# ---------------------------------------------------------------------------
# CLI: `python -m core.schema apply` / `python -m core.schema drop`
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    from .db import get_connection

    cmd = sys.argv[1] if len(sys.argv) > 1 else "apply"
    conn = get_connection()
    try:
        if cmd == "apply":
            create_schema(conn)
            print("Schema applied.")
        elif cmd == "drop":
            drop_schema(conn)
            print("Schema dropped.")
        else:
            print(f"Unknown command: {cmd!r}. Use 'apply' or 'drop'.")
            sys.exit(2)
    finally:
        conn.close()
