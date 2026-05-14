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

    UNIQUE (race_id, horse_id)
);
CREATE INDEX IF NOT EXISTS entry_horse_idx        ON entry (horse_id);
CREATE INDEX IF NOT EXISTS entry_driver_idx       ON entry (driver_id)  WHERE driver_id  IS NOT NULL;
CREATE INDEX IF NOT EXISTS entry_trainer_idx      ON entry (trainer_id) WHERE trainer_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS entry_horse_winner_idx ON entry (horse_id)   WHERE placement = 1;
CREATE INDEX IF NOT EXISTS entry_horse_placed_idx ON entry (horse_id)   WHERE placement BETWEEN 1 AND 3;

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
    pid           INTEGER
);
CREATE INDEX IF NOT EXISTS job_run_started_idx ON job_run (started_at DESC);

-- =====================================================================
-- DERIVED VIEWS — compute on read, do NOT materialise
-- =====================================================================

DROP VIEW IF EXISTS horse_career_stats CASCADE;
CREATE VIEW horse_career_stats AS
WITH computed AS (
    SELECT
        e.horse_id,
        COUNT(*) FILTER (
            WHERE NOT e.withdrawn
              AND COALESCE(e.placement_text, '') !~* '^(gdk|ejg|ejp)'
        )                                                   AS starts,
        COUNT(*) FILTER (
            WHERE e.placement = 1
              AND COALESCE(e.placement_text, '') !~* '^(gdk|ejg|ejp)'
        )                                                   AS wins,
        COUNT(*) FILTER (WHERE e.placement = 2)             AS seconds,
        COUNT(*) FILTER (WHERE e.placement = 3)             AS thirds,
        COUNT(*) FILTER (
            WHERE e.placement BETWEEN 1 AND 3
              AND COALESCE(e.placement_text, '') !~* '^(gdk|ejg|ejp)'
        )                                                   AS placed,
        COUNT(*) FILTER (
            WHERE COALESCE(e.placement_text, '') ~* '^(gdk|ejg|ejp)'
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


def create_schema(conn) -> None:
    """Apply the v2 schema. Idempotent."""
    with conn.cursor() as cur:
        cur.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm")
        cur.execute(SCHEMA_DDL)
        cur.execute(BUFFER_DDL)
    conn.commit()


def drop_schema(conn) -> None:
    """Drop everything in the v2 schema. Use with caution."""
    drop_buffers = "\n".join(
        f"DROP TABLE IF EXISTS {s}_buffer CASCADE;" for s in KNOWN_SOURCES
    )
    with conn.cursor() as cur:
        cur.execute(f"""
            {drop_buffers}
            DROP VIEW  IF EXISTS horse_career_stats     CASCADE;
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
