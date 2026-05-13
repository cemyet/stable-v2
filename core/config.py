"""Configuration for stable-v2.

Database: separate `stable_v2` instance on the same local Postgres.
Web port: 5002 (v1 stays on 5001).

Source-specific URL templates and HTTP headers live here so the scrapers can
import { config }.URL / config.HEADERS without each one reinventing the wheel.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Database + web
# ---------------------------------------------------------------------------

DATABASE_URL = "postgresql://jakob@localhost:5432/stable_v2"

# v1 lives in `jakob`; we read it through postgres_fdw via the v1_raw schema.
# Use unix socket (`/tmp`) rather than TCP — Postgres.app's "trust" auth
# rejects FDW's TCP self-connections without explicit per-app permission.
V1_DATABASE_NAME = "jakob"
V1_DATABASE_HOST = "/tmp"
V1_DATABASE_PORT = 5432
V1_DATABASE_USER = "jakob"

WEB_PORT = 5002

# ---------------------------------------------------------------------------
# Generic HTTP knobs
# ---------------------------------------------------------------------------

REQUEST_DELAY = 0.05
MAX_RETRIES = 3
RETRY_BACKOFF = 2.0
BATCH_SIZE = 200
CONCURRENCY = 10

DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0.0.0 Safari/537.36"
)

# ---------------------------------------------------------------------------
# TravSport ("st")
# ---------------------------------------------------------------------------

ST_HORSE_URL = "https://sportapp.travsport.se/sportinfo/horse/ts{horse_id}/basic"
ST_RACE_API_BASE = (
    "https://api.travsport.se/webapi/raceinfo/results"
    "/organisation/TROT/sourceofdata/SPORT/racedayid/{race_day_id}"
)

ST_HEADERS = {
    "User-Agent": DEFAULT_USER_AGENT,
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "sv-SE,sv;q=0.9,en;q=0.8",
    "Accept-Encoding": "gzip, deflate, br",
}

ST_RACE_HEADERS = {
    "User-Agent": DEFAULT_USER_AGENT,
    "Accept": "application/json",
    "Accept-Language": "sv-SE,sv;q=0.9,en;q=0.8",
}

ST_RACE_CONCURRENCY = 15
ST_RACE_BATCH_SIZE = 100

# ---------------------------------------------------------------------------
# ATG ("atg")
# ---------------------------------------------------------------------------

ATG_BASE = "https://www.atg.se/services/racinginfo/v1/api"
ATG_CALENDAR_URL = ATG_BASE + "/calendar/day/{date}"
ATG_RACE_URL = ATG_BASE + "/races/{atg_race_id}"
ATG_GAME_URL = ATG_BASE + "/games/{atg_game_id}"

ATG_HEADERS = {
    "User-Agent": DEFAULT_USER_AGENT,
    "Accept": "application/json",
    "Accept-Language": "sv-SE,sv;q=0.9,en;q=0.8",
}

ATG_CONCURRENCY = 8
ATG_BATCH_SIZE = 50

# ---------------------------------------------------------------------------
# Source list (used by buffer rotation, source attribution UI, etc.)
# ---------------------------------------------------------------------------

# Each source we know about. The key is what we use as JSONB keys in
# horse.source_data, the table column suffix (st_id, atg_id, ...), and the
# filename suffix in scrapers/ + etl/ (st_horse.py, import_st.py, ...).
KNOWN_SOURCES = ("st", "atg", "usta", "letrot", "kmtid")

# ---------------------------------------------------------------------------
# Buffer retention
# ---------------------------------------------------------------------------

BUFFER_RETENTION_DAYS = 7
