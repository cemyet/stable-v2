"""Configuration for stable-v2.

Database: separate `stable_v2` instance on the same local Postgres.
Web port: 5002 (v1 stays on 5001).

Source-specific URL templates and HTTP headers live here so the scrapers can
import { config }.URL / config.HEADERS without each one reinventing the wheel.
"""

from __future__ import annotations

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# .env loading (optional)
# ---------------------------------------------------------------------------
# Environment overrides let the same code run locally (Postgres.app) or in the
# cloud (Railway → Supabase) without editing this file. A `.env` at the repo
# root is loaded if present; real environment variables always win over it.
# We parse it with a tiny built-in reader so python-dotenv stays optional.


def _load_dotenv() -> None:
    env_path = Path(__file__).resolve().parent.parent / ".env"
    if not env_path.exists():
        return
    try:
        for raw in env_path.read_text().splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            key = key.strip()
            val = val.strip().strip('"').strip("'")
            # Real env vars take precedence over the .env file.
            os.environ.setdefault(key, val)
    except OSError:
        pass


_load_dotenv()


def _env_bool(name: str, default: bool) -> bool:
    val = os.environ.get(name)
    if val is None:
        return default
    return val.strip().lower() in ("1", "true", "yes", "on")


# ---------------------------------------------------------------------------
# Database + web
# ---------------------------------------------------------------------------

# The v2 database. Local default is Postgres.app; set DATABASE_URL in the
# environment (e.g. the Supabase pooler connection string) for cloud runs.
DATABASE_URL = os.environ.get(
    "DATABASE_URL", "postgresql://jakob@localhost:5432/stable_v2"
)

# Supabase / cloud publish target for the local→cloud nightly sync. Only used
# by the publish step (jobs.publish); the app itself talks to DATABASE_URL.
SUPABASE_DATABASE_URL = os.environ.get("SUPABASE_DATABASE_URL", "")

# v1 lives in `jakob`; historically read via postgres_fdw / direct psycopg2.
# Use unix socket (`/tmp`) rather than TCP — Postgres.app's "trust" auth
# rejects FDW's TCP self-connections without explicit per-app permission.
V1_DATABASE_NAME = os.environ.get("V1_DATABASE_NAME", "jakob")
V1_DATABASE_HOST = os.environ.get("V1_DATABASE_HOST", "/tmp")
V1_DATABASE_PORT = int(os.environ.get("V1_DATABASE_PORT", "5432"))
V1_DATABASE_USER = os.environ.get("V1_DATABASE_USER", "jakob")

# Filesystem root of the legacy v1 project (used only by the v1 subprocess
# bridge in jobs.update). Irrelevant once USE_V1_BRIDGE is off.
V1_PROJECT_ROOT = os.environ.get("V1_PROJECT_ROOT", "/Users/jakob/Dev/stable")

# Master switch for the legacy v1 dependency. When False, v2 scrapes ATG
# natively and never shells out to / reads from v1. Defaults to False now that
# native ATG exists; set USE_V1_BRIDGE=1 to fall back to the old bridge path.
USE_V1_BRIDGE = _env_bool("USE_V1_BRIDGE", False)

WEB_PORT = int(os.environ.get("WEB_PORT", "5002"))

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

# Native horse JSON API. The public sportapp pages no longer SSR-embed the
# horse datasets (only `horse-basic-information`); the React app fetches the
# rest from these api.travsport.se endpoints (discovered from the app JS).
# Keys mirror v1's parse_horse_page data_type names so etl.import_st can reuse
# the v1 field map. `{horse_id}` is the TravSport horse id (== our st_id).
ST_API_BASE = "https://api.travsport.se/webapi"
ST_HORSE_API_ENDPOINTS = {
    "horse-basic-information": "horses/basicinformation/organisation/TROT/sourceofdata/SPORT/horseid/{horse_id}",
    "race-results":           "horses/results/organisation/TROT/sourceofdata/SPORT/horseid/{horse_id}",
    "horse-statistics":       "horses/statistics/organisation/TROT/sourceofdata/SPORT/horseid/{horse_id}",
    "horse-history":          "horses/history/organisation/TROT/sourceofdata/SPORT/horseid/{horse_id}",
    "lineage-small":          "horses/pedigree/organisation/TROT/sourceofdata/SPORT/horseid/{horse_id}?pedigreeTree=SMALL",
}

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
KNOWN_SOURCES = ("st", "atg", "usta", "letrot", "kmtid", "hvt", "breedly")

# ---------------------------------------------------------------------------
# kmtid (atgx GPS km-times)
# ---------------------------------------------------------------------------

# kmtid uses YYMMDD in the URL path; the actual data is a JS file under /js/.
KMTID_RACES_URL = "https://kmtid.atgx.se/{yymmdd}/js/races.js"

KMTID_HEADERS = {
    "User-Agent": DEFAULT_USER_AGENT,
    "Accept": "application/javascript,text/javascript,*/*;q=0.8",
    "Accept-Language": "sv-SE,sv;q=0.9,en;q=0.8",
}

# How many days back from "today" the kmtid scraper will probe by default.
# kmtid only publishes ~30 day rolling window, so 35 covers the lag safely.
KMTID_BACKFILL_DAYS = 35

# ---------------------------------------------------------------------------
# HVT online (Hauptverband für Traberzucht — German trotting)
# ---------------------------------------------------------------------------

HVT_BASE = "https://www.hvtonline.de"

HVT_HEADERS = {
    "User-Agent": DEFAULT_USER_AGENT,
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "de-DE,de;q=0.9,en;q=0.8",
}

# ---------------------------------------------------------------------------
# Le Trot (LeTROT — French trotting)
# ---------------------------------------------------------------------------

LETROT_BASE = "https://www.letrot.com"

LETROT_HEADERS = {
    "User-Agent": DEFAULT_USER_AGENT,
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "fr-FR,fr;q=0.9,en;q=0.8",
}

# ---------------------------------------------------------------------------
# Breedly (pedigree-only enrichment, Next.js)
# ---------------------------------------------------------------------------

BREEDLY_BASE = "https://www.breedly.com"

BREEDLY_HEADERS = {
    "User-Agent": DEFAULT_USER_AGENT,
    "Accept": "application/json,text/html;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}

# ---------------------------------------------------------------------------
# Riksbank historical FX (used to convert foreign prize money to SEK)
# ---------------------------------------------------------------------------

RIKSBANK_BASE = "https://api.riksbank.se/swea/v1"
RIKSBANK_SUPPORTED_CCYS = ("EUR", "USD", "GBP", "NOK", "DKK", "CHF", "JPY", "AUD", "CAD")

# ---------------------------------------------------------------------------
# Buffer retention
# ---------------------------------------------------------------------------

BUFFER_RETENTION_DAYS = 7
