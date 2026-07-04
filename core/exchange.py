"""Foreign-exchange helper — ECB primary, Riksbank fallback.

Used to convert foreign prize money (EUR/USD/etc.) into SEK so that v2 can
keep one comparable `prize_kr` column across all sources.

Strategy:
    - `prefetch_range("EUR", start, end)` downloads the ECB's full historical
      reference-rates ZIP (~700 KB, no auth, no rate limit) and populates the
      disk/memory cache with forward-filled daily rates.  One call covers the
      entire range instantly.
    - `get_rate(ccy, date)` looks up the in-memory cache first, then the disk
      cache.  It never makes a blocking API call — if the rate is missing,
      it returns None.
    - For non-EUR currencies the ECB ZIP also contains cross-rates which are
      derived into SEK-per-unit via SEK/EUR ÷ X/EUR.
    - The Riksbank SWEA API remains as a secondary source for `_fetch_window`
      but is only used if explicitly called; the hot path never touches it.

`get_rate(ccy, date)` returns the SEK price of 1 unit of `ccy` on `date`,
or `None` if the cache has nothing.
"""

from __future__ import annotations

import csv
import io
import json
import logging
import threading
import time
import zipfile
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

import httpx

from .config import RIKSBANK_BASE

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# ECB bulk download
# ---------------------------------------------------------------------------
ECB_ZIP_URL = "https://www.ecb.europa.eu/stats/eurofxref/eurofxref-hist.zip"

# JPY rate is published per 100 JPY; everything else is per 1 unit.
_PER_100 = {"JPY"}

# Riksbank series mapping (kept for legacy/fallback use)
_SERIES = {
    "EUR": "SEKEURPMI",
    "USD": "SEKUSDPMI",
    "GBP": "SEKGBPPMI",
    "NOK": "SEKNOKPMI",
    "DKK": "SEKDKKPMI",
    "CHF": "SEKCHFPMI",
    "JPY": "SEKJPYPMI",
    "AUD": "SEKAUDPMI",
    "CAD": "SEKCADPMI",
}

# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------
_CACHE_PATH = Path(__file__).resolve().parent.parent / "logs" / "fx_cache.json"
_LOCK = threading.Lock()
_MEM: dict[str, float] = {}
_LOADED = False


def _load_disk_cache() -> None:
    global _LOADED
    if _LOADED:
        return
    _LOADED = True
    try:
        if _CACHE_PATH.exists():
            with _CACHE_PATH.open() as f:
                data = json.load(f)
            if isinstance(data, dict):
                _MEM.update({k: float(v) for k, v in data.items() if v is not None})
    except Exception as e:
        log.warning("fx cache load failed: %s", e)


def _flush_disk_cache() -> None:
    try:
        _CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        tmp = _CACHE_PATH.with_suffix(".json.tmp")
        with tmp.open("w") as f:
            json.dump(_MEM, f)
        tmp.replace(_CACHE_PATH)
    except Exception as e:
        log.warning("fx cache flush failed: %s", e)


def _key(ccy: str, d: date) -> str:
    return f"{ccy.upper()}:{d.isoformat()}"


# ---------------------------------------------------------------------------
# ECB download + parse (ported from scripts/seed_fx_from_ecb.py)
# ---------------------------------------------------------------------------

_ECB_LOADED = False


def _ecb_download_and_cache(currencies: list[str], start: date, end: date) -> int:
    """Download the ECB historical ZIP once per process and populate the cache.

    Returns the number of new rates written.
    """
    global _ECB_LOADED

    # Only download once per process — the ZIP is the full history anyway.
    if _ECB_LOADED:
        return 0
    _ECB_LOADED = True

    log.info("downloading ECB historical rates: %s", ECB_ZIP_URL)
    try:
        with httpx.Client(timeout=60.0, follow_redirects=True) as c:
            r = c.get(ECB_ZIP_URL)
        r.raise_for_status()
    except Exception as exc:
        log.warning("ECB download failed: %r", exc)
        return 0

    try:
        with zipfile.ZipFile(io.BytesIO(r.content)) as z:
            names = [n for n in z.namelist() if n.lower().endswith(".csv")]
            if not names:
                log.warning("ECB zip contained no CSV")
                return 0
            with z.open(names[0]) as f:
                text = f.read().decode("utf-8")
    except Exception as exc:
        log.warning("ECB zip parse failed: %r", exc)
        return 0

    rdr = csv.reader(io.StringIO(text))
    header = next(rdr)
    rows = [row for row in rdr if row and row[0]]

    try:
        i_sek = header.index("SEK")
    except ValueError:
        log.warning("ECB CSV missing SEK column")
        return 0

    col_idx: dict[str, int | None] = {}
    for ccy in currencies:
        if ccy == "EUR":
            col_idx[ccy] = None
        elif ccy == "SEK":
            continue
        else:
            try:
                col_idx[ccy] = header.index(ccy)
            except ValueError:
                log.warning("ECB CSV missing column for %s — skipping", ccy)

    # Parse raw trading-day rates: {ccy: {date: sek_per_unit}}
    raw: dict[str, dict[date, float]] = {c: {} for c in currencies if c != "SEK"}
    for row in rows:
        try:
            d = date.fromisoformat(row[0])
        except (ValueError, IndexError):
            continue
        try:
            sek_per_eur = float(row[i_sek])
        except (ValueError, IndexError):
            continue
        for ccy in currencies:
            if ccy == "SEK":
                continue
            if ccy == "EUR":
                raw["EUR"][d] = sek_per_eur
                continue
            idx = col_idx.get(ccy)
            if idx is None or idx < 0:
                continue
            try:
                x_per_eur = float(row[idx])
            except (ValueError, IndexError):
                continue
            if x_per_eur == 0:
                continue
            sek_per_x = sek_per_eur / x_per_eur
            if ccy in _PER_100:
                sek_per_x *= 100.0
            raw[ccy][d] = sek_per_x

    # Forward-fill weekends/holidays so every calendar date has a rate.
    count = 0
    with _LOCK:
        for ccy, rates in raw.items():
            if not rates:
                continue
            sorted_dates = sorted(rates.keys())
            fill_start = max(start, sorted_dates[0])
            last_val: float | None = None
            cur = fill_start
            pos = 0
            while cur <= end:
                while pos < len(sorted_dates) and sorted_dates[pos] <= cur:
                    last_val = rates[sorted_dates[pos]]
                    pos += 1
                if last_val is not None:
                    k = _key(ccy, cur)
                    if k not in _MEM:
                        _MEM[k] = last_val
                        count += 1
                cur += timedelta(days=1)
        _flush_disk_cache()

    log.info("ECB: parsed %d trading rows, cached %d new daily rates "
             "(%s..%s)", len(rows), count, start, end)
    return count


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def prefetch_range(ccy: str, start: date, end: date) -> int:
    """Pre-warm the FX cache for a date range.

    Uses the ECB bulk ZIP (one download, full history, no rate limit).
    Returns the number of new observations cached.
    """
    if not ccy:
        return 0
    ccy = ccy.upper()
    if ccy == "SEK":
        return 0
    with _LOCK:
        _load_disk_cache()

    currencies = [ccy]
    return _ecb_download_and_cache(currencies, start, end)


def get_rate(ccy: str, d: date) -> Optional[float]:
    """SEK value of 1 unit of `ccy` on `d`.

    Returns from cache only — never makes a blocking API call. Call
    `prefetch_range` first to populate the cache.
    """
    if not ccy or not d:
        return None
    ccy = ccy.upper()
    if ccy == "SEK":
        return 1.0
    if ccy not in _SERIES:
        return None

    k = _key(ccy, d)
    with _LOCK:
        _load_disk_cache()
        cached = _MEM.get(k)
        if cached is not None:
            return cached

    # Try nearby dates (weekend/holiday → use Friday's rate).
    for delta in range(1, 8):
        fallback_k = _key(ccy, d - timedelta(days=delta))
        with _LOCK:
            fallback = _MEM.get(fallback_k)
            if fallback is not None:
                _MEM[k] = fallback
                return fallback

    return None


def to_sek(amount: Optional[float], ccy: Optional[str], d: Optional[date]) -> tuple[
    Optional[int], Optional[float], Optional[float], Optional[date]
]:
    """Convert (amount, ccy, date) → (sek_int, original_amount, fx_rate, fx_date).

    Always rounds the SEK side to a whole number (we store BIGINT prize_kr).
    For SEK input, returns the input amount unchanged with rate=1.0.
    For unknown / unfetchable currencies, returns (None, amount, None, None)
    so callers can store the original anyway.
    """
    if amount is None:
        return None, None, None, None
    try:
        a = float(amount)
    except (TypeError, ValueError):
        return None, None, None, None
    if not ccy:
        return int(round(a)), a, 1.0, d
    ccy = ccy.upper()
    if ccy == "SEK":
        return int(round(a)), a, 1.0, d
    if not d:
        return None, a, None, None
    rate = get_rate(ccy, d)
    if rate is None:
        return None, a, None, None
    return int(round(a * rate)), a, rate, d


# ---------------------------------------------------------------------------
# Legacy Riksbank helpers (kept for scripts that use them directly)
# ---------------------------------------------------------------------------

_MIN_INTERVAL_S = 15.0
_RATE_LOCK = threading.Lock()
_LAST_CALL_TS = 0.0
_CB_TRIPPED = False
_CB_CONSEC_QUOTA = 0
_CB_THRESHOLD = 3
_NEG_TTL_S = 300.0
_NEG_CACHE: dict[str, float] = {}
_STRICT_CACHE = False


def set_min_interval(seconds: float) -> None:
    """Override the Riksbank min-interval (e.g. lower it if registered)."""
    global _MIN_INTERVAL_S
    _MIN_INTERVAL_S = max(0.0, float(seconds))


def set_strict_cache(enabled: bool) -> None:
    """Enable / disable strict cache mode."""
    global _STRICT_CACHE
    _STRICT_CACHE = bool(enabled)
