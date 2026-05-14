"""Foreign-exchange helper backed by Riksbank's SWEA API.

Used to convert foreign prize money (EUR/USD/etc.) into SEK so that v2 can
keep one comparable `prize_kr` column across all sources.

Strategy:
    - For each (currency, race-date) pair, fetch the SEK reference rate
      published by Riksbank for that date (or, if the date is a weekend or
      bank holiday with no publication, the last earlier publication).
    - Cache aggressively in memory (one process run typically converts a few
      thousand rows for a single source) and on disk under
      `logs/fx_cache.json` so re-runs don't re-hit the API.

Series id reference (Sweden Riksbank SWEA daily middle rates):
    EUR  → SEKEURPMI
    USD  → SEKUSDPMI
    GBP  → SEKGBPPMI
    NOK  → SEKNOKPMI
    DKK  → SEKDKKPMI
    CHF  → SEKCHFPMI
    JPY  → SEKJPYPMI       (rate is per 100 JPY)
    AUD  → SEKAUDPMI
    CAD  → SEKCADPMI

`get_rate(ccy, date)` returns the SEK price of 1 unit of `ccy` on `date`
(taking JPY-per-100 into account), or `None` if the API has nothing.
"""

from __future__ import annotations

import json
import logging
import threading
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

import httpx

from .config import RIKSBANK_BASE

log = logging.getLogger(__name__)

_SERIES = {
    "EUR": "SEKEURPMI",
    "USD": "SEKUSDPMI",
    "GBP": "SEKGBPPMI",
    "NOK": "SEKNOKPMI",
    "DKK": "SEKDKKPMI",
    "CHF": "SEKCHFPMI",
    "JPY": "SEKJPYPMI",   # quoted per 100 JPY
    "AUD": "SEKAUDPMI",
    "CAD": "SEKCADPMI",
}

# JPY rate is published per 100 JPY; everything else is per 1 unit.
_PER_100 = {"JPY"}

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


def _fetch_window(series: str, start: date, end: date) -> list[dict]:
    url = f"{RIKSBANK_BASE}/Observations/{series}/{start.isoformat()}/{end.isoformat()}"
    try:
        with httpx.Client(timeout=15.0) as c:
            r = c.get(url)
        if r.status_code != 200:
            log.warning("riksbank %s %s..%s -> HTTP %s",
                        series, start, end, r.status_code)
            return []
        data = r.json()
        if isinstance(data, list):
            return data
    except Exception as e:
        log.warning("riksbank fetch failed (%s %s..%s): %s",
                    series, start, end, e)
    return []


def get_rate(ccy: str, d: date) -> Optional[float]:
    """SEK value of 1 unit of `ccy` on `d` (or the most recent prior publish)."""
    if not ccy or not d:
        return None
    ccy = ccy.upper()
    if ccy == "SEK":
        return 1.0
    series = _SERIES.get(ccy)
    if not series:
        return None

    with _LOCK:
        _load_disk_cache()
        cached = _MEM.get(_key(ccy, d))
        if cached is not None:
            return cached

    # Fetch a window large enough to cover weekends + holidays.
    start = d - timedelta(days=10)
    obs = _fetch_window(series, start, d)
    if not obs:
        return None

    # Sort observations by date, take the latest one ≤ d.
    obs_sorted = sorted(obs, key=lambda o: o.get("date", ""))
    chosen: Optional[dict] = None
    for o in obs_sorted:
        if o.get("date", "") <= d.isoformat():
            chosen = o
    if not chosen:
        return None

    raw = float(chosen["value"])
    rate = raw / 100.0 if ccy in _PER_100 else raw

    with _LOCK:
        _MEM[_key(ccy, d)] = rate
        # Pre-warm cache for nearby dates from the same fetch window.
        for o in obs_sorted:
            try:
                od = date.fromisoformat(o["date"])
                v = float(o["value"]) / (100.0 if ccy in _PER_100 else 1.0)
                _MEM.setdefault(_key(ccy, od), v)
            except Exception:
                pass
        _flush_disk_cache()

    return rate


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
        return int(round(a)), a, 1.0, d  # assume SEK by default
    ccy = ccy.upper()
    if ccy == "SEK":
        return int(round(a)), a, 1.0, d
    if not d:
        return None, a, None, None
    rate = get_rate(ccy, d)
    if rate is None:
        return None, a, None, None
    return int(round(a * rate)), a, rate, d
