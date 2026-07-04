"""Seed the FX cache (`logs/fx_cache.json`) from the European Central Bank's
historical reference-rates archive.

Why this exists:
    The Riksbank SWEA API is rate-limited (5 calls/min, 1000 calls/day, plus
    a multi-day cooldown once the daily quota is exceeded). When that quota
    is exhausted we can't make progress on the FX backfill — which blocks
    every downstream step that needs `prize_kr`.

    The ECB publishes the full daily history of euro reference rates as a
    single ZIP (`eurofxref-hist.zip`) — ~700 KB, no auth, no rate limit,
    served fresh every business day. For EUR→SEK conversion (the only
    cross we need for LeTrot prize money) the ECB rate is within ~0.2 %
    of Riksbank's middle rate, and the ECB feed actually covers some
    trading days Riksbank skipped.

What this script does:
    1. Download `eurofxref-hist.zip` (or read a local file via --local-csv).
    2. Parse the chosen currencies' columns (default: EUR).
    3. Forward-fill weekends / holidays so every calendar date in the chosen
       range has a rate (uses the most recent prior trading day's rate).
       This is required because `core.exchange.get_rate` in strict-cache
       mode only looks up the exact date — without forward-fill, a Saturday
       race would still come back as "missing rate".
    4. Write the rates into `logs/fx_cache.json` under keys
       `EUR:YYYY-MM-DD` (overwriting any existing values from Riksbank by
       default — pass `--no-overwrite` to keep existing values).

Currencies:
    For non-EUR currencies (USD, GBP, etc.) the ECB publishes X-per-EUR;
    we derive SEK-per-X via `SEK_per_EUR / X_per_EUR`. That keeps the same
    semantics as Riksbank's `SEK<X>PMI` series. JPY is per-100 in our
    cache (matching Riksbank's convention), so we divide accordingly.

Usage:
    python -m scripts.seed_fx_from_ecb                   # EUR only, full history
    python -m scripts.seed_fx_from_ecb --currencies EUR,USD,GBP
    python -m scripts.seed_fx_from_ecb --dry-run         # show deltas, don't write
    python -m scripts.seed_fx_from_ecb --start 2019-01-01
    python -m scripts.seed_fx_from_ecb --no-forward-fill # exact ECB dates only

After this runs, kick off:
    python -m scripts.backfill_fx --skip-prewarm --currencies EUR --recompute-all
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import logging
import sys
import zipfile
from datetime import date, timedelta
from pathlib import Path

import httpx

log = logging.getLogger("seed_fx_from_ecb")

ECB_ZIP_URL = "https://www.ecb.europa.eu/stats/eurofxref/eurofxref-hist.zip"

CACHE_PATH = Path(__file__).resolve().parent.parent / "logs" / "fx_cache.json"

# Currencies our exchange module knows how to look up. Mirrors core.exchange._SERIES.
SUPPORTED_CCYS = {"EUR", "USD", "GBP", "NOK", "DKK", "CHF", "JPY", "AUD", "CAD"}

# JPY in our cache is "SEK per 100 JPY" (matches Riksbank's SEKJPYPMI series).
PER_100 = {"JPY"}


def _download_csv() -> str:
    log.info("downloading ECB historical reference rates: %s", ECB_ZIP_URL)
    with httpx.Client(timeout=60.0, follow_redirects=True) as c:
        r = c.get(ECB_ZIP_URL)
    r.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
        names = [n for n in z.namelist() if n.lower().endswith(".csv")]
        if not names:
            raise RuntimeError("ECB zip contained no CSV")
        with z.open(names[0]) as f:
            return f.read().decode("utf-8")


def _read_csv(text: str) -> tuple[list[str], list[list[str]]]:
    rdr = csv.reader(io.StringIO(text))
    header = next(rdr)
    rows = [r for r in rdr if r and r[0]]
    return header, rows


def _parse_rates(
    header: list[str],
    rows: list[list[str]],
    currencies: list[str],
) -> dict[str, dict[date, float]]:
    """Return {ccy: {date: SEK_per_unit}} based on ECB cross-rates.

    Forward-fill happens later in a separate pass.
    """
    try:
        i_sek = header.index("SEK")
    except ValueError as e:
        raise RuntimeError("ECB CSV missing SEK column") from e

    col_idx: dict[str, int | None] = {}
    for ccy in currencies:
        if ccy == "EUR":
            col_idx[ccy] = None  # derived from i_sek alone
        elif ccy == "SEK":
            continue
        else:
            try:
                col_idx[ccy] = header.index(ccy)
            except ValueError:
                log.warning("ECB CSV missing column for %s — skipping", ccy)
                col_idx[ccy] = -1  # sentinel: skip

    out: dict[str, dict[date, float]] = {c: {} for c in currencies if c != "SEK"}

    for r in rows:
        try:
            d = date.fromisoformat(r[0])
        except (ValueError, IndexError):
            continue
        try:
            sek_per_eur = float(r[i_sek])
        except (ValueError, IndexError):
            continue
        for ccy in currencies:
            if ccy == "SEK":
                continue
            if ccy == "EUR":
                out["EUR"][d] = sek_per_eur
                continue
            idx = col_idx.get(ccy)
            if idx is None or idx < 0:
                continue
            try:
                x_per_eur = float(r[idx])
            except (ValueError, IndexError):
                continue
            if x_per_eur == 0:
                continue
            sek_per_x = sek_per_eur / x_per_eur
            if ccy in PER_100:
                sek_per_x *= 100.0  # cache convention: SEK per 100 JPY
            out[ccy][d] = sek_per_x

    return out


def _forward_fill(
    rates: dict[date, float],
    start: date,
    end: date,
) -> dict[date, float]:
    """Fill weekend/holiday gaps by carrying the most recent prior rate forward."""
    if not rates:
        return {}
    sorted_dates = sorted(rates.keys())
    earliest = sorted_dates[0]
    start = max(start, earliest)
    filled: dict[date, float] = {}
    last_val: float | None = None
    cur = start
    pos = 0
    while cur <= end:
        while pos < len(sorted_dates) and sorted_dates[pos] <= cur:
            last_val = rates[sorted_dates[pos]]
            pos += 1
        if last_val is not None:
            filled[cur] = last_val
        cur += timedelta(days=1)
    return filled


def _load_existing_cache() -> dict[str, float | None]:
    if not CACHE_PATH.exists():
        return {}
    try:
        with CACHE_PATH.open() as f:
            data = json.load(f)
        if isinstance(data, dict):
            return data
    except Exception as e:
        log.warning("could not read existing cache (%s) — treating as empty", e)
    return {}


def _flush_cache(cache: dict[str, float | None]) -> None:
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = CACHE_PATH.with_suffix(".json.tmp")
    with tmp.open("w") as f:
        json.dump(cache, f)
    tmp.replace(CACHE_PATH)


def _apply(
    existing: dict[str, float | None],
    seeded: dict[str, dict[date, float]],
    overwrite: bool,
) -> dict:
    """Merge `seeded` rates into `existing`. Returns a delta report."""
    report: dict = {"by_ccy": {}}
    for ccy, rates in seeded.items():
        added = updated = unchanged = skipped_existing = 0
        deltas: list[tuple[str, float, float]] = []
        for d, new_val in rates.items():
            key = f"{ccy}:{d.isoformat()}"
            old_val = existing.get(key)
            if old_val is None:
                if key in existing and not overwrite:
                    skipped_existing += 1
                    continue
                existing[key] = float(new_val)
                added += 1
            else:
                if abs(float(old_val) - new_val) < 1e-9:
                    unchanged += 1
                    continue
                if not overwrite:
                    skipped_existing += 1
                    continue
                deltas.append((d.isoformat(), float(old_val), float(new_val)))
                existing[key] = float(new_val)
                updated += 1
        report["by_ccy"][ccy] = {
            "added": added,
            "updated": updated,
            "unchanged": unchanged,
            "skipped_existing": skipped_existing,
            "total_in_range": len(rates),
            "sample_deltas": sorted(
                deltas, key=lambda t: abs(t[2] - t[1]), reverse=True
            )[:5],
        }
    return report


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--currencies", default="EUR",
                   help="comma-separated list of currencies to seed "
                        "(default: EUR; supported: " + ",".join(sorted(SUPPORTED_CCYS)) + ")")
    p.add_argument("--start", default="1999-01-01",
                   help="earliest date to seed (YYYY-MM-DD)")
    p.add_argument("--end", default=date.today().isoformat(),
                   help="latest date to seed (YYYY-MM-DD; default: today)")
    p.add_argument("--no-forward-fill", action="store_true",
                   help="don't fill weekend/holiday gaps; only exact ECB dates")
    p.add_argument("--no-overwrite", action="store_true",
                   help="keep existing cache values; only fill new keys")
    p.add_argument("--dry-run", action="store_true",
                   help="report deltas but don't write to fx_cache.json")
    p.add_argument("--local-csv",
                   help="read ECB CSV from a local path instead of downloading")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    try:
        start = date.fromisoformat(args.start)
        end = date.fromisoformat(args.end)
    except ValueError as e:
        log.error("bad date: %s", e)
        return 2
    if end < start:
        log.error("--end before --start")
        return 2

    currencies = [c.strip().upper() for c in args.currencies.split(",") if c.strip()]
    bad = [c for c in currencies if c not in SUPPORTED_CCYS]
    if bad:
        log.error("unsupported currencies: %s (supported: %s)",
                  bad, sorted(SUPPORTED_CCYS))
        return 2

    if args.local_csv:
        text = Path(args.local_csv).read_text(encoding="utf-8")
    else:
        text = _download_csv()
    header, rows = _read_csv(text)
    log.info("ECB CSV: %d header cols, %d data rows", len(header), len(rows))

    raw = _parse_rates(header, rows, currencies)
    for ccy in currencies:
        n = len(raw.get(ccy, {}))
        if n == 0:
            log.warning("no %s rates parsed", ccy)
            continue
        sample_dates = sorted(raw[ccy].keys())
        log.info("parsed %s: %d trading days (%s .. %s)",
                 ccy, n, sample_dates[0], sample_dates[-1])

    if not args.no_forward_fill:
        for ccy in list(raw.keys()):
            filled = _forward_fill(raw[ccy], start, end)
            n_extra = len(filled) - sum(1 for d in raw[ccy] if start <= d <= end)
            raw[ccy] = filled
            log.info("forward-filled %s: %d calendar days in range (added "
                     "%d gap days)", ccy, len(filled), max(0, n_extra))
    else:
        for ccy in list(raw.keys()):
            raw[ccy] = {d: v for d, v in raw[ccy].items() if start <= d <= end}

    existing = _load_existing_cache()
    log.info("existing cache has %d keys total", len(existing))

    snapshot_before = dict(existing) if args.dry_run else None
    report = _apply(existing, raw, overwrite=not args.no_overwrite)

    log.info("=== seed report ===")
    for ccy, r in report["by_ccy"].items():
        log.info(
            "  %s: added=%d  updated=%d  unchanged=%d  skipped_existing=%d  total=%d",
            ccy, r["added"], r["updated"], r["unchanged"],
            r["skipped_existing"], r["total_in_range"],
        )
        for d, old, new in r["sample_deltas"]:
            pct = (new - old) / old * 100 if old else 0.0
            log.info("      delta %s: %.6f → %.6f  (%+.3f %%)", d, old, new, pct)

    if args.dry_run:
        log.info("dry-run: not writing to %s", CACHE_PATH)
        return 0

    _flush_cache(existing)
    log.info("wrote %d keys to %s", len(existing), CACHE_PATH)
    return 0


if __name__ == "__main__":
    sys.exit(main())
