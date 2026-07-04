"""Backfill (or recompute) FX rates on `entry` rows.

Two phases:
    1. Pre-warm `core.exchange` cache for every (currency, year) pair we care
       about — one Riksbank API call per year covers all trading days for that
       currency. With nine currencies × ~30 years that's <300 calls, well
       under the 1000-call daily quota. Skip with `--skip-prewarm` if the
       cache has already been seeded by another means (e.g. ECB via
       `scripts.seed_fx_from_ecb`).
    2. Walk `entry` rows that need FX, look the rate up from the now-warmed
       cache, and UPDATE in batches. By default only rows where
       `prize_fx_rate IS NULL` are touched. With `--recompute-all`, every
       foreign-currency row is re-evaluated against the current cache and
       updated if the rate disagrees (use `--backup-csv` to snapshot the
       old values first).

Usage:
    python -m scripts.backfill_fx                          # all ccys, fill NULLs
    python -m scripts.backfill_fx --currencies EUR --skip-prewarm
    python -m scripts.backfill_fx --dry-run                # report only
    python -m scripts.backfill_fx --recompute-all --currencies EUR \\
        --skip-prewarm --backup-csv logs/fx_recompute_$(date +%s).csv

`--recompute-all` is intended for one-shot source-of-truth swaps (e.g.
"replace every Riksbank-derived rate with the ECB equivalent"); it keeps an
audit trail in the backup CSV so the previous values can be restored.
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
import time
from datetime import date
from pathlib import Path

from core.config import RIKSBANK_SUPPORTED_CCYS
from core.db import get_connection
from core.exchange import _MEM, _load_disk_cache, prefetch_range, set_strict_cache, to_sek

log = logging.getLogger("backfill_fx")

# Riksbank publishes most series back to 1993; EUR only exists from the euro's
# introduction. Anything earlier returns an empty list — harmless but slow, so
# we skip those years up-front.
SERIES_EARLIEST_YEAR = {
    "EUR": 1999,
    "USD": 1993,
    "GBP": 1993,
    "NOK": 1993,
    "DKK": 1993,
    "CHF": 1993,
    "JPY": 1993,
    "AUD": 1993,
    "CAD": 1993,
}

UPDATE_BATCH = 5000


def prewarm(currencies: list[str], start_year: int, end_year: int) -> None:
    """Fetch a full year per (ccy, year) — one API call each.

    Skips years that already have enough observations in the on-disk cache
    (≥200 ≈ a full year of trading days), so re-runs after a partial
    failure don't waste Riksbank quota on data we already have.
    """
    _load_disk_cache()
    plan: list[tuple[str, int]] = []
    skipped: list[tuple[str, int]] = []
    for ccy in currencies:
        first = max(SERIES_EARLIEST_YEAR.get(ccy, 1993), start_year)
        for year in range(first, end_year + 1):
            cached = sum(1 for k in _MEM
                         if k.startswith(f"{ccy}:{year}-"))
            if cached >= 200:
                skipped.append((ccy, year))
            else:
                plan.append((ccy, year))

    log.info("pre-warming FX cache: %d API calls (skipped %d already-cached "
             "years; ≤4/min ≈ %.1f min total)",
             len(plan), len(skipped), len(plan) * 15 / 60)

    for i, (ccy, year) in enumerate(plan, 1):
        yr_start = date(year, 1, 1)
        yr_end = date(year, 12, 31)
        try:
            n = prefetch_range(ccy, yr_start, yr_end)
            log.info("[%d/%d] %s %d → cached %d observations",
                     i, len(plan), ccy, year, n)
        except Exception as exc:
            log.warning("[%d/%d] %s %d failed: %r", i, len(plan), ccy, year, exc)


def backfill_entries(
    conn,
    currencies: list[str],
    dry_run: bool = False,
    recompute_all: bool = False,
    backup_csv: str | None = None,
) -> dict:
    """Walk every entry needing FX and update it from the warmed cache.

    Reads the entire candidate set into memory in one shot (cheap: a few
    tens of MB even for ~1M rows) so we don't have to maintain a
    server-side cursor across COMMITs.

    recompute_all=True selects every foreign-currency row (not just
    `prize_fx_rate IS NULL`) and updates it whenever the cache rate
    differs from the stored value. When `backup_csv` is provided, the
    pre-update values are appended so the operation can be reversed:
        entry_id, prize_kr_old, prize_fx_rate_old, prize_fx_date_old,
                  prize_kr_new, prize_fx_rate_new, prize_fx_date_new
    """
    summary = {
        "scanned":         0,
        "updated":         0,
        "rate_changed":    0,
        "newly_filled":    0,
        "unchanged":       0,
        "still_missing":   0,
        "by_ccy":          {},
    }

    placeholders = ", ".join(["%s"] * len(currencies))
    where_filter = "" if recompute_all else "\n           AND e.prize_fx_rate IS NULL"
    extra_cols = (
        ", e.prize_kr, e.prize_fx_rate, e.prize_fx_date"
        if recompute_all else ""
    )
    select_sql = f"""
        SELECT e.entry_id, e.prize_currency, e.prize_original, r.race_date{extra_cols}
          FROM entry e
          JOIN race  r USING (race_id)
         WHERE e.prize_currency IN ({placeholders})
           AND e.prize_original IS NOT NULL{where_filter}
    """

    update_sql = """
        UPDATE entry
           SET prize_kr      = %s,
               prize_fx_rate = %s,
               prize_fx_date = %s
         WHERE entry_id = %s
    """

    read_cur = conn.cursor()
    read_cur.execute(select_sql, tuple(currencies))
    rows = read_cur.fetchall()
    read_cur.close()
    log.info("fetched %d candidate rows into memory%s",
             len(rows), " (recompute-all mode)" if recompute_all else "")

    backup_fh = None
    backup_writer = None
    if backup_csv and not dry_run:
        backup_path = Path(backup_csv)
        backup_path.parent.mkdir(parents=True, exist_ok=True)
        new_file = not backup_path.exists()
        backup_fh = backup_path.open("a", newline="")
        backup_writer = csv.writer(backup_fh)
        if new_file:
            backup_writer.writerow([
                "entry_id",
                "prize_kr_old", "prize_fx_rate_old", "prize_fx_date_old",
                "prize_kr_new", "prize_fx_rate_new", "prize_fx_date_new",
            ])
        log.info("backup CSV: %s", backup_path)

    write_cur = conn.cursor()
    batch: list[tuple] = []
    t0 = time.time()
    last_log = t0

    try:
        for row in rows:
            if recompute_all:
                (entry_id, ccy, prize_orig, race_date,
                 old_prize_kr, old_fx_rate, old_fx_date) = row
            else:
                entry_id, ccy, prize_orig, race_date = row
                old_prize_kr = old_fx_rate = old_fx_date = None

            summary["scanned"] += 1
            counts = summary["by_ccy"].setdefault(
                ccy,
                {"updated": 0, "missing": 0, "rate_changed": 0,
                 "newly_filled": 0, "unchanged": 0},
            )

            prize_kr, _amt, fx_rate, fx_date = to_sek(prize_orig, ccy, race_date)
            if fx_rate is None:
                summary["still_missing"] += 1
                counts["missing"] += 1
                continue

            if recompute_all and old_fx_rate is not None:
                if (
                    abs(float(old_fx_rate) - float(fx_rate)) < 1e-9
                    and int(old_prize_kr or 0) == int(prize_kr or 0)
                ):
                    summary["unchanged"] += 1
                    counts["unchanged"] += 1
                    continue
                if backup_writer:
                    backup_writer.writerow([
                        entry_id,
                        int(old_prize_kr) if old_prize_kr is not None else "",
                        float(old_fx_rate),
                        old_fx_date.isoformat() if old_fx_date else "",
                        int(prize_kr or 0), float(fx_rate),
                        fx_date.isoformat() if fx_date else "",
                    ])
                summary["rate_changed"] += 1
                counts["rate_changed"] += 1
            else:
                summary["newly_filled"] += 1
                counts["newly_filled"] += 1

            batch.append((int(prize_kr or 0), float(fx_rate), fx_date, entry_id))
            summary["updated"] += 1
            counts["updated"] += 1

            if len(batch) >= UPDATE_BATCH:
                if not dry_run:
                    write_cur.executemany(update_sql, batch)
                    conn.commit()
                    if backup_fh:
                        backup_fh.flush()
                batch.clear()

            now = time.time()
            if now - last_log >= 5.0:
                last_log = now
                rps = summary["scanned"] / max(1e-6, now - t0)
                log.info(
                    "…scanned %d  updated %d  (new=%d changed=%d unchanged=%d "
                    "missing=%d)  (%.0f rows/s)",
                    summary["scanned"], summary["updated"],
                    summary["newly_filled"], summary["rate_changed"],
                    summary["unchanged"], summary["still_missing"], rps,
                )

        if batch and not dry_run:
            write_cur.executemany(update_sql, batch)
            conn.commit()
            if backup_fh:
                backup_fh.flush()
    finally:
        write_cur.close()
        if backup_fh:
            backup_fh.close()

    summary["elapsed_s"] = round(time.time() - t0, 1)
    return summary


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--start", type=int, default=1999,
                   help="earliest year to prefetch (default: 1999)")
    p.add_argument("--end", type=int, default=date.today().year,
                   help="latest year to prefetch (default: current year)")
    p.add_argument("--currencies", default=",".join(RIKSBANK_SUPPORTED_CCYS),
                   help="comma-separated list of currencies to backfill "
                        "(default: all Riksbank-supported)")
    p.add_argument("--dry-run", action="store_true",
                   help="report what would be updated, but make no writes")
    p.add_argument("--skip-prewarm", action="store_true",
                   help="skip the Riksbank prefetch; use only what's already "
                        "in fx_cache.json / in-memory")
    p.add_argument("--recompute-all", action="store_true",
                   help="re-evaluate every foreign-currency row against the "
                        "current cache (not just rows with NULL fx). Use this "
                        "after swapping the FX source (e.g. ECB seeding) to "
                        "guarantee a single source of truth.")
    p.add_argument("--backup-csv", default=None,
                   help="append pre-update (entry_id, old_*, new_*) rows to "
                        "this CSV so --recompute-all can be reversed.")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    currencies = [c.strip().upper() for c in args.currencies.split(",") if c.strip()]
    bad = [c for c in currencies if c not in RIKSBANK_SUPPORTED_CCYS]
    if bad:
        log.error("unsupported currencies: %s (supported: %s)",
                  bad, RIKSBANK_SUPPORTED_CCYS)
        return 2

    if not args.skip_prewarm:
        prewarm(currencies, args.start, args.end)
    else:
        log.info("skipping prewarm; using existing cache only")

    # Lock the exchange module to cache-only for the entry walk — guarantees
    # the per-row to_sek() calls cannot trip the Riksbank daily quota even
    # if the prewarm missed a date or currency.
    set_strict_cache(True)
    log.info("strict-cache enabled: no further API calls will be made")

    if args.backup_csv and not args.recompute_all:
        log.warning("--backup-csv has no effect without --recompute-all")

    conn = get_connection()
    try:
        summary = backfill_entries(
            conn,
            currencies,
            dry_run=args.dry_run,
            recompute_all=args.recompute_all,
            backup_csv=args.backup_csv,
        )
    finally:
        conn.close()

    log.info("=== summary ===")
    log.info("scanned:        %d rows", summary["scanned"])
    log.info("updated:        %d rows%s", summary["updated"],
             " (dry-run, no DB writes)" if args.dry_run else "")
    if args.recompute_all:
        log.info("  newly filled: %d rows (prize_fx_rate was NULL)",
                 summary["newly_filled"])
        log.info("  rate changed: %d rows (replaced existing rate)",
                 summary["rate_changed"])
        log.info("  unchanged:    %d rows (cache agreed with stored rate)",
                 summary["unchanged"])
    log.info("still missing:  %d rows (no rate found in cache)",
             summary["still_missing"])
    for ccy, counts in sorted(summary["by_ccy"].items()):
        log.info(
            "  %s: updated=%d  missing=%d  new=%d  changed=%d  unchanged=%d",
            ccy, counts["updated"], counts.get("missing", 0),
            counts.get("newly_filled", 0), counts.get("rate_changed", 0),
            counts.get("unchanged", 0),
        )
    log.info("elapsed:        %.1fs", summary["elapsed_s"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
