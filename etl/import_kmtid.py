"""Import kmtid GPS sectionals into stable_v2.

kmtid is *enrichment-only*: it never creates new races, horses, or persons.
Each kmtid race is matched to an existing v2 race via the trio
    (track.atg_track_id, race.race_date, race.race_number)
and each start is matched to an existing entry via either
    (race_id, post)            -- preferred (post matches kmtid `number`)
    (race_id, lower(horse.name))  -- fallback when post is missing

When a race or entry can't be matched we just count it as `skipped` and
move on; the next ST/ATG sync will eventually create the missing rows and
the next kmtid pass will pick them up.
"""

from __future__ import annotations

import logging
from datetime import date
from typing import Iterable

from psycopg2.extras import Json

from core.db import buffer_insert, buffer_prune
from scrapers.kmtid import scrape_day, scrape_window

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_race_id(raw_id: str | None) -> tuple[date | None, int | None, int | None]:
    """Decompose '2026-05-09_6_1' -> (date(2026,5,9), atg_track_id=6, race_number=1)."""
    if not raw_id or not isinstance(raw_id, str):
        return None, None, None
    parts = raw_id.split("_")
    if len(parts) != 3:
        return None, None, None
    try:
        d = date.fromisoformat(parts[0])
        return d, int(parts[1]), int(parts[2])
    except (ValueError, TypeError):
        return None, None, None


# kmtid uses Number.MAX_SAFE_INTEGER (~9e15) as a placeholder when GPS data
# is incomplete for a horse. No legitimate timing in ms or distance in m comes
# anywhere near 10 million, so this catches the sentinel cleanly.
_MAX_PLAUSIBLE = 10_000_000.0


def _round_or_none(v):
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    if f != f or abs(f) >= _MAX_PLAUSIBLE:  # NaN or sentinel
        return None
    return f


def _int_or_none(v):
    f = _round_or_none(v)
    return None if f is None else int(round(f))


# ---------------------------------------------------------------------------
# Per-race upsert
# ---------------------------------------------------------------------------

def _find_v2_race(cur, atg_track_id: int, d: date, race_number: int) -> int | None:
    cur.execute(
        """
        SELECT r.race_id
          FROM race r
          JOIN track t ON t.track_id = r.track_id
         WHERE t.atg_track_id = %s
           AND r.race_date    = %s
           AND r.race_number  = %s
         LIMIT 1
        """,
        (atg_track_id, d, race_number),
    )
    row = cur.fetchone()
    return row[0] if row else None


def _stamp_race(cur, race_id: int, kmtid_id: str, race_payload: dict) -> None:
    """Stamp the kmtid id + summary onto race row (kmtid is lowest priority,
    so we never overwrite canonical fields — just record kmtid_id and merge
    raw payload into source_data['kmtid']."""
    # Strip starts[] from the race payload before storing — they're per-entry.
    summary = {k: v for k, v in race_payload.items() if k != "starts"}
    cur.execute(
        """
        UPDATE race
           SET kmtid_id = COALESCE(kmtid_id, %s),
               source_data = COALESCE(source_data, '{}'::jsonb)
                          || jsonb_build_object('kmtid', %s::jsonb),
               last_updated_at = NOW()
         WHERE race_id = %s
        """,
        (kmtid_id, Json(summary), race_id),
    )


def _find_v2_entry_by_post(cur, race_id: int, post: int | None) -> int | None:
    if post is None:
        return None
    cur.execute(
        "SELECT entry_id FROM entry WHERE race_id = %s AND post = %s LIMIT 1",
        (race_id, post),
    )
    row = cur.fetchone()
    return row[0] if row else None


def _find_v2_entry_by_horse_name(cur, race_id: int, horse_name: str | None) -> int | None:
    if not horse_name:
        return None
    cur.execute(
        """
        SELECT e.entry_id
          FROM entry e
          JOIN horse h ON h.horse_id = e.horse_id
         WHERE e.race_id = %s
           AND lower(h.name) = lower(%s)
         LIMIT 1
        """,
        (race_id, horse_name.strip()),
    )
    row = cur.fetchone()
    return row[0] if row else None


def _stamp_entry(cur, entry_id: int, start_payload: dict) -> None:
    timings = start_payload.get("timings") or {}
    intervals = timings.get("intervals") or []

    cur.execute(
        """
        UPDATE entry
           SET kmtid_first_200ms          = %s,
               kmtid_last_200ms           = %s,
               kmtid_best_100ms           = %s,
               kmtid_best_100_start_m     = %s,
               kmtid_actual_distance_m    = %s,
               kmtid_actual_km_time_ms    = %s,
               kmtid_slipstream_distance_m = %s,
               kmtid_intervals            = %s,
               source_data                = COALESCE(source_data, '{}'::jsonb)
                                          || jsonb_build_object('kmtid', %s::jsonb),
               last_updated_at            = NOW()
         WHERE entry_id = %s
        """,
        (
            _round_or_none(timings.get("first200ms")),
            _round_or_none(timings.get("last200ms")),
            _round_or_none(timings.get("best100ms")),
            _int_or_none(timings.get("best100start")),
            _int_or_none(timings.get("actualDistanceRan")),
            _round_or_none(_actual_km_time_ms(timings)),
            _int_or_none(timings.get("slipstreamDistance")),
            Json(intervals),
            Json({
                # keep human-readable variants alongside the raw intervals
                "first200":   timings.get("first200"),
                "last200":    timings.get("last200"),
                "best100":    timings.get("best100"),
                "actualKMTime": timings.get("actualKMTime"),
                "result":     start_payload.get("result"),
                "distance":   start_payload.get("distance"),
                "horseName":  (start_payload.get("horse") or {}).get("name"),
                "driverName": (start_payload.get("driver") or {}).get("name"),
            }),
            entry_id,
        ),
    )


def _actual_km_time_ms(timings: dict) -> float | None:
    """Convert kmtid's 'actualKMTime' string ('1.21,6 min/km') into ms.

    Falls back to None if format unknown."""
    raw = timings.get("actualKMTime")
    if not raw or not isinstance(raw, str):
        return None
    # '1.21,6 min/km' -> 1*60 + 21.6 = 81.6 sec/km -> 81600 ms
    s = raw.strip().split()[0]
    try:
        # pattern: M.SS,T
        if "." in s and "," in s:
            mins_part, rest = s.split(".", 1)
            secs_int_part, tenths_part = rest.split(",", 1)
            total_seconds = int(mins_part) * 60 + int(secs_int_part) + int(tenths_part) / 10.0
            return total_seconds * 1000.0
        # fallback: bare seconds with comma decimal
        if "," in s:
            secs_int, tenths = s.split(",", 1)
            return (int(secs_int) + int(tenths) / 10.0) * 1000.0
    except (ValueError, IndexError):
        return None
    return None


# ---------------------------------------------------------------------------
# Race import
# ---------------------------------------------------------------------------

def import_race(cur, race_payload: dict) -> dict:
    """Import a single kmtid race payload. Returns {matched, skipped, entries_matched, entries_skipped}."""
    out = {"matched": 0, "skipped": 0, "entries_matched": 0, "entries_skipped": 0}

    raw_id = race_payload.get("id")
    d, atg_track, race_number = _parse_race_id(raw_id)
    if d is None:
        out["skipped"] = 1
        return out

    race_id = _find_v2_race(cur, atg_track, d, race_number)
    if not race_id:
        log.debug("kmtid race %s: no v2 race match", raw_id)
        out["skipped"] = 1
        return out

    _stamp_race(cur, race_id, raw_id, race_payload)
    out["matched"] = 1

    for s in race_payload.get("starts") or []:
        post = s.get("number")
        horse_name = (s.get("horse") or {}).get("name")
        entry_id = _find_v2_entry_by_post(cur, race_id, post) \
                   or _find_v2_entry_by_horse_name(cur, race_id, horse_name)
        if not entry_id:
            log.debug("kmtid race %s: no entry for post=%s horse=%s",
                      raw_id, post, horse_name)
            out["entries_skipped"] += 1
            continue
        _stamp_entry(cur, entry_id, s)
        out["entries_matched"] += 1

    return out


# ---------------------------------------------------------------------------
# Day import
# ---------------------------------------------------------------------------

def import_day(conn, d: date) -> dict:
    """Scrape one day from kmtid and import. Returns aggregated counts."""
    summary = {
        "date": d.isoformat(),
        "races_seen": 0,
        "races_matched": 0,
        "races_skipped": 0,
        "entries_matched": 0,
        "entries_skipped": 0,
    }
    races = scrape_day(d)
    summary["races_seen"] = len(races)
    if not races:
        return summary

    with conn.cursor() as cur:
        for r in races:
            res = import_race(cur, r)
            summary["races_matched"]   += res["matched"]
            summary["races_skipped"]   += res["skipped"]
            summary["entries_matched"] += res["entries_matched"]
            summary["entries_skipped"] += res["entries_skipped"]
    conn.commit()
    return summary


def import_window(conn, days: int = 35, end_date: date | None = None) -> dict:
    """Scrape + import the trailing window. Returns aggregated counts."""
    totals = {
        "days_with_data": 0,
        "races_seen": 0,
        "races_matched": 0,
        "races_skipped": 0,
        "entries_matched": 0,
        "entries_skipped": 0,
    }
    for d, races in scrape_window(days=days, end_date=end_date):
        if not races:
            continue
        totals["days_with_data"] += 1
        with conn.cursor() as cur:
            for r in races:
                res = import_race(cur, r)
                totals["races_seen"]       += 1
                totals["races_matched"]    += res["matched"]
                totals["races_skipped"]    += res["skipped"]
                totals["entries_matched"]  += res["entries_matched"]
                totals["entries_skipped"]  += res["entries_skipped"]
        conn.commit()
        log.info("kmtid %s: %s", d.isoformat(), totals)
    buffer_prune(conn, "kmtid")
    return totals


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    import sys
    from core.db import get_connection
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print("usage: python -m etl.import_kmtid <YYMMDD|window>")
        return

    conn = get_connection()
    try:
        arg = sys.argv[1]
        if arg == "window":
            print(import_window(conn))
        else:
            try:
                d = date(2000 + int(arg[0:2]), int(arg[2:4]), int(arg[4:6]))
            except (ValueError, IndexError):
                raise SystemExit(f"bad date arg: {arg!r} (expected YYMMDD)")
            print(import_day(conn, d))
    finally:
        conn.close()


if __name__ == "__main__":
    main()
