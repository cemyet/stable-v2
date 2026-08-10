"""Pull race comments from TR Media and attach them to our entries.

Drives off our own race table rather than the feed's: for every Swedish
raceday in the window we already know which races exist and which of their
entries still lack a comment, so we ask only for those. A day that is fully
written up costs nothing on the next pass, which is what makes a 10-day
re-walk affordable every night.

The re-walk is not optional. Comments are written by hand and land 1-3 days
after the race, so a nightly that only looked at yesterday would see the
occasional early raceday and permanently miss the rest.
"""

from __future__ import annotations

import logging
from datetime import date, timedelta

import httpx

from core.config import TR_COMMENT_WINDOW_DAYS, TR_HEADERS
from etl import comment_match
from scrapers import tr

log = logging.getLogger(__name__)

SOURCE = "tr_api"

_STAGING = "tr_comment_stage"

_CREATE_STAGING_SQL = f"""
DROP TABLE IF EXISTS {_STAGING};
CREATE UNLOGGED TABLE {_STAGING} (
  horse_name   text,
  atg_id       text,
  start_id     bigint,
  race_date    date,
  track_name   text,
  race_number  integer,
  start_number integer,
  comment      text
);
"""

# Races we would gain something by fetching: Swedish, already run, and holding
# at least one entry with no comment yet. `withdrawn` runners are included on
# purpose — the feed comments those too ("Struken: sårskada.").
_TARGETS_SQL = """
SELECT r.race_date, t.name AS track_name, r.race_number,
       COUNT(*) AS entries,
       COUNT(ec.entry_id) AS commented
  FROM race r
  JOIN track t ON t.track_id = r.track_id
  JOIN entry e ON e.race_id = r.race_id
  LEFT JOIN entry_comment ec ON ec.entry_id = e.entry_id
 WHERE r.race_date BETWEEN %(start)s AND %(end)s
   AND t.country = 'SE'
 GROUP BY 1, 2, 3
HAVING COUNT(ec.entry_id) < COUNT(*)
 ORDER BY 1, 2, 3
"""


def _norm(s: str) -> str:
    return "".join(ch for ch in (s or "").lower() if ch.isalnum())


def _slug_map(meets: list[dict]) -> dict[tuple[str, str], str]:
    """(race_date, normalised track name) -> the feed's *track* slug.

    Note this is `track.slug` ("arvika"), not the meet's own slug, which
    already carries the date ("arvika-2026-07-25") and would duplicate it in
    the race URL. Taking the slug from the feed rather than transliterating our
    own track name keeps us honest about spellings we have not seen yet."""
    out: dict[tuple[str, str], str] = {}
    for m in meets:
        track = tr.meet_track(m)
        d = m.get("race_date")
        if track.get("slug") and d and track.get("name"):
            out[(str(d), _norm(track["name"]))] = track["slug"]
    return out


def _targets(conn, start: date, end: date) -> list[tuple[date, str, int]]:
    with conn.cursor() as cur:
        cur.execute(_TARGETS_SQL, {"start": start, "end": end})
        return [(r[0], r[1], r[2]) for r in cur.fetchall()]


def _stage(conn, rows: list[dict]) -> int:
    with conn.cursor() as cur:
        cur.execute(_CREATE_STAGING_SQL)
        if rows:
            args = [(r["horse_name"],
                     str(r["atg_id"]) if r.get("atg_id") is not None else None,
                     r["start_id"], r["race_date"], r["track_name"],
                     r["race_number"], r["start_number"], r["comment"])
                    for r in rows]
            cur.executemany(
                f"INSERT INTO {_STAGING} (horse_name, atg_id, start_id, "
                f"race_date, track_name, race_number, start_number, comment) "
                f"VALUES (%s,%s,%s,%s,%s,%s,%s,%s)", args)
    conn.commit()
    return len(rows)


def import_window(conn, *, start: date, end: date, min_score: int = 3,
                  execute: bool = True, log_fn=None) -> dict:
    """Fetch and attach comments for every incompletely-commented SE race
    whose date falls in [start, end]."""
    say = log_fn or (lambda m: log.info("%s", m))

    targets = _targets(conn, start, end)
    if not targets:
        say(f"[tr] {start}..{end}: every race already fully commented")
        return {"races_targeted": 0, "rows": 0, "written": 0}

    with httpx.Client(headers=TR_HEADERS) as client:
        meets = tr.fetch_meets(start, end, client=client)
        slugs = _slug_map(meets)

        fetch: list[tuple[str, str, int]] = []
        unslugged: set[str] = set()
        for race_date, track_name, race_number in targets:
            slug = slugs.get((race_date.isoformat(), _norm(track_name)))
            if not slug:
                unslugged.add(f"{race_date} {track_name}")
                continue
            fetch.append((slug, race_date.isoformat(), race_number))

        say(f"[tr] {start}..{end}: {len(targets)} races missing comments, "
            f"{len(fetch)} addressable across {len(meets)} SE meets")
        if unslugged:
            say(f"[tr]   no meet in feed for: {sorted(unslugged)[:8]}"
                f"{' ...' if len(unslugged) > 8 else ''}")

        rows = list(tr.scrape_races(fetch, client=client, log_fn=None))

    say(f"[tr]   fetched {len(rows):,} comment rows")
    if not rows:
        return {"races_targeted": len(fetch), "rows": 0, "written": 0}

    _stage(conn, rows)
    stats = comment_match.build_matches(conn, _STAGING)
    say(f"[tr]   matched {stats['matched']:,}/{stats['rows_with_comment']:,} "
        f"(by id {stats['matched_by_atg_id']:,}, by context "
        f"{stats['matched_by_race_context']:,}, unmatched {stats['unmatched']:,})")
    if stats["collisions"]:
        say(f"[tr]   ! {stats['collisions']:,} rows collided on the same entry")

    eligible = comment_match.eligible_count(conn, min_score)
    stats["races_targeted"] = len(fetch)
    stats["rows"] = len(rows)
    stats["eligible"] = eligible

    if not execute:
        say(f"[tr]   DRY RUN — {eligible:,} eligible, nothing written")
        stats["written"] = 0
    else:
        stats["written"] = comment_match.upsert(conn, source=SOURCE,
                                                min_score=min_score)
        say(f"[tr]   wrote/updated {stats['written']:,} rows")

    with conn.cursor() as cur:
        cur.execute(f"DROP TABLE IF EXISTS {_STAGING}")
    comment_match.drop_match_table(conn)
    conn.commit()
    return stats


def run_recent(conn, *, days: int = TR_COMMENT_WINDOW_DAYS, log_fn=None) -> dict:
    """Nightly entry point: re-walk the trailing window."""
    end = date.today()
    return import_window(conn, start=end - timedelta(days=days), end=end,
                         log_fn=log_fn)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    import argparse
    from core.db import get_connection

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--start", type=date.fromisoformat, default=None,
                    help="first race date (default: --days back from today)")
    ap.add_argument("--end", type=date.fromisoformat, default=None,
                    help="last race date (default: today)")
    ap.add_argument("--days", type=int, default=TR_COMMENT_WINDOW_DAYS,
                    help="window size when --start is omitted")
    ap.add_argument("--dry-run", action="store_true",
                    help="match but do not write")
    ap.add_argument("--min-score", type=int, default=3, choices=(1, 2, 3),
                    help="how many of race-number/track/start-number must "
                         "agree before a row is written (default 3 = all)")
    args = ap.parse_args()

    logging.basicConfig(level=logging.WARNING,
                        format="%(asctime)s %(levelname)s %(message)s")

    end = args.end or date.today()
    start = args.start or (end - timedelta(days=args.days))

    conn = get_connection()
    try:
        import_window(conn, start=start, end=end, min_score=args.min_score,
                      execute=not args.dry_run, log_fn=print)
    finally:
        conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
