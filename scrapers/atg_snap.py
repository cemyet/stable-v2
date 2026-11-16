"""How a marking game's odds and stake shares form before the start.

`atg_race_pool` and `atg_game_start` keep only the final win odds and the
final stake shares. This keeps the path to them: every five minutes (launchd,
deploy/com.stable-v2.atgsnap.plist) the open V4 … V86 games of the coming
days are read, and every leg whose own start is still ahead is stored as one
`atg_pool_snap` row: start numbers, win odds, the game's stake shares
(streck), the leg's win pool and the game's turnover so far.

A game is read when one of its legs is due. The cadence follows the leg's
start: every 12 h more than a day out, every 3 h from a day, hourly from 6 h,
every 15 min from 2 h and every run in the last half hour. Each read stores
every upcoming leg, so the moment the game closes (its first leg starts) is
one consistent picture. A leg whose pools haven't moved isn't stored again.

    python -m scrapers.atg_snap             # one pass (what launchd runs)
    python -m scrapers.atg_snap --days 2    # look two days ahead, not five
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timedelta
from pathlib import Path

import httpx
from psycopg2.extras import execute_values

from core.config import ATG_CALENDAR_URL, ATG_GAME_URL, ATG_HEADERS
from core.db import get_connection

GAME_TYPES = ('V86', 'V85', 'V75', 'GS75', 'V64', 'V65', 'V5', 'V4')
OPEN = ('bettable', 'ongoing')
# (at least this many minutes to the leg's start, minutes between rows)
CADENCE = ((1440, 720), (360, 180), (120, 60), (30, 15))
SLACK = 2                     # minutes: launchd's five-minute beat drifts
CALENDAR_CACHE = Path(__file__).resolve().parents[1] / 'logs' / 'atg_snap_calendar.json'
CALENDAR_TTL = 3600           # seconds


def gap_minutes(mins_to_start: float) -> int:
    for ahead, gap in CADENCE:
        if mins_to_start >= ahead:
            return gap
    return 0


def _local(ts: str | None) -> datetime | None:
    if not ts:
        return None
    try:
        return datetime.fromisoformat(ts.replace(' ', 'T')[:19])
    except ValueError:
        return None


def calendar_games(client: httpx.Client, days: int, now: datetime) -> list[tuple[str, str]]:
    """(game id, game type) of every open marking game from today on; the
    calendar is cached for an hour, so a game that opens is read within it."""
    try:
        cache = json.loads(CALENDAR_CACHE.read_text())
    except (OSError, ValueError):
        cache = {}
    out, dirty = [], False
    for d in range(days + 1):
        day = (now.date() + timedelta(days=d)).isoformat()
        hit = cache.get(day)
        if not hit or time.time() - hit['at'] > CALENDAR_TTL:
            r = client.get(ATG_CALENDAR_URL.format(date=day))
            if r.status_code != 200:
                continue
            games = r.json().get('games') or {}
            hit = {'at': time.time(), 'games': [
                [g['id'], gt, g.get('status')] for gt in GAME_TYPES for g in games.get(gt) or []
                if isinstance(g, dict) and g.get('id')]}
            cache[day], dirty = hit, True
        out += [(gid, gt) for gid, gt, status in hit['games'] if status in OPEN]
    if dirty:
        keep = {k: v for k, v in cache.items() if k >= now.date().isoformat()}
        CALENDAR_CACHE.parent.mkdir(parents=True, exist_ok=True)
        CALENDAR_CACHE.write_text(json.dumps(keep))
    return out


def last_rows(cur, gids: list[str]) -> dict[str, dict[int, tuple]]:
    """Per game and leg: (captured_at, start_time, win_turnover, game_turnover) of its last row."""
    cur.execute("""
        SELECT DISTINCT ON (atg_game_id, leg) atg_game_id, leg, captured_at, start_time,
               win_turnover, game_turnover
          FROM atg_pool_snap WHERE atg_game_id = ANY(%s)
         ORDER BY atg_game_id, leg, captured_at DESC
    """, (gids,))
    out: dict[str, dict[int, tuple]] = {}
    for gid, leg, at, start, win_t, game_t in cur.fetchall():
        out.setdefault(gid, {})[leg] = (at, start, win_t, game_t)
    return out


def is_due(legs: dict[int, tuple] | None, now: datetime) -> bool:
    """A game is read when it was never stored, or one of its upcoming legs is due."""
    if not legs:
        return True
    for at, start, _, _ in legs.values():
        if start is None:
            return True
        mins = (start - now).total_seconds() / 60
        if mins < -10:
            continue
        if (now - at).total_seconds() / 60 >= gap_minutes(mins) - SLACK:
            return True
    return False


def rows_for(payload: dict, gid: str, gtype: str, now: datetime,
             prev: dict[int, tuple] | None) -> list[tuple]:
    pool = (payload.get('pools') or {}).get(gtype) or {}
    game_status = pool.get('status') or payload.get('status')
    game_t, systems = pool.get('turnover'), pool.get('systemCount')
    rows = []
    for leg, race in enumerate(payload.get('races') or [], start=1):
        start = _local(race.get('startTime') or race.get('scheduledStartTime'))
        if race.get('status') not in (None, 'upcoming') or (start and start < now - timedelta(minutes=10)):
            continue
        win_t = ((race.get('pools') or {}).get('vinnare') or {}).get('turnover')
        last = (prev or {}).get(leg)
        if last and last[2] == win_t and last[3] == game_t:
            continue
        numbers, odds, streck, scr = [], [], [], []
        for s in race.get('starts') or []:
            sp = s.get('pools') or {}
            o = (sp.get('vinnare') or {}).get('odds')
            b = (sp.get(gtype) or {}).get('betDistribution')
            numbers.append(s.get('number'))
            odds.append(round(o / 100, 2) if isinstance(o, (int, float)) and o > 0 else None)
            streck.append(round(b / 100, 2) if isinstance(b, (int, float)) else None)
            scr.append(bool(s.get('scratched') or (s.get('horse') or {}).get('scratched')))
        if not numbers:
            continue
        rows.append((gid, leg, now, race.get('id'), start, game_status, game_t, systems,
                     win_t, numbers, odds, streck, scr))
    return rows


def run(days: int = 5, log=print) -> dict:
    now = datetime.now().replace(microsecond=0)
    client = httpx.Client(headers=ATG_HEADERS, timeout=20)
    conn = get_connection()
    stats = {'games': 0, 'read': 0, 'rows': 0, 'failed': 0}
    try:
        with conn.cursor() as cur:
            games = calendar_games(client, days, now)
            stats['games'] = len(games)
            prev = last_rows(cur, [g for g, _ in games]) if games else {}
            rows = []
            for gid, gtype in games:
                if not is_due(prev.get(gid), now):
                    continue
                try:
                    r = client.get(ATG_GAME_URL.format(atg_game_id=gid))
                    r.raise_for_status()
                    payload = r.json()
                except (httpx.HTTPError, ValueError) as e:
                    stats['failed'] += 1
                    log(f"[atg-snap] {gid}: {e}")
                    continue
                stats['read'] += 1
                if payload.get('status') not in OPEN:
                    continue
                rows += rows_for(payload, gid, gtype, now, prev.get(gid))
                time.sleep(0.2)
            if rows:
                execute_values(cur, """
                    INSERT INTO atg_pool_snap (atg_game_id, leg, captured_at, atg_race_id, start_time,
                        game_status, game_turnover, systems, win_turnover, numbers, win_odds, streck, scratched)
                    VALUES %s ON CONFLICT DO NOTHING
                """, rows)
            stats['rows'] = len(rows)
        conn.commit()
    finally:
        conn.close()
        client.close()
    if stats['read'] or stats['failed']:
        log(f"[atg-snap] {now:%Y-%m-%d %H:%M} games={stats['games']} read={stats['read']} "
            f"rows={stats['rows']} failed={stats['failed']}")
    return stats


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--days', type=int, default=5, help='days ahead to look (default 5)')
    run(ap.parse_args().days)


if __name__ == '__main__':
    main()
