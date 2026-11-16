"""ATG pool results: what each Swedish trot game paid and how the public
spread its stakes, for backtesting models against real pools.

The games endpoint (`/games/{id}`) carries, per leg, every starter's share
of the game's stakes (`betDistribution`, the "streck"), the final win odds
and the place odds, and per game the turnover, each tier's pot and what a
winning row paid. A day is covered by its marking games (V86 … V3, dd, ld)
plus a single-race `plats_` game for every race none of them contains, so
each Swedish race lands once in `atg_race_pool` however it was played.

Stored compact (atg_game, atg_game_start, atg_race_pool), not raw: a game
payload is ~120 kB and only a few hundred bytes of it matter here.

    python -m scrapers.atg_games                 # last 4 days
    python -m scrapers.atg_games 2025-07-01 2026-11-15
    python -m scrapers.atg_games --load out.jsonl  # compact records from a file
"""

from __future__ import annotations

import json
import time
from datetime import date, datetime, timedelta

import httpx
from psycopg2.extras import Json, execute_values

from core import config
from scrapers.atg import make_client
from scrapers.st_horse import _get_json

# Coverage order: the bigger games first, so their legs need no plats_ fetch.
GAME_TYPES = ('V86', 'V85', 'V75', 'GS75', 'V64', 'V65', 'V5', 'V4', 'V3', 'dd', 'ld')


def _game_date(game_id: str) -> date:
    return datetime.strptime(game_id.split('_')[1], '%Y-%m-%d').date()


def compact(payload: dict, game_type: str, rtp: int | None) -> dict:
    """The parts of a game payload the backtests read."""
    races = []
    for r in payload.get('races', []) or []:
        pools = r.get('pools') or {}
        place = ((pools.get('plats') or {}).get('result') or {}).get('winners') or {}
        leg = (pools.get(game_type) or {}).get('result') or {}
        starts = []
        for s in r.get('starts', []) or []:
            sp = s.get('pools') or {}
            starts.append({
                'n': s.get('number'), 'scr': bool(s.get('scratched')),
                'dist': (sp.get(game_type) or {}).get('betDistribution'),
                'wo': (sp.get('vinnare') or {}).get('odds'),
                'pmin': (sp.get('plats') or {}).get('minOdds'),
                'pmax': (sp.get('plats') or {}).get('maxOdds'),
                'place': (s.get('result') or {}).get('place'),
            })
        races.append({'id': r['id'], 'status': r.get('status'), 'plats': place,
                      'leg': {'winners': leg.get('winners'), 'systems': leg.get('systems'),
                              'reserve': leg.get('reserveOrder')},
                      'starts': starts})
    gp = (payload.get('pools') or {}).get(game_type) or {}
    return {'id': payload['id'], 'type': game_type, 'rtp': rtp, 'status': payload.get('status'),
            'turnover': gp.get('turnover'), 'systemCount': gp.get('systemCount'),
            'result': gp.get('result'), 'payouts': gp.get('payouts'),
            'jackpotAmount': gp.get('jackpotAmount'), 'races': races}


def store_game(conn, g: dict) -> None:
    """Upsert one compact game and the win / place pool of its races."""
    gid = g['id']
    gdate = _game_date(gid)
    payouts = (g.get('result') or {}).get('payouts')
    with conn.cursor() as cur:
        cur.execute("""
            INSERT INTO atg_game (atg_game_id, game_type, game_date, status, rtp, turnover,
                                  system_count, tier_pots, payouts, jackpot, race_ids, legs,
                                  scraped_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW())
            ON CONFLICT (atg_game_id) DO UPDATE SET
                status = EXCLUDED.status, rtp = EXCLUDED.rtp, turnover = EXCLUDED.turnover,
                system_count = EXCLUDED.system_count, tier_pots = EXCLUDED.tier_pots,
                payouts = EXCLUDED.payouts, jackpot = EXCLUDED.jackpot,
                race_ids = EXCLUDED.race_ids, legs = EXCLUDED.legs, scraped_at = NOW()
        """, (gid, g['type'], gdate, g.get('status'), g.get('rtp'), g.get('turnover'),
              g.get('systemCount'), Json(g['payouts']) if g.get('payouts') else None,
              Json(payouts) if payouts else None, g.get('jackpotAmount'),
              [r['id'] for r in g['races']],
              Json([r.get('leg') or {} for r in g['races']]) if g['type'] != 'plats' else None))
        cur.execute("DELETE FROM atg_game_start WHERE atg_game_id = %s", (gid,))
        shares = [(gid, leg + 1, s['n'], (s['dist'] / 1e4) if s['dist'] is not None else None, s['scr'])
                  for leg, r in enumerate(g['races']) for s in r['starts']
                  if s['n'] is not None and g['type'] != 'plats']
        if shares:
            execute_values(cur, "INSERT INTO atg_game_start VALUES %s ON CONFLICT DO NOTHING", shares)
        pool = []
        for r in g['races']:
            pl = r.get('plats') or {}
            if r.get('status') != 'results' or not pl.get('first'):
                continue
            paid = {w['number']: w['odds'] / 100 for k in ('first', 'second', 'third')
                    for w in (pl.get(k) or [])}
            n_paid = sum(1 for k in ('first', 'second', 'third') if pl.get(k))
            for s in r['starts']:
                if s['n'] is None:
                    continue
                pool.append((r['id'], s['n'], gdate,
                             s['wo'] / 100 if s['wo'] else None,
                             s['pmin'] / 100 if s['pmin'] else None,
                             s['pmax'] / 100 if s['pmax'] else None,
                             paid.get(s['n']), n_paid, s['place'] or None, s['scr']))
        if pool:
            execute_values(cur, """
                INSERT INTO atg_race_pool (atg_race_id, number, race_date, win_odds, place_min,
                                           place_max, place_paid, places_paid, finish, scratched)
                VALUES %s
                ON CONFLICT (atg_race_id, number) DO UPDATE SET
                    win_odds = EXCLUDED.win_odds, place_min = EXCLUDED.place_min,
                    place_max = EXCLUDED.place_max, place_paid = EXCLUDED.place_paid,
                    places_paid = EXCLUDED.places_paid, finish = EXCLUDED.finish,
                    scratched = EXCLUDED.scratched
            """, pool)
    conn.commit()


def scrape_day(conn, date_str: str, *, client: httpx.Client | None = None,
               skip_done: bool = True, delay: float | None = None, log=print) -> dict:
    """Every finished Swedish trot game of a day, plus a place pool for each
    race no game covers."""
    delay = config.REQUEST_DELAY if delay is None else delay
    own = client is None
    client = client or make_client()
    counts = {'date': date_str, 'games': 0, 'plats': 0, 'skipped': 0, 'failed': 0}
    try:
        status, cal = _get_json(client, config.ATG_CALENDAR_URL.format(date=date_str))
        if status != 200 or not isinstance(cal, dict):
            counts['calendar_status'] = status
            return counts
        se = {r['id'] for t in cal.get('tracks', []) or []
              if t.get('countryCode') == 'SE' and t.get('sport') == 'trot'
              for r in t.get('races', []) or [] if r.get('status') == 'results'}
        done = set()
        if skip_done:
            with conn.cursor() as cur:
                cur.execute("SELECT atg_game_id FROM atg_game WHERE game_date = %s AND status = 'results'",
                            (date_str,))
                done = {r[0] for r in cur.fetchall()}

        def fetch(gid, gtype, rtp, key):
            if gid in done:
                counts['skipped'] += 1
                return
            st, payload = _get_json(client, config.ATG_GAME_URL.format(atg_game_id=gid))
            if delay:
                time.sleep(delay)
            if st != 200 or not isinstance(payload, dict) or payload.get('status') != 'results':
                counts['failed'] += st != 404
                return
            store_game(conn, compact(payload, gtype, rtp))
            counts[key] += 1

        covered = set()
        games = cal.get('games') or {}
        for gtype in GAME_TYPES:
            for g in games.get(gtype, []) or []:
                legs = set(g.get('races') or [])
                if not legs & se or g.get('status') != 'results':
                    continue
                covered |= legs
                fetch(g['id'], gtype, g.get('returnToPlayer'), 'games')
        for rid in sorted(se - covered):
            fetch(f'plats_{rid}', 'plats', None, 'plats')
    finally:
        if own:
            client.close()
    return counts


def scrape_window(conn, *, days_back: int = 4, end_date: date | None = None,
                  start_date: date | None = None, log=print) -> dict:
    """Yesterday and the days before it (or `start_date` .. `end_date`)."""
    end = end_date or (date.today() - timedelta(days=1))
    start = start_date or (end - timedelta(days=days_back - 1))
    client = make_client()
    totals = {'days': 0, 'games': 0, 'plats': 0, 'skipped': 0, 'failed': 0}
    try:
        d = start
        while d <= end:
            c = scrape_day(conn, d.isoformat(), client=client, log=log)
            totals['days'] += 1
            for k in ('games', 'plats', 'skipped', 'failed'):
                totals[k] += c.get(k, 0)
            log(f"  [atg-games] {d.isoformat()}: games={c['games']} plats={c['plats']} "
                f"skip={c['skipped']} fail={c['failed']}")
            d += timedelta(days=1)
    finally:
        client.close()
    return totals


def load_jsonl(conn, path: str, log=print) -> int:
    """Store compact game records written one per line."""
    n = 0
    with open(path) as f:
        for line in f:
            g = json.loads(line)
            if g.get('status') == 'results':
                store_game(conn, g)
                n += 1
    log(f"  [atg-games] loaded {n:,} games from {path}")
    return n


def main() -> None:
    import sys
    from core.db import get_connection

    args = sys.argv[1:]
    conn = get_connection()
    try:
        if args and args[0] == '--load':
            load_jsonl(conn, args[1])
        elif len(args) == 2:
            print(scrape_window(conn, start_date=date.fromisoformat(args[0]),
                                end_date=date.fromisoformat(args[1])))
        else:
            print(scrape_window(conn, days_back=int(args[0]) if args else 4))
    finally:
        conn.close()


if __name__ == '__main__':
    main()
