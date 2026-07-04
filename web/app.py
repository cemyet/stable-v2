#!/usr/bin/env python3
"""
Stable v2 -- horse data viewer for the flat 5-table v2 schema.

Run with:  python3 -m web.app  (from /Users/jakob/Dev/stable-v2/)

Routes mirror v1 except:
  * `/ml` and `/ml/models`  -> read-only saved-models registry (tabbed).
  * `/api/ml/models*`       -> model metadata for that page.
  * `/api/breed/*`          -> removed.
  * `/horse/<src>/<id>`     -> per-source redirect to canonical /horse/<id>.
  * `/race/st/<id>` and `/race/atg/<id>` redirect helpers.

All SQL is rewritten against the canonical 5-table schema in core.schema:
  horse, person, race, entry, track + horse_owner_history, horse_trainer_history.
Per-person recent-form signals (30-day rolling): tf/df = win-rate; tf_perf/df_perf
= `form` (avg finishing percentile, from entry_perf — actual form, predicts
winning); tf_odds/df_odds = `mkt±` (avg odds-rank outperformance, from
entry_outperf — a market-edge/value signal that anti-correlates with winning).
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
from pathlib import Path

_STABLE_V2_ROOT = Path(__file__).resolve().parent.parent
if str(_STABLE_V2_ROOT) not in sys.path:
    sys.path.insert(0, str(_STABLE_V2_ROOT))

import re as _re_mod

import psycopg2
import psycopg2.extras
from flask import Flask, jsonify, redirect, render_template, request, url_for

from core.config import WEB_PORT
import unicodedata as _unicodedata

from core.db import get_connection, get_v1_connection

app = Flask(__name__)
app.config['TEMPLATES_AUTO_RELOAD'] = True
app.jinja_env.auto_reload = True


# ---------------------------------------------------------------------------
# ATG web URL helper
# Resolves the public atg.se betting page URL for a given atg_race_id by
# looking up the game leg + track slug from v1. Falls back to None if not found.
#
# ATG URL format: https://www.atg.se/spel/{date}/{game_type}/{track_slug}/avd/{leg}
# e.g.            https://www.atg.se/spel/2024-01-28/V4/vincennes/avd/2
# ---------------------------------------------------------------------------

def _track_slug(name: str) -> str:
    """ATG track URL slug: ASCII-lower, strip diacritics, drop everything that
    isn't [a-z0-9]. So 'Åby' → 'aby', 'Saint-Galmier' → 'saintgalmier',
    'Cagnes Sur Mer' → 'cagnessurmer'.
    """
    import re as _re
    nfkd = _unicodedata.normalize('NFKD', name)
    ascii_name = ''.join(c for c in nfkd if not _unicodedata.combining(c))
    return _re.sub(r'[^a-z0-9]', '', ascii_name.lower())


def _atg_slug_for_race(atg_race_id: str) -> str | None:
    """Track slug for a race, sourced from per-race raw JSON (`v2_atg_race_raw`)
    and falling back to the static `atg_track` table. ATG re-uses track ids
    across physical tracks over time, so the raw row is preferred."""
    try:
        v1 = get_v1_connection()
    except Exception:
        return None
    try:
        with v1.cursor() as cur:
            cur.execute(
                "SELECT raw_json->'track'->>'name' FROM v2_atg_race_raw "
                "WHERE atg_race_id = %s",
                (atg_race_id,),
            )
            row = cur.fetchone()
            track_name = row[0] if row else None
            if not track_name:
                cur.execute(
                    "SELECT t.name FROM atg_race ar "
                    "JOIN atg_race_day rd ON rd.atg_race_day_id = ar.atg_race_day_id "
                    "JOIN atg_track t ON t.atg_track_id = rd.atg_track_id "
                    "WHERE ar.atg_race_id = %s",
                    (atg_race_id,),
                )
                row = cur.fetchone()
                track_name = row[0] if row else None
            if not track_name:
                return None
            return _track_slug(track_name) or None
    except Exception:
        return None
    finally:
        v1.close()


# Pool priority for resolving "which game does this race headline?" — biggest
# multi-leg pools first, then small ones. Used for the bet-type deep link and
# the live spelprocent column.
_GAME_LEG_PRIORITY = ['V86', 'V85', 'GS75', 'V75', 'V64', 'V65', 'V5', 'V4']


def _atg_find_game_leg(atg_race_id: str) -> dict | None:
    """Resolve the headline game + leg number for a race via the ATG calendar.

    Returns {game_type, game_id, leg, n_legs} or None. The race is matched
    against each game's `races` list (a list of race-id strings)."""
    if not atg_race_id:
        return None
    parts = atg_race_id.split('_')
    if len(parts) < 3:
        return None
    date = parts[0]
    try:
        track_id = int(parts[1])
    except (TypeError, ValueError):
        return None
    cal = _fetch_atg_calendar(date)
    if not cal:
        return None
    games = cal.get('games', {}) or {}
    for gt in _GAME_LEG_PRIORITY:
        for g in games.get(gt, []) or []:
            if track_id in (g.get('tracks') or []):
                rids = g.get('races') or []
                if atg_race_id in rids:
                    return {'game_type': gt, 'game_id': g.get('id'),
                            'leg': rids.index(atg_race_id) + 1, 'n_legs': len(rids)}
    return None


def _atg_race_url(atg_race_id: str) -> str | None:
    """Public atg.se URL for a race.

    Prefers the headline game-leg deep link
    (`/spel/{date}/{GAME}/{slug}/avd/{leg}`, e.g. V86 leg 6) when the race is
    part of a multi-leg pool; otherwise falls back to the plain win/place page
    (`/spel/{date}/plats/{slug}/lopp/{race_number}`)."""
    if not atg_race_id:
        return None
    parts = atg_race_id.split('_')
    if len(parts) < 3:
        return None
    date = parts[0]
    try:
        race_number = int(parts[-1])
    except (TypeError, ValueError):
        return None
    slug = _atg_slug_for_race(atg_race_id)
    if not slug:
        return None
    leg = _atg_find_game_leg(atg_race_id)
    if leg:
        return f"https://www.atg.se/spel/{date}/{leg['game_type']}/{slug}/avd/{leg['leg']}"
    return f"https://www.atg.se/spel/{date}/plats/{slug}/lopp/{race_number}"


def _atg_live_pools(atg_race_id: str) -> dict | None:
    """Live odds + bet distribution (spelprocent) for a race's starts.

    Pulls the headline game (for spelprocent under the game-type key) when the
    race is a multi-leg leg; otherwise the win pool (odds only). Odds and
    distribution are returned in human units (3.52, 35.45%)."""
    ctx = _atg_find_game_leg(atg_race_id)
    game_type = ctx['game_type'] if ctx else None
    game_id = ctx['game_id'] if ctx else f"vinnare_{atg_race_id}"
    game = _fetch_atg_game(game_id)
    if not game:
        return None
    races = game.get('races') or []
    leg = next((r for r in races
                if isinstance(r, dict) and r.get('id') == atg_race_id), None)
    if leg is None and races and isinstance(races[0], dict):
        leg = races[0]
    if leg is None:
        return None

    def _odds(v):
        return round(v / 100, 2) if isinstance(v, (int, float)) and v > 0 else None

    out_starts = []
    for s in leg.get('starts') or []:
        pools = s.get('pools') or {}
        plats = pools.get('plats') or {}
        bd = trend = None
        if game_type and isinstance(pools.get(game_type), dict):
            bd = pools[game_type].get('betDistribution')
            trend = pools[game_type].get('trend')
        out_starts.append({
            'number': s.get('number'),
            'horseId': (s.get('horse') or {}).get('id'),
            'horseName': (s.get('horse') or {}).get('name'),
            'scratched': bool(s.get('scratched') or (s.get('horse') or {}).get('scratched')),
            'winOdds': _odds((pools.get('vinnare') or {}).get('odds')),
            'platsMin': _odds(plats.get('minOdds')),
            'platsMax': _odds(plats.get('maxOdds')),
            'betDist': round(bd / 100, 2) if isinstance(bd, (int, float)) else None,
            'trend': trend,
        })
    poolinfo = leg.get('pools') or {}
    return {
        'gameType': game_type,
        'leg': ctx['leg'] if ctx else None,
        'nLegs': ctx['n_legs'] if ctx else None,
        'gameId': game_id,
        'atgUrl': _atg_race_url(atg_race_id),
        'startTime': leg.get('startTime') or leg.get('scheduledStartTime'),
        'status': leg.get('status'),
        'turnover': (poolinfo.get('vinnare') or {}).get('turnover'),
        'starts': out_starts,
    }


_GENDER_TEXT = {'H': 'stallion', 'V': 'gelding', 'S': 'mare'}
_BREED_TEXT = {'V': 'varmblodig travare', 'K': 'kallblodig travare'}
_FAST_KM_TIME_FLOOR_SECONDS = 65.0


def fmtName(name: str) -> str:
    """Reverse TravSport 'Lastname Firstname' -> 'Firstname Lastname'."""
    if not name:
        return ''
    parts = name.strip().split()
    if len(parts) < 2:
        return name
    return ' '.join(parts[1:]) + ' ' + parts[0]


def shortName(name: str) -> str:
    """Abbreviate raw 'Lastname First [Middle...]' -> 'Las FM' for tight cells."""
    if not name:
        return ''
    parts = name.strip().split()
    if not parts:
        return ''
    last = parts[0][:3].capitalize()
    if len(parts) == 1:
        return last
    firsts = parts[1:]
    if len(firsts) == 1:
        first_short = firsts[0][:2].capitalize()
    else:
        first_short = ''.join(p[0].upper() for p in firsts if p)
    return f"{last} {first_short}"


def get_db():
    return get_connection()


def _kr(value) -> str:
    if value is None:
        return ''
    return f"{int(value):,} kr".replace(',', ' ')


# =====================================================================
# Home
# =====================================================================

@app.route('/home')
def home_page():
    from datetime import date as _date
    return render_template('home.html', active_tab='home', now=_date.today())


# ---------------------------------------------------------------------------
# ATG live calendar — upcoming trot races for home track spheres + game pages
# ---------------------------------------------------------------------------

_HEADLINE_HIERARCHY = ['V85', 'V86', 'GS75', 'V64', 'V65']
_CYAN_SATELLITE_ORDER = ['vinnare', 'dd', 'ld', 'V5', 'V4', 'plats']
# Bet types ATG publishes as one game *per race* (a separate game id per leg).
# The calendar therefore lists N games for a single race day; we treat the
# whole set as one "game" covering every race of the track that day.
_PER_RACE_GAME_TYPES = {'vinnare', 'plats'}
_ATG_CALENDAR_TTL = 120  # seconds


def _atg_game_id_to_internal(game_type_str: str) -> str:
    """ATG uses 'V64', 'V65', 'dd' etc. Map to our internal lowercase ids."""
    return game_type_str.lower()


def _fetch_atg_calendar(date_str: str) -> dict | None:
    """Fetch ATG calendar for a date, with in-memory cache."""
    import time
    cache_key = f'atg_calendar_{date_str}'
    if cache_key in _leaderboard_cache:
        ts, val = _leaderboard_cache[cache_key]
        if time.time() - ts < _ATG_CALENDAR_TTL:
            return val

    import httpx
    from core.config import ATG_CALENDAR_URL, ATG_HEADERS
    url = ATG_CALENDAR_URL.format(date=date_str)
    try:
        resp = httpx.get(url, headers=ATG_HEADERS, timeout=10.0)
        resp.raise_for_status()
        data = resp.json()
    except Exception:
        return None

    _leaderboard_cache[cache_key] = (time.time(), data)
    return data


def _fetch_atg_game(game_id: str) -> dict | None:
    """Fetch ATG game payload (full races with names, starts, etc.)."""
    import time
    cache_key = f'atg_game_{game_id}'
    if cache_key in _leaderboard_cache:
        ts, val = _leaderboard_cache[cache_key]
        if time.time() - ts < _ATG_CALENDAR_TTL:
            return val

    import httpx
    from core.config import ATG_GAME_URL, ATG_HEADERS
    url = ATG_GAME_URL.format(atg_game_id=game_id)
    try:
        resp = httpx.get(url, headers=ATG_HEADERS, timeout=10.0)
        resp.raise_for_status()
        data = resp.json()
    except Exception:
        return None

    _leaderboard_cache[cache_key] = (time.time(), data)
    return data


def _fetch_atg_race(atg_race_id: str) -> dict | None:
    """Fetch a single ATG race payload (starters, track, etc.)."""
    import time
    cache_key = f'atg_race_{atg_race_id}'
    if cache_key in _leaderboard_cache:
        ts, val = _leaderboard_cache[cache_key]
        if time.time() - ts < _ATG_CALENDAR_TTL:
            return val

    import httpx
    from core.config import ATG_RACE_URL, ATG_HEADERS
    url = ATG_RACE_URL.format(atg_race_id=atg_race_id)
    try:
        resp = httpx.get(url, headers=ATG_HEADERS, timeout=10.0)
        resp.raise_for_status()
        data = resp.json()
    except Exception:
        return None

    _leaderboard_cache[cache_key] = (time.time(), data)
    return data


_ATG_START_METHOD = {'auto': 'A', 'volte': 'V', 'line': 'L'}


def _atg_person_display(block: dict | None) -> tuple[str, str]:
    """Return (full_name, short_name) from an ATG driver/trainer block."""
    if not block:
        return '', ''
    raw = ' '.join(
        p for p in (block.get('lastName'), block.get('firstName')) if p
    ).strip()
    if not raw:
        raw = block.get('shortName') or ''
    full = fmtName(raw) if raw else ''
    short = block.get('shortName') or (shortName(raw) if raw else '')
    return full, short


def _atg_parse_shoes(shoes: dict | None) -> tuple[str | None, bool, bool]:
    if not shoes or not shoes.get('reported'):
        return None, False, False
    front = shoes.get('front') or {}
    back = shoes.get('back') or {}
    fs = bool(front.get('hasShoe'))
    bs = bool(back.get('hasShoe'))
    if fs and bs:
        code = '4'
    elif fs and not bs:
        code = '3'
    elif not fs and bs:
        code = '2'
    else:
        code = '1'
    return code, bool(front.get('changed')), bool(back.get('changed'))


def _resolve_atg_horse_id(cur, atg_horse_id) -> int | None:
    if atg_horse_id is None:
        return None
    cur.execute(
        "SELECT horse_id FROM horse "
        " WHERE atg_id = %s OR st_id = %s LIMIT 1",
        (str(atg_horse_id), atg_horse_id),
    )
    row = cur.fetchone()
    return row['horse_id'] if row else None


def _resolve_atg_person_id(cur, atg_person_id) -> int | None:
    if atg_person_id is None:
        return None
    cur.execute(
        "SELECT person_id FROM person "
        " WHERE atg_id = %s OR st_id = %s LIMIT 1",
        (str(atg_person_id), atg_person_id),
    )
    row = cur.fetchone()
    return row['person_id'] if row else None


def _race_entries_atg_live(cur, atg_race_id: str):
    """Build race page payload from ATG when the race isn't in our DB yet."""
    from datetime import date as _date_cls
    from etl.import_atg import _sex_to_hvs

    raw = _fetch_atg_race(atg_race_id)
    if not raw or raw.get('sport') == 'gallop':
        return jsonify({'error': 'not found'}), 404

    track = raw.get('track') or {}
    sm_raw = (raw.get('startMethod') or '').strip().lower()
    start_method = _ATG_START_METHOD.get(sm_raw)
    race_date_raw = raw.get('date')
    try:
        race_date = _date_cls.fromisoformat(str(race_date_raw)[:10])
    except (TypeError, ValueError):
        race_date = None

    scratched_nums = set(raw.get('result', {}).get('scratchings') or [])
    race_distance = raw.get('distance')

    rows: list[dict] = []
    for s in raw.get('starts') or []:
        h = s.get('horse') or {}
        if not h.get('name'):
            continue
        driver = s.get('driver') or {}
        trainer = h.get('trainer') or {}
        driver_name, driver_short = _atg_person_display(driver)
        trainer_name, trainer_short = _atg_person_display(trainer)
        shoe_code, shoe_front_changed, shoe_back_changed = _atg_parse_shoes(
            h.get('shoes'),
        )
        sulky = h.get('sulky') or {}
        sulky_code = None
        sulky_changed = False
        if sulky.get('reported'):
            stype = sulky.get('type') or {}
            sulky_code = stype.get('code')
            sulky_changed = bool(stype.get('changed'))

        rows.append({
            'entry_id': None,
            'horse_id': _resolve_atg_horse_id(cur, h.get('id')),
            'horse_name': h.get('name') or '',
            'number': s.get('number'),
            'distance': s.get('distance') or race_distance,
            'placement': None,
            'placement_text': None,
            'time_text': None,
            'time_val': None,
            'odds': None,
            'prize_kr': '',
            'age': h.get('age'),
            'sex': _sex_to_hvs(h.get('sex')),
            'sulky': sulky_code,
            'sulky_changed': sulky_changed,
            'shoe_code': shoe_code,
            'shoe_front_changed': shoe_front_changed,
            'shoe_back_changed': shoe_back_changed,
            'driver_name': driver_name,
            'driver_short': driver_short,
            'driver_id': _resolve_atg_person_id(cur, driver.get('id')),
            'trainer_name': trainer_name,
            'trainer_short': trainer_short,
            'trainer_id': _resolve_atg_person_id(cur, trainer.get('id')),
            'dq': False,
            'gal': False,
            'withdrawn': s.get('number') in scratched_nums,
            'primary_source': 'atg',
            'contributors': ['atg'],
            'xgal_track': None,
            'xgal_general': None,
            'pre_starts': 0,
            'pre_wins': 0,
            'post_starts': 0,
            'post_wins': 0,
            'pre_galadj_starts': 0,
            'pre_galadj_wins': 0,
            'post_galadj_starts': 0,
            'post_galadj_wins': 0,
            'd_wr': None,
            't_wr': None,
            'df': None,
            'df_odds': None,
            'df_perf': None,
            'tf': None,
            'tf_odds': None,
            'tf_perf': None,
        })

    if not rows:
        return jsonify({'error': 'not found'}), 404

    horse_ids = [r['horse_id'] for r in rows if r['horse_id']]
    driver_ids = list({r['driver_id'] for r in rows if r['driver_id']})
    trainer_ids = list({r['trainer_id'] for r in rows if r['trainer_id']})

    stats_pre: dict[int, dict] = {}
    if horse_ids and race_date:
        cur.execute(
            """
            SELECT e.horse_id,
                   COUNT(*) FILTER (
                       WHERE NOT e.withdrawn
                         AND """ + _NOT_QUALIFIER + """
                   ) AS starts,
                   COUNT(*) FILTER (
                       WHERE """ + _IS_WIN + """
                   ) AS wins,
                   COUNT(*) FILTER (
                       WHERE NOT e.withdrawn
                         AND NOT e.galopp
                         AND NOT COALESCE(e.disqualified, false)
                         AND """ + _NOT_QUALIFIER + """
                   ) AS clean_starts,
                   COUNT(*) FILTER (
                       WHERE """ + _IS_WIN + """
                         AND NOT e.galopp
                   ) AS clean_wins
            FROM entry e
            JOIN race  r2 ON r2.race_id = e.race_id
            WHERE e.horse_id = ANY(%s)
              AND r2.race_date < %s
            GROUP BY e.horse_id
            """,
            (horse_ids, race_date),
        )
        for srow in cur.fetchall():
            stats_pre[srow['horse_id']] = {
                'starts': srow['starts'] or 0,
                'wins': srow['wins'] or 0,
                'clean_starts': srow['clean_starts'] or 0,
                'clean_wins': srow['clean_wins'] or 0,
            }

    d_wr_map = _person_win_rates(cur.connection, driver_ids, 'driver')
    t_wr_map = _person_win_rates(cur.connection, trainer_ids, 'trainer')
    df_map = _batch_person_form_at_date(cur.connection, driver_ids, 'driver', race_date)
    tf_map = _batch_person_form_at_date(cur.connection, trainer_ids, 'trainer', race_date)

    for r in rows:
        pre = stats_pre.get(r['horse_id'], {
            'starts': 0,
            'wins': 0,
            'clean_starts': 0,
            'clean_wins': 0,
        })
        r['pre_starts'] = pre['starts']
        r['pre_wins'] = pre['wins']
        r['post_starts'] = pre['starts']
        r['post_wins'] = pre['wins']
        r['pre_galadj_starts'] = pre['clean_starts']
        r['pre_galadj_wins'] = pre['clean_wins']
        r['post_galadj_starts'] = pre['clean_starts']
        r['post_galadj_wins'] = pre['clean_wins']
        if r['driver_id']:
            r['d_wr'] = d_wr_map.get(r['driver_id'])
            r['df'] = df_map.get(r['driver_id'], {}).get('form')
            r['df_odds'] = df_map.get(r['driver_id'], {}).get('form_odds')
            r['df_perf'] = df_map.get(r['driver_id'], {}).get('form_perf')
        if r['trainer_id']:
            r['t_wr'] = t_wr_map.get(r['trainer_id'])
            r['tf'] = tf_map.get(r['trainer_id'], {}).get('form')
            r['tf_odds'] = tf_map.get(r['trainer_id'], {}).get('form_odds')
            r['tf_perf'] = tf_map.get(r['trainer_id'], {}).get('form_perf')

    _today = _date_cls.today()
    is_upcoming = bool(race_date and race_date >= _today)
    source_pills = [{
        'key': 'atg',
        'source_id': atg_race_id,
        'url': _atg_race_url(atg_race_id),
    }]
    return jsonify({
        'race_date': race_date.isoformat() if race_date else None,
        'track': (track.get('name') or '').strip().title(),
        'country': track.get('countryCode'),
        'race_number': raw.get('number'),
        'distance': race_distance,
        'start_method': start_method,
        'race_class': None,
        'victory_margin': None,
        'atg_url': _atg_race_url(atg_race_id),
        'is_upcoming': is_upcoming,
        'gal_models': None,
        'has_kmtid': False,
        'primary_source': 'atg',
        'contributors': ['atg'],
        'source_pills': source_pills,
        'results': rows,
    })


def _format_atg_race_display_name(name: str) -> str:
    """Strip ATG qualifier parens; keep title after the last ' - ' separator."""
    s = _re_mod.sub(r'\([^)]*\)', '', name or '').strip()
    s = _re_mod.sub(r'\s+', ' ', s)
    if ' - ' in s:
        s = s.rsplit(' - ', 1)[-1].strip()
    return s


def _format_atg_race_distance_label(distance, start_method) -> str | None:
    """Format ATG distance + start method, e.g. '2140m auto'."""
    if not distance:
        return None
    label = f"{distance}m"
    method = (start_method or '').strip().lower()
    if method == 'volte':
        method = 'volt'
    if method:
        label += f" {method}"
    return label


def _parse_atg_game_races(data: dict) -> list[dict]:
    """Extract panel fields from an ATG game response."""
    races_raw = data.get('races') or []
    panels = []
    for i, race in enumerate(races_raw):
        if isinstance(race, str):
            continue
        starts = race.get('starts') or []
        raw_name = (race.get('name') or '').strip()
        panels.append({
            'raceNumber': i + 1,
            'atgRaceId': race.get('id'),
            'name': _format_atg_race_display_name(raw_name),
            'starters': len(starts),
            'distanceLabel': _format_atg_race_distance_label(
                race.get('distance'),
                race.get('startMethod'),
            ),
        })
    return panels


def _per_race_game_ids(date_str: str, game_type_raw: str, track_id: int) -> list[tuple[int, str]]:
    """All (raceNumber, gameId) for a per-race bet type (vinnare/plats) on a
    given track/date, ordered by race number. ATG ids look like
    'vinnare_<date>_<track>_<raceNum>'."""
    data = _fetch_atg_calendar(date_str)
    if not data:
        return []
    game_list = (data.get('games') or {}).get(game_type_raw) or []
    out: list[tuple[int, str]] = []
    for g in game_list:
        if track_id not in (g.get('tracks') or []):
            continue
        gid = g.get('id') or ''
        parts = gid.split('_')
        try:
            rnum = int(parts[-1])
        except (ValueError, IndexError):
            rnum = len(out) + 1
        out.append((rnum, gid))
    out.sort(key=lambda x: x[0])
    return out


def _aggregate_per_race_panels(date_str: str, game_type_raw: str, track_id: int) -> list[dict]:
    """Merge every per-race vinnare/plats game for a track/date into a single
    ordered race-panel list (one panel per race), so the bubble shows the full
    card instead of a single race."""
    ids = _per_race_game_ids(date_str, game_type_raw, track_id)
    if not ids:
        return []

    def _one(item: tuple[int, str]) -> dict | None:
        rnum, gid = item
        panels = _parse_atg_game_races(_fetch_atg_game(gid) or {})
        if not panels:
            return None
        panel = panels[0]
        panel['raceNumber'] = rnum
        return panel

    from concurrent.futures import ThreadPoolExecutor
    panels: list[dict] = []
    with ThreadPoolExecutor(max_workers=8) as ex:
        for panel in ex.map(_one, ids):
            if panel:
                panels.append(panel)
    panels.sort(key=lambda p: p.get('raceNumber') or 0)
    return panels


def _atg_game_has_jackpot(game: dict | None) -> bool:
    """True when ATG reports a rollover jackpot on this headline game."""
    if not game:
        return False
    amount = game.get('jackpotAmount')
    return isinstance(amount, (int, float)) and amount > 0


def _build_upcoming_tracks(data: dict) -> list[dict]:
    """
    Parse ATG calendar into structured track entries for the home page.

    Returns a list of dicts sorted by headline hierarchy:
      {
        trackName, trackId, headline (id like 'v64' or None),
        headlineGameId (ATG game id),
        headlineStartTime, headlineRaceCount,
        jackpot (bool — headline game has rollover pool),
        jackpotAmount, estimatedJackpot (öre, when jackpot),
        satellites [{id, gameId, startTime, raceCount}]
      }
    """
    if not data:
        return []

    tracks_raw = data.get('tracks', [])
    games_raw = data.get('games', {})

    # Only trot tracks
    trot_tracks = {t['id']: t for t in tracks_raw if t.get('sport') == 'trot'}
    if not trot_tracks:
        return []

    # Build a map: track_id -> {game_type -> best game info}
    # "best" = the first game of that type assigned to the track
    track_games: dict[int, dict[str, dict]] = {tid: {} for tid in trot_tracks}
    # Per-race bet types appear once per leg; track how many games (= races)
    # each track has so satellite bubbles show the full card count.
    track_game_counts: dict[int, dict[str, int]] = {tid: {} for tid in trot_tracks}

    for game_type, game_list in games_raw.items():
        if not isinstance(game_list, list):
            continue
        for game in game_list:
            for tid in game.get('tracks', []):
                if tid in track_games:
                    track_game_counts[tid][game_type] = (
                        track_game_counts[tid].get(game_type, 0) + 1
                    )
                    if game_type not in track_games[tid]:
                        track_games[tid][game_type] = game

    # Build output per track
    result = []
    for tid, track_info in trot_tracks.items():
        track_name = track_info.get('name', f'Track {tid}')
        tg = track_games.get(tid, {})

        # Determine headline
        headline = None
        headline_game = None
        for h in _HEADLINE_HIERARCHY:
            if h in tg:
                headline = h
                headline_game = tg[h]
                break

        jackpot = _atg_game_has_jackpot(headline_game)

        # Determine cyan satellites (only the 6 we care about)
        satellites = []
        for cyan_type in _CYAN_SATELLITE_ORDER:
            if cyan_type in tg:
                g = tg[cyan_type]
                if cyan_type in _PER_RACE_GAME_TYPES:
                    race_count = track_game_counts.get(tid, {}).get(cyan_type, 1)
                else:
                    race_count = len(g.get('races', []))
                satellites.append({
                    'id': _atg_game_id_to_internal(cyan_type),
                    'gameId': g.get('id', ''),
                    'startTime': g.get('startTime', ''),
                    'raceCount': race_count,
                })

        headline_game_id = headline_game.get('id', '') if headline_game else ''
        # "posted" = ATG has published a playable game (with an id) for this
        # track/date. For dates further out the calendar lists the scheduled
        # game *types* (e.g. V85) but no game id yet — those render greyed out
        # and non-clickable on the home page.
        posted = bool(headline_game_id) or any(s.get('gameId') for s in satellites)

        entry = {
            'trackName': track_name,
            'trackId': tid,
            'headline': _atg_game_id_to_internal(headline) if headline else None,
            'headlineGameId': headline_game_id or None,
            'headlineStartTime': headline_game.get('startTime', '') if headline_game else None,
            'headlineRaceCount': len(headline_game.get('races', [])) if headline_game else 0,
            'jackpot': jackpot,
            'jackpotAmount': headline_game.get('jackpotAmount') if jackpot else None,
            'estimatedJackpot': headline_game.get('estimatedJackpot') if jackpot else None,
            'posted': posted,
            'satellites': satellites,
        }
        result.append(entry)

    # Sort by headline hierarchy (tracks with bigger headlines first)
    def sort_key(entry):
        h = (entry.get('headline') or '').lower()
        hierarchy_map = {'v85': 0, 'v86': 1, 'gs75': 2, 'v64': 3, 'v65': 4}
        return hierarchy_map.get(h, 99)

    result.sort(key=sort_key)
    return result


_UPCOMING_MAX_DAYS_AHEAD = 7


@app.route('/api/home/upcoming')
def home_upcoming():
    """Return trot track entries for the home page sphere row.

    Defaults to today; accepts ?date=YYYY-MM-DD to look ahead. The date is
    clamped to [today, today + _UPCOMING_MAX_DAYS_AHEAD]; past dates and
    malformed input fall back to today."""
    from datetime import date as _date, datetime as _dt, timedelta as _td
    today = _date.today()
    req = request.args.get('date')
    target = today
    if req:
        try:
            parsed = _dt.strptime(req, '%Y-%m-%d').date()
            if parsed < today:
                target = today
            elif parsed > today + _td(days=_UPCOMING_MAX_DAYS_AHEAD):
                target = today + _td(days=_UPCOMING_MAX_DAYS_AHEAD)
            else:
                target = parsed
        except ValueError:
            target = today
    data = _fetch_atg_calendar(str(target))
    if not data:
        return jsonify([])
    return jsonify(_build_upcoming_tracks(data))


@app.route('/api/atg/game/<game_id>')
def atg_game_panels(game_id):
    """Return race panel fields for a game (names, starter counts).

    For per-race bet types (vinnare/plats) ATG exposes one game per race, so we
    aggregate every leg of the track/date into a single ordered race list."""
    parts = game_id.split('_')
    game_type_raw = parts[0] if parts else ''
    game_type = _atg_game_id_to_internal(game_type_raw)

    if game_type_raw in _PER_RACE_GAME_TYPES and len(parts) >= 3:
        date_str = parts[1]
        try:
            track_id = int(parts[2])
        except ValueError:
            track_id = None
        if track_id is not None:
            races = _aggregate_per_race_panels(date_str, game_type_raw, track_id)
            if races:
                return jsonify({'gameId': game_id, 'gameType': game_type, 'races': races})

    data = _fetch_atg_game(game_id)
    if not data:
        return jsonify({'error': 'not found'}), 404
    return jsonify({
        'gameId': game_id,
        'gameType': game_type,
        'races': _parse_atg_game_races(data),
    })


@app.route('/game/<game_id>')
def game_page(game_id):
    """
    Game page for a specific ATG game.

    game_id is the ATG game identifier like 'V64_2026-06-08_15_4'.
    """
    from datetime import date as _date, datetime as _datetime

    # Parse game_id to extract date and type
    parts = game_id.split('_')
    game_type_raw = parts[0] if parts else ''
    date_str = parts[1] if len(parts) > 1 else str(_date.today())

    # Fetch calendar to get game details
    data = _fetch_atg_calendar(date_str)
    game_info = None
    track_name = None

    if data:
        games_raw = data.get('games', {})
        tracks_raw = data.get('tracks', [])
        track_map = {t['id']: t for t in tracks_raw}

        # Find the specific game
        for game_type, game_list in games_raw.items():
            if not isinstance(game_list, list):
                continue
            for game in game_list:
                if game.get('id') == game_id:
                    game_info = game
                    # Get track name
                    for tid in game.get('tracks', []):
                        if tid in track_map:
                            track_name = track_map[tid].get('name', '')
                            break
                    break
            if game_info:
                break

    # Build template context
    game_type_label = game_type_raw.upper()
    if game_type_label == 'GS75':
        game_type_label = 'GS75'

    race_count = len(game_info.get('races', [])) if game_info else 0
    start_time = game_info.get('startTime', '') if game_info else ''

    # Per-race bet types (vinnare/plats) list one game per leg; count them all
    # so the header reflects the full card rather than a single race.
    if game_type_raw in _PER_RACE_GAME_TYPES and len(parts) >= 3:
        try:
            _ids = _per_race_game_ids(date_str, game_type_raw, int(parts[2]))
            if _ids:
                race_count = len(_ids)
        except ValueError:
            pass

    # Format start time for display (Swedish style)
    time_display = ''
    if start_time:
        try:
            dt = _datetime.fromisoformat(start_time)
            time_display = dt.strftime('%Y-%m-%d %H:%M')
        except (ValueError, TypeError):
            time_display = start_time

    game_type_id = _atg_game_id_to_internal(game_type_raw)

    return render_template('game.html',
                           active_tab='home',
                           game_id=game_id,
                           game_type_id=game_type_id,
                           game_type_label=game_type_label,
                           track_name=track_name or '',
                           race_count=race_count,
                           time_display=time_display,
                           date_str=date_str)


_leaderboard_cache: dict[str, tuple[float, object]] = {}
_CACHE_TTL = 300


def _cached(key, fn):
    import time
    now = time.time()
    if key in _leaderboard_cache:
        ts, val = _leaderboard_cache[key]
        if now - ts < _CACHE_TTL:
            return val
    val = fn()
    _leaderboard_cache[key] = (now, val)
    return val


_QUALIFIER_RE = r"^(gdk|ejg|ejp|gd|gk|egk|egdk|gkd|gdj|gdb|gdek|gdgk|gdl|ddk|frdk|erj|ejk|ejgk|ejgd|ej|EJ|EJG|EJP|GDK|Gdk|g)[0-9]?$|^[12]?[pP][0-9]?$"
_NOT_QUALIFIER = f"COALESCE(e.placement_text,'') !~ '{_QUALIFIER_RE}'"
_IS_WIN   = f"e.placement_text = '1' AND NOT COALESCE(e.disqualified, false) AND {_NOT_QUALIFIER}"
_IS_PLACED = f"e.placement_text IN ('1','2','3') AND NOT COALESCE(e.disqualified, false) AND {_NOT_QUALIFIER}"

_FORM_MIN_STARTS = 7  # baseline used by recent-entries form cells (30d rolling)
_FORM_WIN_P_CAP = 0.7


def _permille_form(starts: int | None, wins: int | None) -> int | None:
    if not starts or starts < _FORM_MIN_STARTS:
        return None
    return round((wins or 0) * 1000 / starts)


def _permille_s_form(n: int | None, sum_outperf) -> int | None:
    """mkt± in permille: average per-start market outperformance (odds-rank vs
    finish-rank) scaled ×1000. >0 = beats the market. A value/edge signal, NOT
    recent form — it anti-correlates with winning (winning favourites score ~0).
    See scripts/refresh_entry_outperf and the trainer-form experiment."""
    if not n or n < _FORM_MIN_STARTS:
        return None
    return max(-999, min(999, round((float(sum_outperf or 0) / n) * 1000)))


def _permille_perf(n: int | None, sum_perf) -> int | None:
    """`form` (actual recent form) in permille: average per-start finishing
    percentile (1000 = won every start, 0 = always last) scaled ×1000. Unlike
    mkt± this rewards winning favourites and PREDICTS winning (trainer-form
    experiment: y_win AUC ≈ 0.67). Powered by entry_perf; display value/10 as %."""
    if not n or n < _FORM_MIN_STARTS:
        return None
    return max(0, min(1000, round((float(sum_perf or 0) / n) * 1000)))


def _batch_person_form_at_date(conn, person_ids: list[int], role: str,
                               as_of) -> dict[int, dict[str, int | None]]:
    """Leak-free 30-day rolling form ending the day before `as_of`."""
    if not person_ids or not as_of:
        return {}
    id_col = 'driver_id' if role == 'driver' else 'trainer_id'
    with conn.cursor() as cur:
        cur.execute(f"""
            SELECT e.{id_col} AS pid,
                   COUNT(*) FILTER (
                       WHERE NOT COALESCE(e.withdrawn, false) AND {_NOT_QUALIFIER}
                   ) AS starts,
                   COUNT(*) FILTER (WHERE {_IS_WIN}) AS wins,
                   COUNT(eo.market_outperf)            AS n_of,
                   COALESCE(SUM(eo.market_outperf), 0) AS sum_of,
                   COUNT(ep.perf)                       AS n_pf,
                   COALESCE(SUM(ep.perf), 0)            AS sum_pf
            FROM entry e
            LEFT JOIN entry_outperf eo ON eo.entry_id = e.entry_id
            LEFT JOIN entry_perf    ep ON ep.entry_id = e.entry_id
            WHERE e.{id_col} = ANY(%s)
              AND e.race_date >= %s - INTERVAL '30 days'
              AND e.race_date < %s
            GROUP BY e.{id_col}
        """, (person_ids, as_of, as_of))
        out: dict[int, dict[str, int | None]] = {}
        for pid, starts, wins, n_of, sum_of, n_pf, sum_pf in cur.fetchall():
            out[pid] = {
                'form': _permille_form(starts, wins),
                'form_odds': _permille_s_form(n_of, sum_of),
                'form_perf': _permille_perf(n_pf, sum_pf),
            }
        return out


def _batch_person_form_multi(conn, person_ids: list[int], role: str,
                             as_ofs: list) -> dict[tuple[int, object], dict[str, int | None]]:
    """Same as above but keyed on (person_id, as_of_date) for history rows."""
    if not person_ids or not as_ofs or len(person_ids) != len(as_ofs):
        return {}
    id_col = 'driver_id' if role == 'driver' else 'trainer_id'
    with conn.cursor() as cur:
        cur.execute(f"""
            WITH targets AS (
                SELECT * FROM unnest(%s::int[], %s::date[]) AS t(pid, as_of)
            )
            SELECT t.pid, t.as_of,
                   COUNT(*) FILTER (
                       WHERE NOT COALESCE(e.withdrawn, false) AND {_NOT_QUALIFIER}
                   ) AS starts,
                   COUNT(*) FILTER (WHERE {_IS_WIN}) AS wins,
                   COUNT(eo.market_outperf)            AS n_of,
                   COALESCE(SUM(eo.market_outperf), 0) AS sum_of,
                   COUNT(ep.perf)                       AS n_pf,
                   COALESCE(SUM(ep.perf), 0)            AS sum_pf
            FROM targets t
            JOIN entry e ON e.{id_col} = t.pid
             AND e.race_date >= t.as_of - INTERVAL '30 days'
             AND e.race_date < t.as_of
            LEFT JOIN entry_outperf eo ON eo.entry_id = e.entry_id
            LEFT JOIN entry_perf    ep ON ep.entry_id = e.entry_id
            GROUP BY t.pid, t.as_of
        """, (person_ids, as_ofs))
        out: dict[tuple[int, object], dict[str, int | None]] = {}
        for pid, as_of, starts, wins, n_of, sum_of, n_pf, sum_pf in cur.fetchall():
            out[(pid, as_of)] = {
                'form': _permille_form(starts, wins),
                'form_odds': _permille_s_form(n_of, sum_of),
                'form_perf': _permille_perf(n_pf, sum_pf),
            }
        return out


# Min-starts thresholds scale with window length so a small-sample 7d list
# stays meaningful but a 90d list filters out very low-volume drivers.
# Lowered the 30d bar from 10 → 7 so high-quality trainers with small
# stables (e.g. Timo Nurmos) actually surface when they're hot. The 30d
# value also matches `_FORM_MIN_STARTS` so the in-page form cells on the
# trainer/driver pages line up with the home leaderboard.
_FORM_MIN_STARTS_BY_DAYS = {7: 3, 30: 7, 90: 15}


@app.route('/api/home/form-leaders')
def home_form_leaders():
    """Rolling win-rate leaderboard with odds-weighted form.

    Query params:
      days        — window length in days, one of {7, 30, 90}. Default 30.
                    Outside that set, falls back to 30.
      min_starts  — override the default min-starts threshold for that window.
      sort        — one of {form, delta, form_odds}. Default delta.

    Always returns *both* hot and cold lists for the window — the frontend
    picks which one to display via its own toggle.
    """
    try:
        days = int(request.args.get('days', 30))
    except (TypeError, ValueError):
        days = 30
    if days not in _FORM_MIN_STARTS_BY_DAYS:
        days = 30
    try:
        min_starts = int(request.args.get('min_starts',
                                          _FORM_MIN_STARTS_BY_DAYS[days]))
    except (TypeError, ValueError):
        min_starts = _FORM_MIN_STARTS_BY_DAYS[days]

    sort_key = request.args.get('sort', 'delta')
    if sort_key not in ('form', 'delta', 'form_odds', 'form_perf'):
        sort_key = 'delta'

    cache_key = f'form-leaders:{days}:{min_starts}:{sort_key}'

    def fetch():
        conn = get_db()
        try:
            results: dict[str, list | int] = {'days': days, 'min_starts': min_starts}
            for role, id_col in (('trainer', 'trainer_id'),
                                 ('driver',  'driver_id')):
                with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                    cur.execute(f"""
                        WITH
                        recent AS (
                            SELECT e.{id_col}                       AS pid,
                                   COUNT(*) FILTER (
                                       WHERE NOT e.withdrawn
                                         AND {_NOT_QUALIFIER}
                                   )                                AS starts,
                                   COUNT(*) FILTER (
                                       WHERE {_IS_WIN}
                                   )                                AS wins,
                                   COUNT(eo.market_outperf)            AS n_of,
                                   COALESCE(SUM(eo.market_outperf), 0) AS sum_of,
                                   COUNT(ep.perf)                       AS n_pf,
                                   COALESCE(SUM(ep.perf), 0)            AS sum_pf
                            FROM entry e
                            LEFT JOIN entry_outperf eo ON eo.entry_id = e.entry_id
                            LEFT JOIN entry_perf    ep ON ep.entry_id = e.entry_id
                            WHERE e.{id_col} IS NOT NULL
                              AND e.race_date >= CURRENT_DATE - (%s || ' days')::interval
                            GROUP BY e.{id_col}
                        )
                        SELECT r.pid,
                               p.name,
                               p.short_name,
                               r.starts        AS recent_starts,
                               r.wins          AS recent_wins,
                               r.n_of,
                               r.sum_of,
                               r.n_pf,
                               r.sum_pf,
                               c.starts        AS career_starts,
                               c.wins          AS career_wins
                        FROM recent r
                        LEFT JOIN person_career_stats c
                               ON c.role = %s AND c.person_id = r.pid
                        JOIN person p ON p.person_id = r.pid
                        WHERE r.starts >= %s
                    """, (days, role, min_starts))
                    rows = cur.fetchall()

                def fmt(row_list):
                    out = []
                    for r in row_list:
                        rs = r['recent_starts'] or 0
                        rw = r['recent_wins'] or 0
                        cs = r['career_starts'] or 0
                        cw = r['career_wins'] or 0
                        form = round(rw * 1000 / rs) if rs else 0
                        wr   = round(cw * 1000 / cs) if cs else 0
                        out.append({
                            'id':        r['pid'],
                            'name':      fmtName(r['name'] or ''),
                            'short':     r['short_name'],
                            'form':      form,
                            'wr':        wr,
                            'form_odds': (_permille_s_form(r['n_of'], r['sum_of'])
                                          if (r['n_of'] or 0) >= min_starts else None),
                            'form_perf': (_permille_perf(r['n_pf'], r['sum_pf'])
                                          if (r['n_pf'] or 0) >= min_starts else None),
                            'delta':     form - wr,
                            'starts':    rs,
                        })
                    return out

                # "Hot" = highest first for the selected column. "Cold" =
                # lowest first, but we filter out anyone whose career win rate
                # is already low — otherwise the cold list just surfaces the
                # same low-volume drivers every time. We want drivers who are
                # *underperforming relative to themselves*.
                all_rows = fmt(rows)
                def sort_value(row):
                    value = row.get(sort_key)
                    # Null odds-delta goes last in both hot and cold lists.
                    if value is None:
                        return None
                    return value

                def hot_key(row):
                    value = sort_value(row)
                    return (
                        value is not None,
                        value if value is not None else -10**9,
                        row.get('starts') or 0,
                    )

                def cold_key(row):
                    value = sort_value(row)
                    return (
                        value is None,
                        value if value is not None else 10**9,
                        -(row.get('starts') or 0),
                    )

                hot = sorted(
                    all_rows,
                    key=hot_key,
                    reverse=True,
                )[:10]
                cold_pool = [x for x in all_rows if x['wr'] >= 50]  # >= 5% career
                cold = sorted(
                    cold_pool,
                    key=cold_key,
                )[:10]
                results[f'{role}_hot']  = hot
                results[f'{role}_cold'] = cold
        finally:
            conn.close()
        return results

    return jsonify(_cached(cache_key, fetch))


@app.route('/api/home/top-horses')
def home_top_horses():
    """Top horses by earnings / fastest km-time / win rate."""
    sort = request.args.get('sort', 'earnings')
    if sort not in {'earnings', 'time', 'win_rate'}:
        sort = 'earnings'
    period = request.args.get('period', 'ytd')
    breed = request.args.get('breed', 'standardbred')
    breed_code = {'standardbred': 'V', 'coldblood': 'K'}.get(breed, 'V')
    limit = min(int(request.args.get('limit', 10)), 50)
    cache_key = f'horses:{sort}:{period}:{limit}:{breed_code}'

    def fetch():
        conn = get_db()
        try:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                if sort == 'time':
                    date_filter = (
                        "TRUE" if period == 'all'
                        else "r.race_date >= date_trunc('year', CURRENT_DATE)"
                    )
                    cur.execute(f"""
                        SELECT e.horse_id,
                               h.name          AS horse_name,
                               t.name          AS track_name,
                               t.track_id      AS track_id,
                               r.race_date,
                               e.distance,
                               e.time_text,
                               e.time_seconds
                        FROM entry e
                        JOIN horse h ON h.horse_id = e.horse_id
                        JOIN race  r ON r.race_id  = e.race_id
                        LEFT JOIN track t ON t.track_id = r.track_id
                        WHERE {date_filter}
                          AND NOT COALESCE(e.withdrawn, false)
                          AND NOT COALESCE(e.disqualified, false)
                          AND {_NOT_QUALIFIER}
                          AND h.breed_code = %s
                          AND e.time_seconds IS NOT NULL
                          AND e.time_seconds >= {_FAST_KM_TIME_FLOOR_SECONDS}
                        ORDER BY e.time_seconds ASC,
                                 r.race_date DESC NULLS LAST,
                                 e.entry_id DESC
                        LIMIT {limit}
                    """, (breed_code,))
                elif period == 'all':
                    cur.execute(f"""
                        SELECT h.horse_id,
                               h.name                     AS horse_name,
                               s.starts                   AS starts,
                               s.wins                     AS wins,
                               s.prize_money_kr           AS earnings
                        FROM horse h
                        JOIN horse_career_stats s ON s.horse_id = h.horse_id
                        WHERE h.breed_code = %s
                        ORDER BY
                          CASE %s
                            WHEN 'earnings'  THEN s.prize_money_kr
                            ELSE 0
                          END DESC,
                          CASE WHEN %s = 'win_rate'
                            THEN CASE WHEN s.starts > 0
                                 THEN s.wins::numeric / s.starts
                                 ELSE -1 END
                            ELSE 0
                          END DESC,
                          s.prize_money_kr DESC
                        LIMIT {limit}
                    """, (breed_code, sort, sort))
                elif sort != 'time':
                    cur.execute(f"""
                        SELECT e.horse_id,
                               h.name                                                        AS horse_name,
                               SUM(CASE WHEN NOT COALESCE(e.withdrawn,false) THEN 1 ELSE 0 END) AS starts,
                               SUM(CASE WHEN {_IS_WIN} THEN 1 ELSE 0 END)                    AS wins,
                               SUM(COALESCE(e.prize_kr, 0))                                  AS earnings
                        FROM entry e
                        JOIN horse h ON h.horse_id = e.horse_id
                        JOIN race  r ON r.race_id  = e.race_id
                        WHERE r.race_date >= date_trunc('year', CURRENT_DATE)
                          AND NOT COALESCE(e.withdrawn, false)
                          AND {_NOT_QUALIFIER}
                          AND h.breed_code = %s
                        GROUP BY e.horse_id, h.name
                        ORDER BY
                          CASE %s
                            WHEN 'earnings'  THEN SUM(COALESCE(e.prize_kr, 0))
                            ELSE 0
                          END DESC,
                          CASE WHEN %s = 'win_rate'
                            THEN SUM(CASE WHEN {_IS_WIN} THEN 1 ELSE 0 END)::numeric
                                 / NULLIF(SUM(CASE WHEN NOT COALESCE(e.withdrawn,false) THEN 1 ELSE 0 END), 0)
                            ELSE 0
                          END DESC,
                          SUM(COALESCE(e.prize_kr, 0)) DESC
                        LIMIT {limit}
                    """, (breed_code, sort, sort))
                rows = cur.fetchall()
        finally:
            conn.close()

        if sort == 'time':
            return [{
                'horse_id': r['horse_id'],
                'name': r['horse_name'] or '',
                'track': (r['track_name'] or '').strip().title(),
                'track_id': r['track_id'],
                'race_date': r['race_date'].isoformat() if r['race_date'] else None,
                'distance': r['distance'],
                'time_text': r['time_text'],
                'time_seconds': float(r['time_seconds']) if r['time_seconds'] is not None else None,
            } for r in rows]

        return [{
            'horse_id': r['horse_id'],
            'name':     r['horse_name'] or '',
            'starts':   r['starts'] or 0,
            'wins':     r['wins'] or 0,
            'earnings': int(r['earnings'] or 0),
            'win_rate': round((r['wins'] or 0) * 1000 / (r['starts'] or 1)) if r['starts'] else 0,
        } for r in rows]

    return jsonify(_cached(cache_key, fetch))


@app.route('/api/home/top-people')
def home_top_people():
    sort = request.args.get('sort', 'earnings')
    role = request.args.get('role', 'driver')
    period = request.args.get('period', 'ytd')
    cache_key = f'people:{role}:{sort}:{period}'

    def fetch():
        id_col = 'driver_id' if role == 'driver' else 'trainer_id'
        date_filter = ("TRUE" if period == 'all'
                       else "r.race_date >= date_trunc('year', CURRENT_DATE)")
        min_starts = 5 if period == 'ytd' else 20

        conn = get_db()
        try:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(f"""
                    SELECT e.{id_col}                                                    AS pid,
                           p.name,
                           SUM(CASE WHEN NOT COALESCE(e.withdrawn,false) THEN 1 ELSE 0 END) AS starts,
                           SUM(CASE WHEN {_IS_WIN} THEN 1 ELSE 0 END)                    AS wins,
                           SUM(COALESCE(e.prize_kr, 0))                                  AS earnings
                    FROM entry e
                    JOIN person p ON p.person_id = e.{id_col}
                    JOIN race   r ON r.race_id   = e.race_id
                    WHERE e.{id_col} IS NOT NULL
                      AND {date_filter}
                      AND NOT COALESCE(e.withdrawn, false)
                      AND {_NOT_QUALIFIER}
                    GROUP BY e.{id_col}, p.name
                    HAVING SUM(CASE WHEN NOT COALESCE(e.withdrawn,false) THEN 1 ELSE 0 END) >= {min_starts}
                    ORDER BY
                      CASE %(sort)s
                        WHEN 'earnings'  THEN SUM(COALESCE(e.prize_kr, 0))
                        WHEN 'wins'      THEN SUM(CASE WHEN {_IS_WIN} THEN 1 ELSE 0 END)
                        WHEN 'starts'    THEN SUM(CASE WHEN NOT COALESCE(e.withdrawn,false) THEN 1 ELSE 0 END)
                        ELSE 0
                      END DESC,
                      SUM(COALESCE(e.prize_kr, 0)) DESC
                    LIMIT 10
                """, {'sort': sort})
                rows = cur.fetchall()
        finally:
            conn.close()
        return [{
            'id':       r['pid'],
            'name':     fmtName(r['name'] or ''),
            'starts':   r['starts'] or 0,
            'wins':     r['wins'] or 0,
            'earnings': int(r['earnings'] or 0),
            'win_rate': round((r['wins'] or 0) * 1000 / (r['starts'] or 1)) if r['starts'] else 0,
        } for r in rows]

    return jsonify(_cached(cache_key, fetch))


@app.route('/api/home/top-offspring')
def home_top_offspring():
    """Top sires/dams by their offspring's race results."""
    sort = request.args.get('sort', 'earnings')
    if sort not in {'earnings', 'avg_earnings', 'win_rate'}:
        sort = 'earnings'
    role = request.args.get('role', 'sire')
    if role not in {'sire', 'dam', 'all'}:
        role = 'sire'
    period = request.args.get('period', 'ytd')
    breed = request.args.get('breed', 'standardbred')
    breed_code = {'standardbred': 'V', 'coldblood': 'K'}.get(breed, 'V')
    limit = min(int(request.args.get('limit', 10)), 50)
    cache_key = f"offspring:{role}:{sort}:{period}:{limit}:{breed_code}"

    def fetch_role(parent_col: str):
        # Win% needs a stronger sample floor for sires than dams. Otherwise
        # small imported sire groups with 3 good runners dominate all-time.
        if parent_col == 'sire_id':
            min_win_starts = 50 if period == 'ytd' else 300
            min_offspring = 10 if period == 'ytd' else 25
        else:
            min_win_starts = 20 if period == 'ytd' else 75
            min_offspring = 2 if period == 'ytd' else 3
        having = ""
        if sort == 'win_rate':
            having = f"""
                HAVING SUM(starts) >= {min_win_starts}
                   AND COUNT(DISTINCT child_id) >= {min_offspring}
            """
        elif sort == 'avg_earnings':
            having = f"""
                HAVING COUNT(DISTINCT child_id) >= {min_offspring}
            """

        conn = get_db()
        try:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                if period == 'all':
                    # Use the precomputed horse career view. Also treat missing
                    # child breed_code as the parent's breed, which avoids
                    # under-counting imported French offspring with NULL breed.
                    source_sql = f"""
                        SELECT parent.horse_id AS parent_id,
                               parent.name AS parent_name,
                               child.horse_id AS child_id,
                               s.starts,
                               s.wins,
                               s.prize_money_kr AS earnings
                        FROM horse child
                        JOIN horse parent ON parent.horse_id = child.{parent_col}
                        JOIN horse_career_stats s ON s.horse_id = child.horse_id
                        WHERE child.{parent_col} IS NOT NULL
                          AND COALESCE(child.breed_code, parent.breed_code) = %(breed)s
                          AND s.starts > 0
                    """
                else:
                    source_sql = f"""
                        SELECT parent.horse_id AS parent_id,
                               parent.name AS parent_name,
                               child.horse_id AS child_id,
                               y.starts,
                               y.wins,
                               y.prize_money_kr AS earnings
                        FROM horse child
                        JOIN horse parent ON parent.horse_id = child.{parent_col}
                        JOIN horse_year_stats y ON y.horse_id = child.horse_id
                        WHERE child.{parent_col} IS NOT NULL
                          AND COALESCE(child.breed_code, parent.breed_code) = %(breed)s
                          AND y.race_year = EXTRACT(YEAR FROM CURRENT_DATE)::integer
                          AND y.starts > 0
                    """

                cur.execute(f"""
                    WITH child_stats AS (
                        {source_sql}
                    )
                    SELECT parent_id,
                           parent_name,
                           COUNT(DISTINCT child_id) AS offspring_count,
                           SUM(starts) AS starts,
                           SUM(wins) AS wins,
                           SUM(earnings) AS earnings,
                           SUM(earnings)::numeric / NULLIF(COUNT(DISTINCT child_id), 0) AS avg_earnings
                    FROM child_stats
                    GROUP BY parent_id, parent_name
                    {having}
                    ORDER BY
                      CASE %(sort)s
                        WHEN 'earnings' THEN SUM(earnings)
                        WHEN 'avg_earnings'
                            THEN SUM(earnings)::numeric / NULLIF(COUNT(DISTINCT child_id), 0)
                        ELSE 0
                      END DESC,
                      CASE WHEN %(sort)s = 'win_rate'
                        THEN SUM(wins)::numeric / NULLIF(SUM(starts), 0)
                        ELSE 0
                      END DESC,
                      SUM(earnings) DESC
                    LIMIT {limit}
                """, {'sort': sort, 'breed': breed_code})
                rows = cur.fetchall()
        finally:
            conn.close()

        return [{
            'horse_id': int(r['parent_id']),
            'name': r['parent_name'] or '',
            'offspring_count': int(r['offspring_count'] or 0),
            'starts': int(r['starts'] or 0),
            'wins': int(r['wins'] or 0),
            'earnings': int(r['earnings'] or 0),
            'avg_earnings': int(r['avg_earnings'] or 0),
            'win_rate': round((r['wins'] or 0) * 1000 / (r['starts'] or 1)) if r['starts'] else 0,
        } for r in rows]

    def fetch():
        if role == 'all':
            return {
                'sires': fetch_role('sire_id'),
                'dams': fetch_role('dam_id'),
            }
        return fetch_role('sire_id' if role == 'sire' else 'dam_id')

    return jsonify(_cached(cache_key, fetch))


# =====================================================================
# Stable (search + watchlist)
# =====================================================================

@app.route('/')
def index():
    return render_template('index.html', active_tab='stable_search')


@app.route('/watchlist')
def watchlist_page():
    return render_template('watchlist.html', active_tab='stable_watchlist')


@app.route('/api/watchlist', methods=['GET'])
def api_watchlist_list():
    """List all watchlisted horses with active/inactive status."""
    conn = get_db()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("""
                SELECT w.horse_id, h.name, h.breed_code, h.gender_code,
                       h.date_of_birth, h.is_dead,
                       COALESCE(sire_h.name, h.sire_name) AS sire_name,
                       h.sire_id,
                       s.starts, s.wins, s.prize_money_kr,
                       s.last_start,
                       w.added_at, w.note
                FROM watchlist w
                JOIN horse h ON h.horse_id = w.horse_id
                LEFT JOIN horse sire_h ON sire_h.horse_id = h.sire_id
                LEFT JOIN horse_career_stats s ON s.horse_id = w.horse_id
                ORDER BY s.prize_money_kr DESC NULLS LAST, h.name
            """)
            rows = cur.fetchall()
    finally:
        conn.close()

    from datetime import date as _d, timedelta
    cutoff = _d.today() - timedelta(days=365)
    result = []
    for r in rows:
        last = r['last_start']
        is_active = bool(last and last >= cutoff and not r['is_dead'])
        starts = r['starts'] or 0
        wins = r['wins'] or 0
        result.append({
            'horse_id': r['horse_id'],
            'name': r['name'] or '',
            'year': str(r['date_of_birth'].year) if r['date_of_birth'] else '',
            'sire_name': r['sire_name'] or '',
            'sire_id': r['sire_id'],
            'gender': _GENDER_TEXT.get(r['gender_code'] or '', ''),
            'starts': starts,
            'wins': wins,
            'win_rate': f"{round(100 * wins / starts)}" if starts else '',
            'earnings': _kr(r['prize_money_kr']) if r['prize_money_kr'] else '',
            'earnings_raw': int(r['prize_money_kr'] or 0),
            'last_start': last.isoformat() if last else None,
            'added_at': r['added_at'].isoformat() if r['added_at'] else None,
            'note': r['note'],
            'active': is_active,
        })
    return jsonify(result)


@app.route('/api/watchlist/<int:horse_id>', methods=['PUT', 'DELETE'])
def api_watchlist_toggle(horse_id):
    """Add (PUT) or remove (DELETE) a horse from the watchlist."""
    conn = get_db()
    try:
        with conn.cursor() as cur:
            if request.method == 'PUT':
                note = (request.get_json(silent=True) or {}).get('note')
                cur.execute("""
                    INSERT INTO watchlist (horse_id, note)
                    VALUES (%s, %s)
                    ON CONFLICT (horse_id) DO UPDATE SET note = EXCLUDED.note
                """, (horse_id, note))
                conn.commit()
                return jsonify({'status': 'added', 'horse_id': horse_id})
            else:
                cur.execute("DELETE FROM watchlist WHERE horse_id = %s", (horse_id,))
                conn.commit()
                return jsonify({'status': 'removed', 'horse_id': horse_id})
    finally:
        conn.close()


@app.route('/api/watchlist/<int:horse_id>/status', methods=['GET'])
def api_watchlist_status(horse_id):
    """Check if a horse is on the watchlist."""
    conn = get_db()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT 1 FROM watchlist WHERE horse_id = %s", (horse_id,))
            on_list = cur.fetchone() is not None
    finally:
        conn.close()
    return jsonify({'horse_id': horse_id, 'watched': on_list})


@app.route('/api/search')
def search():
    """Stable-front-page search.

    ?q=<query>&kind=horse|driver|trainer|track (default: horse).

    The horse response shape is unchanged for backward compatibility.
    Each other kind returns its own simple per-row shape — see the search
    page JS for the rendering contract.
    """
    q = request.args.get('q', '').strip()
    kind = (request.args.get('kind') or 'horse').lower()
    if len(q) < 2:
        return jsonify([])
    if kind not in ('horse', 'driver', 'trainer', 'track'):
        kind = 'horse'

    conn = get_db()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            # Accent-insensitive matching: unaccent('Redén') = 'Reden', so a
            # search for `reden` also finds `Redén Daniel`. unaccent() comes
            # from the extension already installed for v2_normalize_name.
            pattern = f'%{q}%'
            if kind == 'horse':
                cur.execute(
                    """
                    SELECT h.horse_id,
                           h.name,
                           to_char(h.date_of_birth, 'YYYY-MM-DD') AS date_of_birth,
                           h.color,
                           CASE h.gender_code WHEN 'H' THEN 'hingst'
                                              WHEN 'V' THEN 'valack'
                                              WHEN 'S' THEN 'sto' END AS gender,
                           CASE h.breed_code  WHEN 'V' THEN 'varmblodig travare'
                                              WHEN 'K' THEN 'kallblodig travare' END AS breed,
                           s.prize_money_kr AS earnings
                    FROM horse h
                    LEFT JOIN horse_career_stats s ON s.horse_id = h.horse_id
                    WHERE unaccent(h.name) ILIKE unaccent(%s)
                    ORDER BY s.prize_money_kr DESC NULLS LAST, h.name
                    LIMIT 50
                    """,
                    (pattern,),
                )
                results = [dict(r) for r in cur.fetchall()]
                for r in results:
                    r['earnings'] = _kr(r['earnings']) if r['earnings'] is not None else None
                return jsonify(results)

            if kind in ('driver', 'trainer'):
                role_flag = 'is_driver' if kind == 'driver' else 'is_trainer'
                role_col  = 'driver_id' if kind == 'driver' else 'trainer_id'
                cur.execute(
                    f"""
                    SELECT p.person_id,
                           COALESCE(p.name, p.short_name) AS name,
                           p.short_name,
                           p.license_country,
                           (SELECT COUNT(*) FROM entry e
                             WHERE e.{role_col} = p.person_id) AS entries
                      FROM person p
                     WHERE p.{role_flag} = TRUE
                       AND (   unaccent(p.name)       ILIKE unaccent(%s)
                            OR unaccent(p.short_name) ILIKE unaccent(%s) )
                     ORDER BY (SELECT COUNT(*) FROM entry e
                                WHERE e.{role_col} = p.person_id) DESC NULLS LAST,
                              COALESCE(p.name, p.short_name)
                     LIMIT 50
                    """,
                    (pattern, pattern),
                )
                return jsonify([dict(r) for r in cur.fetchall()])

            # kind == 'track'
            cur.execute(
                """
                SELECT t.track_id, t.name, t.country, t.sport,
                       t.st_code, t.atg_track_id,
                       (SELECT COUNT(*) FROM race r WHERE r.track_id = t.track_id) AS races
                  FROM track t
                 WHERE unaccent(t.name) ILIKE unaccent(%s)
                 ORDER BY (SELECT COUNT(*) FROM race r WHERE r.track_id = t.track_id) DESC,
                          t.name
                 LIMIT 50
                """,
                (pattern,),
            )
            return jsonify([dict(r) for r in cur.fetchall()])
    finally:
        conn.close()


# =====================================================================
# Horse page
# =====================================================================

# Per-source redirect helpers. The canonical id is `horse.horse_id`; each
# source contributes a per-source key column. We resolve the source id to
# the canonical row and 302-redirect so the URL bar matches what the rest
# of the app links to.

_HORSE_SOURCE_COLS = {
    'st': ('st_id', int),
    'atg': ('atg_id', str),
    'usta': ('usta_id', str),
    'letrot': ('letrot_id', str),
    'kmtid': ('kmtid_id', str),
}


@app.route('/horse/st/<int:src_id>')
def horse_by_st(src_id):
    return _horse_redirect('st', src_id)


@app.route('/horse/atg/<path:src_id>')
def horse_by_atg(src_id):
    return _horse_redirect('atg', src_id)


@app.route('/horse/usta/<path:src_id>')
def horse_by_usta(src_id):
    return _horse_redirect('usta', src_id)


@app.route('/horse/letrot/<path:src_id>')
def horse_by_letrot(src_id):
    return _horse_redirect('letrot', src_id)


@app.route('/horse/kmtid/<path:src_id>')
def horse_by_kmtid(src_id):
    return _horse_redirect('kmtid', src_id)


def _horse_redirect(source: str, src_id):
    col, kind = _HORSE_SOURCE_COLS[source]
    try:
        val = kind(src_id)
    except (TypeError, ValueError):
        return 'invalid id', 400
    conn = get_db()
    try:
        with conn.cursor() as cur:
            cur.execute(f"SELECT horse_id FROM horse WHERE {col} = %s LIMIT 1", (val,))
            row = cur.fetchone()
    finally:
        conn.close()
    if not row:
        return 'not found', 404
    return redirect(url_for('horse_page', horse_id=row[0]), code=302)


@app.route('/horse/<int:horse_id>')
def horse_page(horse_id):
    conn = get_db()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                """
                SELECT h.*,
                       gf.name AS mother_father_name,
                       s.starts, s.wins, s.placed, s.seconds, s.thirds, s.qualifiers,
                       s.prize_money_kr,
                       s.first_start, s.last_start,
                       -- Prefer the maintained trainer history; fall back to
                       -- the trainer on the horse's most recent race entry so
                       -- the passport stays populated for ATG/LeTrot-only
                       -- horses that have no horse_trainer_history rows.
                       COALESCE(
                         (SELECT th.trainer_name
                          FROM horse_trainer_history th
                          WHERE th.horse_id = h.horse_id
                          ORDER BY (th.to_date IS NULL) DESC, th.from_date DESC
                          LIMIT 1),
                         (SELECT pe.name
                          FROM entry e
                          JOIN race re ON re.race_id = e.race_id
                          JOIN person pe ON pe.person_id = e.trainer_id
                          WHERE e.horse_id = h.horse_id AND e.trainer_id IS NOT NULL
                          ORDER BY re.race_date DESC NULLS LAST
                          LIMIT 1)
                       ) AS trainer_name,
                       COALESCE(
                         (SELECT th.trainer_id
                          FROM horse_trainer_history th
                          WHERE th.horse_id = h.horse_id
                          ORDER BY (th.to_date IS NULL) DESC, th.from_date DESC
                          LIMIT 1),
                         (SELECT e.trainer_id
                          FROM entry e
                          JOIN race re ON re.race_id = e.race_id
                          WHERE e.horse_id = h.horse_id AND e.trainer_id IS NOT NULL
                          ORDER BY re.race_date DESC NULLS LAST
                          LIMIT 1)
                       ) AS trainer_id,
                       (SELECT oh.owner_name
                        FROM horse_owner_history oh
                        WHERE oh.horse_id = h.horse_id
                        ORDER BY (oh.to_date IS NULL) DESC, oh.from_date DESC
                        LIMIT 1) AS owner_name,
                       (SELECT pe.person_type
                        FROM horse_owner_history oh
                        JOIN person pe ON pe.person_id = oh.owner_id
                        WHERE oh.horse_id = h.horse_id
                        ORDER BY (oh.to_date IS NULL) DESC, oh.from_date DESC
                        LIMIT 1) AS owner_type
                FROM horse h
                LEFT JOIN horse  m  ON m.horse_id = h.dam_id
                LEFT JOIN horse  gf ON gf.horse_id = m.sire_id
                LEFT JOIN horse_career_stats s ON s.horse_id = h.horse_id
                WHERE h.horse_id = %s
                """,
                (horse_id,),
            )
            row = cur.fetchone()
            if not row:
                return 'not found', 404

            # Canonical reg/UELN can be empty when the horse was created from
            # ATG before ST passport was scraped — fall back to ST basic blob.
            display_reg = row.get('registration_number') or ''
            display_ueln = row.get('ueln_number') or ''
            if row.get('st_id') and (not display_reg or not display_ueln):
                cur.execute(
                    """
                    SELECT raw_json->>'registrationNumber' AS registration_number,
                           raw_json->>'uelnNumber' AS ueln_number
                      FROM st_horse_raw
                     WHERE horse_id = %s
                       AND data_type = 'horse-basic-information'
                    """,
                    (row['st_id'],),
                )
                st_ids = cur.fetchone()
                if st_ids:
                    display_reg = display_reg or (st_ids.get('registration_number') or '')
                    display_ueln = display_ueln or (st_ids.get('ueln_number') or '')

    finally:
        conn.close()

    starts  = row['starts']  or 0
    wins    = row['wins']    or 0
    seconds = row['seconds'] or 0
    thirds  = row['thirds']  or 0
    placed  = seconds + thirds
    unplaced = max(starts - wins - placed, 0)
    win_rate = f"{round(100 * wins / starts)}" if starts else ''

    owner_name = row['owner_name'] or ''
    owner_type = (row['owner_type'] or '').upper()
    is_stable  = owner_type == 'LEGAL'

    source_data = row.get('source_data') or {}
    sources = sorted(
        k for k in source_data.keys()
        if isinstance(k, str) and not k.startswith('_')
    ) if isinstance(source_data, dict) else []

    horse = {
        'horse_id':      row['horse_id'],
        'st_id':         row['st_id'],
        'sources':       sources,
        'name':                  row['name'] or '',
        'registration_number':   display_reg,
        'ueln_number':           display_ueln,
        'date_of_birth': str(row['date_of_birth']) if row['date_of_birth'] else '',
        'gender':        _GENDER_TEXT.get(row['gender_code'] or '', ''),
        'breed':         _BREED_TEXT.get(row['breed_code'] or '', ''),
        'color':         row['color'] or '',
        # v2.horse has no breeder_id column (breeder lives in source_data
        # for some sources but isn't a canonical FK). Leave blank.
        'breeder':       '',
        'trainer':       fmtName(row['trainer_name'] or ''),
        'trainer_id':    row['trainer_id'],
        'stable':        owner_name if is_stable else '',
        'owner':         owner_name if not is_stable else '',
        # Template keys named after v1's "father/mother" terminology; values
        # are v2's canonical sire/dam id + name (so /horse/<id> links work).
        'father':        row['sire_name'] or '',
        'father_id':     row['sire_id'],
        'mother':        row['dam_name'] or '',
        'mother_id':     row['dam_id'],
        'mother_father': row['mother_father_name'] or '',
        'earnings':      _kr(row['prize_money_kr']),
        'starts':        starts,
        'wins':          wins,
        'placed':        placed,
        'unplaced':      unplaced,
        'win_rate':      win_rate,
    }
    return render_template('horse.html', horse=horse, active_tab='stable_search')


@app.route('/api/horse/<int:horse_id>/races')
def horse_races(horse_id):
    """Race history for a horse, most recent first."""
    conn = get_db()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                """
                SELECT r.race_date,
                       t.name                                AS track_label,
                       t.country                             AS track_country,
                       t.track_id                            AS track_id,
                       e.race_id,
                       r.atg_race_id                         AS atg_id,
                       r.race_number,
                       e.distance,
                       e.auto                                AS start_method_auto,
                       e.placement_text,
                       e.placement,
                       e.time_text                           AS km_time_text,
                       e.time_seconds                        AS time_val,
                       e.program_number,
                       e.post                                AS start_position,
                       e.odds,
                       e.prize_kr                            AS prize_money_kr,
                       p.name                                AS driver_name,
                       e.driver_id,
                       pt.name                               AS trainer_name,
                       e.trainer_id,
                       e.disqualified                        AS dq,
                       e.galopp                              AS gal,
                       e.withdrawn,
                       e.kmtid_actual_distance_m             AS kmtid_actual_m,
                       e.kmtid_actual_km_time_ms             AS kmtid_actual_km_ms,
                       e.kmtid_best_100ms                    AS kmtid_best_100ms,
                       e.kmtid_slipstream_distance_m         AS kmtid_slip_m,
                       e.primary_source,
                       e.source_data->'_contributors'        AS contributors
                FROM entry e
                JOIN race    r ON r.race_id    = e.race_id
                LEFT JOIN track   t ON t.track_id   = r.track_id
                LEFT JOIN person  p ON p.person_id  = e.driver_id
                LEFT JOIN person  pt ON pt.person_id = e.trainer_id
                WHERE e.horse_id = %s
                ORDER BY r.race_date DESC NULLS LAST, e.race_id DESC NULLS LAST
                LIMIT 500
                """,
                (horse_id,),
            )
            rows = []
            for r in cur.fetchall():
                contribs = r.get('contributors') or []
                # Fall back to primary_source when no column-merge has
                # happened yet — keeps pills visible from day one.
                if not contribs and r.get('primary_source'):
                    contribs = [r['primary_source']]
                # xLabs (kmtid) GPS sectionals live on the entry, not in
                # `contributors`. Surface them as a source pill so the src
                # column flags GPS-covered runs, mirroring the race page.
                if (r.get('kmtid_actual_m') is not None
                        or r.get('kmtid_best_100ms') is not None) \
                        and 'kmtid' not in contribs:
                    contribs = [*contribs, 'kmtid']
                rows.append({
                    'race_date': r['race_date'].isoformat() if r['race_date'] else None,
                    'track_label': (r['track_label'] or '').strip().title(),
                    'track_country': r['track_country'],
                    'track_id': r['track_id'],
                    'race_id': r['race_id'],
                    'atg_id': r['atg_id'],
                    'race_number': r['race_number'],
                    'placement_text': r['placement_text'],
                    'placement_num': r['placement'],
                    'km_time_text': r['km_time_text'],
                    'time_val': float(r['time_val']) if r['time_val'] is not None else None,
                    'program_number': r['program_number'],
                    'start_position': r['start_position'],
                    'distance': r['distance'],
                    'start_method': ('A' if r['start_method_auto'] is True
                                     else 'V' if r['start_method_auto'] is False
                                     else None),
                    'odds_text': str(r['odds']) if r['odds'] else '',
                    'prize_kr': _kr(r['prize_money_kr']) if r['prize_money_kr'] else '',
                    'driver_name': r['driver_name'],
                    'driver_id': r['driver_id'],
                    'trainer_name': r['trainer_name'],
                    'trainer_id': r['trainer_id'],
                    'disqualified': r['dq'],
                    'galopp': r['gal'],
                    'withdrawn': r['withdrawn'],
                    'kmtid_actual_m':     r['kmtid_actual_m'],
                    'kmtid_actual_km_ms': float(r['kmtid_actual_km_ms']) if r['kmtid_actual_km_ms'] is not None else None,
                    'kmtid_best_100ms':   float(r['kmtid_best_100ms'])   if r['kmtid_best_100ms']   is not None else None,
                    'kmtid_slip_m':       r['kmtid_slip_m'],
                    'sources':            contribs,
                })
    finally:
        conn.close()
    return jsonify(rows)


# =====================================================================
# Track pages
# =====================================================================

def _track_redirect_id(conn, column, value):
    with conn.cursor() as cur:
        cur.execute(f"SELECT track_id FROM track WHERE {column} = %s", (value,))
        row = cur.fetchone()
    return row[0] if row else None


@app.route('/track/atg/<int:atg_track_id>')
def track_page_atg(atg_track_id):
    conn = get_db()
    try:
        tid = _track_redirect_id(conn, 'atg_track_id', atg_track_id)
    finally:
        conn.close()
    if tid is None:
        return 'not found', 404
    return redirect(url_for('track_page', track_id=tid), code=302)


@app.route('/track/<int:track_id>')
def track_page(track_id):
    conn = get_db()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("SELECT * FROM track WHERE track_id = %s", (track_id,))
            row = cur.fetchone()
            if not row:
                return 'not found', 404

            # Prefer the precomputed track_stats matview (one row, instant).
            # Fall back to a live aggregate if the row is missing (e.g. a brand
            # new track not yet in the matview, or matview not built yet).
            ea = None
            try:
                cur.execute("SELECT * FROM track_stats WHERE track_id = %s", (track_id,))
                ea = cur.fetchone()
            except Exception:
                conn.rollback()
            if ea is None:
                cur.execute(
                    "SELECT COUNT(*) AS races, MIN(race_date) AS first_date, "
                    "MAX(race_date) AS last_date FROM race WHERE track_id = %s",
                    (track_id,),
                )
                rc = cur.fetchone()
                cur.execute(
                    """
                    SELECT
                        COUNT(*) FILTER (WHERE NOT e.withdrawn)                          AS starts,
                        COUNT(*) FILTER (WHERE NOT e.withdrawn AND e.galopp)             AS gals,
                        COUNT(*) FILTER (WHERE NOT e.withdrawn AND e.auto)               AS starts_auto,
                        COUNT(*) FILTER (WHERE NOT e.withdrawn AND e.auto AND e.galopp)  AS gals_auto,
                        COUNT(*) FILTER (WHERE NOT e.withdrawn AND e.auto = false)              AS starts_volt,
                        COUNT(*) FILTER (WHERE NOT e.withdrawn AND e.auto = false AND e.galopp) AS gals_volt
                    FROM race r JOIN entry e ON e.race_id = r.race_id
                    WHERE r.track_id = %s
                    """,
                    (track_id,),
                )
                ea = cur.fetchone()
                ea['races'] = rc['races']
                ea['first_date'] = rc['first_date']
                ea['last_date'] = rc['last_date']

            # Average winner odds from precomputed track_post_stats (instant).
            cur.execute("""
                SELECT SUM(winner_odds_sum) AS odds_sum,
                       SUM(winner_cnt)      AS odds_cnt
                FROM track_post_stats
                WHERE track_id = %s
            """, (track_id,))
            odds_agg = cur.fetchone()
            avg_winner_odds = (
                round(float(odds_agg['odds_sum']) / int(odds_agg['odds_cnt']), 1)
                if odds_agg and odds_agg['odds_cnt'] else None
            )
    finally:
        conn.close()

    def _rate(num, den):
        return round(100 * num / den, 1) if den else None

    _starts = int(ea['starts'] or 0)
    _races = ea['races'] or 0

    track = {
        'track_id':      row['track_id'],
        'name':          (row['name'] or '').strip().title(),
        'country':       row['country'],
        'sport':         row['sport'],
        'st_code':       row['st_code'],
        'atg_track_id':  row['atg_track_id'],
        # static physical attributes
        'track_length_m':     row.get('track_length_m'),
        'home_stretch_m':     row.get('home_stretch_m'),
        'num_open_stretches': row.get('num_open_stretches'),
        'track_width_m':      row.get('track_width_m'),
        'auto_car_wings':     row.get('auto_car_wings'),
        'surface':            row.get('surface'),
        'shape':              row.get('shape'),
        'opened_year':        row.get('opened_year'),
        # derived stats (our data)
        'races':         _races,
        'first_year':    ea['first_date'].year if ea.get('first_date') else None,
        'last_year':     ea['last_date'].year if ea.get('last_date') else None,
        'starts':        _starts,
        'gal_rate':      _rate(ea['gals'], ea['starts']),
        'gal_rate_auto': _rate(ea['gals_auto'], ea['starts_auto']),
        'gal_rate_volt': _rate(ea['gals_volt'], ea['starts_volt']),
        'avg_winner_odds': avg_winner_odds,
    }
    return render_template('track.html', track=track, active_tab='stable_search')


@app.route('/api/track/<int:track_id>/races')
def track_races(track_id):
    """Recent races run at this track, most recent first (one row per race)."""
    conn = get_db()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                """
                SELECT r.race_id, r.atg_race_id AS atg_id, r.race_date, r.race_number,
                       agg.starters, agg.gals, agg.distance, agg.auto,
                       w.horse_id AS winner_id, w.horse_name AS winner_name,
                       w.km_time_text, w.time_seconds,
                       w.driver_id, w.driver_name, w.winner_odds
                FROM race r
                JOIN LATERAL (
                    SELECT COUNT(*) FILTER (WHERE NOT e.withdrawn)                AS starters,
                           COUNT(*) FILTER (WHERE NOT e.withdrawn AND e.galopp)   AS gals,
                           MIN(e.distance) FILTER (WHERE NOT e.withdrawn)         AS distance,
                           bool_or(e.auto)                                        AS auto
                    FROM entry e WHERE e.race_id = r.race_id
                ) agg ON TRUE
                LEFT JOIN LATERAL (
                    SELECT e2.horse_id, h.name AS horse_name,
                           e2.time_text AS km_time_text, e2.time_seconds,
                           e2.driver_id, p.name AS driver_name,
                           e2.odds AS winner_odds
                    FROM entry e2
                    JOIN horse h ON h.horse_id = e2.horse_id
                    LEFT JOIN person p ON p.person_id = e2.driver_id
                    WHERE e2.race_id = r.race_id
                      AND e2.placement_text = '1' AND NOT e2.disqualified
                    ORDER BY e2.entry_id
                    LIMIT 1
                ) w ON TRUE
                WHERE r.track_id = %s
                ORDER BY r.race_date DESC NULLS LAST, r.race_number DESC NULLS LAST
                LIMIT 300
                """,
                (track_id,),
            )
            rows = []
            for r in cur.fetchall():
                rows.append({
                    'race_date':   r['race_date'].isoformat() if r['race_date'] else None,
                    'race_id':     r['race_id'],
                    'atg_id':      r['atg_id'],
                    'race_number': r['race_number'],
                    'starters':    r['starters'],
                    'gals':        r['gals'],
                    'distance':    r['distance'],
                    'start_method': ('A' if r['auto'] is True else 'V' if r['auto'] is False else None),
                    'winner_id':   r['winner_id'],
                    'winner_name': r['winner_name'],
                    'km_time_text': r['km_time_text'],
                    'time_val':    float(r['time_seconds']) if r['time_seconds'] is not None else None,
                    'driver_id':   r['driver_id'],
                    'driver_name': r['driver_name'],
                    'winner_odds': float(r['winner_odds']) if r.get('winner_odds') else None,
                })
    finally:
        conn.close()
    return jsonify(rows)


# =====================================================================
# STATS — browse pages (horse / driver / trainer / track) backed by the
# horse_stats / person_stats / track_post_stats materialized views.
# =====================================================================

@app.route('/stats')
@app.route('/stats/horse')
def stats_horse_page():
    return render_template('stats_horse.html', active_tab='stats_horse')


@app.route('/stats/driver')
def stats_driver_page():
    return render_template('stats_person.html', active_tab='stats_driver',
                           role='driver')


@app.route('/stats/trainer')
def stats_trainer_page():
    return render_template('stats_person.html', active_tab='stats_trainer',
                           role='trainer')


@app.route('/stats/track')
def stats_track_page():
    return render_template('stats_track.html', active_tab='stats_track')


def _f(name, default=None):
    """Parse a float query arg, treating '' / missing as default."""
    v = request.args.get(name)
    if v is None or v == '':
        return default
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _i(name, default=None):
    v = _f(name, None)
    return int(v) if v is not None else default


# Sortable columns for the horse browse → safe SQL expressions. win%/top3%/gal%
# are computed; NULLIF guards against division by zero (0 starts).
_HORSE_SORTS = {
    'name':            'lower(name)',
    'breed':           'breed_code',
    'year':            'dob_year',
    'sex':             'gender_code',
    'starts':          'starts',
    'wins':            'wins',
    'winrate':         '(wins::numeric / NULLIF(starts,0))',
    'top3rate':        '(placed::numeric / NULLIF(starts,0))',
    'earnings':        'prize_money_kr',
    'offspring':       'offspring_count',
    'offspring_sek':   'offspring_prize',
    'avg_offspring':   '(offspring_prize::numeric / NULLIF(offspring_count,0))',
    'galrate':         '(gals::numeric / NULLIF(starts,0))',
    'record':          'best_time_s',
}

# active = raced within ~15 months; else inactive; deceased trumps both.
_ACTIVE_CUTOFF_SQL = "(CURRENT_DATE - INTERVAL '15 months')"


@app.route('/api/stats/horses')
def api_stats_horses():
    """Filtered + sorted + paginated horse list off the horse_stats MV.

    All filter bounds are optional; only the ones supplied constrain the set.
    Returns {rows, total, offset, limit}."""
    where, params = ['TRUE'], []

    breed = (request.args.get('breed') or '').upper()
    if breed in ('V', 'K'):
        where.append('breed_code = %s')
        params.append(breed)

    # sex: H stallion / V gelding / S mare (gender_code on the MV).
    sex = (request.args.get('sex') or '').upper()
    if sex in ('H', 'V', 'S'):
        where.append('gender_code = %s')
        params.append(sex)

    # Pedigree filters (by canonical horse id). 'after' maps to the maternal
    # grandsire column (mgs_id = dam.sire_id) exposed by the horse_stats MV.
    for col, arg in (('sire_id', 'sire_id'), ('dam_id', 'dam_id'), ('mgs_id', 'after_id')):
        pid = _i(arg)
        if pid:
            where.append(f'{col} = %s')
            params.append(pid)

    def rng(col_sql, lo_arg, hi_arg, scale=1.0):
        lo, hi = _f(lo_arg), _f(hi_arg)
        if lo is not None:
            where.append(f'{col_sql} >= %s')
            params.append(lo * scale)
        if hi is not None:
            where.append(f'{col_sql} <= %s')
            params.append(hi * scale)

    rng('starts', 'starts_min', 'starts_max')
    rng('wins', 'wins_min', 'wins_max')
    # win% / top3% / gal% arrive as 0..100; stored ratios are 0..1.
    rng('(wins::numeric / NULLIF(starts,0))', 'winrate_min', 'winrate_max', 0.01)
    rng('(placed::numeric / NULLIF(starts,0))', 'top3_min', 'top3_max', 0.01)
    rng('prize_money_kr', 'earnings_min', 'earnings_max')
    rng('offspring_count', 'offspring_min', 'offspring_max')
    rng('offspring_prize', 'offspring_sek_min', 'offspring_sek_max')
    rng('(offspring_prize::numeric / NULLIF(offspring_count,0))',
        'avg_offspring_min', 'avg_offspring_max')
    rng('(gals::numeric / NULLIF(starts,0))', 'galrate_min', 'galrate_max', 0.01)

    # record (best km time, seconds). Optional because most rows have no time
    # and distances differ — only filters when explicitly bounded, and rows
    # with no record pass through unless the user excludes them.
    rec_lo, rec_hi = _f('record_min'), _f('record_max')
    if rec_lo is not None or rec_hi is not None:
        clause = ['best_time_s IS NOT NULL']
        if rec_lo is not None:
            clause.append('best_time_s >= %s'); params.append(rec_lo)
        if rec_hi is not None:
            clause.append('best_time_s <= %s'); params.append(rec_hi)
        sub = ' AND '.join(clause)
        if request.args.get('record_exclude_null') == '1':
            where.append(f'({sub})')
        else:
            where.append(f'(best_time_s IS NULL OR ({sub}))')

    status = request.args.get('status')
    if status == 'active':
        where.append(f'(NOT is_dead AND last_start >= {_ACTIVE_CUTOFF_SQL})')
    elif status == 'inactive':
        where.append(f'(NOT is_dead AND (last_start IS NULL OR last_start < {_ACTIVE_CUTOFF_SQL}))')
    elif status == 'deceased':
        where.append('is_dead')

    # breeding status: geldings (V) have none. active = produced a foal in the
    # last 2 seasons; deceased trumps; else inactive.
    cur_year_sql = 'EXTRACT(YEAR FROM CURRENT_DATE)::int'
    breeding = request.args.get('breeding')
    if breeding == 'active':
        where.append(f"(gender_code <> 'V' AND NOT is_dead AND last_offspring_year >= {cur_year_sql} - 2)")
    elif breeding == 'inactive':
        where.append(f"(gender_code <> 'V' AND NOT is_dead AND (last_offspring_year IS NULL OR last_offspring_year < {cur_year_sql} - 2))")
    elif breeding == 'deceased':
        where.append("(gender_code <> 'V' AND is_dead)")

    where_sql = ' AND '.join(where)

    sort = request.args.get('sort', 'earnings')
    sort_sql = _HORSE_SORTS.get(sort, 'prize_money_kr')
    direction = 'ASC' if request.args.get('dir') == 'asc' else 'DESC'
    # record sorts ascending by default (fastest first); flip the implicit sense.
    nulls = 'NULLS LAST'

    limit = min(max(_i('limit', 50) or 50, 1), 200)
    offset = max(_i('offset', 0) or 0, 0)

    conn = get_db()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(f"SELECT COUNT(*) AS n FROM horse_stats WHERE {where_sql}", params)
            total = cur.fetchone()['n']
            cur.execute(
                f"""
                SELECT horse_id, name, breed_code, gender_code, dob_year, is_dead,
                       sire_id, sire_name, dam_id, dam_name, mgs_id, mgs_name,
                       trainer_id, trainer_name,
                       starts, wins, placed, gals, best_time_s, last_start,
                       prize_money_kr, offspring_count, offspring_prize,
                       last_offspring_year,
                       (last_start >= {_ACTIVE_CUTOFF_SQL}) AS recent_start,
                       {cur_year_sql} AS cur_year
                FROM horse_stats
                WHERE {where_sql}
                ORDER BY {sort_sql} {direction} {nulls}, prize_money_kr DESC, horse_id
                LIMIT %s OFFSET %s
                """,
                params + [limit, offset],
            )
            rows = [dict(r) for r in cur.fetchall()]
    finally:
        conn.close()

    cur_year = rows[0]['cur_year'] if rows else None
    out = []
    for r in rows:
        starts = r['starts'] or 0
        gender = r['gender_code']
        is_dead = r['is_dead']
        recent = r['recent_start']
        loy = r['last_offspring_year']
        if is_dead:
            status_v = 'deceased'
        elif recent:
            status_v = 'active'
        else:
            status_v = 'inactive'
        if gender == 'V':
            breeding_v = None
        elif is_dead:
            breeding_v = 'deceased'
        elif loy is not None and cur_year is not None and loy >= cur_year - 2:
            breeding_v = 'active'
        else:
            breeding_v = 'inactive'
        out.append({
            'horse_id': r['horse_id'],
            'name': r['name'],
            'breed': r['breed_code'],
            'year': r['dob_year'],
            'sex': (r['gender_code'] or '').lower(),
            'sire_id': r['sire_id'], 'sire_name': r['sire_name'],
            'dam_id': r['dam_id'], 'dam_name': r['dam_name'],
            'mgs_id': r['mgs_id'], 'mgs_name': r['mgs_name'],
            'trainer_id': r['trainer_id'], 'trainer_name': r['trainer_name'],
            'starts': starts,
            'wins': r['wins'] or 0,
            'winrate': round(100 * r['wins'] / starts, 1) if starts else None,
            'top3rate': round(100 * r['placed'] / starts, 1) if starts else None,
            'earnings': int(r['prize_money_kr'] or 0),
            'offspring': r['offspring_count'] or 0,
            'offspring_sek': int(r['offspring_prize'] or 0),
            'avg_offspring': int(r['offspring_prize'] / r['offspring_count']) if r['offspring_count'] else None,
            'galrate': round(100 * r['gals'] / starts, 1) if starts else None,
            'record': float(r['best_time_s']) if r['best_time_s'] is not None else None,
            'status': status_v,
            'breeding': breeding_v,
        })
    return jsonify({'rows': out, 'total': total, 'offset': offset, 'limit': limit})


_PERSON_SORTS = {
    'name':     'lower(ps.name)',
    'starts':   'ps.starts',
    'wins':     'ps.wins',
    'winrate':  '(ps.wins::numeric / NULLIF(ps.starts,0))',
    'top3rate': '(ps.placed::numeric / NULLIF(ps.starts,0))',
    'earnings': 'ps.prize',
    'galrate':  '(ps.gals::numeric / NULLIF(ps.starts,0))',
    's_form':   's_form',
    'form':     'form',
}


@app.route('/api/stats/persons')
def api_stats_persons():
    """Filtered/sorted driver or trainer list off person_stats. ?role=driver|trainer."""
    role = 'trainer' if request.args.get('role') == 'trainer' else 'driver'
    where, params = ['role = %s'], [role]

    def rng(col_sql, lo_arg, hi_arg, scale=1.0):
        lo, hi = _f(lo_arg), _f(hi_arg)
        if lo is not None:
            where.append(f'{col_sql} >= %s'); params.append(lo * scale)
        if hi is not None:
            where.append(f'{col_sql} <= %s'); params.append(hi * scale)

    rng('starts', 'starts_min', 'starts_max')
    rng('wins', 'wins_min', 'wins_max')
    rng('(wins::numeric / NULLIF(starts,0))', 'winrate_min', 'winrate_max', 0.01)
    rng('(placed::numeric / NULLIF(starts,0))', 'top3_min', 'top3_max', 0.01)
    rng('prize', 'earnings_min', 'earnings_max')
    rng('(gals::numeric / NULLIF(starts,0))', 'galrate_min', 'galrate_max', 0.01)

    country = (request.args.get('country') or '').upper()
    if country:
        # Trainer license_country is known to be heavily polluted after merges;
        # don't let it drive filtering on the trainer stats page.
        if role != 'trainer':
            where.append('license_country = %s')
            params.append(country)

    where_sql = ' AND '.join(where)
    sort = request.args.get('sort', 'earnings')
    try:
        sform_days = int(request.args.get('sform_days', 30))
    except (TypeError, ValueError):
        sform_days = 30
    if sform_days not in _FORM_MIN_STARTS_BY_DAYS:
        sform_days = 30
    sort_sql = _PERSON_SORTS.get(sort, 'prize')
    direction = 'ASC' if request.args.get('dir') == 'asc' else 'DESC'
    limit = min(max(_i('limit', 50) or 50, 1), 200)
    offset = max(_i('offset', 0) or 0, 0)

    conn = get_db()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(f"SELECT COUNT(*) AS n FROM person_stats WHERE {where_sql}", params)
            total = cur.fetchone()['n']
            cur.execute(
                f"""
                WITH sform AS (
                    SELECT e.trainer_id AS person_id,
                           COUNT(eo.market_outperf)            AS n_sform,
                           COALESCE(SUM(eo.market_outperf), 0) AS sum_sform,
                           COUNT(ep.perf)                      AS n_form,
                           COALESCE(SUM(ep.perf), 0)           AS sum_form
                    FROM entry e
                    LEFT JOIN entry_outperf eo ON eo.entry_id = e.entry_id
                    LEFT JOIN entry_perf    ep ON ep.entry_id = e.entry_id
                    WHERE %s = 'trainer'
                      AND e.trainer_id IS NOT NULL
                      AND e.race_date >= CURRENT_DATE - (%s || ' days')::interval
                    GROUP BY e.trainer_id
                )
                SELECT ps.person_id, ps.name, ps.short_name, ps.license_country,
                       ps.starts, ps.wins, ps.placed, ps.gals, ps.prize, ps.last_start,
                       sf.n_sform,
                       CASE WHEN sf.n_sform >= %s
                            THEN round((sf.sum_sform / sf.n_sform) * 1000)::int
                       END AS s_form,
                       CASE WHEN sf.n_form >= %s
                            THEN round((sf.sum_form / sf.n_form) * 1000)::int
                       END AS form
                FROM person_stats ps
                LEFT JOIN sform sf ON sf.person_id = ps.person_id
                WHERE {where_sql}
                ORDER BY {sort_sql} {direction} NULLS LAST, ps.prize DESC, ps.person_id
                LIMIT %s OFFSET %s
                """,
                [role, sform_days, _FORM_MIN_STARTS_BY_DAYS[sform_days],
                 _FORM_MIN_STARTS_BY_DAYS[sform_days]] + params + [limit, offset],
            )
            rows = [dict(r) for r in cur.fetchall()]
    finally:
        conn.close()

    out = []
    for r in rows:
        starts = r['starts'] or 0
        out.append({
            'person_id': r['person_id'],
            'name': r['name'],
            'country': r['license_country'],
            'starts': starts,
            'wins': r['wins'] or 0,
            'winrate': round(100 * r['wins'] / starts, 1) if starts else None,
            'top3rate': round(100 * r['placed'] / starts, 1) if starts else None,
            'earnings': int(r['prize'] or 0),
            'galrate': round(100 * r['gals'] / starts, 1) if starts else None,
            's_form': r.get('s_form') if role == 'trainer' else None,
            'form': r.get('form') if role == 'trainer' else None,
        })
    return jsonify({'rows': out, 'total': total, 'offset': offset, 'limit': limit, 'role': role})


_TRACK_POST_MIN_STARTS = 50  # ignore thin posts when picking a "top" post


def _track_post_rows(cur, country='SE'):
    """All per-(track, method, post) rows for tracks in `country`, joined to
    track meta. Small (≤ ~4k rows) so aggregation happens in Python."""
    cur.execute(
        """
        SELECT tps.track_id, tps.auto, tps.post,
               tps.starts, tps.wins, tps.placed, tps.gals,
               tps.winner_odds_sum, tps.winner_cnt
        FROM track_post_stats tps
        JOIN track t ON t.track_id = tps.track_id
        WHERE t.country = %s
        """,
        (country,),
    )
    return cur.fetchall()


def _post_win_index(post_rows, method):
    """{track_id: {post: win_rate}} for one method (auto/volt/any)."""
    idx = {}
    for r in post_rows:
        if method == 'auto' and r['auto'] is not True:
            continue
        if method == 'volt' and r['auto'] is not False:
            continue
        d = idx.setdefault(r['track_id'], {})
        cur = d.get(r['post'])
        if cur is None:
            d[r['post']] = {'starts': 0, 'wins': 0}
        d[r['post']]['starts'] += r['starts'] or 0
        d[r['post']]['wins'] += r['wins'] or 0
    return idx


def _top_posts(post_idx):
    """{track_id: (top_post, top_post_relative)} from a {track:{post:{starts,wins}}}.

    top_post        = post with the highest win% (min sample).
    top_post_rel    = the post where THIS track ranks highest vs other tracks
                      on the same post (percentile of win% across tracks)."""
    # win% per track/post
    rate = {}
    for tid, posts in post_idx.items():
        for p, agg in posts.items():
            if agg['starts'] >= _TRACK_POST_MIN_STARTS:
                rate.setdefault(tid, {})[p] = agg['wins'] / agg['starts']
    # cross-track distribution per post for percentile ranking
    by_post = {}
    for tid, posts in rate.items():
        for p, wr in posts.items():
            by_post.setdefault(p, []).append(wr)
    for p in by_post:
        by_post[p].sort()
    out = {}
    for tid, posts in rate.items():
        if not posts:
            continue
        top_post = max(posts, key=lambda p: posts[p])
        best_pct, best_rel_post = -1.0, None
        for p, wr in posts.items():
            arr = by_post.get(p) or []
            if len(arr) < 5:
                continue
            below = sum(1 for x in arr if x < wr)
            pct = below / (len(arr) - 1) if len(arr) > 1 else 0.0
            if pct > best_pct:
                best_pct, best_rel_post = pct, p
        out[tid] = (top_post, best_rel_post if best_rel_post is not None else top_post)
    return out


@app.route('/api/stats/tracks')
def api_stats_tracks():
    """Track browse. Locked to country=SE for now. ?method=any|auto|volt
    drives gal% and the top-post columns."""
    method = request.args.get('method', 'any')
    if method not in ('any', 'auto', 'volt'):
        method = 'any'

    conn = get_db()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                """
                SELECT t.track_id, t.name, t.country,
                       t.track_length_m, t.track_width_m, t.surface,
                       ts.races,
                       ts.starts, ts.gals,
                       ts.starts_auto, ts.gals_auto,
                       ts.starts_volt, ts.gals_volt
                FROM track t
                JOIN track_stats ts ON ts.track_id = t.track_id
                WHERE t.country = 'SE'
                """,
            )
            tracks = cur.fetchall()
            post_rows = _track_post_rows(cur, 'SE')
    finally:
        conn.close()

    post_idx = _post_win_index(post_rows, method)
    tops = _top_posts(post_idx)
    # avg winner odds per track for the selected method
    odds = {}
    for r in post_rows:
        if method == 'auto' and r['auto'] is not True:
            continue
        if method == 'volt' and r['auto'] is not False:
            continue
        d = odds.setdefault(r['track_id'], [0.0, 0])
        d[0] += float(r['winner_odds_sum'] or 0)
        d[1] += int(r['winner_cnt'] or 0)

    def _rate(n, d):
        return round(100 * n / d, 1) if d else None

    rows = []
    for t in tracks:
        if method == 'auto':
            g_starts, g_gals = t['starts_auto'], t['gals_auto']
        elif method == 'volt':
            g_starts, g_gals = t['starts_volt'], t['gals_volt']
        else:
            g_starts, g_gals = t['starts'], t['gals']
        od = odds.get(t['track_id'])
        avg_odds = round(od[0] / od[1], 1) if od and od[1] else None
        tp = tops.get(t['track_id'])
        rows.append({
            'track_id': t['track_id'],
            'name': (t['name'] or '').strip().title(),
            'country': t['country'],
            'length': t['track_length_m'],
            'width': t['track_width_m'],
            'surface': t['surface'],
            'races': t['races'] or 0,
            'galrate': _rate(g_gals, g_starts),
            'galrate_auto': _rate(t['gals_auto'], t['starts_auto']),
            'galrate_volt': _rate(t['gals_volt'], t['starts_volt']),
            'avg_winner_odds': avg_odds,
            'top_post': tp[0] if tp else None,
            'top_post_rel': tp[1] if tp else None,
            'starts': g_starts or 0,
        })

    # filters
    def passes(r):
        for key, col in (('races_min', 'races'), ('races_max', 'races'),
                         ('galrate_min', 'galrate'), ('galrate_max', 'galrate'),
                         ('length_min', 'length'), ('length_max', 'length'),
                         ('odds_min', 'avg_winner_odds'), ('odds_max', 'avg_winner_odds')):
            v = _f(key)
            if v is None:
                continue
            cv = r[col]
            if cv is None:
                return False
            if key.endswith('_min') and cv < v:
                return False
            if key.endswith('_max') and cv > v:
                return False
        surf = request.args.get('surface')
        if surf and (r['surface'] or '') != surf:
            return False
        return True

    rows = [r for r in rows if passes(r)]

    sort = request.args.get('sort', 'races')
    keymap = {
        'name': lambda r: (r['name'] or '').lower(),
        'length': lambda r: (r['length'] is None, r['length'] or 0),
        'width': lambda r: (r['width'] is None, r['width'] or 0),
        'races': lambda r: r['races'] or 0,
        'galrate': lambda r: (r['galrate'] is None, r['galrate'] or 0),
        'avg_winner_odds': lambda r: (r['avg_winner_odds'] is None, r['avg_winner_odds'] or 0),
    }
    keyfn = keymap.get(sort, keymap['races'])
    reverse = request.args.get('dir', 'desc') != 'asc'
    # keep NULLs last regardless of direction for the (is_none, val) tuples
    rows.sort(key=keyfn, reverse=reverse)
    return jsonify({'rows': rows, 'total': len(rows), 'method': method})


@app.route('/api/stats/track-options')
def api_stats_track_options():
    """SE tracks with enough data for the post-position visualisation dropdown."""
    conn = get_db()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                """
                SELECT t.track_id, t.name, ts.races
                FROM track t
                JOIN track_stats ts ON ts.track_id = t.track_id
                WHERE t.country = 'SE' AND ts.races > 1000
                ORDER BY ts.races DESC, t.name
                """,
            )
            rows = [{'track_id': r['track_id'],
                     'name': (r['name'] or '').strip().title(),
                     'races': r['races']} for r in cur.fetchall()]
    finally:
        conn.close()
    return jsonify(rows)


_POST_MIN_STARTS_VIZ = 50  # ignore thin posts in the visualisation


@app.route('/api/stats/track-posts')
def api_stats_track_posts():
    """Per-post win%/gal% for one track + the cross-track SE average + global
    max per post (for consistent bar scaling). Caps auto at 12 posts (13-15
    are statistical noise in autostart). Volt shows all 15.

    ?track_id=&method=auto|volt&metric=win|gal."""
    track_id = _i('track_id')
    method = request.args.get('method', 'auto')
    if method not in ('auto', 'volt'):
        method = 'auto'
    metric = request.args.get('metric', 'win')
    if metric not in ('win', 'gal'):
        metric = 'win'
    auto_val = (method == 'auto')
    max_post = 12 if auto_val else 15

    conn = get_db()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            # this track
            this = {}
            if track_id is not None:
                cur.execute(
                    """
                    SELECT post, starts, wins, gals
                    FROM track_post_stats
                    WHERE track_id = %s AND auto = %s AND post <= %s
                    """,
                    (track_id, auto_val, max_post),
                )
                for r in cur.fetchall():
                    this[r['post']] = r
            # cross-track average per post (SE, >1000 races, per-post >=50 starts)
            cur.execute(
                """
                SELECT tps.post,
                       SUM(tps.starts) AS starts,
                       SUM(tps.wins)   AS wins,
                       SUM(tps.gals)   AS gals
                FROM track_post_stats tps
                JOIN track t       ON t.track_id = tps.track_id
                JOIN track_stats ts ON ts.track_id = tps.track_id
                WHERE t.country = 'SE' AND ts.races > 1000
                      AND tps.auto = %s AND tps.post <= %s
                      AND tps.starts >= %s
                GROUP BY tps.post
                """,
                (auto_val, max_post, _POST_MIN_STARTS_VIZ),
            )
            avg = {r['post']: r for r in cur.fetchall()}

            # global max per post across all qualifying SE tracks (for bar ceiling)
            col = 'wins' if metric == 'win' else 'gals'
            cur.execute(
                f"""
                SELECT tps.post,
                       MAX(100.0 * tps.{col} / NULLIF(tps.starts, 0)) AS max_val
                FROM track_post_stats tps
                JOIN track t       ON t.track_id = tps.track_id
                JOIN track_stats ts ON ts.track_id = tps.track_id
                WHERE t.country = 'SE' AND ts.races > 1000
                      AND tps.auto = %s AND tps.post <= %s
                      AND tps.starts >= %s
                GROUP BY tps.post
                """,
                (auto_val, max_post, _POST_MIN_STARTS_VIZ),
            )
            maxvals = {r['post']: float(r['max_val']) if r['max_val'] else None
                       for r in cur.fetchall()}
    finally:
        conn.close()

    def val(r):
        if not r or not r['starts'] or int(r['starts']) < _POST_MIN_STARTS_VIZ:
            return None
        num = float(r['wins'] if metric == 'win' else r['gals'])
        return round(100 * num / float(r['starts']), 2)

    posts = []
    for p in range(1, max_post + 1):
        posts.append({
            'post': p,
            'value': val(this.get(p)),
            'starts': (this.get(p) or {}).get('starts') or 0,
            'avg': val(avg.get(p)),
            'max': round(maxvals.get(p) or 0, 2),
        })
    return jsonify({'track_id': track_id, 'method': method, 'metric': metric,
                    'max_post': max_post, 'posts': posts})


@app.route('/api/horse/<int:horse_id>/offspring')
def horse_offspring(horse_id):
    """Offspring of a horse, ordered by lifetime earnings."""
    conn = get_db()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                """
                SELECT h.horse_id,
                       h.name,
                       to_char(h.date_of_birth, 'YYYY') AS year,
                       h.gender_code,
                       CASE WHEN h.sire_id = %s THEN COALESCE(dam_h.name, h.dam_name)
                            ELSE COALESCE(sire_h.name, h.sire_name) END
                         AS other_parent_name,
                       CASE WHEN h.sire_id = %s THEN h.dam_id ELSE h.sire_id END
                         AS other_parent_id,
                       CASE WHEN h.sire_id = %s THEN COALESCE(dam_sire_h.name, dam_h.sire_name)
                            ELSE COALESCE(sire_sire_h.name, sire_h.sire_name) END
                         AS after_parent_name,
                       CASE WHEN h.sire_id = %s THEN dam_h.sire_id ELSE sire_h.sire_id END
                         AS after_parent_id,
                       s.starts,
                       s.wins,
                       s.prize_money_kr
                FROM horse h
                LEFT JOIN horse sire_h ON sire_h.horse_id = h.sire_id
                LEFT JOIN horse dam_h  ON dam_h.horse_id  = h.dam_id
                LEFT JOIN horse sire_sire_h ON sire_sire_h.horse_id = sire_h.sire_id
                LEFT JOIN horse dam_sire_h  ON dam_sire_h.horse_id  = dam_h.sire_id
                LEFT JOIN horse_career_stats s ON s.horse_id = h.horse_id
                WHERE h.sire_id = %s
                   OR h.dam_id  = %s
                ORDER BY s.prize_money_kr DESC NULLS LAST, h.date_of_birth DESC NULLS LAST
                LIMIT 500
                """,
                (horse_id, horse_id, horse_id, horse_id, horse_id, horse_id),
            )
            rows = []
            for r in cur.fetchall():
                starts = r['starts'] or 0
                wins = r['wins'] or 0
                rows.append({
                    'horse_id': r['horse_id'],
                    'name': r['name'],
                    'year': r['year'],
                    'gender': _GENDER_TEXT.get(r['gender_code'] or '', ''),
                    'other_parent_name': r['other_parent_name'] or '',
                    'other_parent_id': r['other_parent_id'],
                    'after_parent_name': r['after_parent_name'] or '',
                    'after_parent_id': r['after_parent_id'],
                    'starts': starts,
                    'wins': wins,
                    'win_rate': f"{round(100 * wins / starts)}" if starts else '',
                    'earnings': _kr(r['prize_money_kr']) if r['prize_money_kr'] else '',
                    'earnings_raw': int(r['prize_money_kr']) if r['prize_money_kr'] else 0,
                })
    finally:
        conn.close()
    return jsonify(rows)


@app.route('/api/horse/<int:horse_id>/siblings')
def horse_siblings(horse_id):
    """Siblings of a horse: other horses sharing its sire and/or dam.

    `rel` is 'full' (both parents shared), 'paternal' (sire only) or
    'maternal' (dam only). Full siblings are returned first."""
    conn = get_db()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("SELECT sire_id, dam_id FROM horse WHERE horse_id = %s", (horse_id,))
            base = cur.fetchone()
            if not base or (base['sire_id'] is None and base['dam_id'] is None):
                return jsonify([])
            sx, dx = base['sire_id'], base['dam_id']
            cur.execute(
                """
                SELECT h.horse_id,
                       h.name,
                       to_char(h.date_of_birth, 'YYYY') AS year,
                       h.gender_code,
                       h.sire_id,
                       COALESCE(sire_h.name, h.sire_name) AS sire_name,
                       h.dam_id,
                       COALESCE(dam_h.name, h.dam_name) AS dam_name,
                       (%(sx)s IS NOT NULL AND h.sire_id = %(sx)s) AS shares_sire,
                       (%(dx)s IS NOT NULL AND h.dam_id  = %(dx)s) AS shares_dam,
                       s.starts,
                       s.wins,
                       s.prize_money_kr
                FROM horse h
                LEFT JOIN horse sire_h ON sire_h.horse_id = h.sire_id
                LEFT JOIN horse dam_h  ON dam_h.horse_id  = h.dam_id
                LEFT JOIN horse_career_stats s ON s.horse_id = h.horse_id
                WHERE h.horse_id <> %(x)s
                  AND ( (%(sx)s IS NOT NULL AND h.sire_id = %(sx)s)
                     OR (%(dx)s IS NOT NULL AND h.dam_id  = %(dx)s) )
                ORDER BY COALESCE(%(sx)s IS NOT NULL AND h.sire_id = %(sx)s
                          AND %(dx)s IS NOT NULL AND h.dam_id = %(dx)s, FALSE) DESC,
                         s.prize_money_kr DESC NULLS LAST,
                         h.date_of_birth DESC NULLS LAST
                LIMIT 500
                """,
                {'x': horse_id, 'sx': sx, 'dx': dx},
            )
            rows = []
            for r in cur.fetchall():
                starts = r['starts'] or 0
                wins = r['wins'] or 0
                shares_sire = bool(r['shares_sire'])
                shares_dam = bool(r['shares_dam'])
                rel = 'full' if (shares_sire and shares_dam) else ('paternal' if shares_sire else 'maternal')
                rows.append({
                    'horse_id': r['horse_id'],
                    'name': r['name'],
                    'year': r['year'],
                    'gender': _GENDER_TEXT.get(r['gender_code'] or '', ''),
                    'sire_id': r['sire_id'],
                    'sire_name': r['sire_name'] or '',
                    'dam_id': r['dam_id'],
                    'dam_name': r['dam_name'] or '',
                    'rel': rel,
                    'is_full': shares_sire and shares_dam,
                    'starts': starts,
                    'wins': wins,
                    'win_rate': f"{round(100 * wins / starts)}" if starts else '',
                    'earnings': _kr(r['prize_money_kr']) if r['prize_money_kr'] else '',
                    'earnings_raw': int(r['prize_money_kr']) if r['prize_money_kr'] else 0,
                })
    finally:
        conn.close()
    return jsonify(rows)


# =====================================================================
# Race (single)
# =====================================================================

@app.route('/race/<int:race_id>')
def race_page_by_id(race_id):
    return render_template('race.html', active_tab='race',
                           race_key=str(race_id), is_atg=False)


@app.route('/race/atg/<path:atg_race_id>')
def race_page_by_atg(atg_race_id):
    return render_template('race.html', active_tab='race',
                           race_key=atg_race_id, is_atg=True)


@app.route('/race/st/<int:st_race_id>')
def race_by_st(st_race_id):
    conn = get_db()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT race_id FROM race WHERE st_race_id = %s LIMIT 1",
                        (st_race_id,))
            row = cur.fetchone()
    finally:
        conn.close()
    if not row:
        return 'not found', 404
    return redirect(url_for('race_page_by_id', race_id=row[0]), code=302)


@app.route('/api/race/<int:race_id>')
def race_detail_by_id(race_id):
    return _race_entries(race_id=race_id)


@app.route('/api/race/atg/<path:atg_race_id>')
def race_detail_by_atg(atg_race_id):
    return _race_entries(atg_race_id=atg_race_id)


@app.route('/api/race/atg/<path:atg_race_id>/live')
def race_live_by_atg(atg_race_id):
    """Live win/place odds + bet distribution (spelprocent) for an upcoming
    race, pulled from the ATG game pools."""
    data = _atg_live_pools(atg_race_id)
    if not data:
        return jsonify({'error': 'no live pools'}), 404
    return jsonify(data)


def _race_entries(*, race_id=None, atg_race_id=None):
    conn = get_db()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            if atg_race_id is not None:
                cur.execute("SELECT race_id FROM race WHERE atg_race_id = %s",
                            (atg_race_id,))
                row = cur.fetchone()
                if not row:
                    return _race_entries_atg_live(cur, atg_race_id)
                race_id = row['race_id']

            cur.execute(
                """
                SELECT r.race_date,
                       r.race_number, r.distance, r.start_method, r.race_class,
                       r.victory_margin,
                       r.track_id,
                       t.name    AS track_name,
                       t.country AS track_country,
                       r.atg_race_id, r.atg_race_day_id, r.st_race_id,
                       r.letrot_race_id,
                       r.kmtid_id,
                       r.primary_source,
                       r.source_data->'_contributors' AS contributors
                FROM race r
                LEFT JOIN track t ON t.track_id = r.track_id
                WHERE r.race_id = %s
                """,
                (race_id,),
            )
            head = cur.fetchone()
            if not head:
                return jsonify({'error': 'not found'}), 404

            cur.execute(
                """
                SELECT e.entry_id,
                       e.horse_id,
                       h.name                          AS horse_name,
                       e.program_number                AS number,
                       e.distance,
                       e.placement,
                       e.placement_text,
                       e.time_text,
                       e.time_seconds                  AS time_val,
                       e.odds,
                       e.prize_kr                      AS prize,
                       e.disqualified                  AS dq,
                       e.galopp                        AS gal,
                       e.withdrawn,
                       e.age, e.sex,
                       e.sulky, e.sulky_changed,
                       e.tillagg,
                       e.shoe_code, e.shoe_front_changed, e.shoe_back_changed,
                       e.kmtid_actual_distance_m       AS kmtid_actual_m,
                       e.kmtid_actual_km_time_ms       AS kmtid_actual_km_ms,
                       e.kmtid_best_100ms              AS kmtid_best_100ms,
                       e.kmtid_first_200ms             AS kmtid_first_200ms,
                       e.kmtid_last_200ms              AS kmtid_last_200ms,
                       e.kmtid_slipstream_distance_m   AS kmtid_slip_m,
                       e.primary_source,
                       e.source_data->'_contributors' AS contributors,
                       pd.name AS driver_name,  pd.short_name AS driver_short_name,  e.driver_id,
                       pt.name AS trainer_name, pt.short_name AS trainer_short_name, e.trainer_id
                FROM entry e
                LEFT JOIN horse  h  ON h.horse_id   = e.horse_id
                LEFT JOIN person pd ON pd.person_id = e.driver_id
                LEFT JOIN person pt ON pt.person_id = e.trainer_id
                WHERE e.race_id = %s
                ORDER BY COALESCE(e.program_number, 999), e.horse_id
                """,
                (race_id,),
            )
            entry_rows = cur.fetchall()
            entry_ids = [r['entry_id'] for r in entry_rows if r.get('entry_id')]
            gal_track_map, gal_general_map = {}, {}
            gal_models = {'track': None, 'general': None, 'has_track_model': False}
            if entry_ids:
                cur.execute(
                    """
                    SELECT p.entry_id, p.target, p.prob,
                           m.model_id, m.name
                    FROM ml_prediction p
                    LEFT JOIN ml_model m ON m.model_id = p.model_id
                    WHERE p.entry_id = ANY(%s)
                      AND p.target IN ('y_gal_track', 'y_gal_general', 'y_gal')
                    """,
                    (entry_ids,),
                )
                for r in cur.fetchall():
                    prob = float(r['prob']) if r['prob'] is not None else None
                    if r['target'] in ('y_gal_track', 'y_gal'):
                        gal_track_map[r['entry_id']] = prob
                        if r['target'] == 'y_gal_track' and r['name']:
                            gal_models['track'] = {'model_id': r['model_id'], 'name': r['name']}
                    if r['target'] in ('y_gal_general', 'y_gal'):
                        gal_general_map[r['entry_id']] = prob
                        if r['target'] == 'y_gal_general' and r['name']:
                            gal_models['general'] = {'model_id': r['model_id'], 'name': r['name']}
                # Legacy y_gal-only rows: treat as both track and general.
                for eid in entry_ids:
                    if eid not in gal_track_map and eid in gal_general_map:
                        gal_track_map[eid] = gal_general_map[eid]
                    if eid not in gal_general_map and eid in gal_track_map:
                        gal_general_map[eid] = gal_track_map[eid]

            track_id = head.get('track_id')
            if track_id and not gal_models['track']:
                cur.execute("""
                    SELECT model_id, name
                    FROM ml_model
                    WHERE target = 'y_gal' AND scope = 'track' AND track_id = %s
                    ORDER BY (metrics->'test'->>'roc_auc')::float DESC NULLS LAST
                    LIMIT 1
                """, (track_id,))
                tm = cur.fetchone()
                if tm:
                    gal_models['track'] = {'model_id': tm['model_id'], 'name': tm['name']}
            if not gal_models['general']:
                cur.execute("""
                    SELECT model_id, name FROM ml_model
                    WHERE target = 'y_gal' AND scope = 'general'
                      AND COALESCE(slice_def->>'method', 'any') = 'any'
                    ORDER BY (metrics->'test'->>'roc_auc')::float DESC NULLS LAST
                    LIMIT 1
                """)
                gm = cur.fetchone()
                if gm:
                    gal_models['general'] = {'model_id': gm['model_id'], 'name': gm['name']}
            if gal_models['track'] and gal_models['general']:
                gal_models['has_track_model'] = (
                    gal_models['track']['model_id'] != gal_models['general']['model_id']
                )
            horse_ids = [r['horse_id'] for r in entry_rows if r['horse_id']]
            driver_ids = list({r['driver_id'] for r in entry_rows if r['driver_id']})
            trainer_ids = list({r['trainer_id'] for r in entry_rows if r['trainer_id']})
            head_race_date = head.get('race_date')
            d_wr_map = _person_win_rates(conn, driver_ids, 'driver')
            t_wr_map = _person_win_rates(conn, trainer_ids, 'trainer')
            df_map = _batch_person_form_at_date(conn, driver_ids, 'driver', head_race_date)
            tf_map = _batch_person_form_at_date(conn, trainer_ids, 'trainer', head_race_date)

            stats_pre: dict[int, dict] = {}
            if horse_ids and head_race_date:
                cur.execute(
                    """
                    SELECT e.horse_id,
                           COUNT(*) FILTER (
                               WHERE NOT e.withdrawn
                                 AND COALESCE(e.placement_text, '') !~ '{_QUALIFIER_RE}'
                           ) AS starts,
                           COUNT(*) FILTER (
                               WHERE e.placement_text = '1'
                                 AND NOT COALESCE(e.disqualified, false)
                                 AND COALESCE(e.placement_text, '') !~ '{_QUALIFIER_RE}'
                           ) AS wins,
                           COUNT(*) FILTER (
                               WHERE NOT e.withdrawn
                                 AND NOT e.galopp
                                 AND NOT COALESCE(e.disqualified, false)
                                 AND """ + _NOT_QUALIFIER + """
                           ) AS clean_starts,
                           COUNT(*) FILTER (
                               WHERE e.placement_text = '1'
                                 AND NOT COALESCE(e.disqualified, false)
                                 AND NOT e.galopp
                                 AND """ + _NOT_QUALIFIER + """
                           ) AS clean_wins
                    FROM entry e
                    JOIN race  r2 ON r2.race_id = e.race_id
                    WHERE e.horse_id = ANY(%s)
                      AND r2.race_date < %s
                    GROUP BY e.horse_id
                    """,
                    (horse_ids, head_race_date),
                )
                for srow in cur.fetchall():
                    stats_pre[srow['horse_id']] = {
                        'starts': srow['starts'] or 0,
                        'wins':   srow['wins'] or 0,
                        'clean_starts': srow['clean_starts'] or 0,
                        'clean_wins':   srow['clean_wins'] or 0,
                    }

            rows = []
            for r in entry_rows:
                pre = stats_pre.get(r['horse_id'],
                                    {'starts': 0, 'wins': 0,
                                     'clean_starts': 0, 'clean_wins': 0})
                pt = r['placement_text'] or ''
                qual = bool(_re_mod.match(_QUALIFIER_RE, pt))
                contributes_start = (not r['withdrawn']) and (not qual)
                won_this_race = pt == '1' and not r['dq'] and not qual
                post_starts = pre['starts'] + (1 if contributes_start else 0)
                post_wins   = pre['wins']   + (1 if won_this_race    else 0)
                # Gal-adjusted: a clean start excludes galopp + DQ races.
                clean_this_race = contributes_start and (not r['gal']) and (not r['dq'])
                clean_won_this_race = won_this_race and (not r['gal'])
                post_clean_starts = pre['clean_starts'] + (1 if clean_this_race else 0)
                post_clean_wins   = pre['clean_wins']   + (1 if clean_won_this_race else 0)
                rows.append({
                    'entry_id': r['entry_id'],
                    'horse_id': r['horse_id'],
                    'horse_name': r['horse_name'],
                    'number': r['number'],
                    'xgal_track': gal_track_map.get(r['entry_id']),
                    'xgal_general': gal_general_map.get(r['entry_id']),
                    'distance': r['distance'],
                    'primary_source': r['primary_source'],
                    'contributors': r.get('contributors') or [],
                    'kmtid_actual_m':     r['kmtid_actual_m'],
                    'kmtid_actual_km_ms': float(r['kmtid_actual_km_ms']) if r['kmtid_actual_km_ms'] is not None else None,
                    'kmtid_best_100ms':   float(r['kmtid_best_100ms'])   if r['kmtid_best_100ms']   is not None else None,
                    'kmtid_first_200ms':  float(r['kmtid_first_200ms'])  if r['kmtid_first_200ms']  is not None else None,
                    'kmtid_last_200ms':   float(r['kmtid_last_200ms'])   if r['kmtid_last_200ms']   is not None else None,
                    'kmtid_slip_m':       r['kmtid_slip_m'],
                    'placement': r['placement'],
                    'placement_text': r['placement_text'],
                    'time_text': r['time_text'],
                    'time_val': float(r['time_val']) if r['time_val'] is not None else None,
                    'odds': float(r['odds']) if r['odds'] is not None else None,
                    'prize_kr': _kr(r['prize']) if r['prize'] else '',
                    'age': r['age'],
                    'sex': r['sex'],
                    'sulky': r['sulky'],
                    'sulky_changed': r['sulky_changed'],
                    'tillagg': r['tillagg'],
                    'shoe_code': r['shoe_code'],
                    'shoe_front_changed': r['shoe_front_changed'],
                    'shoe_back_changed': r['shoe_back_changed'],
                    'tf': tf_map.get(r['trainer_id'], {}).get('form'),
                    'tf_odds': tf_map.get(r['trainer_id'], {}).get('form_odds'),
                    'tf_perf': tf_map.get(r['trainer_id'], {}).get('form_perf'),
                    'df': df_map.get(r['driver_id'], {}).get('form'),
                    'df_odds': df_map.get(r['driver_id'], {}).get('form_odds'),
                    'df_perf': df_map.get(r['driver_id'], {}).get('form_perf'),
                    'd_wr': d_wr_map.get(r['driver_id']),
                    't_wr': t_wr_map.get(r['trainer_id']),
                    'driver_name':  fmtName(r['driver_name'] or ''),
                    'driver_short': r['driver_short_name'] or shortName(r['driver_name'] or ''),
                    'driver_id':    r['driver_id'],
                    'trainer_name': fmtName(r['trainer_name'] or ''),
                    'trainer_short': r['trainer_short_name'] or shortName(r['trainer_name'] or ''),
                    'trainer_id':   r['trainer_id'],
                    'dq': r['dq'],
                    'gal': r['gal'],
                    'withdrawn': r['withdrawn'],
                    'pre_starts':  pre['starts'],
                    'pre_wins':    pre['wins'],
                    'post_starts': post_starts,
                    'post_wins':   post_wins,
                    'pre_galadj_starts':  pre['clean_starts'],
                    'pre_galadj_wins':    pre['clean_wins'],
                    'post_galadj_starts': post_clean_starts,
                    'post_galadj_wins':   post_clean_wins,
                })
    finally:
        conn.close()

    race_date = head.get('race_date')
    from datetime import date as _date_cls
    _today = _date_cls.today()
    is_upcoming = bool(
        race_date and hasattr(race_date, '__ge__') and race_date >= _today
        and not any(r.get('placement') is not None for r in rows)
    )
    atg_rid   = head.get('atg_race_id')
    st_rid    = head.get('st_race_id')
    let_rid   = head.get('letrot_race_id')
    contributors = head.get('contributors') or []
    # Surface every source attached to this race row so the frontend can
    # render a pill per source.
    source_pills = []
    if atg_rid:
        source_pills.append({'key': 'atg',    'source_id': atg_rid,
                             'url': _atg_race_url(atg_rid)})
    if st_rid:
        source_pills.append({'key': 'st',     'source_id': st_rid, 'url': None})
    if let_rid:
        source_pills.append({'key': 'letrot', 'source_id': let_rid, 'url': None})
    if head.get('kmtid_id'):
        source_pills.append({'key': 'kmtid',  'source_id': head.get('kmtid_id'),
                             'url': None})
    return jsonify({
        'race_date': race_date.isoformat() if hasattr(race_date, 'isoformat') else race_date,
        'track': (head.get('track_name') or '').strip().title(),
        'track_id': head.get('track_id'),
        'country': head.get('track_country'),
        'race_number': head.get('race_number'),
        'distance': head.get('distance'),
        'start_method': head.get('start_method'),
        'race_class': head.get('race_class'),
        'victory_margin': head.get('victory_margin'),
        'atg_url': _atg_race_url(atg_rid) if atg_rid else None,
        'is_upcoming': is_upcoming,
        'gal_models': gal_models if is_upcoming else None,
        'has_kmtid': head.get('kmtid_id') is not None,
        'primary_source': head.get('primary_source'),
        'contributors': contributors,
        'source_pills': source_pills,
        'results': rows,
    })


# =====================================================================
# Driver / Trainer
# =====================================================================

@app.route('/driver/<int:driver_id>')
def driver_page(driver_id):
    return render_template('driver.html', active_tab='driver', driver_id=driver_id)


@app.route('/trainer/<int:trainer_id>')
def trainer_page(trainer_id):
    return render_template('trainer.html', active_tab='trainer', trainer_id=trainer_id)


def _person_win_rates(conn, person_ids: list[int], role: str) -> dict[int, int]:
    """Return role-specific career win rate (permille) for a list of person IDs.

    role: 'driver' or 'trainer' — picks the matching id column.
    """
    if not person_ids:
        return {}
    # Served from the person_career_stats materialized view (refreshed each
    # update run). Its win/qualifier predicates are identical to _IS_WIN /
    # _NOT_QUALIFIER, so the permille values are the same as a live aggregate
    # — minus a 7.5M-row scan per page load.
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT person_id, starts, wins
            FROM person_career_stats
            WHERE role = %s AND person_id = ANY(%s)
            """,
            (role, person_ids),
        )
        out: dict[int, int] = {}
        for pid, starts, wins in cur.fetchall():
            if pid and starts:
                out[pid] = round(wins * 1000 / starts)
    return out


def _person_recent_entries(conn, person_id: int, role: str) -> list:
    """Last 50 entries for a driver or trainer (role in {'driver','trainer'})."""
    id_col = 'driver_id' if role == 'driver' else 'trainer_id'
    other_col = 'trainer_id' if role == 'driver' else 'driver_id'
    other_alias = 'trainer' if role == 'driver' else 'driver'

    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(
            f"""
            SELECT h.name AS horse_name, e.horse_id,
                   e.program_number  AS number,
                   e.placement, e.placement_text,
                   e.time_seconds    AS time_val, e.time_text,
                   e.odds, e.disqualified AS dq, e.galopp AS gal, e.withdrawn,
                   e.sulky, e.sulky_changed,
                   e.shoe_code, e.shoe_front_changed, e.shoe_back_changed,
                   e.sex, e.age,
                   e.primary_source,
                   e.source_data->'_contributors' AS contributors,
                   e.{other_col} AS other_id,
                   r.atg_race_id AS atg_id,
                   r.race_id     AS race_id,
                   r.race_date,
                   t.name        AS track,
                   t.track_id    AS track_id,
                   po.name       AS other_name,
                   po.short_name AS other_short_name
            FROM entry e
            JOIN race r        ON r.race_id    = e.race_id
            LEFT JOIN horse  h  ON h.horse_id   = e.horse_id
            LEFT JOIN person po ON po.person_id  = e.{other_col}
            LEFT JOIN track  t  ON t.track_id    = r.track_id
            WHERE e.{id_col} = %s
            ORDER BY e.race_date DESC NULLS LAST, e.entry_id DESC
            LIMIT 50
            """,
            (person_id,),
        )
        rows = cur.fetchall()

    # Batch-fetch career win rates: self in own role, others in opposite role.
    other_role = 'trainer' if role == 'driver' else 'driver'
    other_ids = list({rr['other_id'] for rr in rows if rr['other_id']})
    self_wr_map = _person_win_rates(conn, [person_id], role)
    other_wr_map = _person_win_rates(conn, other_ids, other_role)
    self_wr = self_wr_map.get(person_id)

    pids: list[int] = []
    as_ofs: list = []
    other_pids: list[int] = []
    other_asofs: list = []
    for rr in rows:
        if rr['race_date']:
            pids.append(person_id)
            as_ofs.append(rr['race_date'])
            if rr['other_id']:
                other_pids.append(rr['other_id'])
                other_asofs.append(rr['race_date'])
    form_map = _batch_person_form_multi(conn, pids, role, as_ofs) if pids else {}
    other_form_map = (
        _batch_person_form_multi(conn, other_pids, other_role, other_asofs)
        if other_pids else {}
    )

    recent = []
    for rr in rows:
        other_id = rr['other_id']
        other_wr = other_wr_map.get(other_id) if other_id else None
        # d_wr / t_wr: career win rate of driver and trainer on this entry
        d_wr = self_wr if role == 'driver' else other_wr
        t_wr = self_wr if role == 'trainer' else other_wr
        self_form = form_map.get((person_id, rr['race_date']), {})
        other_form = (
            other_form_map.get((other_id, rr['race_date']), {})
            if other_id and rr['race_date'] else {}
        )
        if role == 'driver':
            df, df_odds, df_perf = (self_form.get('form'), self_form.get('form_odds'),
                                    self_form.get('form_perf'))
            tf, tf_odds, tf_perf = (other_form.get('form'), other_form.get('form_odds'),
                                    other_form.get('form_perf'))
        else:
            tf, tf_odds, tf_perf = (self_form.get('form'), self_form.get('form_odds'),
                                    self_form.get('form_perf'))
            df, df_odds, df_perf = (other_form.get('form'), other_form.get('form_odds'),
                                    other_form.get('form_perf'))
        item = {
            'horse_name': rr['horse_name'],
            'horse_id':   rr['horse_id'],
            'number':     rr['number'],
            'placement':  rr['placement'],
            'placement_text': rr['placement_text'],
            'time_val':   float(rr['time_val']) if rr['time_val'] is not None else None,
            'time_text':  rr['time_text'],
            'odds':       float(rr['odds']) if rr['odds'] is not None else None,
            'dq':         rr['dq'],
            'gal':        rr['gal'],
            'withdrawn':  rr['withdrawn'],
            'tf': tf, 'tf_odds': tf_odds, 'tf_perf': tf_perf,
            'df': df, 'df_odds': df_odds, 'df_perf': df_perf,
            'd_wr': d_wr, 't_wr': t_wr,
            'sex':        rr['sex'],
            'age':        rr['age'],
            'sulky':      rr['sulky'],
            'sulky_changed': rr['sulky_changed'],
            'shoe_code':  rr['shoe_code'],
            'shoe_front_changed': rr['shoe_front_changed'],
            'shoe_back_changed':  rr['shoe_back_changed'],
            'race_date':  rr['race_date'].isoformat() if rr['race_date'] else None,
            'track':      (rr['track'] or '').strip().lower(),
            'track_id':   rr['track_id'],
            'atg_id':     rr['atg_id'],
            'race_id':    rr['race_id'],
            'primary_source': rr['primary_source'],
            'contributors':   rr.get('contributors') or (
                [rr['primary_source']] if rr.get('primary_source') else []
            ),
            f'{other_alias}_id':    other_id,
            f'{other_alias}_name':  fmtName(rr['other_name'] or ''),
            f'{other_alias}_short': rr['other_short_name'] or shortName(rr['other_name'] or ''),
        }
        recent.append(item)
    return recent


def _person_form_series(conn, person_id: int, id_col: str) -> list:
    """Monthly win-rate series for the chart (last 3 years).

    Returns points with keys compatible with the driver/trainer chart JS:
    median_df (or median_tf), scored, starts, median_d_wr (or median_t_wr).
    All rates in permille (×1000).
    """
    with conn.cursor() as cur:
        cur.execute(f"""
            SELECT
                DATE_TRUNC('month', e.race_date)::date            AS month,
                COUNT(*) FILTER (
                    WHERE NOT COALESCE(e.withdrawn, false)
                      AND {_NOT_QUALIFIER}
                )                                                  AS starts,
                COUNT(*) FILTER (WHERE {_IS_WIN})                 AS wins
            FROM entry e
            WHERE e.{id_col} = %s
              AND e.race_date >= CURRENT_DATE - INTERVAL '3 years'
            GROUP BY 1
            ORDER BY 1
        """, (person_id,))
        monthly = cur.fetchall()

    if not monthly:
        return []

    # Compute cumulative win rate and rolling 3-month win rate.
    cum_starts = cum_wins = 0
    result = []
    n = len(monthly)
    for i, (month, starts, wins) in enumerate(monthly):
        cum_starts += starts
        cum_wins += wins
        # Rolling 3-month window.
        window_starts = sum(monthly[j][1] for j in range(max(0, i - 2), i + 1))
        window_wins   = sum(monthly[j][2] for j in range(max(0, i - 2), i + 1))
        roll_wr = round(window_wins * 1000 / window_starts) if window_starts else None
        cum_wr  = round(cum_wins * 1000 / cum_starts)       if cum_starts  else None
        result.append({
            'month':         month.isoformat(),
            'starts':        starts,
            'scored':        wins,
            'median_df':     roll_wr,
            'median_tf':     roll_wr,
            'scored_wr':     wins,
            'median_d_wr':   cum_wr,
            'median_t_wr':   cum_wr,
            'median_df_odds': None,
            'median_tf_odds': None,
            'scored_odds':    0,
        })
    return result


@app.route('/api/driver/<int:driver_id>')
def driver_api(driver_id):
    """Driver header + monthly form-series + recent entries."""
    conn = get_db()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("SELECT person_id, name, short_name FROM person WHERE person_id = %s",
                        (driver_id,))
            head = cur.fetchone()
            if not head:
                return jsonify({'error': 'not found'}), 404
        series = _person_form_series(conn, driver_id, 'driver_id')
        recent = _person_recent_entries(conn, driver_id, 'driver')
    finally:
        conn.close()
    return jsonify({
        'driver_id': head['person_id'],
        'name':      fmtName(head['name'] or ''),
        'short':     head['short_name'],
        'series':    series,
        'recent':    recent,
    })


def _trainer_top_horses(conn, trainer_id: int, limit: int = 5) -> list:
    """Top horses by earnings *while in this trainer's barn*.

    We sum `entry.prize_kr` over entries where this trainer was listed,
    rather than the horse's lifetime earnings — that way the credit goes
    to the trainer who was actually responsible.
    """
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(
            """
            SELECT h.horse_id,
                   h.name,
                   h.gender_code,
                   to_char(h.date_of_birth, 'YYYY') AS year,
                   COUNT(*) FILTER (
                       WHERE NOT COALESCE(e.withdrawn, false)
                         AND COALESCE(e.placement_text, '') !~ %s
                   )                                    AS starts,
                   COUNT(*) FILTER (
                       WHERE e.placement_text = '1'
                         AND NOT COALESCE(e.disqualified, false)
                         AND COALESCE(e.placement_text, '') !~ %s
                   )                                    AS wins,
                   COALESCE(SUM(e.prize_kr), 0)::bigint AS earnings_kr
              FROM entry e
              JOIN horse h ON h.horse_id = e.horse_id
             WHERE e.trainer_id = %s
             GROUP BY h.horse_id, h.name, h.gender_code, h.date_of_birth
             HAVING COALESCE(SUM(e.prize_kr), 0) > 0
             ORDER BY earnings_kr DESC
             LIMIT %s
            """,
            (_QUALIFIER_RE, _QUALIFIER_RE, trainer_id, limit),
        )
        rows = []
        for r in cur.fetchall():
            rows.append({
                'horse_id':    r['horse_id'],
                'name':        r['name'],
                'gender_code': r['gender_code'],
                'year':        r['year'],
                'starts':      r['starts'] or 0,
                'wins':        r['wins']   or 0,
                'earnings':    _kr(r['earnings_kr']),
                'earnings_kr': int(r['earnings_kr']),
            })
        return rows


@app.route('/api/trainer/<int:trainer_id>')
def trainer_api(trainer_id):
    """Trainer header + monthly form-series + recent entries + top horses."""
    conn = get_db()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("SELECT person_id, name, short_name FROM person WHERE person_id = %s",
                        (trainer_id,))
            head = cur.fetchone()
            if not head:
                return jsonify({'error': 'not found'}), 404
        series      = _person_form_series(conn, trainer_id, 'trainer_id')
        recent      = _person_recent_entries(conn, trainer_id, 'trainer')
        top_horses  = _trainer_top_horses(conn, trainer_id)
    finally:
        conn.close()
    return jsonify({
        'trainer_id': head['person_id'],
        'name':       fmtName(head['name'] or ''),
        'short':      head['short_name'],
        'series':     series,
        'recent':     recent,
        'top_horses': top_horses,
    })


# =====================================================================
# ML  —  models registry (read-only)
# =====================================================================
# The manual data-explorer / training UI was removed — models are trained
# programmatically via the CLI scripts (scripts/train_model.py,
# scripts/eval_trainer_form.py, …) and surface here once registered in
# ml_model. The models page groups them into tabs (xgal / trainer form /
# upcoming) client-side off /api/ml/models.


@app.route('/ml')
def ml_page():
    return redirect(url_for('ml_models_page'), code=302)


@app.route('/ml/models')
def ml_models_page():
    return render_template('ml_models.html', active_tab='ml_models')


@app.route('/ml/models/<int:model_id>')
def ml_model_detail_page(model_id):
    return render_template('ml_model_detail.html', active_tab='ml_models', model_id=model_id)


@app.route('/api/ml/models')
def api_ml_models():
    """List registered models with metadata + evaluation metrics."""
    conn = get_db()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT model_id, name, scope, track_id, track_name, target,
                       algo, slice_def, metrics, created_at
                FROM ml_model
                ORDER BY created_at DESC, model_id DESC
            """)
            cols = [d[0] for d in cur.description]
            rows = [dict(zip(cols, r)) for r in cur.fetchall()]
        for r in rows:
            r['created_at'] = r['created_at'].isoformat() if r['created_at'] else None
        general = [r for r in rows if r['scope'] != 'track']
        track = [r for r in rows if r['scope'] == 'track']
        return jsonify({'general': general, 'track': track, 'total': len(rows)})
    finally:
        conn.close()


@app.route('/api/ml/models/<int:model_id>')
def api_ml_model_detail(model_id):
    """Single model details."""
    conn = get_db()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT model_id, name, scope, track_id, track_name, target,
                       algo, slice_def, metrics, artifact_path, created_at
                FROM ml_model WHERE model_id = %s
            """, (model_id,))
            row = cur.fetchone()
        if not row:
            return jsonify({'error': 'not found'}), 404
        cols = [d[0] for d in cur.description]
        m = dict(zip(cols, row))
        m['created_at'] = m['created_at'].isoformat() if m['created_at'] else None
        return jsonify(m)
    finally:
        conn.close()


# =====================================================================
# Admin
# =====================================================================

@app.route('/admin')
def admin_page():
    return redirect(url_for('admin_design_page'), code=302)


@app.route('/admin/update')
def admin_update_page():
    return render_template('admin.html', active_tab='update')


@app.route('/admin/matching')
def admin_matching_page():
    return render_template('matching.html', active_tab='matching')


@app.route('/admin/design')
def admin_design_page():
    return render_template('design.html', active_tab='design')




# ---------------------------------------------------------------------------
# Matching API — health dashboard, browse, dry-run / execute merge,
# script runner, rollback. Backed by scripts/audit_matching.py for the
# heavy queries.
# ---------------------------------------------------------------------------

# In-process cache for the health snapshot (60s TTL). The audit queries
# are expensive on the 4M-row entry table; we cache the JSON result so
# the dashboard reload doesn't re-pay each time.
_MATCHING_HEALTH_CACHE: dict = {"ts": 0.0, "data": None}
_MATCHING_HEALTH_TTL = 60.0


@app.route('/api/admin/matching/health')
def admin_matching_health():
    """Return per-category audit counts + recent merge activity."""
    import time as _time
    force = request.args.get('refresh') == '1'
    now = _time.time()
    if (not force and _MATCHING_HEALTH_CACHE["data"]
            and now - _MATCHING_HEALTH_CACHE["ts"] < _MATCHING_HEALTH_TTL):
        return jsonify(_MATCHING_HEALTH_CACHE["data"])

    from scripts.audit_matching import collect_health
    conn = get_db()
    try:
        data = collect_health(conn)
    finally:
        conn.close()
    _MATCHING_HEALTH_CACHE["ts"] = now
    _MATCHING_HEALTH_CACHE["data"] = data
    return jsonify(data)


@app.route('/api/admin/matching/browse/<category>')
def admin_matching_browse(category: str):
    """Top 50 affected horses for the given category card."""
    from scripts.audit_matching import browse_category
    conn = get_db()
    try:
        rows = browse_category(conn, category, limit=int(request.args.get('limit', 50)))
    finally:
        conn.close()
    return jsonify({'category': category, 'rows': rows})


@app.route('/api/admin/matching/preview-merge')
def admin_matching_preview_merge():
    """Dry-run merge preview."""
    from core.identity import merge_horses
    try:
        frm = int(request.args.get('from'))
        to  = int(request.args.get('to'))
    except (TypeError, ValueError):
        return jsonify({'error': 'from + to are required ints'}), 400
    conn = get_db()
    try:
        with conn.cursor() as cur:
            res = merge_horses(cur, frm, to,
                               reason='admin preview', method='manual',
                               merged_by='admin', dry_run=True)
        return jsonify(res)
    finally:
        conn.close()


@app.route('/api/admin/matching/merge', methods=['POST'])
def admin_matching_execute_merge():
    """Execute one horse merge. Body: { from_horse_id, to_horse_id, reason, method? }."""
    from core.identity import merge_horses
    payload = request.get_json(silent=True) or {}
    try:
        frm = int(payload['from_horse_id'])
        to  = int(payload['to_horse_id'])
    except (KeyError, TypeError, ValueError):
        return jsonify({'error': 'missing from/to'}), 400
    reason = (payload.get('reason') or 'manual merge').strip()
    method = (payload.get('method') or 'manual').strip()
    conn = get_db()
    try:
        with conn.cursor() as cur:
            res = merge_horses(cur, frm, to,
                               reason=reason, method=method,
                               merged_by='admin', dry_run=False)
        conn.commit()
        _MATCHING_HEALTH_CACHE["data"] = None
        return jsonify(res)
    except Exception as exc:
        conn.rollback()
        return jsonify({'error': str(exc)}), 500
    finally:
        conn.close()


@app.route('/api/admin/matching/rollback/<int:merge_id>', methods=['POST'])
def admin_matching_rollback(merge_id: int):
    from core.identity import rollback_horse_merge
    conn = get_db()
    try:
        with conn.cursor() as cur:
            res = rollback_horse_merge(cur, merge_id)
        if 'error' in res:
            conn.rollback()
            return jsonify(res), 400
        conn.commit()
        _MATCHING_HEALTH_CACHE["data"] = None
        return jsonify(res)
    finally:
        conn.close()


@app.route('/api/admin/matching/horse/<int:horse_id>/candidates')
def admin_matching_candidates(horse_id: int):
    """Suggest merge-target horses for a given horse_id.

    Heuristics, ranked weakest to strongest:
      - Same name (normalised) — always shown, never auto-merge candidate.
      - Same name + same birth_country — better.
      - Same name + same birth_year + same sire_name + same dam_name — strong (pedigree).
    """
    conn = get_db()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM horse WHERE horse_id = %s", (horse_id,))
            row = cur.fetchone()
            if not row:
                return jsonify({'error': 'horse not found'}), 404
            cols = [d.name for d in cur.description]
            base = dict(zip(cols, row))

            cur.execute(
                """
                SELECT h.horse_id, h.name, h.birth_country, h.date_of_birth,
                       h.sire_name, h.dam_name,
                       h.st_id, h.atg_id, h.letrot_id, h.hvt_id, h.usta_id,
                       h.primary_source,
                       (SELECT COUNT(*) FROM entry e WHERE e.horse_id = h.horse_id) AS entries,
                       CASE
                         WHEN EXTRACT(year FROM h.date_of_birth)
                            = EXTRACT(year FROM %s::date)
                          AND upper(coalesce(h.sire_name,'')) = upper(coalesce(%s,''))
                          AND upper(coalesce(h.dam_name,'')) = upper(coalesce(%s,''))
                          AND coalesce(h.sire_name, '') <> ''
                          AND coalesce(h.dam_name, '') <> ''
                         THEN 'pedigree'
                         WHEN h.birth_country = %s THEN 'country'
                         ELSE 'name'
                       END AS match_kind
                  FROM horse h
                 WHERE h.horse_id <> %s
                   AND v2_normalize_name(h.name) = v2_normalize_name(%s)
                   AND v2_normalize_name(h.name) <> ''
                 ORDER BY match_kind, entries DESC
                 LIMIT 30
                """,
                (base.get('date_of_birth'), base.get('sire_name'), base.get('dam_name'),
                 base.get('birth_country'), horse_id, base.get('name')),
            )
            rows = []
            for r in cur.fetchall():
                rows.append({
                    'horse_id':       r[0],
                    'name':           r[1],
                    'birth_country':  r[2],
                    'date_of_birth':  r[3].isoformat() if r[3] else None,
                    'sire_name':      r[4],
                    'dam_name':       r[5],
                    'st_id':          r[6],
                    'atg_id':         r[7],
                    'letrot_id':      r[8],
                    'hvt_id':         r[9],
                    'usta_id':        r[10],
                    'primary_source': r[11],
                    'entries':        int(r[12] or 0),
                    'match_kind':     r[12 + 1],
                })
            return jsonify({
                'base': {
                    'horse_id':       base['horse_id'],
                    'name':           base.get('name'),
                    'birth_country':  base.get('birth_country'),
                    'date_of_birth':  base['date_of_birth'].isoformat() if base.get('date_of_birth') else None,
                    'sire_name':      base.get('sire_name'),
                    'dam_name':       base.get('dam_name'),
                    'st_id':          base.get('st_id'),
                    'atg_id':         base.get('atg_id'),
                    'letrot_id':      base.get('letrot_id'),
                    'hvt_id':         base.get('hvt_id'),
                    'usta_id':        base.get('usta_id'),
                    'primary_source': base.get('primary_source'),
                },
                'candidates': rows,
            })
    finally:
        conn.close()


# Scripts that the admin runner is allowed to spawn.
_MATCHING_SCRIPTS = {
    'audit':              'scripts/audit_matching.py',
    'pedigree':           'scripts/merge_pedigree_duplicates.py',
    'synth_pairs':        'scripts/merge_synth_pairs.py',
    'split_polluted':     'scripts/split_polluted_atg_ids.py',
    'same_row':           'scripts/clean_same_row_atg_ids.py',
    'merge_races':        'scripts/merge_duplicate_races.py',
    'person_pairs':       'scripts/merge_synth_pairs_persons.py',
}


@app.route('/api/admin/matching/script/<name>', methods=['POST'])
def admin_matching_run_script(name: str):
    """Spawn one of the Act 2 cleanup scripts (dry-run by default).

    Body: { execute: bool }. With execute=true, passes --execute to the
    script; otherwise dry-run. Reuses the job_run infrastructure for
    progress polling.
    """
    if name not in _MATCHING_SCRIPTS:
        return jsonify({'error': f'unknown script: {name}'}), 400
    payload = request.get_json(silent=True) or {}
    execute = bool(payload.get('execute'))
    job_name = f"merge_{name}{'' if execute else '_dryrun'}"

    conn = get_db()
    try:
        _reap_zombie_runs(conn)
        with conn.cursor() as cur:
            cur.execute(
                "SELECT job_run_id FROM job_run "
                " WHERE job_name = %s AND status = 'running' "
                " ORDER BY job_run_id DESC LIMIT 1",
                (job_name,),
            )
            row = cur.fetchone()
            if row:
                return jsonify({'job_run_id': row[0], 'already_running': True})

            cur.execute(
                "INSERT INTO job_run (job_name, status, log) "
                "VALUES (%s, 'running', %s) RETURNING job_run_id",
                (job_name, f'[admin] queued {name} (execute={execute})...\n'),
            )
            rid = cur.fetchone()[0]
        conn.commit()

        script = _MATCHING_SCRIPTS[name]
        cmd = [sys.executable, str(_STABLE_V2_ROOT / script),
               '--job-run-id', str(rid)]
        if execute:
            cmd.append('--execute')

        env = os.environ.copy()
        env['PYTHONUNBUFFERED'] = '1'

        log_dir = _STABLE_V2_ROOT / 'logs'
        log_dir.mkdir(exist_ok=True)
        log_file = log_dir / f'{job_name}_{rid}.log'
        try:
            with open(log_file, 'wb') as f:
                proc = subprocess.Popen(
                    cmd, cwd=str(_STABLE_V2_ROOT),
                    stdout=f, stderr=subprocess.STDOUT,
                    env=env, close_fds=True,
                )
        except Exception as exc:
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE job_run SET status='failed', finished_at=NOW(), "
                    "log = COALESCE(log,'') || %s WHERE job_run_id=%s",
                    (f'[admin] failed to spawn subprocess: {exc}\n', rid),
                )
            conn.commit()
            return jsonify({'job_run_id': rid, 'error': str(exc)}), 500

        with conn.cursor() as cur:
            cur.execute("UPDATE job_run SET pid = %s WHERE job_run_id = %s",
                        (proc.pid, rid))
        conn.commit()
    finally:
        conn.close()

    return jsonify({'job_run_id': rid})


# ---------------------------------------------------------------------------
# Unified entity search — drives the drag-and-drop search panel on
# /admin/matching. Returns up to `limit` hits across horses + persons,
# matching by id OR by name (ILIKE / trigram). Tracks are out of scope
# (they have their own dedupe script).
#
# Response shape:
#   {
#     "query":   "<original query>",
#     "results": [
#       { "entity": "horse",  "id": 123, "name": "...", "subtitle": "...",
#         "ids": {"st_id": 5, "atg_id": "x:FR:..."}, "primary_source": "atg" },
#       { "entity": "person", "id": 234, "name": "...", "subtitle": "driver / trainer",
#         "ids": {"st_id": 9}, "primary_source": "st" },
#       ...
#     ]
#   }
# ---------------------------------------------------------------------------


def _entity_search_horse(cur, q: str, limit: int, by_id: int | None) -> list[dict]:
    if by_id is not None:
        cur.execute(
            """
            SELECT h.horse_id, h.name, h.birth_country, h.date_of_birth,
                   h.st_id, h.atg_id, h.letrot_id, h.hvt_id, h.usta_id,
                   h.kmtid_id, h.breedly_id, h.primary_source,
                   (SELECT COUNT(*) FROM entry e WHERE e.horse_id = h.horse_id) AS entries
              FROM horse h WHERE h.horse_id = %s
            """,
            (by_id,),
        )
    else:
        cur.execute(
            """
            SELECT h.horse_id, h.name, h.birth_country, h.date_of_birth,
                   h.st_id, h.atg_id, h.letrot_id, h.hvt_id, h.usta_id,
                   h.kmtid_id, h.breedly_id, h.primary_source,
                   (SELECT COUNT(*) FROM entry e WHERE e.horse_id = h.horse_id) AS entries
              FROM horse h
             WHERE h.name ILIKE %s
             ORDER BY (SELECT COUNT(*) FROM entry e WHERE e.horse_id = h.horse_id) DESC,
                      h.name
             LIMIT %s
            """,
            (f'%{q}%', limit),
        )
    out: list[dict] = []
    for r in cur.fetchall():
        ids: dict = {}
        for col, val in [('st_id', r[4]), ('atg_id', r[5]), ('letrot_id', r[6]),
                         ('hvt_id', r[7]), ('usta_id', r[8]),
                         ('kmtid_id', r[9]), ('breedly_id', r[10])]:
            if val is not None:
                ids[col] = val
        dob = r[3].isoformat() if r[3] else ''
        country = r[2] or ''
        out.append({
            'entity':         'horse',
            'id':             r[0],
            'name':           r[1] or '',
            'subtitle':       ' · '.join(s for s in [country, dob, f'{int(r[12] or 0)} entries'] if s),
            'ids':            ids,
            'primary_source': r[11],
        })
    return out


def _entity_search_person(cur, q: str, limit: int, by_id: int | None) -> list[dict]:
    if by_id is not None:
        cur.execute(
            """
            SELECT p.person_id, p.name, p.short_name, p.license_country,
                   p.is_driver, p.is_trainer, p.is_owner, p.is_breeder,
                   p.st_id, p.atg_id, p.letrot_id, p.hvt_id, p.usta_id,
                   p.primary_source,
                   (SELECT COUNT(*) FROM entry e
                     WHERE e.driver_id = p.person_id OR e.trainer_id = p.person_id) AS entries
              FROM person p WHERE p.person_id = %s
            """,
            (by_id,),
        )
    else:
        # name OR short_name ILIKE
        cur.execute(
            """
            SELECT p.person_id, p.name, p.short_name, p.license_country,
                   p.is_driver, p.is_trainer, p.is_owner, p.is_breeder,
                   p.st_id, p.atg_id, p.letrot_id, p.hvt_id, p.usta_id,
                   p.primary_source,
                   (SELECT COUNT(*) FROM entry e
                     WHERE e.driver_id = p.person_id OR e.trainer_id = p.person_id) AS entries
              FROM person p
             WHERE p.name       ILIKE %s
                OR p.short_name ILIKE %s
             ORDER BY (SELECT COUNT(*) FROM entry e
                        WHERE e.driver_id = p.person_id OR e.trainer_id = p.person_id) DESC,
                      p.name
             LIMIT %s
            """,
            (f'%{q}%', f'%{q}%', limit),
        )
    out: list[dict] = []
    for r in cur.fetchall():
        roles = []
        if r[4]: roles.append('driver')
        if r[5]: roles.append('trainer')
        if r[6]: roles.append('owner')
        if r[7]: roles.append('breeder')
        ids: dict = {}
        for col, val in [('st_id', r[8]), ('atg_id', r[9]), ('letrot_id', r[10]),
                         ('hvt_id', r[11]), ('usta_id', r[12])]:
            if val is not None:
                ids[col] = val
        country = r[3] or ''
        out.append({
            'entity':         'person',
            'id':             r[0],
            'name':           r[1] or r[2] or '',
            'subtitle':       ' · '.join(s for s in [
                                 '/'.join(roles) or 'no role',
                                 country,
                                 f'{int(r[14] or 0)} entries',
                             ] if s),
            'ids':            ids,
            'primary_source': r[13],
        })
    return out


@app.route('/api/admin/matching/search')
def admin_matching_search():
    q = (request.args.get('q') or '').strip()
    if len(q) < 2:
        return jsonify({'query': q, 'results': []})
    etype = (request.args.get('type') or 'all').lower()
    try:
        limit = max(1, min(int(request.args.get('limit', 30)), 100))
    except ValueError:
        limit = 30

    # If the query is purely numeric, treat as id lookup as well.
    by_id = int(q) if q.isdigit() else None

    conn = get_db()
    try:
        with conn.cursor() as cur:
            results: list[dict] = []
            if etype in ('horse', 'all'):
                results.extend(_entity_search_horse(cur, q, limit, by_id))
            if etype in ('person', 'all'):
                results.extend(_entity_search_person(cur, q, limit, by_id))
        return jsonify({'query': q, 'type': etype, 'results': results})
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Person merge endpoints — mirror the horse ones but use core.identity's
# merge_persons / rollback_person_merge.
# ---------------------------------------------------------------------------

@app.route('/api/admin/matching/person/<int:person_id>/candidates')
def admin_matching_person_candidates(person_id: int):
    """Return basic person info + same-name candidate persons.

    Persons don't have pedigree, so candidates are scored as:
      - same name + same license_country  → 'country'
      - same name only                    → 'name'
    """
    conn = get_db()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM person WHERE person_id = %s", (person_id,))
            row = cur.fetchone()
            if not row:
                return jsonify({'error': 'person not found'}), 404
            cols = [d.name for d in cur.description]
            base = dict(zip(cols, row))

            cur.execute(
                """
                SELECT p.person_id, p.name, p.short_name, p.license_country,
                       p.is_driver, p.is_trainer, p.is_owner, p.is_breeder,
                       p.st_id, p.atg_id, p.letrot_id, p.hvt_id, p.usta_id,
                       p.primary_source,
                       (SELECT COUNT(*) FROM entry e
                         WHERE e.driver_id = p.person_id OR e.trainer_id = p.person_id) AS entries,
                       CASE
                         WHEN p.license_country = %s AND p.license_country IS NOT NULL
                           THEN 'country'
                         ELSE 'name'
                       END AS match_kind
                  FROM person p
                 WHERE p.person_id <> %s
                   AND v2_normalize_name(p.name) <> ''
                   AND ( v2_normalize_name(coalesce(p.name,''))       = v2_normalize_name(coalesce(%s,''))
                      OR v2_normalize_name(coalesce(p.short_name,'')) = v2_normalize_name(coalesce(%s,'')) )
                 ORDER BY match_kind, entries DESC
                 LIMIT 30
                """,
                (base.get('license_country'), person_id,
                 base.get('name'), base.get('short_name')),
            )
            rows: list[dict] = []
            for r in cur.fetchall():
                roles = []
                if r[4]: roles.append('driver')
                if r[5]: roles.append('trainer')
                if r[6]: roles.append('owner')
                if r[7]: roles.append('breeder')
                rows.append({
                    'person_id':       r[0],
                    'name':            r[1],
                    'short_name':      r[2],
                    'license_country': r[3],
                    'roles':           roles,
                    'st_id':           r[8],
                    'atg_id':          r[9],
                    'letrot_id':       r[10],
                    'hvt_id':          r[11],
                    'usta_id':         r[12],
                    'primary_source':  r[13],
                    'entries':         int(r[14] or 0),
                    'match_kind':      r[15],
                })
            return jsonify({
                'base': {
                    'person_id':       base['person_id'],
                    'name':            base.get('name'),
                    'short_name':      base.get('short_name'),
                    'license_country': base.get('license_country'),
                    'roles': [k for k in ('driver','trainer','owner','breeder')
                              if base.get(f'is_{k}')],
                    'st_id':           base.get('st_id'),
                    'atg_id':          base.get('atg_id'),
                    'letrot_id':       base.get('letrot_id'),
                    'hvt_id':          base.get('hvt_id'),
                    'usta_id':         base.get('usta_id'),
                    'primary_source':  base.get('primary_source'),
                },
                'candidates': rows,
            })
    finally:
        conn.close()


@app.route('/api/admin/matching/preview-merge-person')
def admin_matching_preview_merge_person():
    from core.identity import merge_persons
    try:
        frm = int(request.args.get('from'))
        to  = int(request.args.get('to'))
    except (TypeError, ValueError):
        return jsonify({'error': 'from + to are required ints'}), 400
    conn = get_db()
    try:
        with conn.cursor() as cur:
            res = merge_persons(cur, frm, to,
                                reason='admin preview', method='manual',
                                merged_by='admin', dry_run=True)
        return jsonify(res)
    finally:
        conn.close()


@app.route('/api/admin/matching/merge-person', methods=['POST'])
def admin_matching_execute_merge_person():
    """Body: { from_person_id, to_person_id, reason, method? }."""
    from core.identity import merge_persons
    payload = request.get_json(silent=True) or {}
    try:
        frm = int(payload['from_person_id'])
        to  = int(payload['to_person_id'])
    except (KeyError, TypeError, ValueError):
        return jsonify({'error': 'missing from/to'}), 400
    reason = (payload.get('reason') or 'manual merge').strip()
    method = (payload.get('method') or 'manual').strip()
    conn = get_db()
    try:
        with conn.cursor() as cur:
            res = merge_persons(cur, frm, to,
                                reason=reason, method=method,
                                merged_by='admin', dry_run=False)
        conn.commit()
        _MATCHING_HEALTH_CACHE["data"] = None
        return jsonify(res)
    except Exception as exc:
        conn.rollback()
        return jsonify({'error': str(exc)}), 500
    finally:
        conn.close()


@app.route('/api/admin/matching/rollback-person/<int:merge_id>', methods=['POST'])
def admin_matching_rollback_person(merge_id: int):
    from core.identity import rollback_person_merge
    conn = get_db()
    try:
        with conn.cursor() as cur:
            res = rollback_person_merge(cur, merge_id)
        if 'error' in res:
            conn.rollback()
            return jsonify(res), 400
        conn.commit()
        _MATCHING_HEALTH_CACHE["data"] = None
        return jsonify(res)
    finally:
        conn.close()




def _pid_alive(pid: int | None) -> bool:
    if not pid or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    # os.kill(0) succeeds for zombie (defunct) processes — detect those
    # via waitpid(WNOHANG) which reaps them and returns the pid.
    try:
        wpid, _ = os.waitpid(pid, os.WNOHANG)
        if wpid != 0:
            return False  # was a zombie, now reaped
    except ChildProcessError:
        pass  # not our child — can't waitpid, but kill(0) said alive
    return True


def _reap_zombie_runs(conn) -> list[int]:
    reaped: list[int] = []
    with conn.cursor() as cur:
        cur.execute(
            "SELECT job_run_id, pid, started_at FROM job_run "
            "WHERE status = 'running'"
        )
        rows = cur.fetchall()
        for rid, pid, started in rows:
            if pid is None:
                cur.execute("SELECT (NOW() - %s) > INTERVAL '60 seconds'", (started,))
                stale = cur.fetchone()[0]
                if not stale:
                    continue
            elif _pid_alive(pid):
                continue
            cur.execute(
                "UPDATE job_run SET status = 'failed', "
                "finished_at = NOW(), "
                "log = COALESCE(log,'') || %s "
                "WHERE job_run_id = %s",
                (f'[admin] reaped zombie row (pid {pid} not alive)\n', rid),
            )
            reaped.append(rid)
    if reaped:
        conn.commit()
    return reaped


@app.route('/api/admin/run', methods=['POST'])
def admin_run():
    """Spawn `python -m jobs.update --job-run-id <id>`.

    `jobs.update` is not implemented in v2 yet — if it can't be spawned,
    the run row gets marked failed and the API still returns the id.
    """
    payload = request.get_json(silent=True) or {}
    mode = payload.get('mode', 'atg')
    if mode not in ('atg', 'st', 'kmtid', 'letrot', 'cleanup', 'all'):
        return jsonify({'error': 'invalid mode'}), 400
    job_name = {
        'atg':     'update',
        'st':      'update_st',
        'kmtid':   'update_kmtid',
        'letrot':  'update_letrot',
        'cleanup': 'update_cleanup',
        'all':     'update_all',
    }[mode]

    conn = get_db()
    try:
        _reap_zombie_runs(conn)
        with conn.cursor() as cur:
            cur.execute(
                "SELECT job_run_id FROM job_run "
                "WHERE job_name = %s AND status = 'running' "
                "ORDER BY job_run_id DESC LIMIT 1",
                (job_name,),
            )
            row = cur.fetchone()
            if row:
                return jsonify({'job_run_id': row[0], 'already_running': True})

            cur.execute(
                "INSERT INTO job_run (job_name, status, log) "
                "VALUES (%s, 'running', %s) RETURNING job_run_id",
                (job_name, f'[admin] queued ({mode}), spawning subprocess...\n'),
            )
            rid = cur.fetchone()[0]
        conn.commit()

        cmd = [
            sys.executable, '-m', 'jobs.update',
            '--job-run-id', str(rid),
            '--mode', mode,
        ]
        env = os.environ.copy()
        env['PYTHONUNBUFFERED'] = '1'

        log_dir = _STABLE_V2_ROOT / 'logs'
        log_dir.mkdir(exist_ok=True)
        log_file = log_dir / f'{job_name}_{rid}.log'
        try:
            with open(log_file, 'wb') as f:
                proc = subprocess.Popen(
                    cmd,
                    cwd=str(_STABLE_V2_ROOT),
                    stdout=f, stderr=subprocess.STDOUT,
                    env=env, close_fds=True,
                )
        except Exception as exc:
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE job_run SET status='failed', finished_at=NOW(), "
                    "log = COALESCE(log,'') || %s WHERE job_run_id=%s",
                    (f'[admin] failed to spawn subprocess: {exc}\n', rid),
                )
            conn.commit()
            return jsonify({'job_run_id': rid, 'error': str(exc)}), 500

        with conn.cursor() as cur:
            cur.execute("UPDATE job_run SET pid = %s WHERE job_run_id = %s",
                        (proc.pid, rid))
        conn.commit()
    finally:
        conn.close()

    return jsonify({'job_run_id': rid})


@app.route('/api/admin/cancel', methods=['POST'])
def admin_cancel():
    payload = request.get_json(silent=True) or {}
    rid = payload.get('job_run_id')
    if not rid:
        return jsonify({'error': 'missing job_run_id'}), 400

    conn = get_db()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT status, pid FROM job_run WHERE job_run_id = %s", (rid,))
            row = cur.fetchone()
            if not row:
                return jsonify({'error': 'not found'}), 404
            status, pid = row
            if status != 'running':
                return jsonify({'job_run_id': rid, 'status': status,
                                'cancelled': False, 'note': 'not running'})

            killed = False
            if _pid_alive(pid):
                try:
                    os.kill(pid, signal.SIGTERM)
                    killed = True
                except OSError:
                    pass

            cur.execute(
                "UPDATE job_run SET status = 'failed', "
                "finished_at = NOW(), "
                "log = COALESCE(log,'') || %s "
                "WHERE job_run_id = %s",
                (f'[admin] cancelled by user (sigterm sent={killed})\n', rid),
            )
        conn.commit()
    finally:
        conn.close()
    return jsonify({'job_run_id': rid, 'cancelled': True})


_SOURCE_LABELS = {
    'atg':     'ATG',
    'letrot':  'LeTrot',
    'kmtid':   'XLabs',
    'hvt':     'HVT',
    'breedly': 'Breedly',
}


@app.route('/api/admin/sources')
def admin_sources():
    """Per-source freshness summary for the admin update page.

    Returns label, last_date (most recent race we have for this source),
    coverage count, and `runnable` (False = button is dimmed for now).
    """
    out = []
    conn = get_db()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT MAX(r.race_date), COUNT(*)
                  FROM race r
                 WHERE r.primary_source = 'st' OR r.atg_race_id IS NOT NULL
            """)
            d, n = cur.fetchone()
            out.append({'id': 'atg', 'label': _SOURCE_LABELS['atg'],
                        'last_date': d.isoformat() if d else None,
                        'count': int(n or 0), 'runnable': True})

            cur.execute("""
                SELECT MAX(r.race_date), COUNT(*)
                  FROM race r
                 WHERE r.letrot_race_id IS NOT NULL
            """)
            d, n = cur.fetchone()
            out.append({'id': 'letrot', 'label': _SOURCE_LABELS['letrot'],
                        'last_date': d.isoformat() if d else None,
                        'count': int(n or 0), 'runnable': True})

            cur.execute("""
                SELECT MAX(r.race_date), COUNT(DISTINCT e.race_id)
                  FROM entry e JOIN race r ON r.race_id = e.race_id
                 WHERE e.kmtid_actual_km_time_ms IS NOT NULL
            """)
            d, n = cur.fetchone()
            out.append({'id': 'kmtid', 'label': _SOURCE_LABELS['kmtid'],
                        'last_date': d.isoformat() if d else None,
                        'count': int(n or 0), 'runnable': True})

            cur.execute("""
                SELECT MAX(r.race_date), COUNT(*)
                  FROM race r
                 WHERE r.hvt_race_id IS NOT NULL
            """)
            d, n = cur.fetchone()
            out.append({'id': 'hvt', 'label': _SOURCE_LABELS['hvt'],
                        'last_date': d.isoformat() if d else None,
                        'count': int(n or 0), 'runnable': False})

            cur.execute("""
                SELECT MAX(last_updated_at)::date, COUNT(*)
                  FROM horse
                 WHERE breedly_id IS NOT NULL
            """)
            d, n = cur.fetchone()
            out.append({'id': 'breedly', 'label': _SOURCE_LABELS['breedly'],
                        'last_date': d.isoformat() if d else None,
                        'count': int(n or 0), 'runnable': False})
    finally:
        conn.close()
    return jsonify(out)


@app.route('/api/admin/status')
def admin_status():
    rid = request.args.get('id', type=int)
    conn = get_db()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            if rid:
                cur.execute("SELECT * FROM job_run WHERE job_run_id = %s", (rid,))
                row = cur.fetchone()
                if not row:
                    return jsonify({'error': 'not found'}), 404
                return jsonify(_serialize_run(row))
            cur.execute("SELECT * FROM job_run ORDER BY job_run_id DESC LIMIT 10")
            return jsonify([_serialize_run(r) for r in cur.fetchall()])
    finally:
        conn.close()


def _serialize_run(row) -> dict:
    return {
        'job_run_id':  row['job_run_id'],
        'job_name':    row['job_name'],
        'started_at':  row['started_at'].isoformat()  if row['started_at']  else None,
        'finished_at': row['finished_at'].isoformat() if row['finished_at'] else None,
        'status':      row['status'],
        'phase':       row.get('phase') or '',
        'summary':     row['summary'] or {},
        'log':         row['log'] or '',
    }


if __name__ == '__main__':
    # debug=False mirrors v1 — the admin page polls the DB so we don't need
    # autoreload, and avoiding the reloader prevents log writes from re-spawning
    # the parent process.
    app.run(debug=False, port=WEB_PORT, threaded=True)
