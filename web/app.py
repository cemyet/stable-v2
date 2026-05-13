#!/usr/bin/env python3
"""
Stable v2 -- horse data viewer for the flat 5-table v2 schema.

Run with:  python3 -m web.app  (from /Users/jakob/Dev/stable-v2/)

Routes mirror v1 except:
  * `/ml*`                -> single "coming soon" placeholder.
  * `/api/ml/*`           -> removed.
  * `/api/breed/*`        -> removed.
  * `/horse/<src>/<id>`   -> per-source redirect to canonical /horse/<id>.
  * `/race/st/<id>` and `/race/atg/<id>` redirect helpers.

All SQL is rewritten against the canonical 5-table schema in core.schema:
  horse, person, race, entry, track + horse_owner_history, horse_trainer_history.
ML feature columns (tf, df, tf_odds, df_odds, t_wr, d_wr, ...) from v1 are
gone in v2; the templates that referenced them render null cells, and the
trainer/driver form charts simply show "no data".
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

import psycopg2
import psycopg2.extras
from flask import Flask, jsonify, redirect, render_template, request, url_for

from core.config import WEB_PORT
from core.db import get_connection

app = Flask(__name__)
app.config['TEMPLATES_AUTO_RELOAD'] = True
app.jinja_env.auto_reload = True


_GENDER_TEXT = {'H': 'stallion', 'V': 'gelding', 'S': 'mare'}
_BREED_TEXT = {'V': 'varmblodig travare', 'K': 'kallblodig travare'}


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


_NOT_QUALIFIER = "COALESCE(e.placement_text,'') !~* '^(gdk|ejg|ejp)'"


@app.route('/api/home/form-leaders')
def home_form_leaders():
    """Simplified rolling 30-day win-rate leaderboard.

    v1 had tf/df/tf_odds/df_odds (odds-weighted form columns) — those are
    gone in v2. We compute career and 30-day win% on the fly. Out-of-form
    lists are returned empty ("coming soon" — needs more data design).
    """
    def fetch():
        conn = get_db()
        try:
            results: dict[str, list] = {}
            for role, id_col in (('trainer', 'trainer_id'),
                                 ('driver',  'driver_id')):
                with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                    cur.execute(f"""
                        WITH
                        career AS (
                            SELECT e.{id_col}                       AS pid,
                                   COUNT(*) FILTER (
                                       WHERE NOT e.withdrawn
                                         AND {_NOT_QUALIFIER}
                                   )                                AS starts,
                                   COUNT(*) FILTER (
                                       WHERE e.placement = 1
                                         AND {_NOT_QUALIFIER}
                                   )                                AS wins
                            FROM entry e
                            WHERE e.{id_col} IS NOT NULL
                            GROUP BY e.{id_col}
                        ),
                        recent AS (
                            SELECT e.{id_col}                       AS pid,
                                   COUNT(*) FILTER (
                                       WHERE NOT e.withdrawn
                                         AND {_NOT_QUALIFIER}
                                   )                                AS starts,
                                   COUNT(*) FILTER (
                                       WHERE e.placement = 1
                                         AND {_NOT_QUALIFIER}
                                   )                                AS wins
                            FROM entry e
                            JOIN race  r ON r.race_id = e.race_id
                            WHERE e.{id_col} IS NOT NULL
                              AND r.race_date >= CURRENT_DATE - INTERVAL '30 days'
                            GROUP BY e.{id_col}
                        )
                        SELECT r.pid,
                               p.name,
                               p.short_name,
                               r.starts        AS recent_starts,
                               r.wins          AS recent_wins,
                               c.starts        AS career_starts,
                               c.wins          AS career_wins
                        FROM recent r
                        LEFT JOIN career c ON c.pid = r.pid
                        JOIN person p ON p.person_id = r.pid
                        WHERE r.starts >= 10
                        ORDER BY r.wins::numeric / NULLIF(r.starts, 0) DESC,
                                 r.wins DESC
                        LIMIT 10
                    """)
                    rows = cur.fetchall()

                def fmt(rows):
                    out = []
                    for r in rows:
                        rs = r['recent_starts'] or 0
                        rw = r['recent_wins'] or 0
                        cs = r['career_starts'] or 0
                        cw = r['career_wins'] or 0
                        # Templates display rate via fmtWr(v/10) — pass permille (×10).
                        form = round(rw * 1000 / rs) if rs else 0
                        wr   = round(cw * 1000 / cs) if cs else 0
                        out.append({
                            'id':        r['pid'],
                            'name':      fmtName(r['name'] or ''),
                            'short':     r['short_name'],
                            'form':      form,
                            'wr':        wr,
                            'form_odds': None,
                            'delta':     form - wr,
                        })
                    return out

                results[f'{role}_hot']  = fmt(rows)
                results[f'{role}_cold'] = []
        finally:
            conn.close()
        return results

    return jsonify(_cached('form-leaders', fetch))


@app.route('/api/home/top-horses')
def home_top_horses():
    """Top horses by lifetime earnings / wins / win rate. Uses horse_career_stats."""
    sort = request.args.get('sort', 'earnings')
    period = request.args.get('period', 'ytd')
    cache_key = f'horses:{sort}:{period}'

    def fetch():
        conn = get_db()
        try:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                if period == 'all':
                    cur.execute(f"""
                        SELECT h.horse_id,
                               h.name                     AS horse_name,
                               s.starts                   AS starts,
                               s.wins                     AS wins,
                               s.prize_money_kr           AS earnings
                        FROM horse h
                        JOIN horse_career_stats s ON s.horse_id = h.horse_id
                        WHERE COALESCE(s.starts, 0) >= 10
                        ORDER BY
                          CASE %s
                            WHEN 'earnings'  THEN s.prize_money_kr
                            WHEN 'wins'      THEN s.wins
                            ELSE 0
                          END DESC,
                          CASE WHEN %s = 'win_rate'
                            THEN s.wins::numeric / NULLIF(s.starts, 0)
                            ELSE 0
                          END DESC,
                          s.prize_money_kr DESC
                        LIMIT 10
                    """, (sort, sort))
                else:
                    cur.execute(f"""
                        SELECT e.horse_id,
                               h.name                                                        AS horse_name,
                               SUM(CASE WHEN NOT COALESCE(e.withdrawn,false) THEN 1 ELSE 0 END) AS starts,
                               SUM(CASE WHEN e.placement = 1 THEN 1 ELSE 0 END)              AS wins,
                               SUM(COALESCE(e.prize_kr, 0))                                  AS earnings
                        FROM entry e
                        JOIN horse h ON h.horse_id = e.horse_id
                        JOIN race  r ON r.race_id  = e.race_id
                        WHERE r.race_date >= date_trunc('year', CURRENT_DATE)
                          AND NOT COALESCE(e.withdrawn, false)
                          AND {_NOT_QUALIFIER}
                        GROUP BY e.horse_id, h.name
                        HAVING SUM(CASE WHEN NOT COALESCE(e.withdrawn,false) THEN 1 ELSE 0 END) >= 3
                        ORDER BY
                          CASE %s
                            WHEN 'earnings'  THEN SUM(COALESCE(e.prize_kr, 0))
                            WHEN 'wins'      THEN SUM(CASE WHEN e.placement = 1 THEN 1 ELSE 0 END)
                            ELSE 0
                          END DESC,
                          CASE WHEN %s = 'win_rate'
                            THEN SUM(CASE WHEN e.placement = 1 THEN 1 ELSE 0 END)::numeric
                                 / NULLIF(SUM(CASE WHEN NOT COALESCE(e.withdrawn,false) THEN 1 ELSE 0 END), 0)
                            ELSE 0
                          END DESC,
                          SUM(COALESCE(e.prize_kr, 0)) DESC
                        LIMIT 10
                    """, (sort, sort))
                rows = cur.fetchall()
        finally:
            conn.close()

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
                           SUM(CASE WHEN e.placement = 1 THEN 1 ELSE 0 END)              AS wins,
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
                        WHEN 'wins'      THEN SUM(CASE WHEN e.placement = 1 THEN 1 ELSE 0 END)
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


# =====================================================================
# Stable (search)
# =====================================================================

@app.route('/')
def index():
    return render_template('index.html', active_tab='stable')


@app.route('/api/search')
def search():
    q = request.args.get('q', '').strip()
    if len(q) < 2:
        return jsonify([])

    conn = get_db()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
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
                WHERE h.name ILIKE %s
                ORDER BY s.prize_money_kr DESC NULLS LAST, h.name
                LIMIT 50
                """,
                (f'%{q}%',),
            )
            results = [dict(r) for r in cur.fetchall()]
            for r in results:
                r['earnings'] = _kr(r['earnings']) if r['earnings'] is not None else None
    finally:
        conn.close()
    return jsonify(results)


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
                       (SELECT th.trainer_name
                        FROM horse_trainer_history th
                        WHERE th.horse_id = h.horse_id
                        ORDER BY (th.to_date IS NULL) DESC, th.from_date DESC
                        LIMIT 1) AS trainer_name,
                       (SELECT th.trainer_id
                        FROM horse_trainer_history th
                        WHERE th.horse_id = h.horse_id
                        ORDER BY (th.to_date IS NULL) DESC, th.from_date DESC
                        LIMIT 1) AS trainer_id,
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
    sources = sorted(source_data.keys()) if isinstance(source_data, dict) else []

    horse = {
        'horse_id':      row['horse_id'],
        'st_id':         row['st_id'],
        'sources':       sources,
        'name':          row['name'] or '',
        'date_of_birth': str(row['date_of_birth']) if row['date_of_birth'] else '',
        'gender':        _GENDER_TEXT.get(row['gender_code'] or '', ''),
        'breed':         _BREED_TEXT.get(row['breed_code'] or '', ''),
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
    return render_template('horse.html', horse=horse, active_tab='stable')


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
                       e.disqualified                        AS dq,
                       e.withdrawn
                FROM entry e
                JOIN race    r ON r.race_id    = e.race_id
                LEFT JOIN track   t ON t.track_id   = r.track_id
                LEFT JOIN person  p ON p.person_id  = e.driver_id
                WHERE e.horse_id = %s
                ORDER BY r.race_date DESC NULLS LAST, e.race_id DESC NULLS LAST
                LIMIT 500
                """,
                (horse_id,),
            )
            rows = []
            for r in cur.fetchall():
                rows.append({
                    'race_date': r['race_date'].isoformat() if r['race_date'] else None,
                    'track_label': (r['track_label'] or '').strip().title(),
                    'track_country': r['track_country'],
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
                    'disqualified': r['dq'],
                    'withdrawn': r['withdrawn'],
                })
    finally:
        conn.close()
    return jsonify(rows)


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
                       h.sire_name                       AS father_name,
                       h.dam_name                        AS mother_name,
                       s.starts,
                       s.wins,
                       s.prize_money_kr
                FROM horse h
                LEFT JOIN horse_career_stats s ON s.horse_id = h.horse_id
                WHERE h.sire_id = %s
                   OR h.dam_id  = %s
                ORDER BY s.prize_money_kr DESC NULLS LAST, h.date_of_birth DESC NULLS LAST
                LIMIT 500
                """,
                (horse_id, horse_id),
            )
            rows = []
            for r in cur.fetchall():
                rows.append({
                    'horse_id': r['horse_id'],
                    'name': r['name'],
                    'year': r['year'],
                    'gender': _GENDER_TEXT.get(r['gender_code'] or '', ''),
                    'starts': r['starts'] or 0,
                    'wins':   r['wins'] or 0,
                    'earnings': _kr(r['prize_money_kr']) if r['prize_money_kr'] else '',
                    'earnings_raw': int(r['prize_money_kr']) if r['prize_money_kr'] else 0,
                    'is_father': True,
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


def _race_entries(*, race_id=None, atg_race_id=None):
    conn = get_db()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            if atg_race_id is not None:
                cur.execute("SELECT race_id FROM race WHERE atg_race_id = %s",
                            (atg_race_id,))
                row = cur.fetchone()
                if not row:
                    return jsonify({'error': 'not found'}), 404
                race_id = row['race_id']

            cur.execute(
                """
                SELECT r.race_date,
                       r.race_number, r.distance, r.start_method, r.race_class,
                       r.victory_margin,
                       t.name    AS track_name,
                       t.country AS track_country
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
                SELECT e.horse_id,
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
            horse_ids = [r['horse_id'] for r in entry_rows if r['horse_id']]

            stats_pre: dict[int, dict] = {}
            head_race_date = head.get('race_date')
            if horse_ids and head_race_date:
                cur.execute(
                    """
                    SELECT e.horse_id,
                           COUNT(*) FILTER (
                               WHERE NOT e.withdrawn
                                 AND COALESCE(e.placement_text, '') !~* '^(gdk|ejg|ejp)'
                           ) AS starts,
                           COUNT(*) FILTER (
                               WHERE e.placement = 1
                                 AND COALESCE(e.placement_text, '') !~* '^(gdk|ejg|ejp)'
                           ) AS wins
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
                    }

            rows = []
            for r in entry_rows:
                pre = stats_pre.get(r['horse_id'], {'starts': 0, 'wins': 0})
                pt = r['placement_text']
                qual = bool(pt) and pt[:3].lower() in ('gdk', 'ejg', 'ejp')
                contributes_start = (not r['withdrawn']) and (not qual)
                won_this_race = r['placement'] == 1 and not qual
                post_starts = pre['starts'] + (1 if contributes_start else 0)
                post_wins   = pre['wins']   + (1 if won_this_race    else 0)
                rows.append({
                    'horse_id': r['horse_id'],
                    'horse_name': r['horse_name'],
                    'number': r['number'],
                    'distance': r['distance'],
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
                    # ML-feature columns dropped in v2 — pass nulls so the
                    # existing race.html JS renders empty cells gracefully.
                    'tf': None, 'tf_odds': None, 'df': None, 'df_odds': None,
                    'd_wr': None, 't_wr': None,
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
                })
    finally:
        conn.close()

    race_date = head.get('race_date')
    return jsonify({
        'race_date': race_date.isoformat() if hasattr(race_date, 'isoformat') else race_date,
        'track': (head.get('track_name') or '').strip().title(),
        'country': head.get('track_country'),
        'race_number': head.get('race_number'),
        'distance': head.get('distance'),
        'start_method': head.get('start_method'),
        'race_class': head.get('race_class'),
        'victory_margin': head.get('victory_margin'),
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
                   e.{other_col} AS other_id,
                   r.atg_race_id AS atg_id,
                   r.race_id     AS race_id,
                   r.race_date,
                   t.name        AS track,
                   po.name       AS other_name,
                   po.short_name AS other_short_name
            FROM entry e
            JOIN race r       ON r.race_id    = e.race_id
            LEFT JOIN horse  h ON h.horse_id   = e.horse_id
            LEFT JOIN person po ON po.person_id = e.{other_col}
            LEFT JOIN track  t  ON t.track_id   = r.track_id
            WHERE e.{id_col} = %s
            ORDER BY r.race_date DESC NULLS LAST, e.entry_id DESC
            LIMIT 50
            """,
            (person_id,),
        )
        recent = []
        for rr in cur.fetchall():
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
                # ML feature cols dropped in v2.
                'tf': None, 'tf_odds': None, 'df': None, 'df_odds': None,
                'd_wr': None, 't_wr': None,
                'sex':        rr['sex'],
                'age':        rr['age'],
                'sulky':      rr['sulky'],
                'sulky_changed': rr['sulky_changed'],
                'shoe_code':  rr['shoe_code'],
                'shoe_front_changed': rr['shoe_front_changed'],
                'shoe_back_changed':  rr['shoe_back_changed'],
                'race_date':  rr['race_date'].isoformat() if rr['race_date'] else None,
                'track':      (rr['track'] or '').strip().lower(),
                'atg_id':     rr['atg_id'],
                'race_id':    rr['race_id'],
                f'{other_alias}_id':    rr['other_id'],
                f'{other_alias}_name':  fmtName(rr['other_name'] or ''),
                f'{other_alias}_short': rr['other_short_name'] or shortName(rr['other_name'] or ''),
            }
            recent.append(item)
        return recent


@app.route('/api/driver/<int:driver_id>')
def driver_api(driver_id):
    """Driver header + weekly form-series (empty in v2) + recent entries."""
    conn = get_db()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("SELECT person_id, name, short_name FROM person WHERE person_id = %s",
                        (driver_id,))
            head = cur.fetchone()
            if not head:
                return jsonify({'error': 'not found'}), 404
        recent = _person_recent_entries(conn, driver_id, 'driver')
    finally:
        conn.close()
    return jsonify({
        'driver_id': head['person_id'],
        'name':      fmtName(head['name'] or ''),
        'short':     head['short_name'],
        # v2 doesn't track tf/df/df_odds yet — series is empty. The chart
        # JS handles this by showing "no data".
        'series':    [],
        'recent':    recent,
    })


@app.route('/api/trainer/<int:trainer_id>')
def trainer_api(trainer_id):
    """Trainer header + weekly form-series (empty in v2) + recent entries."""
    conn = get_db()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("SELECT person_id, name, short_name FROM person WHERE person_id = %s",
                        (trainer_id,))
            head = cur.fetchone()
            if not head:
                return jsonify({'error': 'not found'}), 404
        recent = _person_recent_entries(conn, trainer_id, 'trainer')
    finally:
        conn.close()
    return jsonify({
        'trainer_id': head['person_id'],
        'name':       fmtName(head['name'] or ''),
        'short':      head['short_name'],
        'series':     [],
        'recent':     recent,
    })


# =====================================================================
# ML (placeholder)
# =====================================================================

def _ml_placeholder():
    return render_template('ml.html', active_tab='ml')


app.add_url_rule('/ml',         'ml_page',        _ml_placeholder)
app.add_url_rule('/ml/race',    'ml_race_page',   _ml_placeholder)
app.add_url_rule('/ml/breed',   'ml_breed_page',  _ml_placeholder)
app.add_url_rule('/ml/play',    'ml_play_page',   _ml_placeholder)
app.add_url_rule('/ml/models',  'ml_models_page', _ml_placeholder)


# =====================================================================
# Admin
# =====================================================================

@app.route('/admin')
def admin_page():
    return render_template('admin.html', active_tab='admin')


def _pid_alive(pid: int | None) -> bool:
    if not pid or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
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
    if mode not in ('atg', 'st'):
        return jsonify({'error': 'invalid mode'}), 400
    job_name = 'update' if mode == 'atg' else 'update_st'

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
        'summary':     row['summary'] or {},
        'log':         row['log'] or '',
    }


if __name__ == '__main__':
    # debug=False mirrors v1 — the admin page polls the DB so we don't need
    # autoreload, and avoiding the reloader prevents log writes from re-spawning
    # the parent process.
    app.run(debug=False, port=WEB_PORT, threaded=True)
