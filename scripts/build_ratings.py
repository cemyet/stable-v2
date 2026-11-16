#!/usr/bin/env python3
"""Build the horse ratings: one chronological pass over every result since
2010 (core/ratings/engine.py), then pedigree, the overall's weights and the
backtest. Writes rating_entry, rating_person, rating_pedigree and
rating_model; the tables are built in staging copies and swapped in at the
end, so the card keeps reading the previous build meanwhile.

    python scripts/build_ratings.py              # full build (~10 min)
    python scripts/build_ratings.py --since 2018-01-01 --dry-run
"""

from __future__ import annotations

import argparse
import datetime as dt
import io
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from core.db import get_connection  # noqa: E402
from core.ratings import keys as K  # noqa: E402
from core.ratings import pedigree as P  # noqa: E402
from core.ratings.compose import horse_values, rollup  # noqa: E402
from core.ratings.engine import (  # noqa: E402
    FRESH_BUCKETS, R_DID, R_HID, R_RID, R_TID, Engine, _robust)
from core.ratings.fit import fit_weights, prepare  # noqa: E402

STREAM_SQL = """
    SELECT e.entry_id, e.race_id, e.race_date, e.horse_id, e.driver_id, e.trainer_id,
           e.post, e.auto, e.distance, e.tillagg, e.placement, e.placement_text,
           COALESCE(e.galopp, false), COALESCE(e.disqualified, false), e.time_seconds,
           e.odds, e.prize_kr, e.kmtid_first_200ms, e.kmtid_last_200ms, e.kmtid_best_100ms,
           e.kmtid_intervals, ec.comment, h.breed_code, h.date_of_birth, r.track_id
    FROM entry e
    JOIN race r USING (race_id)
    LEFT JOIN horse h ON h.horse_id = e.horse_id
    LEFT JOIN entry_comment ec ON ec.entry_id = e.entry_id
    WHERE e.race_date >= %s AND e.race_date <= %s
      AND NOT COALESCE(e.withdrawn, false) AND e.horse_id IS NOT NULL
    ORDER BY e.race_date, e.race_id
"""


def km_offsets(cur):
    """Winners' km time by breed × start × distance band and by track, as
    offsets from a 2140-ish auto start, so times compare across races."""
    cur.execute("""
        SELECT COALESCE(h.breed_code, 'V'), e.auto,
               CASE WHEN e.distance < 1800 THEN 0 WHEN e.distance < 2400 THEN 1
                    WHEN e.distance < 2900 THEN 2 ELSE 3 END,
               r.track_id, count(*), avg(e.time_seconds)
        FROM entry e JOIN race r USING (race_id) JOIN horse h ON h.horse_id = e.horse_id
        WHERE e.placement = 1 AND e.time_seconds BETWEEN 55 AND 130
          AND e.race_date >= '2015-01-01' AND NOT COALESCE(e.galopp, false)
        GROUP BY 1, 2, 3, 4
    """)
    cells = cur.fetchall()
    band = defaultdict(lambda: [0.0, 0])
    for b, a, db, tr, n, m in cells:
        b = b if b in ('V', 'K') else 'V'
        band[(b, bool(a), db)][0] += n * float(m)
        band[(b, bool(a), db)][1] += n
    mean = {k: s / n for k, (s, n) in band.items() if n}
    off = {}
    for (b, a, db), m in mean.items():
        base = mean.get((b, True, 1)) or mean.get((b, False, 1))
        off[(b, a, db)] = m - base
    track = defaultdict(lambda: [0.0, 0])
    for b, a, db, tr, n, m in cells:
        b = b if b in ('V', 'K') else 'V'
        bm = mean.get((b, bool(a), db))
        if bm is None or tr is None:
            continue
        track[tr][0] += n * (float(m) - bm)
        track[tr][1] += n
    toff = {tr: s / (n + 200) for tr, (s, n) in track.items()}
    return off, toff


def post_offsets(cur):
    """Mean xLabs first-200 pace by start method and post, relative to the
    front row of an auto start."""
    cur.execute("""
        SELECT e.auto, LEAST(e.post, 15), count(*), avg(e.kmtid_first_200ms)
        FROM entry e
        WHERE e.kmtid_first_200ms BETWEEN 55000 AND 95000 AND e.post IS NOT NULL
          AND NOT COALESCE(e.galopp, false)
        GROUP BY 1, 2
    """)
    rows = cur.fetchall()
    front = [(n, float(m)) for a, p, n, m in rows if a and 1 <= p <= 5]
    base = sum(n * m for n, m in front) / max(1, sum(n for n, _ in front))
    off = {}
    tot = defaultdict(lambda: [0.0, 0])
    for a, p, n, m in rows:
        tot[bool(a)][0] += n * float(m)
        tot[bool(a)][1] += n
        if n >= 20:
            off[(bool(a), p)] = float(m) - base
    for a, (s, n) in tot.items():
        if n:
            off[(a, None)] = s / n - base
    return off


NULL = '\\N'


def _arr(vals, fmt=str):
    return '{' + ','.join('NULL' if v is None else fmt(v) for v in vals) + '}'


def _f(x):
    return f'{x:.5g}'


class Writer:
    def __init__(self, conn, dry):
        self.conn = conn
        self.dry = dry
        self.n_entry = 0
        self.n_person = 0
        if not dry:
            with conn.cursor() as cur:
                for t in ('rating_entry', 'rating_person', 'rating_pedigree'):
                    cur.execute(f"DROP TABLE IF EXISTS {t}_stage")
                    cur.execute(f"CREATE UNLOGGED TABLE {t}_stage (LIKE {t} INCLUDING DEFAULTS)")
            conn.commit()

    def entries(self, rows):
        self.n_entry += len(rows)
        if self.dry or not rows:
            return
        buf = io.StringIO()
        for eid, hid, d, elo, starts, hv, hc, raw, resid, pw, breed in rows:
            resid = NULL if resid is None else resid
            pw = NULL if pw is None else pw
            buf.write(f"{eid}\t{hid}\t{d.isoformat()}\t{elo}\t{starts}\t{_arr(hv)}\t{_arr(hc)}\t"
                      f"{_arr(raw, _f)}\t{resid}\t{pw}\t{breed}\n")
        buf.seek(0)
        with self.conn.cursor() as cur:
            cur.copy_expert("COPY rating_entry_stage FROM STDIN", buf)
        self.conn.commit()

    def persons(self, rows):
        self.n_person += len(rows)
        if self.dry or not rows:
            return
        buf = io.StringIO()
        for r in rows:
            buf.write('\t'.join(NULL if v is None else (v.isoformat() if isinstance(v, dt.date) else str(v))
                                for v in r) + '\n')
        buf.seek(0)
        with self.conn.cursor() as cur:
            cur.copy_expert("COPY rating_person_stage FROM STDIN", buf)
        self.conn.commit()

    def pedigree(self, ped):
        if self.dry:
            return
        buf = io.StringIO()
        for hid, (v, c, info) in ped.items():
            buf.write(f"{hid}\t{_arr(v)}\t{_arr(c)}\t{json.dumps(info, ensure_ascii=False).replace(chr(92), chr(92) * 2)}\n")
        buf.seek(0)
        with self.conn.cursor() as cur:
            cur.copy_expert("COPY rating_pedigree_stage FROM STDIN", buf)
        self.conn.commit()

    def swap(self, model: dict):
        if self.dry:
            return
        with self.conn.cursor() as cur:
            for t in ('rating_entry', 'rating_person', 'rating_pedigree'):
                cur.execute(f"TRUNCATE {t}")
                cur.execute(f"INSERT INTO {t} SELECT * FROM {t}_stage")
            for k, v in model.items():
                cur.execute("""
                    INSERT INTO rating_model (key, value, updated_at) VALUES (%s, %s, NOW())
                    ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value, updated_at = NOW()
                """, (k, json.dumps(v)))
        self.conn.commit()
        with self.conn.cursor() as cur:
            for t in ('rating_entry', 'rating_person', 'rating_pedigree'):
                cur.execute(f"DROP TABLE IF EXISTS {t}_stage")
            cur.execute("ANALYZE rating_entry")
            cur.execute("ANALYZE rating_person")
            cur.execute("ANALYZE rating_pedigree")
        self.conn.commit()


def overall_norm(eng: Engine, ped, weights, last_conn, t_end):
    """Per-breed spread of the overall among horses active in the last year,
    so the overall lands on the same 65 + 10·z scale as its parts."""
    by_breed = defaultdict(list)
    for hid, (hv, hc, starts, t) in eng.last_snap.items():
        if t < t_end - 365:
            continue
        h = eng.H.get(hid)
        if h is None:
            continue
        vals = horse_values(hv, hc)
        tid, did = last_conn.get(hid, (None, None))
        tv, tc, sv, sc, dv, dc = eng.person_now(tid, did, t_end)
        vals['trainer'] = (tv or None, tc / 100 if tv else 0.0)
        vals['stable'] = (sv or None, sc / 100 if sv else 0.0)
        vals['driver'] = (dv or None, dc / 100 if dv else 0.0)
        p = ped.get(hid)
        if p:
            for k, v, c in zip(K.PEDIGREE_KEYS, p[0], p[1]):
                vals[k] = (v or None, c / 100 if v else 0.0)
        _, (ovr, conf) = rollup(vals, weights, starts)
        if ovr is not None and conf >= 0.4:
            by_breed[h.breed].append(ovr)
    out = {}
    for b, vals in by_breed.items():
        if len(vals) >= 100:
            mu, sd = _robust(vals)
            out[b] = {'mu': round(mu, 3), 'sd': round(sd, 3), 'n': len(vals)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--since', default='2010-01-01')
    ap.add_argument('--until', default=dt.date.today().isoformat())
    ap.add_argument('--snap-start', default='2023-01-01')
    ap.add_argument('--bt-start', default='2024-01-01')
    ap.add_argument('--holdout', default='2026-05-01')
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--dump-backtest', help='pickle the prepared backtest races here')
    args = ap.parse_args()
    t_start = time.time()

    def log(msg):
        print(f"[{time.time() - t_start:6.0f}s] {msg}", flush=True)

    read = get_connection()
    write = get_connection()
    with read.cursor() as cur:
        km_off, track_off = km_offsets(cur)
        post_off = post_offsets(cur)
    log(f"offsets: {len(km_off)} distance bands, {len(track_off)} tracks, {len(post_off)} posts")

    eng = Engine(snap_start=dt.date.fromisoformat(args.snap_start),
                 bt_start=dt.date.fromisoformat(args.bt_start),
                 km_off=km_off, track_off=track_off, post_off=post_off, log=log)
    out = Writer(write, args.dry_run)
    last_conn = {}
    snap_t = eng.snap_t

    cur = read.cursor(name='ratings_stream')
    cur.itersize = 20000
    cur.execute(STREAM_SQL, (args.since, args.until))
    batch = []
    rid = None
    year = None
    for row in cur:
        if row[R_RID] != rid:
            if batch:
                eng.race(batch)
            batch = []
            rid = row[R_RID]
            if row[2].year != year:
                year = row[2].year
                log(f"{year}: {eng.races:,} races, {eng.entries:,} starts so far, {len(eng.H):,} horses")
            if len(eng.snap_rows) >= 100000:
                out.entries(eng.snap_rows)
                eng.snap_rows = []
            if eng.person_rows:
                out.persons(eng.person_rows)
                eng.person_rows = []
        batch.append(row)
        if row[2].toordinal() >= snap_t:
            last_conn[row[R_HID]] = (row[R_TID], row[R_DID])
    if batch:
        eng.race(batch)
    cur.close()
    out.entries(eng.snap_rows)
    eng.snap_rows = []
    out.persons(eng.person_rows)
    eng.person_rows = []
    log(f"pass done: {eng.races:,} races, {eng.entries:,} starts, {len(eng.H):,} horses, "
        f"{out.n_entry:,} snapshots, {out.n_person:,} person-months")

    with read.cursor() as cur2:
        ped = P.build(cur2, eng.H, today=dt.date.fromisoformat(args.until), log=log)
    out.pedigree(ped)

    races = prepare(eng.bt, eng.fresh, ped)
    log(f"backtest: {len(races):,} races with one winner and 5+ runners")
    weights, report = fit_weights(races, holdout_t=dt.date.fromisoformat(args.holdout).toordinal(), log=log)
    log("weights: " + ', '.join(f"{k} {v:.3f}" for k, v in sorted(weights.items(), key=lambda x: -x[1])))
    log("holdout: " + json.dumps(report['holdout']))
    log("holdout with market odds: " + json.dumps(report['holdout_with_market']))
    log("joint coefficients: " + json.dumps(report['coef']))
    log("one-at-a-time coefficients: " + json.dumps(report['solo_coef']))
    log("coverage (any value): " + json.dumps(report['coverage']))
    log("mean confidence: " + json.dumps(report['mean_conf']))
    if args.dump_backtest:
        import pickle
        with open(args.dump_backtest, 'wb') as fh:
            pickle.dump(races, fh)

    t_end = dt.date.fromisoformat(args.until).toordinal()
    norm = overall_norm(eng, ped, weights, last_conn, t_end)
    log(f"overall norm: {norm}")

    census = {b: {'elo': p['elo'], 'active': p['active'],
                  'subs': {k: {'mu': round(s[0], 5), 'sd': round(s[1], 5), 'rho': round(s[2], 3), 'n': s[3]}
                           for k, s in zip(K.HORSE_KEYS, p['subs'])}}
              for b, p in eng.params.items()}
    model = {
        'weights': weights,
        'weights_default': K.DEFAULT_WEIGHTS,
        'norm': norm,
        'fresh': {'buckets': list(FRESH_BUCKETS), 'table': eng.fresh},
        'census': census,
        'backtest': report,
        'meta': {'built': dt.datetime.now().isoformat(timespec='seconds'), 'since': args.since,
                 'until': args.until, 'snap_start': args.snap_start, 'races': eng.races,
                 'starts': eng.entries, 'horse_keys': list(K.HORSE_KEYS),
                 'pedigree_keys': list(K.PEDIGREE_KEYS)},
    }
    out.swap(model)

    top = sorted(((h.R, hid) for hid, h in eng.H.items()
                  if h.breed == 'V' and h.last is not None and h.last >= t_end - 180 and h.n >= 8),
                 reverse=True)[:20]
    with read.cursor() as cur3:
        cur3.execute("SELECT horse_id, name FROM horse WHERE horse_id = ANY(%s)", ([h for _, h in top],))
        names = dict(cur3.fetchall())
    log("top Elo (V, raced in the last 180 days): " +
        '; '.join(f"{names.get(h, h)} {r:.0f}" for r, h in top))
    log("done" + (" (dry run, nothing written)" if args.dry_run else ""))


if __name__ == '__main__':
    main()
