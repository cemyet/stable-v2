"""The horse card's rating payload, as of a race day.

The horse's values come from its last snapshot before the day (so a card
for a past race shows what was known then), the trainer and driver from the
month's person ratings for the people named on that start, freshness from
the days since that last start, and pairing from how the horse has run with
today's driver.
"""

from __future__ import annotations

import datetime as dt
import threading
import time

from core.ratings import keys as K
from core.ratings.compose import (display_overall, fresh_value, horse_values, pair_value,
                                  rollup)
from core.ratings.engine import fresh_bucket

_model_lock = threading.Lock()
_model_cache: dict = {'at': 0.0, 'value': None}


def load_model(cur, ttl=300.0):
    with _model_lock:
        if _model_cache['value'] is not None and time.time() - _model_cache['at'] < ttl:
            return _model_cache['value']
    cur.execute("SELECT key, value FROM rating_model WHERE key IN ('weights', 'norm', 'fresh', 'meta', 'backtest')")
    model = dict(cur.fetchall())
    with _model_lock:
        _model_cache.update(at=time.time(), value=model)
    return model


def _km(sec_per_km):
    """Seconds per km as a Swedish km time: 68.3 -> 1.08,3."""
    if sec_per_km is None:
        return '–'
    tenths = int(round(abs(sec_per_km) * 10))
    m, rest = divmod(tenths, 600)
    return f"{m}.{rest // 10:02d},{rest % 10}"


def _n(c, k):
    """Evidence behind a confidence c = n / (n + k)."""
    if c <= 0:
        return 0
    if c >= 0.995:
        return None
    return k * c / (1 - c)


def _runs(c, k, word='starts'):
    n = _n(c, k)
    if n is None:
        return f"many {word}"
    n = max(1, round(n))
    return f"≈{n} {word}" if n != 1 else f"1 {word[:-1] if word.endswith('s') else word}"


def _signed(x, digits=2):
    return f"{x:+.{digits}f}"


def basis(key, raw, c, extra=None):
    """One line on what a sub-rating rests on."""
    s = K.SUB_BY_KEY[key]
    if raw is None:
        return 'no evidence yet'
    ev = _runs(c, s.k)
    if key == 'gate':
        return f"first 200 m at {_km(-raw)} at best, post-adjusted · {_runs(c, s.k, 'xLabs runs')}"
    if key == 'early':
        return f"{abs(raw) * 100:.0f}% of the field {'further forward' if raw >= 0 else 'further back'} early than its posts predict · {_runs(c, s.k, 'commented starts')}"
    if key == 'duels':
        return f"start Elo {raw:.0f} from early positions against the horses beside it · {ev}"
    if key == 'top':
        return f"best 100 m at {_km(-raw)} · {_runs(c, s.k, 'xLabs runs')}"
    if key == 'accel':
        return f"picks up {raw:.1f} s/km over 200 m when it goes · {_runs(c, s.k, 'xLabs runs')}"
    if key == 'record':
        return f"record {_km(-raw)} adjusted to a 2140 m auto start on an average track · {ev}"
    if key == 'cruise':
        return f"typical {_km(-raw)} adjusted km time over its recent starts · {ev}"
    if key == 'closing':
        return f"last 200 m at {_km(-raw)} at best · {_runs(c, s.k, 'xLabs runs')}"
    if key == 'sectors':
        return f"closing sector {_km(-raw)} at best, scaled to the last 500 m · {_runs(c, s.k, 'noted sectors')}"
    if key == 'gains':
        return (f"{'makes up' if raw >= 0 else 'loses'} {abs(raw) * 100:.0f}% of the field from its early position "
                f"when it starts from midfield or behind · {ev}")
    if key in ('fverdict', 'sverdict', 'fight', 'rate', 'gait'):
        what = {'fverdict': 'finish', 'sverdict': 'staying', 'fight': 'fighting',
                'rate': 'rateability', 'gait': 'gait and temper'}[key]
        return f"{what} verdicts in race comments average {_signed(raw)} (−1 to +1) · {_runs(c, s.k, 'commented starts')}"
    if key == 'death':
        return f"runs {_signed(raw)} against expectation after a death-seat trip · {_runs(c, s.k, 'such trips')}"
    if key == 'wide':
        return f"runs {_signed(raw)} against expectation after wide or uncovered trips · {_runs(c, s.k, 'such trips')}"
    if key == 'fade':
        return f"last 200 m {abs(raw):.1f} s/km {'faster' if raw >= 0 else 'slower'} than its mid-race pace · {_runs(c, s.k, 'xLabs runs')}"
    if key == 'tempo':
        return f"runs {_signed(raw)} against expectation from the lead, more credit at a hard tempo · {_runs(c, s.k, 'starts in front')}"
    if key == 'distance':
        return f"{_signed(raw)} against expectation at 2500 m+ compared with all its starts · {_runs(c, s.k, 'long starts')}"
    if key == 'gallop':
        return f"breaks into a gallop of its own doing in {-raw * 100:.0f}% of starts · {ev}"
    if key == 'consistency':
        return f"its results scatter ±{-raw:.2f} around expectation · {ev}"
    if key == 'close':
        return f"close finishes net {_signed(raw)} per finish (won +1, lost −1) · {_runs(c, s.k, 'close finishes')}"
    if key == 'holding':
        return (f"when it leads all the way it wins {abs(raw) * 100:.0f} percentage points "
                f"{'more' if raw >= 0 else 'less'} often than its Elo chance · {_runs(c, s.k, 'such starts')}")
    if key == 'elo':
        return f"Elo {raw:.0f} from every finishing order since 2010 · {ev}"
    if key == 'peak':
        return f"best Elo held over three straight starts: {raw:.0f}"
    if key == 'tested':
        return f"opponents average Elo {raw:.0f} · {ev}"
    if key == 'prize':
        return f"typical winner's purse in its races {10 ** raw / 1000:,.0f} tkr · {ev}"
    if key == 'trend':
        return f"Elo {_signed(raw, 0)} over the last {max(1, round(_n(c, s.k) or 5))} starts"
    if key == 'recent':
        return f"recent runs {_signed(raw)} against expectation, last two months weigh most · {ev}"
    if key == 'fresh':
        return extra or ''
    if key in ('trainer', 'stable', 'driver', 'pairing'):
        return extra or ''
    return extra or ''


def _ped_basis(key, info):
    i = (info or {}).get(key) or {}
    if key == 'own':
        return f"BLUP index {i['blup']:.0f}, reliability {i['rel']:.2f}" if i.get('blup') is not None else 'no BLUP index'
    if key == 'sire':
        parts = []
        if i.get('blup') is not None:
            parts.append(f"BLUP {i['blup']:.0f}")
        if i.get('foals'):
            parts.append(f"{i['foals']} foals rated at the same age")
        return ' · '.join(parts) or 'unknown sire'
    if key == 'dam':
        parts = []
        if i.get('blup') is not None:
            parts.append(f"BLUP {i['blup']:.0f}")
        parts.append(f"{i.get('foals') or 0} other foals")
        return ' · '.join(parts)
    if key == 'damsire':
        parts = []
        if i.get('blup') is not None:
            parts.append(f"BLUP {i['blup']:.0f}")
        if i.get('grandfoals'):
            parts.append(f"{i['grandfoals']:,} foals out of his daughters")
        return ' · '.join(parts) or 'unknown damsire'
    return ''


def payload(cur, horse_id: int, *, as_of: dt.date, atg_race_id: str | None = None):
    model = load_model(cur)
    if not model.get('weights'):
        return None
    weights = model['weights']
    norm = model.get('norm') or {}

    cur.execute("""
        SELECT race_date, elo, starts, hv, hc, hr, entry_id, breed
        FROM rating_entry WHERE horse_id = %s AND race_date < %s
        ORDER BY race_date DESC, entry_id DESC LIMIT 1
    """, (horse_id, as_of))
    snap = cur.fetchone()

    trainer_id = driver_id = None
    if atg_race_id:
        cur.execute("""
            SELECT e.trainer_id, e.driver_id FROM entry e JOIN race r USING (race_id)
            WHERE r.atg_race_id = %s AND e.horse_id = %s LIMIT 1
        """, (atg_race_id, horse_id))
        row = cur.fetchone()
        if row:
            trainer_id, driver_id = row
    if snap and (trainer_id is None or driver_id is None):
        cur.execute("SELECT trainer_id, driver_id FROM entry WHERE entry_id = %s", (snap[6],))
        row = cur.fetchone()
        if row:
            trainer_id = trainer_id or row[0]
            driver_id = driver_id or row[1]

    if snap and snap[7] in ('V', 'K'):
        breed = snap[7]
    else:
        cur.execute("SELECT breed_code FROM horse WHERE horse_id = %s", (horse_id,))
        row = cur.fetchone()
        breed = row[0] if row and row[0] in ('V', 'K') else 'V'

    values = {}
    raws = {}
    notes = {}
    starts = 0
    if snap:
        _d, elo, starts, hv, hc, hr, _eid, _b = snap
        values.update(horse_values(hv, hc))
        for i, k in enumerate(K.HORSE_KEYS):
            raws[k] = hr[i] if hr and i < len(hr) else None

    def person(pid, role):
        if not pid:
            return None
        cur.execute("""
            SELECT v, c, raw, v2, c2, raw2, n, p.name FROM rating_person rp
            LEFT JOIN person p ON p.person_id = rp.person_id
            WHERE rp.person_id = %s AND rp.role = %s AND rp.month <= %s
            ORDER BY rp.month DESC LIMIT 1
        """, (pid, role, as_of))
        return cur.fetchone()

    tr = person(trainer_id, 'T')
    if tr:
        v, c, raw, v2, c2, raw2, n, name = tr
        values['trainer'] = (v if c else None, (c or 0) / 100)
        values['stable'] = (v2 if c2 else None, (c2 or 0) / 100)
        notes['trainer'] = (f"{name or 'trainer'}: {raw:.2f}× the wins the horses' Elo expects, complete fields, "
                            f"18-month half-life · ≈{n:.0f} expected wins")
        notes['stable'] = f"{name or 'trainer'}: {raw2:.2f}× expected wins in the last weeks (45-day half-life)" if raw2 else ''
    dr = person(driver_id, 'D')
    if dr:
        v, c, raw, _v2, _c2, _r2, n, name = dr
        values['driver'] = (v if c else None, (c or 0) / 100)
        notes['driver'] = f"{name or 'driver'}: {raw:+.0f} Elo on top of the horse's · {n:,.0f} drives"

    if snap and driver_id:
        cur.execute("""
            SELECT avg(re.resid) FILTER (WHERE e.driver_id = %s), count(*) FILTER (WHERE e.driver_id = %s),
                   avg(re.resid)
            FROM rating_entry re JOIN entry e USING (entry_id)
            WHERE re.horse_id = %s AND re.race_date < %s AND re.resid IS NOT NULL
        """, (driver_id, driver_id, horse_id, as_of))
        w, nw, a = cur.fetchone()
        if nw:
            diff = (w or 0) - (a or 0)
            values['pairing'] = pair_value(diff, nw)
            notes['pairing'] = (f"{nw} start{'s' if nw != 1 else ''} with this driver since 2023, "
                                f"{_signed(diff)} against expectation compared with its other starts")
        else:
            notes['pairing'] = 'first time together since 2023'

    if snap:
        days = (as_of - snap[0]).days
        fr = model.get('fresh') or {}
        values['fresh'] = fresh_value(fr.get('table'), fresh_bucket(days))
        notes['fresh'] = f"{days} days since its last start"

    cur.execute("SELECT v, c, info FROM rating_pedigree WHERE horse_id = %s", (horse_id,))
    ped = cur.fetchone()
    ped_info = None
    if ped:
        pv, pc, ped_info = ped
        for k, v, c in zip(K.PEDIGREE_KEYS, pv, pc):
            values[k] = (v or None, c / 100 if v else 0.0)

    mains, (ovr_raw, ovr_c) = rollup(values, weights, starts)
    ovr, pm = display_overall(ovr_raw, ovr_c, norm.get(breed))

    out_mains = []
    for m in K.MAINS:
        mv, mc = mains[m.key]
        subs = []
        for s in K.SUBS:
            if s.main != m.key:
                continue
            v, c = values.get(s.key, (None, 0.0))
            if s.scope == 'horse':
                b = basis(s.key, raws.get(s.key), c)
            elif s.scope == 'pedigree':
                b = _ped_basis(s.key, ped_info)
            else:
                b = notes.get(s.key) or 'no evidence yet'
            subs.append({'key': s.key, 'label': s.label, 'v': v, 'c': round(c, 2), 'basis': b})
        out_mains.append({'key': m.key, 'code': m.code, 'label': m.label, 'group': m.group,
                          'v': round(mv) if mv is not None else None, 'c': round(mc, 2), 'subs': subs})

    return {
        'horse_id': horse_id,
        'as_of': as_of.isoformat(),
        'snapshot': snap[0].isoformat() if snap else None,
        'starts': starts,
        'elo': round(snap[1]) if snap else None,
        'overall': {'v': ovr, 'pm': pm, 'c': round(ovr_c, 2)},
        'weights': weights,
        'mains': out_mains,
        'built': (model.get('meta') or {}).get('built'),
    }
