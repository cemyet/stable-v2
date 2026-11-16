"""What a model's probabilities would have returned in real ATG pools.

Every strategy is settled against what the pool actually paid
(scrapers/atg_games.py), with our own stake and winning rows added to the
pool so a hit dilutes itself. The same races / games are always scored with
two references next to the model: the market (closing win odds) and, in the
marking pools, the public (its stake share per horse). Pools are heavy
tailed, so each ROI comes with a game-bootstrap 90% interval, the ROI of the
two halves of the period, and the ROI without the single biggest return.

  win     flat 1 unit where p × closing odds clears a bar; the favourite;
          the favourite where it isn't the market's; a stake ∝ p on everyone
  place   the top-2/3 chance from p (Harville) × the place odds on the board
  V4 …    cover X% per leg: horses in order of p until their sum reaches X;
          and a banker on the leg where p's favourite (at least 35%) is most
          underbet by the public, covering 80% elsewhere. Games with a top-pool option
          (V64 / V65 V6, GS75 V7) also played top-only.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from core.reduce_system import top_pool

ROW_PRICE = {'V4': 2.0, 'V5': 1.0, 'V64': 1.0, 'V65': 1.0,
             'V75': 0.5, 'GS75': 1.0, 'V85': 0.5, 'V86': 0.25}
# ATG's price changes: (first game day, row price from then on).
PRICE_FROM = {'V75': (('2026-11-28', 0.6),)}


def row_price(gtype: str, day) -> float:
    """The row price of a game type on a given day (date, Timestamp or ISO string)."""
    price, d = ROW_PRICE[gtype], str(day)[:10]
    for since, p in PRICE_FROM.get(gtype, ()):
        if d >= since:
            price = p
    return price
COVER = (0.6, 0.7, 0.8)
WIN_EDGES = (1.0, 1.2, 1.5)
PLACE_EDGES = (1.0, 1.2)
N_BOOT = 1000


# ── data ─────────────────────────────────────────────────────────────────

def load(conn, since):
    """Win / place pool per starter and the marking games, from `since` on."""
    with conn.cursor() as cur:
        cur.execute("""
            SELECT e.entry_id, p.atg_race_id, p.number, p.race_date, p.win_odds, p.place_min,
                   p.place_paid, p.places_paid, p.finish, p.scratched
            FROM atg_race_pool p
            JOIN race r ON r.atg_race_id = p.atg_race_id
            JOIN entry e ON e.race_id = r.race_id AND e.program_number = p.number
            WHERE p.race_date >= %s
        """, (since,))
        st = pd.DataFrame(cur.fetchall(), columns=['entry_id', 'race', 'number', 'date', 'o', 'pmin',
                                                   'ppaid', 'k', 'finish', 'scr'])
        cur.execute("""
            SELECT g.atg_game_id, g.game_type, g.game_date, g.rtp, g.turnover, g.tier_pots,
                   g.payouts, g.race_ids, g.legs
            FROM atg_game g
            WHERE g.game_type = ANY(%s) AND g.status = 'results' AND g.game_date >= %s
              AND g.payouts IS NOT NULL
        """, (list(ROW_PRICE), since))
        games = cur.fetchall()
        cur.execute("""
            SELECT s.atg_game_id, s.leg, s.number, s.bet_share, s.scratched
            FROM atg_game_start s JOIN atg_game g USING (atg_game_id)
            WHERE g.game_type = ANY(%s) AND g.game_date >= %s
        """, (list(ROW_PRICE), since))
        shares = {}
        for gid, leg, n, share, scr in cur.fetchall():
            shares.setdefault((gid, leg), {})[n] = (share, scr)
    st['entry_id'] = st['entry_id'].astype(int)
    st['date'] = pd.to_datetime(st['date'])
    out = []
    for gid, gtype, gdate, rtp, turnover, pots, payouts, race_ids, legs in games:
        out.append({'id': gid, 'type': gtype, 'date': pd.Timestamp(gdate), 'rtp': (rtp or 65) / 100,
                    'turnover': (turnover or 0) / 100,
                    'pots': {int(k): v / 100 for k, v in (pots or {}).items()},
                    'payouts': {int(k): v for k, v in (payouts or {}).items()},
                    'races': list(race_ids), 'winners': [set((lg or {}).get('winners') or []) for lg in (legs or [])],
                    'shares': [shares.get((gid, i + 1), {}) for i in range(len(race_ids))]})
    return st, out


def tilted(st, tilt: dict) -> dict:
    """xedge's chances from its stored tilts (entry id -> tilt, any scale per
    race): the closing odds' shares times the tilt, renormalised per race."""
    d = st[~st['scr'] & (st['o'] > 1) & st['entry_id'].isin(tilt)]
    inv = 1 / d['o']
    p = inv / inv.groupby(d['race']).transform('sum') * d['entry_id'].map(tilt)
    p = p / p.groupby(d['race']).transform('sum')
    return dict(zip(d['entry_id'].astype(int).tolist(), p.astype(float).tolist()))


# ── helpers ──────────────────────────────────────────────────────────────

def _roi_stats(cost, ret, half):
    """ROI, bootstrap 90% interval, the halves' ROI, ROI without the top return."""
    cost, ret = np.asarray(cost, float), np.asarray(ret, float)
    c = cost.sum()
    if c <= 0:
        return None
    rng = np.random.default_rng(7)
    idx = rng.integers(0, len(cost), (N_BOOT, len(cost)))
    bc = cost[idx].sum(1)
    br = np.where(bc > 0, ret[idx].sum(1) / np.maximum(bc, 1e-12) - 1, np.nan)
    h = np.asarray(half, bool)

    def r(m):
        cc = cost[m].sum()
        return round(float(ret[m].sum() / cc - 1), 4) if cc > 0 else None

    return {'cost': round(float(c), 2), 'ret': round(float(ret.sum()), 2),
            'roi': round(float(ret.sum() / c - 1), 4),
            'lo': round(float(np.nanquantile(br, 0.05)), 4), 'hi': round(float(np.nanquantile(br, 0.95)), 4),
            'roi_h1': r(~h), 'roi_h2': r(h),
            'roi_wo_top': round(float((ret.sum() - ret.max()) / c - 1), 4)}


def harville(p, k):
    """P(top k) per runner of one race from its win probabilities."""
    p = np.asarray(p, float)
    out = p.copy()
    if k < 2:
        return out
    a = p[:, None] * p[None, :] / np.maximum(1 - p[:, None], 1e-9)    # a[i, j]: i wins, j second
    np.fill_diagonal(a, 0.0)
    out = out + a.sum(0)
    if k >= 3:
        # b[i, m] = P(i 1st, m 2nd) / (1 - p_i - p_m): times p_j gives j 3rd
        b = a / np.maximum(1 - p[:, None] - p[None, :], 1e-9)
        np.fill_diagonal(b, 0.0)
        tot = b.sum()
        out = out + p * (tot - b.sum(0) - b.sum(1))
    return out


def _pick(p, numbers, c):
    """Numbers in descending p until their p sums to c."""
    order = np.argsort(-p)
    k = int(np.searchsorted(np.cumsum(p[order]), c - 1e-12) + 1)
    return set(numbers[order[:k]].tolist())


def _settle(g, sets, top_only=False):
    """(cost, return, rows, all right) of one flat system in a finished game."""
    rtp, T = g['rtp'], g['turnover']
    rows = int(np.prod([len(s) for s in sets]))
    cost = rows * row_price(g['type'], g['date'])
    poly = np.array([1.0])
    for s, win in zip(sets, g['winners']):
        a = 1 if s & win else 0
        poly = np.convolve(poly, [len(s) - a, a])                      # index = legs right
    tp = top_pool(g['type']) if top_only else None
    tiers = [tp['tier']] if tp else sorted(g['payouts'])
    ret = 0.0
    for t in tiers:
        mine = poly[t] if t < len(poly) else 0.0
        if not mine:
            continue
        info = g['payouts'].get(t) or {}
        sys_n = info.get('systems') or 0
        if info.get('jackpot') and sys_n:
            continue                                                   # below the minimum: rolled over
        pot = g['pots'].get(t, T * rtp)
        net = (info.get('payout') or 0) / 100 * sys_n if sys_n else pot
        if tp:
            net += cost * rtp
            units = mine * tp['factor']
        else:
            net += cost * (pot / T if T else rtp)
            units = mine
        ret += net / (sys_n + units) * units
    return cost, ret, rows, bool(poly[-1] > 0)


# ── evaluation ───────────────────────────────────────────────────────────

def _race_frame(st, probs):
    """Starters of races where every runner has a model probability."""
    d = st[~st['scr'] & (st['o'] > 1)].copy()
    d['p'] = d['entry_id'].map(probs)
    ok = d.groupby('race')['p'].transform(lambda s: s.notna().all())
    d = d[ok & (d.groupby('race')['race'].transform('size') >= 4)].copy()
    d['p'] /= d.groupby('race')['p'].transform('sum')
    inv = 1 / d['o']
    d['q'] = inv / inv.groupby(d['race']).transform('sum')
    d['won'] = (d['finish'] == 1).astype(float)
    d['ret'] = d['won'] * d['o']
    for c in ('p', 'q'):
        d['fav_' + c] = d.groupby('race')[c].rank(ascending=False, method='first') == 1
    return d.sort_values(['race', 'number'], kind='stable')


def _rule(d, stake, label, half_of):
    s = np.asarray(stake, float)
    if s.sum() <= 0:
        return None
    by = pd.DataFrame({'race': d['race'].values, 'c': s, 'r': s * d['_ret'].values,
                       'h': half_of(d)}).groupby('race').agg(c=('c', 'sum'), r=('r', 'sum'), h=('h', 'first'))
    by = by[by['c'] > 0]
    stats = _roi_stats(by['c'], by['r'], by['h'])
    hit = float((d['_ret'].values[s > 0] > 0).mean())
    return {'rule': label, 'bets': int((s > 0).sum()), 'hit': round(hit, 4), **stats}


def evaluate(st, games, probs: dict, coupons: list | None = None) -> dict:
    """Win, place and marking-pool results for one model's probabilities
    (entry id -> p), with market / public references on the same races.
    `coupons`, if given, collects every marking-game system that was settled:
    (game id, source, rule, rows, cost, return, legs right, picks per leg)."""
    d = _race_frame(st, probs)
    if d.empty:
        return {}
    mid = d['date'].min() + (d['date'].max() - d['date'].min()) / 2

    def half_of(x):
        return (x['date'] >= mid).values

    out = {'from': str(d['date'].min().date()), 'to': str(d['date'].max().date()),
           'races': int(d['race'].nunique())}

    # win
    d['_ret'] = d['ret']
    win = {'model': [], 'market': []}
    for e in WIN_EDGES:
        win['model'].append(_rule(d, d['p'] * d['o'] > e, f'p × odds > {e:g}', half_of))
    win['model'].append(_rule(d, d['fav_p'], 'favourite', half_of))
    win['model'].append(_rule(d, d['fav_p'] & ~d['fav_q'], 'favourite ≠ market favourite', half_of))
    win['model'].append(_rule(d, d['p'], 'stake ∝ p, every runner', half_of))
    win['market'].append(_rule(d, np.ones(len(d)), 'every runner', half_of))
    win['market'].append(_rule(d, d['fav_q'], 'favourite', half_of))
    win['market'].append(_rule(d, d['q'], 'stake ∝ p, every runner', half_of))
    out['win'] = {k: [r for r in v if r] for k, v in win.items()}

    # place
    pl = d[(d['k'] >= 2) & (d['pmin'] > 1)].copy()
    pl = pl[pl.groupby('race')['race'].transform('size') >= 5]
    if not pl.empty:
        for c in ('p', 'q'):
            pl['pp_' + c] = np.concatenate([harville(g[c].values, int(g['k'].iloc[0]))
                                            for _, g in pl.groupby('race', sort=False)])
        pl['_ret'] = pl['ppaid'].fillna(0.0)
        place = {'model': [], 'market': []}
        for src, key in (('p', 'model'), ('q', 'market')):
            for e in PLACE_EDGES:
                place[key].append(_rule(pl, pl['pp_' + src] * pl['pmin'] > e,
                                        f'top-k chance × place odds > {e:g}', half_of))
            place[key].append(_rule(pl, pl['fav_' + src], 'favourite', half_of))
        place['market'].append(_rule(pl, np.ones(len(pl)), 'every runner', half_of))
        out['place'] = {k: [r for r in v if r] for k, v in place.items()}

    # marking pools
    by_race = {r: g for r, g in d.groupby('race')}
    for gtype in ROW_PRICE:
        recs = []
        for g in games:
            if (g['type'] != gtype or len(g['winners']) != len(g['races'])
                    or not all(g['winners'])):
                continue
            legs = []
            for i, rid in enumerate(g['races']):
                f = by_race.get(rid)
                sh = g['shares'][i]
                if f is None or not sh:
                    legs = None
                    break
                pub = np.array([(sh.get(n) or (0, 0))[0] or 0.0 for n in f['number']]) + 1e-4
                legs.append((f['number'].values, f['p'].values, f['q'].values, pub / pub.sum()))
            if not legs:
                continue
            tp = top_pool(gtype)
            for src, col in (('model', 1), ('market', 2), ('public', 3)):
                for name, sets in _strategies(legs, col).items():
                    right = sum(1 for s, win in zip(sets, g['winners']) if s & win)
                    for top in ((False, True) if tp and name in ('cover 70%', 'cover 80%') else (False,)):
                        cost, ret, rows, hit = _settle(g, sets, top)
                        rule = name + (f" · {tp['label']}" if top else '')
                        recs.append((src, rule, g['date'] >= mid, cost, ret, rows, hit))
                        if coupons is not None:
                            coupons.append((g['id'], src, rule, rows, cost, ret, right,
                                            [sorted(int(n) for n in s) for s in sets]))
        if not recs:
            continue
        R = pd.DataFrame(recs, columns=['src', 'rule', 'h', 'cost', 'ret', 'rows', 'hit'])
        block = {'games': int(R.groupby(['src', 'rule']).size().max())}
        for src, sub in R.groupby('src', sort=False):
            block[src] = [{'rule': rule, 'rows': int(x['rows'].mean()), 'hit': round(float(x['hit'].mean()), 4),
                           **_roi_stats(x['cost'], x['ret'], x['h'])}
                          for rule, x in sub.groupby('rule', sort=False)]
        out[gtype] = block
    return out


def banker_leg(legs, col):
    """The leg to bank on: among legs whose favourite (by `col`) has at least
    35%, the one where the public underbets it most; the strongest favourite
    breaks ties (the public's own picks are never underbet by the public)."""
    p1 = np.array([lg[col].max() for lg in legs])
    under = np.array([lg[col].max() / lg[3][np.argmax(lg[col])] for lg in legs])
    return int(np.argmax(np.where(p1 >= 0.35, 10 + under, 0) + 1e-3 * p1))


def _strategies(legs, col):
    out = {f'cover {c:.0%}': [_pick(lg[col], lg[0], c) for lg in legs] for c in COVER}
    b = banker_leg(legs, col)
    out['banker + cover 80%'] = [{int(lg[0][np.argmax(lg[col])])} if i == b else _pick(lg[col], lg[0], 0.8)
                                 for i, lg in enumerate(legs)]
    return out
