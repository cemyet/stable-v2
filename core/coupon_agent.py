"""The coupon agent's policies, shared by scripts/build_coupon_agent (fitting,
testing) and the game page (today's coupon). numpy only.

Every coupon × multiplier of a game's menu gets a linear score over its
features: what it's expected to return under the agent's own chances and
under the alternative belief, its stake against the bankroll, the
alternative's chance of any payout, and its shape. The best is played if it
scores above zero, else the game is passed. A coupon may cost at most
STAKE_CAP of the bankroll.

There are two policies with the same features. 'live' plays xedge's chances
against the win odds, once every leg's win pool holds money; 'forecast'
plays xwin against xodds before that. Without the market the forecasts'
expected returns run several times higher than what such coupons return,
so each policy is fitted on its own inputs.
"""
from __future__ import annotations

import datetime as dt

import numpy as np

from core import coupon_sim as CS
from core.reduce_system import top_pool

FEATS = ['ev_alt', 'ev_own', 'stake', 'phit', 'bankers', 'cover', 'weighted', 'top', 'units']
SHAPE = ['bankers', 'cover', 'weighted', 'top', 'units']
STAKE_CAP = 0.25
BANKROLL = 10_000.0
LUCKY = 3
MULTS = np.asarray(CS.MULTS, float)
# The game page's coupon scale (stake units × copies): 0.05 … 2 in 5% steps, then 3, 5, 10.
SCALES = np.round(np.r_[np.arange(1, 41) / 20, 3, 5, 10], 2)


def item_arrays(items) -> dict:
    """Shape columns of the menu's items (Item objects or their dicts)."""
    get = (lambda it, k: it[k]) if items and isinstance(items[0], dict) else (lambda it, k: getattr(it, k))
    return {'bankers': np.array([get(it, 'bankers') for it in items], float),
            'cover': np.array([get(it, 'cover') for it in items], float),
            'weighted': np.array([get(it, 'weight') != 'flat' for it in items], float),
            'top': np.array([bool(get(it, 'top')) for it in items], float),
            'units': np.array([get(it, 'units') for it in items], float)}


def parts(cost, ev_own, ev_alt, phit_alt, units, bankroll=BANKROLL):
    """The menu-dependent features, kept apart so a policy scores every game
    at once without a [G, I, M, F] tensor. cost [..., I] at ×1, ev_* [..., I, M],
    phit_alt [..., I], units [I]; NaN cost = the coupon isn't on this game's menu."""
    c = np.asarray(cost, float)[..., None] * MULTS
    scale = np.isin(np.round(np.asarray(units, float)[:, None] * MULTS, 2), SCALES)
    ok = np.isfinite(c) & (c <= STAKE_CAP * bankroll) & scale
    c = np.where(ok, c, 1.0)
    lg = lambda x, lo: np.log(np.clip(np.nan_to_num(np.asarray(x, float), nan=lo), lo, None))  # noqa: E731
    return {'ev_alt': (lg(ev_alt, 1e-6) - np.log(c)).astype(np.float32),
            'ev_own': (lg(ev_own, 1e-6) - np.log(c)).astype(np.float32),
            'stake': np.log(c / bankroll).astype(np.float32),
            'phit': np.minimum(lg(phit_alt, 1e-4), 0.0).astype(np.float32),
            'ok': ok}


def scores(theta, P, shape, type_ix):
    """Score per coupon × multiplier ([..., I, M]); type_ix is the game type's bias index."""
    theta = np.asarray(theta, np.float32)
    w, bias = theta[:len(FEATS)], theta[len(FEATS):]
    s = w[0] * P['ev_alt'] + w[1] * P['ev_own'] + w[2] * P['stake'] + w[3] * P['phit'][..., None]
    item = sum(w[4 + j] * shape[k] for j, k in enumerate(SHAPE)).astype(np.float32)
    s = s + item[:, None] + bias[np.asarray(type_ix)][..., None, None]
    return np.where(P['ok'], s, np.float32(-np.inf))


def choose(theta, P, shape, type_ix):
    """(item index, multiplier index, score) of one game's best coupon, or None to pass."""
    s = scores(theta, P, shape, type_ix)
    i, k = np.unravel_index(int(np.argmax(s)), s.shape)
    if not np.isfinite(s[i, k]) or s[i, k] <= 0:
        return None
    return int(i), int(k), float(s[i, k])


# ── fitting: every game at once (the bankroll is fixed, so games don't interact) ──

def play(theta, P, shape, type_ix, cost, ret):
    """Per game: the chosen item and multiplier (-1 = pass), its cost and
    return. P from parts() over [G, I], cost [G, I], ret [G, I, M]."""
    s = scores(theta, P, shape, type_ix)
    G = s.shape[0]
    flat = s.reshape(G, -1)
    j = np.argmax(flat, axis=1)
    best = flat[np.arange(G), j]
    i, k = np.unravel_index(j, s.shape[1:])
    on = np.isfinite(best) & (best > 0)
    c = np.where(on, np.asarray(cost)[np.arange(G), i] * MULTS[k], 0.0)
    r = np.where(on, np.asarray(ret)[np.arange(G), i, k], 0.0)
    return np.where(on, i, -1), np.where(on, k, -1), np.nan_to_num(c), np.nan_to_num(r)


def growth(cost, ret, bankroll=BANKROLL):
    """Log growth per game of a fixed bankroll (a game can't lose more than all of it)."""
    return np.log1p(np.maximum((ret - cost) / bankroll, -0.999))


def objective(theta, P, shape, type_ix, cost, ret, robust=True):
    """Log growth over the games, the LUCKY best left out when `robust`, so a
    policy can't be chosen for the jackpots it happened to catch."""
    _, _, c, r = play(theta, P, shape, type_ix, cost, ret)
    g = growth(c, r)[c > 0]
    if not len(g):
        return 0.0
    return float(g.sum() - (np.sort(g)[-LUCKY:].clip(0, None).sum() if robust else 0.0))


def cem(P, shape, type_ix, cost, ret, n_types, log=print, iters=40, pop=64, elite=8, seed=0):
    """The cross-entropy method on the robust log growth: (best weights, final mean)."""
    rng = np.random.default_rng(seed)
    d = len(FEATS) + n_types
    mu = np.zeros(d)
    mu[0] = 2.0          # start as an EV filter: play when the market expects ~1.3× the stake
    mu[len(FEATS):] = -0.5
    sd = np.ones(d)
    best = (-np.inf, mu.copy())
    for t in range(iters):
        th = mu + sd * rng.standard_normal((pop, d))
        th[0] = mu
        sc = np.array([objective(x, P, shape, type_ix, cost, ret) for x in th])
        top = np.argsort(-sc)[:elite]
        if sc[top[0]] > best[0]:
            best = (float(sc[top[0]]), th[top[0]].copy())
        mu = 0.7 * th[top].mean(axis=0) + 0.3 * mu
        sd = 0.7 * th[top].std(axis=0) + 0.3 * sd + 0.02
        if t % 5 == 0 or t == iters - 1:
            log(f"    CEM {t + 1}/{iters}: best robust log growth {sc[top[0]]:+.4f} · elite mean {sc[top].mean():+.4f}")
    return best[1], mu


# ── today's coupon ───────────────────────────────────────────────────────

WIN_POOL_LIVE = 50_000       # kr in every leg's win pool before the live policy plays


def _filled(x):
    """Chances with the missing ones at half the smallest, summing to one."""
    x = np.asarray(x, float)
    ok = np.isfinite(x) & (x > 0)
    x = np.where(ok, x, 0.5 * x[ok].min())
    return x / x.sum()


def live_game(payload, preds, price, tier_share, turnover_est, rtp=0.65):
    """An open marking game as core.coupon_sim sees it, from ATG's game
    payload and the stored predictions ((atg race id, number) -> {'xwin',
    'xodds', 'xedge' tilt}). Once every leg's win pool holds WIN_POOL_LIVE
    the game is 'live': xedge over the win odds, against the odds. Before
    that it is 'forecast': xwin against xodds. The public's shares are the
    game's stake distribution so far (the alternative belief until there
    is one).

    Returns (game, mode, alternative belief per leg, per-leg facts);
    ValueError when a leg has nothing to play on."""
    gt = str(payload.get('type') or '').upper()
    pool = (payload.get('pools') or {}).get(gt) or {}
    found = []
    for i, race in enumerate(r for r in payload.get('races') or [] if isinstance(r, dict)):
        rid = race.get('id')
        win_pool = (((race.get('pools') or {}).get('vinnare') or {}).get('turnover') or 0) / 100
        nums, odds, dist = [], [], []
        for s in race.get('starts') or []:
            if s.get('number') is None or s.get('scratched') or (s.get('horse') or {}).get('scratched'):
                continue
            sp = s.get('pools') or {}
            o = (sp.get('vinnare') or {}).get('odds')
            b = (sp.get(gt) or {}).get('betDistribution')
            nums.append(int(s['number']))
            odds.append(o / 100 if isinstance(o, (int, float)) and o > 100 else np.nan)
            dist.append(b / 10_000 if isinstance(b, (int, float)) else np.nan)
        if len(nums) < 2:
            raise ValueError(f"leg {i + 1} has fewer than two starters")
        nums, odds, dist = np.array(nums), np.array(odds), np.array(dist)
        pr = [preds.get((rid, int(n))) or {} for n in nums]
        got = {k: np.array([d.get(k, np.nan) for d in pr], float) for k in ('xwin', 'xodds', 'xedge')}
        live = win_pool >= WIN_POOL_LIVE and np.isfinite(odds).all() and np.isfinite(got['xedge']).mean() >= 0.75
        fc = np.isfinite(got['xwin']).mean() >= 0.75 and np.isfinite(got['xodds']).mean() >= 0.75
        found.append((i, rid, win_pool, nums, odds, dist, got, live, fc))
    if not found:
        raise ValueError('the game has no legs')
    mode = 'live' if all(f[7] for f in found) else 'forecast'
    legs, alt, facts = [], [], []
    for i, rid, win_pool, nums, odds, dist, got, live, fc in found:
        if mode == 'live':
            q = (1 / odds) / (1 / odds).sum()
            tilt = np.where(np.isfinite(got['xedge']), got['xedge'], np.nanmedian(got['xedge']))
            p, a = _filled(q * tilt), q
        elif fc:
            p, a = _filled(got['xwin']), _filled(got['xodds'])
        else:
            raise ValueError(f"leg {i + 1} has no model chances yet")
        streck = bool(np.isfinite(dist).all() and dist.sum() > 0)
        s = dist if streck else a
        s = (s + 1e-4) / (s + 1e-4).sum()
        legs.append(CS.Leg(nums, p, a, s, np.zeros(len(nums), bool)))
        alt.append(a)
        facts.append({'leg': i + 1, 'raceId': rid, 'odds': bool(live), 'winPool': round(win_pool),
                      'starters': len(nums), 'streck': streck})
    turnover = (pool.get('turnover') or 0) / 100
    T = max(turnover, float(turnover_est or 0))
    tiers = np.array(sorted(int(t) for t in tier_share), int)
    pot = np.array([T * rtp * tier_share[t] for t in tiers])
    pot[-1] += float(pool.get('jackpotAmount') or 0)
    L = len(legs)
    kappa = np.array([CS.KAPPA.get((gt, L - int(t)), 1.0) for t in tiers])
    game = CS.Game(str(payload.get('id') or ''), gt, dt.datetime.now(), legs, price, rtp, T, tiers, pot.copy(),
                   np.zeros(len(tiers)), np.zeros(len(tiers), bool), pot, kappa, top_pool(gt))
    return game, mode, alt, {'legs': facts, 'turnover': round(turnover), 'turnover_used': round(T)}


def suggest(game, alt, policy, seed=0):
    """The policy's coupon for one game: every coupon of its menu expected
    under the agent's chances and the alternative, scored, the best kept
    (even when the agent itself would pass, so there's a coupon to start
    from). `policy` is one of the registered policies ({'types', 'theta'})."""
    types = list(policy['types'])
    if game.gtype not in types:
        raise ValueError(f"the agent wasn't trained on {game.gtype}")
    items = CS.items_for(game.gtype)
    shape = item_arrays(items)
    I, M = len(items), len(MULTS)
    cost = np.full(I, np.nan)
    ev = {b: np.full((I, M), np.nan) for b in ('own', 'alt')}
    phit = {b: np.full(I, np.nan) for b in ('own', 'alt')}
    cps = [None] * I
    smp = {'own': CS.Sampler(game, seed=seed), 'alt': CS.Sampler(game, seed=seed, belief=alt)}
    for top in (False, True):
        js = [j for j, it in enumerate(items) if it.top == top]
        if not js:
            continue
        for j in js:
            cps[j] = CS.coupon(game, items[j])
            cost[j] = cps[j][1] * game.price
        for b, s in smp.items():
            e, h = s.expected_many([cps[j][0] for j in js], [cps[j][1] for j in js], top=top)
            ev[b][js], phit[b][js] = e, h
    P = parts(cost, ev['own'], ev['alt'], phit['alt'], shape['units'], policy.get('bankroll', BANKROLL))
    s = scores(policy['theta'], P, shape, types.index(game.gtype))
    i, k = np.unravel_index(int(np.argmax(s)), s.shape)
    if not np.isfinite(s[i, k]):
        raise ValueError('no coupon fits the stake cap')
    it, (W, U, rows) = items[i], cps[i]
    m = float(MULTS[k])
    legs = []
    for lg, w in zip(game.legs, W):
        on = np.nonzero(w > 0)[0]
        on = on[np.argsort(-lg.p[on], kind='stable')]
        legs.append([{'number': int(lg.numbers[j]), 'weight': round(float(w[j]) * 100, 4),
                      'p': round(float(lg.p[j]), 4), 'alt': round(float(lg.q[j]), 4),
                      'streck': round(float(lg.s[j]), 4)} for j in on])
    return {'item': {'bankers': it.bankers, 'rule': it.rule, 'cover': it.cover, 'weight': it.weight,
                     'units': it.units, 'top': it.top, 'label': it.label},
            'mult': int(m), 'scale': round(it.units * m, 2), 'rows': int(rows),
            'cost': round(float(cost[i] * m), 2),
            'ev': {'own': round(float(ev['own'][i, k]), 2), 'alt': round(float(ev['alt'][i, k]), 2)},
            'phit': {'own': round(float(phit['own'][i]), 4), 'alt': round(float(phit['alt'][i]), 4)},
            'score': round(float(s[i, k]), 4), 'play': bool(s[i, k] > 0),
            'menu': int(np.isfinite(s).sum()), 'legs': legs}
