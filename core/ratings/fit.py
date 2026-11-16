"""Backtest and overall weights.

Every runner in the backtest window carries its pre-race snapshot. The ten
mains are rolled up exactly as the card does it, then a conditional logit
(one winner per race) is fitted on the earlier part of the window to learn
how much each main says about who wins. Those coefficients become the
overall's weights; pedigree keeps a fixed share because its source (the
current BLUP index) already knows how the horse raced later.
"""

from __future__ import annotations

import math
import random

from core.ratings import keys as K
from core.ratings.compose import fresh_value, horse_values, pair_value, rollup
from core.ratings.engine import fresh_bucket

FIT_MAINS = [m.key for m in K.MAINS if m.key != 'pedigree']
PEDIGREE_SHARE = 0.12


def runner_values(run, fresh_table, ped):
    hid, won, odds, pelo, snap, person, days, pair, breed, starts = run
    vals = horse_values(snap[0], snap[1]) if snap else {}
    tv, tc, sv, sc, dv, dc = person
    vals['trainer'] = (tv or None, tc / 100 if tv else 0.0)
    vals['stable'] = (sv or None, sc / 100 if sv else 0.0)
    vals['driver'] = (dv or None, dc / 100 if dv else 0.0)
    vals['fresh'] = fresh_value(fresh_table, fresh_bucket(days)) if days is not None else (None, 0.0)
    vals['pairing'] = pair_value(*pair)
    p = ped.get(hid)
    if p:
        for k, v, c in zip(K.PEDIGREE_KEYS, p[0], p[1]):
            vals[k] = (v or None, c / 100 if v else 0.0)
    return vals


def _solve(A, b):
    n = len(b)
    M = [row[:] + [b[i]] for i, row in enumerate(A)]
    for c in range(n):
        p = max(range(c, n), key=lambda r: abs(M[r][c]))
        M[c], M[p] = M[p], M[c]
        if abs(M[c][c]) < 1e-12:
            continue
        for r in range(n):
            if r != c:
                f = M[r][c] / M[c][c]
                if f:
                    for k in range(c, n + 1):
                        M[r][k] -= f * M[c][k]
    return [M[i][n] / M[i][i] if abs(M[i][i]) > 1e-12 else 0.0 for i in range(n)]


def clogit(races, dim, iters=8, l2=1.0, log=None):
    """races: [(X, y)] with X a list of feature vectors and y the winner's index."""
    beta = [0.0] * dim
    for it in range(iters):
        g = [0.0] * dim
        H = [[0.0] * dim for _ in range(dim)]
        ll = 0.0
        for X, y in races:
            s = [sum(b * x for b, x in zip(beta, xi)) for xi in X]
            m = max(s)
            e = [math.exp(v - m) for v in s]
            Z = sum(e)
            p = [v / Z for v in e]
            ll += s[y] - m - math.log(Z)
            mean = [sum(p[i] * X[i][k] for i in range(len(X))) for k in range(dim)]
            xy = X[y]
            for k in range(dim):
                g[k] += xy[k] - mean[k]
            for i, xi in enumerate(X):
                pi = p[i]
                for k in range(dim):
                    pk = pi * xi[k]
                    if pk:
                        row = H[k]
                        for l in range(k, dim):
                            row[l] += pk * xi[l]
            for k in range(dim):
                row = H[k]
                for l in range(k, dim):
                    row[l] -= mean[k] * mean[l]
        for k in range(dim):
            for l in range(k):
                H[k][l] = H[l][k]
            g[k] -= l2 * beta[k]
            H[k][k] += l2
        step = _solve(H, g)
        beta = [b + s for b, s in zip(beta, step)]
        if log:
            log(f"[fit] iter {it + 1}: log-lik/race {ll / max(1, len(races)):.4f}")
        if max(abs(s) for s in step) < 1e-4:
            break
    return beta


def _ll(probs, y):
    return -math.log(max(1e-9, probs[y]))


def evaluate(races, score):
    """Mean log-loss and top-1 accuracy of a scorer: races -> probabilities."""
    ll = 0.0
    hit = 0
    n = 0
    for race in races:
        p = score(race)
        if p is None:
            continue
        y = race['y']
        ll += _ll(p, y)
        hit += 1 if max(range(len(p)), key=lambda i: p[i]) == y else 0
        n += 1
    return {'races': n, 'logloss': round(ll / n, 4) if n else None,
            'top1': round(hit / n, 4) if n else None}


def softmax(s):
    m = max(s)
    e = [math.exp(v - m) for v in s]
    Z = sum(e)
    return [v / Z for v in e]


def prepare(bt, fresh_table, ped):
    """Backtest races -> per-runner mains, starts and market data."""
    out = []
    for t, runners in bt:
        if len(runners) < 5:
            continue
        wins = [i for i, r in enumerate(runners) if r[1]]
        if len(wins) != 1:
            continue
        rows = []
        for run in runners:
            vals = runner_values(run, fresh_table, ped)
            mains, _ = rollup(vals, K.DEFAULT_WEIGHTS, run[9])
            rows.append({'mains': mains, 'starts': run[9], 'debut': run[4] is None,
                         'odds': run[2], 'pelo': run[3], 'breed': run[8]})
        out.append({'t': t, 'y': wins[0], 'rows': rows})
    return out


def overall_of(row, weights):
    num = den = 0.0
    for key, w in weights.items():
        if key == 'pedigree':
            w *= K.PEDIGREE_K / (row['starts'] + K.PEDIGREE_K)
        v, c = row['mains'].get(key, (None, 0.0))
        if v is None or c <= 0:
            continue
        num += w * c * v
        den += w * c
    return num / den if den > 0 else None


def _features(row):
    zs = {}
    num = den = 0.0
    for k in FIT_MAINS:
        v, c = row['mains'][k]
        if v is not None and c > 0:
            z = (v - K.SCALE_MID) / K.SCALE_SD
            zs[k] = (z, c)
            w = K.DEFAULT_WEIGHTS[k]
            num += w * c * z
            den += w * c
    zbar = num / den if den > 0 else 0.0
    x = []
    for k in FIT_MAINS:
        z, c = zs.get(k, (zbar, 0.0))
        x.append(c * z + (1 - c) * zbar)
    x.append(1.0 if row['debut'] else 0.0)
    return x


def fit_weights(races, *, holdout_t: int, log=print, max_fit=30000):
    fit_set = [r for r in races if r['t'] < holdout_t]
    hold = [r for r in races if r['t'] >= holdout_t]
    random.seed(7)
    sample = fit_set if len(fit_set) <= max_fit else random.sample(fit_set, max_fit)
    data = [([_features(row) for row in r['rows']], r['y']) for r in sample]
    log(f"[fit] conditional logit on {len(data):,} races ({len(fit_set):,} in the fit window, "
        f"{len(hold):,} held out)")
    beta = clogit(data, len(FIT_MAINS) + 1, log=log)
    coef = dict(zip(FIT_MAINS + ['debut'], beta))
    pos = {k: max(coef[k], 0.0) for k in FIT_MAINS}
    tot = sum(pos.values()) or 1.0
    fitted = {k: round((1 - PEDIGREE_SHARE) * pos[k] / tot, 4) for k in FIT_MAINS}
    fitted['pedigree'] = PEDIGREE_SHARE
    # The overall describes the horse, not only who wins today: the fit
    # alone zeroes mains that add nothing beyond class and form for picking
    # winners, so the card uses an even blend with the hand-set weights.
    weights = {k: round(0.5 * fitted[k] + 0.5 * K.DEFAULT_WEIGHTS[k], 4) for k in fitted}

    def calib(ws):
        d = []
        for r in sample:
            X = []
            for row in r['rows']:
                o = overall_of(row, ws)
                X.append([((o if o is not None else K.SCALE_MID) - K.SCALE_MID) / K.SCALE_SD,
                          1.0 if row['debut'] or o is None else 0.0])
            d.append((X, r['y']))
        return clogit(d, 2, iters=10, l2=0.01)

    g_fit, d_fit = calib(fitted)
    g_bl, d_bl = calib(weights)
    g_def, d_def = calib(K.DEFAULT_WEIGHTS)
    log(f"[fit] overall temperature: fitted γ={g_fit:.3f}, blend γ={g_bl:.3f}, default γ={g_def:.3f}")

    elo_d = [([[math.log(max(1e-9, row['pelo']))] for row in r['rows']], r['y']) for r in sample]
    g_elo = clogit(elo_d, 1, iters=10, l2=0.01)[0]

    def by_overall(ws, gam, dlt):
        def score(r):
            s = []
            for row in r['rows']:
                o = overall_of(row, ws)
                deb = 1.0 if row['debut'] or o is None else 0.0
                s.append(gam * ((o if o is not None else K.SCALE_MID) - K.SCALE_MID) / K.SCALE_SD + dlt * deb)
            return softmax(s)
        return score

    def market(r):
        inv = [1 / row['odds'] if row['odds'] and row['odds'] > 1.0 else 0.0 for row in r['rows']]
        if min(inv) <= 0:
            return None
        Z = sum(inv)
        return [v / Z for v in inv]

    with_mkt = [r for r in hold if market(r) is not None]
    elo_score = lambda r: softmax([g_elo * math.log(max(1e-9, x['pelo'])) for x in r['rows']])  # noqa: E731
    report = {
        'holdout_from': holdout_t,
        'coef': {k: round(v, 4) for k, v in coef.items()},
        'weights_fitted': fitted,
        'gamma': round(g_bl, 4), 'debut': round(d_bl, 4), 'gamma_elo': round(g_elo, 4),
        'holdout': {
            'uniform': evaluate(hold, lambda r: [1 / len(r['rows'])] * len(r['rows'])),
            'elo': evaluate(hold, elo_score),
            'overall_default': evaluate(hold, by_overall(K.DEFAULT_WEIGHTS, g_def, d_def)),
            'overall_fitted': evaluate(hold, by_overall(fitted, g_fit, d_fit)),
            'overall_blend': evaluate(hold, by_overall(weights, g_bl, d_bl)),
        },
        'holdout_with_market': {
            'market': evaluate(with_mkt, market),
            'elo': evaluate(with_mkt, elo_score),
            'overall_fitted': evaluate(with_mkt, by_overall(fitted, g_fit, d_fit)),
            'overall_blend': evaluate(with_mkt, by_overall(weights, g_bl, d_bl)),
        },
        'coverage': {k: round(sum(1 for r in hold for row in r['rows'] if row['mains'][k][0] is not None)
                               / max(1, sum(len(r['rows']) for r in hold)), 3) for k in K.MAIN_BY_KEY},
        'mean_conf': {k: round(sum(row['mains'][k][1] for r in hold for row in r['rows'])
                                / max(1, sum(len(r['rows']) for r in hold)), 3) for k in K.MAIN_BY_KEY},
    }
    # one main at a time: how much each says on its own
    solo = {}
    for i, k in enumerate(FIT_MAINS):
        d1 = [([[x[i]] for x in X], y) for X, y in data[:8000]]
        solo[k] = round(clogit(d1, 1, iters=6, l2=0.01)[0], 4)
    report['solo_coef'] = solo
    return weights, report
