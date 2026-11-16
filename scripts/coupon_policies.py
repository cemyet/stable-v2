"""Coupon policies over the menus of scripts/coupon_lab.py.

static  every fixed recipe (coupon × multiplier × budget) on the training
        games, the best per question carried to the test games untouched:
        how many bankers and by which rule, flat or weighted rows, how many
        copies, and an expected-value filter (play only when the blend
        expects the coupon to return at least τ × its cost).
agent   a first try at learning the betting policy: a linear score over
        each coupon's features (expected ROI under the blend and under the
        source, stake against the bankroll, chance of any payout, its
        shape), the best-scoring coupon played if its score clears zero.
        Its weights are found by the cross-entropy method on the training
        games' log growth, then frozen and run over the test games.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from core import coupon_sim as CS
from core.pools import _roi_stats
from scripts.coupon_lab import OUT, SPLIT, load_menu

BANKROLL = 10_000.0
BUDGETS = (100, 250, 500, 1000, 2500, np.inf)
TAUS = (0.0, 0.9, 1.0, 1.2, 1.5, 2.0)
MULTS = np.asarray(CS.MULTS, float)


def _drawdown(cost, ret):
    eq = np.cumsum(ret - cost)
    return float((np.maximum.accumulate(np.concatenate([[0.0], eq]))[1:] - eq).max()) if len(eq) else 0.0


def _growth(cost, ret):
    """Log growth per game of a fixed bankroll (a game can't lose more than all of it)."""
    return np.log1p(np.maximum((ret - cost) / BANKROLL, -0.999))


def _summary(cost, ret, dates, half):
    """ROI and friends for one played sequence (cost / return per game)."""
    played = cost > 0
    if played.sum() == 0:
        return None
    c, r = cost[played], ret[played]
    st = _roi_stats(c, r, half[played])
    return {'games': int(played.sum()), 'hit': round(float((r > 0).mean()), 3),
            'profit': round(float(r.sum() - c.sum())), 'avg_cost': round(float(c.mean()), 1),
            'max_drawdown': round(_drawdown(c, r)),
            'log_growth': round(float(_growth(c, r).sum()), 4), **st}


def _recipe(A, gi, i, k, budget, tau, belief='ev_blend'):
    """Cost / return per game of one fixed recipe over games gi."""
    c = A['cost'][gi, i] * MULTS[k]
    ok = np.isfinite(c) & (c <= budget)
    if tau > 0:
        ok &= A[belief][gi, i, k] >= tau * np.where(ok, c, 1.0)
    cost = np.where(ok, c, 0.0)
    ret = np.where(ok, A['ret'][gi, i, k], 0.0)
    return np.nan_to_num(cost), np.nan_to_num(ret)


def static_report(args, log):
    out = {}
    for src in args.sources:
        A, meta, items = load_menu(src)
        it = items['items']
        res = {}
        for gt, sub in meta.groupby('type'):
            gi = sub.index.values
            tr, te = gi[sub['date'] < SPLIT], gi[sub['date'] >= SPLIT]
            if len(tr) < 15 or len(te) < 10:
                continue
            half_tr = (meta.loc[tr, 'date'] >= meta.loc[tr, 'date'].median()).values
            half_te = (meta.loc[te, 'date'] >= meta.loc[te, 'date'].median()).values
            valid = [i for i in range(len(it)) if np.isfinite(A['cost'][gi, i]).all()]
            cand = []
            n_min = max(20, 0.2 * len(tr))
            for i in valid:
                for k in range(len(MULTS)):
                    for b in BUDGETS:
                        for tau in TAUS:
                            c, r = _recipe(A, tr, i, k, b, tau)
                            n = int((c > 0).sum())
                            if n < n_min:
                                continue
                            g = _growth(c, r)
                            ct, rt = _recipe(A, te, i, k, b, tau)
                            cand.append((i, k, b, tau, n, c.sum(), r.sum(), float(g.sum()),
                                         float(g.sum() - g.max()), int((ct > 0).sum()), ct.sum(), rt.sum(),
                                         float(_growth(ct, rt).sum())))
            C = pd.DataFrame(cand, columns=['i', 'k', 'budget', 'tau', 'n', 'cost', 'ret', 'lg_all', 'lg',
                                            'n_te', 'cost_te', 'ret_te', 'lg_te'])
            C['roi'] = C['ret'] / C['cost'] - 1
            C['roi_te'] = C['ret_te'] / C['cost_te'].where(C['cost_te'] > 0) - 1
            for col in ('bankers', 'rule', 'cover', 'weight', 'units', 'top'):
                C[col] = [it[i][col] for i in C['i']]
            C['mult'] = MULTS[C['k'].values]

            def carry(row):
                c, r = _recipe(A, te, int(row.i), int(row.k), row.budget, row.tau)
                c0, r0 = _recipe(A, tr, int(row.i), int(row.k), row.budget, row.tau)
                return {'recipe': it[int(row.i)]['label'], 'mult': int(row.mult),
                        'budget': None if not np.isfinite(row.budget) else int(row.budget), 'tau': row.tau,
                        'train': _summary(c0, r0, None, half_tr), 'test': _summary(c, r, None, half_te)}

            q = {}
            # the best by training log growth, overall and per question
            q['best'] = carry(C.loc[C['lg'].idxmax()])
            q['by_bankers'] = [carry(s.loc[s['lg'].idxmax()]) for _, s in C.groupby('bankers')]
            q['by_rule'] = [carry(s.loc[s['lg'].idxmax()]) for _, s in C[C['bankers'] > 0].groupby('rule')]
            q['by_weight'] = [carry(s.loc[s['lg'].idxmax()]) for _, s in C.groupby(['weight', 'units'])]
            q['by_mult'] = [carry(s.loc[s['lg'].idxmax()]) for _, s in C.groupby('mult')]
            q['by_tau'] = [carry(s.loc[s['lg'].idxmax()]) for _, s in C.groupby('tau')]
            if C['top'].any():
                q['by_top'] = [carry(s.loc[s['lg'].idxmax()]) for _, s in C.groupby('top')]
            # the plain reference: cover 70%, flat, no banker, every game
            ref = C[(C['bankers'] == 0) & (C['cover'] == 0.7) & (C['weight'] == 'flat') & (C['mult'] == 1)
                    & ~C['top'] & ~np.isfinite(C['budget']) & (C['tau'] == 0)]
            if len(ref):
                q['plain'] = carry(ref.iloc[0])
            # whole families, nothing selected: every coupon at ×1, every game
            plain = C[(C['tau'] == 0) & ~np.isfinite(C['budget']) & (C['mult'] == 1)]
            q['families'] = {
                col: [{'value': v if not isinstance(v, (np.floating, np.integer)) else v.item(),
                       'recipes': int(len(s)),
                       'median_roi_train': round(float(s['roi'].median()), 4),
                       'median_roi_test': round(float(s['roi_te'].median()), 4),
                       'pooled_roi_test': round(float(s['ret_te'].sum() / s['cost_te'].sum() - 1), 4),
                       'share_positive_test': round(float((s['roi_te'] > 0).mean()), 3)}
                      for v, s in plain.groupby(col)]
                for col in ('bankers', 'rule', 'weight', 'cover', 'top')}
            res[gt] = {'games_train': int(len(tr)), 'games_test': int(len(te)), **q}
            C.assign(type=gt, label=[it[i]['label'] for i in C['i']]).to_pickle(OUT / f"recipes_{src}_{gt}.pkl")
            b = q['best']
            log(f"[{src}] {gt}: best on train '{b['recipe']}' ×{b['mult']} budget {b['budget']} τ {b['tau']} · "
                f"train ROI {b['train']['roi']:+.1%} ({b['train']['games']}) → test "
                + (f"{b['test']['roi']:+.1%} ({b['test']['games']} games)" if b['test'] else 'not played'))
        out[src] = res
    path = OUT / 'static.json'
    path.write_text(json.dumps(out, indent=1, default=float))
    log(f"-> {path}")


# ── agent ────────────────────────────────────────────────────────────────

FEATS = ['ev_blend', 'ev_own', 'stake', 'phit', 'bankers', 'cover', 'weighted', 'top', 'units']


def _features(A, items, g, B):
    """[I, M, F] features of every coupon × multiplier of game g at bankroll B."""
    cost = A['cost'][g][:, None] * MULTS[None, :]
    ok = np.isfinite(cost) & (cost <= 0.25 * B)
    c = np.where(ok, cost, 1.0)
    f = np.stack([
        np.log(np.clip(A['ev_blend'][g], 1e-6, None) / c),
        np.log(np.clip(A['ev_own'][g], 1e-6, None) / c),
        np.log(c / B),
        np.broadcast_to(np.log(np.clip(A['phit_blend'][g], 1e-4, 1))[:, None], c.shape),
        np.broadcast_to(items['bankers'][:, None], c.shape),
        np.broadcast_to(items['cover'][:, None], c.shape),
        np.broadcast_to(items['weighted'][:, None], c.shape),
        np.broadcast_to(items['top'][:, None], c.shape),
        np.broadcast_to(items['units'][:, None], c.shape),
    ], axis=-1)
    return np.nan_to_num(f, nan=-50.0), ok


LUCKY = 3


def run_policy(theta, A, items, gi, compound=False, record=False, robust=False):
    """One pass over games gi; theta = [weights per feature, bias per game type].
    `robust` leaves out the LUCKY best games' growth, so a policy can't be
    chosen for the jackpots it happened to catch."""
    w, bias = theta[:len(FEATS)], theta[len(FEATS):]
    B = BANKROLL
    lg, rec, per = 0.0, [], []
    for g in gi:
        Bnow = B if compound else BANKROLL
        f, ok = _features(A, items, g, Bnow)
        s = f @ w + bias[items['type_ix'][g]]
        s = np.where(ok, s, -np.inf)
        j = np.unravel_index(np.argmax(s), s.shape)
        if not np.isfinite(s[j]) or s[j] <= 0:
            if record:
                rec.append((g, -1, -1, 0.0, 0.0))
            continue
        cost = float(A['cost'][g, j[0]] * MULTS[j[1]])
        ret = float(A['ret'][g, j[0], j[1]])
        x = np.log(max(Bnow - cost + ret, 1e-9) / Bnow)
        lg += x
        per.append(x)
        if compound:
            B = max(B - cost + ret, 1e-9)
            if B < 50:
                lg -= 20
                break
        if record:
            rec.append((g, int(j[0]), int(j[1]), cost, ret))
    if robust and per:
        lg -= float(np.sort(per)[-LUCKY:].clip(0, None).sum())
    return lg, rec


def cem(A, items, gi, log, iters=30, pop=48, elite=8, seed=0):
    rng = np.random.default_rng(seed)
    d = len(FEATS) + items['n_types']
    mu = np.zeros(d)
    mu[0] = 2.0          # start as an EV filter: play when the blend expects ~1.3× the stake
    mu[len(FEATS):] = -0.5
    sd = np.full(d, 1.0)
    best = (-np.inf, mu.copy())
    for t in range(iters):
        th = mu + sd * rng.standard_normal((pop, d))
        th[0] = mu
        sc = np.array([run_policy(x, A, items, gi, robust=True)[0] for x in th])
        top = np.argsort(-sc)[:elite]
        if sc[top[0]] > best[0]:
            best = (float(sc[top[0]]), th[top[0]].copy())
        mu = 0.7 * th[top].mean(axis=0) + 0.3 * mu
        sd = 0.7 * th[top].std(axis=0) + 0.3 * sd + 0.02
        if t % 5 == 0 or t == iters - 1:
            log(f"    CEM {t + 1}/{iters}: best train log growth {sc[top[0]]:+.4f} · elite mean {sc[top].mean():+.4f}")
    return best[1], mu


def agent_report(args, log):
    out = {}
    for src in args.sources:
        A, meta, items = load_menu(src)
        it = items['items']
        types = sorted(meta['type'].unique())
        I = {'bankers': np.array([x['bankers'] for x in it], float),
             'cover': np.array([x['cover'] for x in it], float),
             'weighted': np.array([x['weight'] != 'flat' for x in it], float),
             'top': np.array([x['top'] for x in it], float),
             'units': np.array([x['units'] for x in it], float),
             'type_ix': meta['type'].map({t: i for i, t in enumerate(types)}).values,
             'n_types': len(types)}
        tr = meta.index[meta['date'] < SPLIT].values
        te = meta.index[meta['date'] >= SPLIT].values
        log(f"[{src}] agent: {len(tr):,} training games, {len(te):,} test games")
        theta, mu = cem(A, I, tr, log)
        res = {}
        for name, th in (('best', theta), ('mean', mu)):
            for part, gi in (('train', tr), ('test', te)):
                lg, rec = run_policy(th, A, I, gi, record=True)
                R = pd.DataFrame(rec, columns=['g', 'i', 'k', 'cost', 'ret'])
                half = (meta.loc[R['g'], 'date'] >= meta.loc[gi, 'date'].median()).values
                s = _summary(R['cost'].values, R['ret'].values, None, half)
                played = R[R['i'] >= 0]
                mix = played['i'].map(lambda i: it[i]['bankers']).value_counts().sort_index().to_dict()
                res[f"{name}_{part}"] = {'log_growth': round(lg, 4), **(s or {}),
                                         'bankers_played': {int(k): int(v) for k, v in mix.items()},
                                         'mults_played': {int(MULTS[k]): int(v) for k, v in
                                                          played['k'].value_counts().sort_index().items()},
                                         'by_type': {t: _summary(R['cost'].values[m], R['ret'].values[m], None,
                                                                 half[m])
                                                     for t in types
                                                     for m in [(meta.loc[R['g'], 'type'] == t).values]
                                                     if (R['cost'].values[m] > 0).any()}}
                sm = res[f"{name}_{part}"]
                log(f"  {name:4s} {part:5s}: games played {sm.get('games', 0)}, ROI {sm.get('roi', 0):+.1%}, "
                    f"profit {sm.get('profit', 0):+,} kr, log growth {lg:+.4f}, bankers {sm['bankers_played']}, "
                    f"mults {sm['mults_played']}")
        res['theta'] = {f: round(float(v), 3) for f, v in zip(FEATS + [f"bias {t}" for t in types], theta)}
        out[src] = res
    path = OUT / 'agent.json'
    path.write_text(json.dumps(out, indent=1, default=float))
    log(f"-> {path}")
