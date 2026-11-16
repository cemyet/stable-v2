"""The coupon agent: which coupon to play in each marking game, or none.

    .venv-ml/bin/python -m scripts.build_coupon_agent menus --mode race_day
    .venv-ml/bin/python -m scripts.build_coupon_agent menus --mode forecast
    .venv-ml/bin/python -m scripts.build_coupon_agent train      # fit, test, register

Menus (core/coupon_sim: 0–3 bankers by four rules, five coverages, flat or
weighted rows, the top pool where there is one, each at five multipliers)
cover every finished V4 … V86 game since 2021, from the live race models'
out-of-sample predictions (ml_backtest):

  race_day  coupons picked by xedge's chances, each expected under xedge
            and under the closing odds alone
  forecast  coupons picked by xwin's chances, expected under xwin and
            under xodds (the forecast of the odds): what the agent has
            before the win pools hold money. Both use the stake shares the
            pool closed with, so for days ahead this is an upper bound.

Two policies (core/coupon_agent), one per menu, are fitted by the
cross-entropy method on the games before the race models' forward test and
frozen, then played over every later game. They're registered together as
target 'coupon_agent' with their weights in slice_def, so the web app runs
them without the artifact: 'live' once every leg's win pool holds money,
'forecast' (preliminary) before that.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from core import coupon_agent as CA  # noqa: E402
from core import coupon_sim as CS  # noqa: E402
from core import pools as PL  # noqa: E402
from core.db import get_connection  # noqa: E402

OUT = _ROOT / 'out' / 'coupon_agent'
SINCE = '2021-01-01'
ITEMS = sorted({it for gt in PL.ROW_PRICE for it in CS.items_for(gt)},
               key=lambda it: (it.bankers, it.rule, it.cover, it.weight, -it.units, it.top))
ITEM_IX = {it: i for i, it in enumerate(ITEMS)}
MODES = {'race_day': ('xedge', None), 'forecast': ('xwin', 'xodds')}


def _log_factory():
    t0 = time.time()

    def log(msg):
        print(f"[{time.time() - t0:6.0f}s] {msg}", flush=True)
    return log


def live_model(cur, target):
    cur.execute("""SELECT model_id, name, slice_def->>'test_since' FROM ml_model
                   WHERE target = %s ORDER BY created_at DESC, model_id DESC LIMIT 1""", (target,))
    return cur.fetchone()


def backtest(cur, model_id):
    cur.execute("SELECT entry_id, prob FROM ml_backtest WHERE model_id = %s", (model_id,))
    return {int(e): float(p) for e, p in cur.fetchall()}


def _aligned(g, h):
    return h is not None and len(h.legs) == len(g.legs) and all(
        np.array_equal(a.numbers, b.numbers) for a, b in zip(g.legs, h.legs))


def build_menu(mode, log):
    own_t, alt_t = MODES[mode]
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            own_m = live_model(cur, own_t)
            own = backtest(cur, own_m[0])
            alt_m = live_model(cur, alt_t) if alt_t else None
            alt = backtest(cur, alt_m[0]) if alt_m else None
        st, raw = PL.load(conn, SINCE)
    finally:
        conn.close()
    log(f"[{mode}] {own_t} #{own_m[0]} {own_m[1]} ({len(own):,} predictions) against "
        + (f"{alt_t} #{alt_m[0]} {alt_m[1]} ({len(alt):,})" if alt_m else 'the closing odds')
        + f" · {len(raw):,} finished games since {SINCE}")
    games = CS.build_games(st, raw, own)
    by = None
    if alt is not None:
        by = {h.gid: h for h in CS.build_games(st, raw, alt)}
        games = [g for g in games if _aligned(g, by.get(g.gid))]
    G, I, M = len(games), len(ITEMS), len(CS.MULTS)
    A = {'cost': np.full((G, I), np.nan), 'rows': np.zeros((G, I)),
         'ret': np.full((G, I, M), np.nan, np.float32),
         'ev_own': np.full((G, I, M), np.nan, np.float32), 'ev_alt': np.full((G, I, M), np.nan, np.float32),
         'phit_own': np.full((G, I), np.nan, np.float32), 'phit_alt': np.full((G, I), np.nan, np.float32)}
    t0 = time.time()
    for gi, g in enumerate(games):
        belief = [lg.q for lg in g.legs] if by is None else [lg.p for lg in by[g.gid].legs]
        smp = {'own': CS.Sampler(g, seed=gi), 'alt': CS.Sampler(g, seed=gi, belief=belief)}
        items = CS.items_for(g.gtype)
        for top in (False, True):
            its = [it for it in items if it.top == top]
            if not its:
                continue
            cps = [CS.coupon(g, it) for it in its]
            js = [ITEM_IX[it] for it in its]
            for j, it, (W, U, nr) in zip(js, its, cps):
                c, r = CS.settle(g, W, U, top=top)
                A['cost'][gi, j], A['rows'][gi, j], A['ret'][gi, j] = c[0], nr, r
            for b, s in smp.items():
                ev, ph = s.expected_many([cp[0] for cp in cps], [cp[1] for cp in cps], top=top)
                A['ev_' + b][gi, js], A['phit_' + b][gi, js] = ev, ph
        if gi % 500 == 0:
            log(f"[{mode}] {gi:,}/{G:,} games ({time.time() - t0:.0f}s)")
    meta = pd.DataFrame({'gid': [g.gid for g in games], 'type': [g.gtype for g in games],
                         'date': [g.date for g in games], 'turnover': [g.T for g in games],
                         'legs': [len(g.legs) for g in games]})
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / f"menu_{mode}.npz", **A)
    meta.to_pickle(OUT / f"games_{mode}.pkl")
    src = {'own': {'target': own_t, 'model_id': own_m[0], 'name': own_m[1], 'test_since': own_m[2]},
           'alt': ({'target': alt_t, 'model_id': alt_m[0], 'name': alt_m[1]} if alt_m else {'target': 'market'}),
           'since': SINCE, 'games': G}
    (OUT / f"menu_{mode}.json").write_text(json.dumps(src, indent=1))
    log(f"[{mode}] {G:,} games × {I} coupons -> {OUT / f'menu_{mode}.npz'}")


def load(mode):
    z = np.load(OUT / f"menu_{mode}.npz")
    return {k: z[k] for k in z.files}, pd.read_pickle(OUT / f"games_{mode}.pkl"), \
        json.loads((OUT / f"menu_{mode}.json").read_text())


def turnover_estimates(conn, recent=30, per_track=10):
    """What an open game is expected to turn over (kr): the median of the
    last games of its type, and of its type at its track where that has four."""
    with conn.cursor() as cur:
        cur.execute("""SELECT game_type, atg_game_id, turnover FROM atg_game
                       WHERE status = 'results' AND game_type = ANY(%s) AND turnover > 0
                       ORDER BY game_date DESC""", (list(PL.ROW_PRICE),))
        rows = cur.fetchall()
    out = {}
    for gt in PL.ROW_PRICE:
        games = [(gid.split('_')[2] if gid.count('_') >= 3 else '', t / 100) for t_, gid, t in rows if t_ == gt]
        if not games:
            continue
        tracks = {}
        for tr, t in games:
            tracks.setdefault(tr, []).append(t)
        out[gt] = {'all': round(float(np.median([t for _, t in games[:recent]]))),
                   'tracks': {tr: round(float(np.median(v[:per_track]))) for tr, v in tracks.items() if len(v) >= 4}}
    return out


def _drawdown(cost, ret):
    eq = np.cumsum(ret - cost)
    return float((np.maximum.accumulate(np.concatenate([[0.0], eq]))[1:] - eq).max()) if len(eq) else 0.0


def summarize(i, k, cost, ret, meta):
    """What the frozen policy did over one set of games, in date order."""
    on = cost > 0
    out = {'from': str(meta['date'].min().date()), 'to': str(meta['date'].max().date()),
           'games': int(len(meta)), 'played': int(on.sum())}
    if not on.any():
        return out
    half = (meta['date'] >= meta['date'].median()).values
    out.update(PL._roi_stats(cost[on], ret[on], half[on]) or {})
    out.update({'profit': round(float(ret.sum() - cost.sum())), 'hit': round(float((ret[on] > 0).mean()), 3),
                'avg_cost': round(float(cost[on].mean()), 1), 'max_drawdown': round(_drawdown(cost[on], ret[on])),
                'log_growth': round(float(CA.growth(cost[on], ret[on]).sum()), 4)})
    by = []
    for t, m in meta.groupby('type').groups.items():
        m = meta.index.get_indexer(m)
        p = on[m]
        row = {'type': t, 'games': int(len(m)), 'played': int(p.sum()),
               'cost': round(float(cost[m].sum()), 2), 'ret': round(float(ret[m].sum()), 2)}
        if p.any():
            c, r = cost[m][p], ret[m][p]
            row.update(roi=round(float(r.sum() / c.sum() - 1), 4),
                       roi_wo_top=round(float((r.sum() - r.max()) / c.sum() - 1), 4))
        by.append(row)
    out['by_type'] = sorted(by, key=lambda r: list(PL.ROW_PRICE).index(r['type']))
    chosen = [ITEMS[x] for x in i[on]]
    hist = lambda v: {str(a): int(b) for a, b in sorted(pd.Series(v).value_counts().items())}  # noqa: E731
    out['plays'] = {'bankers': hist([it.bankers for it in chosen]),
                    'mult': hist([int(CA.MULTS[x]) for x in k[on]]),
                    'cover': hist([f"{it.cover:.0%}" for it in chosen]),
                    'weighted': round(float(np.mean([it.weight != 'flat' for it in chosen])), 3),
                    'top': round(float(np.mean([it.top for it in chosen])), 3)}
    return out


def train(args, log):
    from scripts.build_race_models import register
    menus = {'live': load('race_day'), 'forecast': load('forecast')}
    src = menus['live'][2]
    split = pd.Timestamp(src['own']['test_since'])
    types = sorted(set(menus['live'][1]['type']) | set(menus['forecast'][1]['type']), key=list(PL.ROW_PRICE).index)
    tix = {t: n for n, t in enumerate(types)}
    shape = CA.item_arrays(ITEMS)
    sub = lambda P, m: {k: v[m] for k, v in P.items()}  # noqa: E731
    D = {}
    for mode, (A, meta, _) in menus.items():
        tr = (meta['date'] < split).values
        D[mode] = {'P': CA.parts(A['cost'], A['ev_own'], A['ev_alt'], A['phit_alt'], shape['units']),
                   'A': A, 'meta': meta, 'ty': meta['type'].map(tix).values, 'train': tr, 'test': ~tr}
    log(f"agent: {int(D['live']['train'].sum()):,} training games (< {split.date()}), "
        f"{int(D['live']['test'].sum()):,} test games · {len(ITEMS)} coupons × {len(CA.MULTS)} copies")

    def run(theta, mode, part):
        d = D[mode]
        m = d[part]
        i, k, c, r = CA.play(theta, sub(d['P'], m), shape, d['ty'][m], d['A']['cost'][m], d['A']['ret'][m])
        s = summarize(i, k, c, r, d['meta'][m].reset_index(drop=True))
        log(f"    {mode:8s} {part:5s}: played {s['played']:,}/{s['games']:,}, ROI {s.get('roi', 0):+.1%} "
            f"({s.get('lo', 0):+.1%} … {s.get('hi', 0):+.1%}), w/o top {s.get('roi_wo_top', 0):+.1%}, "
            f"profit {s.get('profit', 0):+,} kr, log growth {s.get('log_growth', 0):+.3f}")
        return s

    # The final mean of each search is the policy; its single best sample is
    # the luckier draw on the same games.
    thetas, runs = {}, {}
    for mode in ('live', 'forecast'):
        d = D[mode]
        log(f"  {mode} policy ({menus[mode][2]['own']['target']} against {menus[mode][2]['alt']['target']})")
        _, mu = CA.cem(sub(d['P'], d['train']), shape, d['ty'][d['train']], d['A']['cost'][d['train']],
                       d['A']['ret'][d['train']], len(types), log, iters=args.iters, pop=args.pop, seed=args.seed)
        thetas[mode] = mu
        runs[mode] = {part: run(mu, mode, part) for part in ('train', 'test')}
    log('  the live policy on the forecasts, for comparison')
    cross = run(thetas['live'], 'forecast', 'test')

    feats = CA.FEATS + [f"bias {t}" for t in types]
    coefs = lambda th: sorted([{'feature': f, 'coef': round(float(v), 4)} for f, v in zip(feats, th)],  # noqa: E731
                              key=lambda d: -abs(d['coef']))
    metrics = {'test': runs['live']['test'], 'train': runs['live']['train'],
               'preliminary': runs['forecast']['test'], 'preliminary_train': runs['forecast']['train'],
               'live_on_forecasts': cross,
               'theta': coefs(thetas['live']), 'theta_forecast': coefs(thetas['forecast']),
               'cem': {'iters': args.iters, 'pop': args.pop, 'robust_drop': CA.LUCKY, 'seed': args.seed}}
    conn = get_connection()
    try:
        turnover = turnover_estimates(conn)
    finally:
        conn.close()
    one = lambda th: {'types': types, 'theta': [round(float(v), 6) for v in th], 'bankroll': CA.BANKROLL}  # noqa: E731
    policy = {'feats': CA.FEATS, 'types': types, 'live': one(thetas['live']), 'forecast': one(thetas['forecast']),
              'mults': [int(m) for m in CA.MULTS], 'bankroll': CA.BANKROLL, 'stake_cap': CA.STAKE_CAP,
              'mc': CS.MC, 'win_pool_live': CA.WIN_POOL_LIVE, 'turnover': turnover}
    srcf = menus['forecast'][2]
    slice_def = {
        'label': 'coupon agent (linear policies over the coupon menu, cross-entropy method)',
        'source': src['own'], 'alt': src['alt'], 'forecast': {'own': srcf['own'], 'alt': srcf['alt']},
        'beliefs': ['xedge', 'market'], 'test_since': str(split.date()), 'since': SINCE,
        'items': len(ITEMS), 'bankroll': CA.BANKROLL, 'stake_cap': CA.STAKE_CAP,
        'features': feats, 'n_features': len(feats), 'win_pool_live': CA.WIN_POOL_LIVE,
        'n_train': int(D['live']['train'].sum()), 'n_test': int(D['live']['test'].sum()), 'policy': policy,
        'note': ('two policies over the same features, each fitted on games before the forward test and frozen: '
                 'live (xedge’s out-of-fold chances against the closing odds) once every leg’s win pool holds '
                 'money, preliminary (xwin against xodds) before that. Settled against what the pool paid with '
                 'the agent’s rows added; both use the stake shares the pool closed with.'),
    }
    if args.dry_run:
        log('dry run: not registered')
        return
    conn = get_connection()
    try:
        register(conn, 'coupon_agent', 'CEM linear policy', 'coupon_agent', policy, slice_def, metrics, log)
    finally:
        conn.close()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('cmd', choices=['menus', 'train'])
    ap.add_argument('--mode', choices=list(MODES), default='race_day')
    ap.add_argument('--iters', type=int, default=40)
    ap.add_argument('--pop', type=int, default=64)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()
    log = _log_factory()
    if args.cmd == 'menus':
        build_menu(args.mode, log)
    else:
        train(args, log)
    return 0


if __name__ == '__main__':
    sys.exit(main())
