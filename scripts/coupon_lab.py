"""Spelteknik lab: which coupons to play in the marking pools, how big, and when.

    .venv-ml/bin/python -m scripts.coupon_lab menus     # coupon menus per game -> out/coupon_lab/
    .venv-ml/bin/python -m scripts.coupon_lab static    # every fixed coupon recipe, train / test
    .venv-ml/bin/python -m scripts.coupon_lab agent     # a betting agent tuned on train, scored on test

Sources (what picks the horses): xwin's forward test (ml_backtest), the
market (closing win odds) and a blend of the two (winner logit on log market
and log xwin, fitted on the training games' races only).

Every game gets the same menu (core/coupon_sim.items_for): 0–3 bankers by
four rules, five coverages, flat or weighted rows, the top pool where the
game has one; each at five multipliers. For each coupon it stores the cost,
the realised return and what the coupon was expected to return before the
race under two beliefs: the source itself and the blend.
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

from core import coupon_sim as CS  # noqa: E402
from core import pools as PL  # noqa: E402
from core.db import get_connection  # noqa: E402

OUT = _ROOT / 'out' / 'coupon_lab'
SPLIT = pd.Timestamp('2026-03-01')
XWIN_MODEL = 76
ALL_ITEMS = sorted({it for gt in PL.ROW_PRICE for it in CS.items_for(gt)},
                   key=lambda it: (it.bankers, it.rule, it.cover, it.weight, -it.units, it.top))
ITEM_IX = {it: i for i, it in enumerate(ALL_ITEMS)}


def _log_factory():
    t0 = time.time()

    def log(msg):
        print(f"[{time.time() - t0:6.0f}s] {msg}", flush=True)
    return log


def fit_blend(st, probs, until):
    """Winner logit on [log market, log model] over races before `until`."""
    from scipy.optimize import minimize
    d = PL._race_frame(st, probs)
    d = d[d['date'] < until]
    lq, lp = np.log(d['q'].clip(1e-9)).values, np.log(d['p'].clip(1e-9)).values
    race = pd.factorize(d['race'])[0]
    won = d['won'].values

    def nll(b):
        z = b[0] * lq + b[1] * lp
        zm = pd.Series(z).groupby(race).transform('max').values
        lse = np.log(pd.Series(np.exp(z - zm)).groupby(race).transform('sum').values) + zm
        return -float(((z - lse) * won).sum())

    r = minimize(nll, np.array([0.8, 0.3]), method='Nelder-Mead', options={'xatol': 1e-4, 'fatol': 1e-3})
    return r.x


def blend_probs(st, probs, b):
    d = PL._race_frame(st, probs)
    z = np.exp(b[0] * np.log(d['q'].clip(1e-9)) + b[1] * np.log(d['p'].clip(1e-9)))
    p = z / z.groupby(d['race']).transform('sum')
    return dict(zip(d['entry_id'].astype(int), p.astype(float)))


def sources(conn, st, log):
    with conn.cursor() as cur:
        cur.execute("SELECT entry_id, prob FROM ml_backtest WHERE model_id = %s", (XWIN_MODEL,))
        xwin = {int(e): float(p) for e, p in cur.fetchall()}
    d = st[~st['scr'] & (st['o'] > 1)]
    inv = 1 / d['o']
    market = dict(zip(d['entry_id'].astype(int), (inv / inv.groupby(d['race']).transform('sum')).astype(float)))
    b = fit_blend(st, xwin, SPLIT)
    log(f"blend on races before {SPLIT.date()}: log p = {b[0]:.3f}·log market + {b[1]:.3f}·log xwin")
    out = {'xwin': xwin, 'market': market, 'blend': blend_probs(st, xwin, b)}
    oof = _ROOT / 'out' / 'race_lab' / 'oof.npz'
    if oof.exists():
        z = np.load(oof)
        out['edge'] = dict(zip(z['eids'].astype(int).tolist(), z['edge'].astype(float).tolist()))
    return out, [float(x) for x in b]


def build_menus(args, log):
    OUT.mkdir(parents=True, exist_ok=True)
    conn = get_connection()
    try:
        st, raw = PL.load(conn, args.since)
        srcs, coef = sources(conn, st, log)
    finally:
        conn.close()
    blend_games = {g.gid: g for g in CS.build_games(st, raw, srcs['blend'])}
    for name in args.sources:
        games = [g for g in CS.build_games(st, raw, srcs[name]) if g.gid in blend_games]
        G, I, M = len(games), len(ALL_ITEMS), len(CS.MULTS)
        cost = np.full((G, I), np.nan)
        rows = np.zeros((G, I))
        ret = np.full((G, I, M), np.nan)
        ev = {b: np.full((G, I, M), np.nan) for b in ('own', 'blend')}
        phit = {b: np.full((G, I), np.nan) for b in ('own', 'blend')}
        t0 = time.time()
        for gi, g in enumerate(games):
            bl = [lg.p for lg in blend_games[g.gid].legs]
            samplers = {'own': CS.Sampler(g, seed=gi), 'blend': CS.Sampler(g, seed=gi, belief=bl)}
            for it in CS.items_for(g.gtype):
                j = ITEM_IX[it]
                W, U, nr = CS.coupon(g, it)
                c, r = CS.settle(g, W, U, top=it.top)
                cost[gi, j], rows[gi, j], ret[gi, j] = c[0], nr, r
                for b, smp in samplers.items():
                    e, ph, _ = smp.expected(W, U, top=it.top)
                    ev[b][gi, j], phit[b][gi, j] = e, ph
            if gi % 250 == 0:
                log(f"[{name}] {gi:,}/{G:,} games ({time.time() - t0:.0f}s)")
        meta = pd.DataFrame({'gid': [g.gid for g in games], 'type': [g.gtype for g in games],
                             'date': [g.date for g in games], 'turnover': [g.T for g in games],
                             'top_pot': [float(g.pot[-1]) for g in games], 'legs': [len(g.legs) for g in games]})
        np.savez_compressed(OUT / f"menu_{name}.npz", cost=cost, rows=rows, ret=ret,
                            ev_own=ev['own'], ev_blend=ev['blend'],
                            phit_own=phit['own'], phit_blend=phit['blend'])
        meta.to_pickle(OUT / f"games_{name}.pkl")
        log(f"[{name}] {G:,} games × {I} coupons -> {OUT / f'menu_{name}.npz'}")
    items = [{'bankers': it.bankers, 'rule': it.rule, 'cover': it.cover, 'weight': it.weight,
              'units': it.units, 'top': it.top, 'label': it.label} for it in ALL_ITEMS]
    (OUT / 'items.json').write_text(json.dumps({'items': items, 'mults': CS.MULTS, 'blend': coef,
                                                'split': str(SPLIT.date())}, indent=1))


def load_menu(name):
    z = np.load(OUT / f"menu_{name}.npz")
    A = {k: z[k] for k in z.files}
    meta = pd.read_pickle(OUT / f"games_{name}.pkl")
    items = json.loads((OUT / 'items.json').read_text())
    return A, meta, items


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('cmd', choices=['menus', 'static', 'agent'])
    ap.add_argument('--since', default='2025-07-01')
    ap.add_argument('--sources', nargs='*', default=['xwin', 'blend', 'market'])
    args = ap.parse_args()
    log = _log_factory()
    if args.cmd == 'menus':
        build_menus(args, log)
    elif args.cmd == 'static':
        from scripts.coupon_policies import static_report
        static_report(args, log)
    else:
        from scripts.coupon_policies import agent_report
        agent_report(args, log)
    return 0


if __name__ == '__main__':
    sys.exit(main())
