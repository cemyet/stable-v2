"""Experiments on the race-model chain, from a cache built once.

    .venv-ml/bin/python -m scripts.race_lab build        # passes + extra inputs -> out/race_lab/cache.npz
    .venv-ml/bin/python -m scripts.race_lab exp [names]  # xwin variants on the forward test

`build` runs the rating pass and the xlead pass the way build_race_models
does, and joins what the live models stored in ml_backtest (the rating's,
xlead's and xodds' out-of-fold shares, xwin's forward test). On top it
makes the inputs the chain doesn't have yet, all out of fold on the same
time blocks:

  xfront   who is in front at 700 m (core/xlead.py front_labels), with the
           xlead pass's features
  xgal     the chance of a gallop, from entry_features (the inputs of the
           y_gal model)
  trainer  the trainer-form table (scripts/build_trainer_form_exp.py)

`exp` retrains xwin per variant on the out-of-fold years and scores the
forward test, next to the market and to a market + model blend.
"""
from __future__ import annotations

import argparse
import datetime as dt
import gc
import json
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from core import xlead as XL  # noqa: E402
from core import xodds as XO  # noqa: E402
from core import xstack as XS  # noqa: E402
from core.db import get_connection  # noqa: E402
from scripts import build_race_models as BR  # noqa: E402

OUT = _ROOT / 'out' / 'race_lab'
CACHE = OUT / 'cache.npz'

GAL_FEATURES = [
    'horse_galrate', 'is_auto', 'galrate_recent_10', 'horse_starts', 'clean_streak', 'starters',
    'horse_trainer_starts', 'races_with_trainer', 'races_with_driver', 'driver_galrate',
    'horse_galrate_auto', 'race_month', 'horse_galrate_barefoot', 'post_galrate_volt', 'distance_m',
    'gal_rank_in_field', 'galrate_method', 'earnings_pct_in_race', 'days_since_last_start', 'track_id',
    'gal_pre', 'horse_winrate', 'driver_winrate', 'trainer_winrate', 'shoe_code', 'breed_code',
]
GAL_CATEGORICAL = ['track_id', 'shoe_code', 'breed_code']
GAL_GBM = dict(learning_rate=0.08, max_iter=600, max_leaf_nodes=63, min_samples_leaf=200,
               l2_regularization=1.0)
TRAINER = ['tf_perf_90', 'tf_perf_last50', 'tf_perf_decay365', 'tf_winrate_90', 'tf_top3rate_90',
           'tf_posbeat_90', 'tf_posbeat_decay365', 'tf_beatrate_90', 'tf_winresid_90', 'tf_n_180']
TRAINER_DAY = ['tf_perf_90', 'tf_winrate_90', 'tf_top3rate_90', 'tf_posbeat_90', 'tf_beatrate_90',
               'tf_winresid_90', 'tf_n_180']
LIVE = {'rating': 72, 'lead_auto': 73, 'lead_volt': 74, 'xodds': 75, 'xwin': 76}


def _log_factory():
    t0 = time.time()

    def log(msg):
        print(f"[{time.time() - t0:6.0f}s] {msg}", flush=True)
    return log


def _table(eids, mask, d, fill=np.nan):
    v = np.full(mask.shape, fill, dtype=float)
    idx = np.nonzero(mask)
    v[idx] = [d.get(int(e), fill) for e in eids[idx]]
    return v


# ── build ────────────────────────────────────────────────────────────────

def _backtest(cur, model_id):
    cur.execute("SELECT entry_id, prob FROM ml_backtest WHERE model_id = %s", (model_id,))
    return {int(e): float(p) for e, p in cur.fetchall()}


def _xgal_oof(conn, start, cut, K, log):
    """Out-of-fold gallop chances per entry id, rows from `start` on."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    cols = ', '.join(f'ef.{c}' for c in GAL_FEATURES)
    df = pd.read_sql(f"""
        SELECT ef.entry_id, ef.race_date, ef.y_gal::int AS y, {cols}
        FROM entry_features ef
        JOIN track t ON t.track_id = ef.track_id
        JOIN entry e ON e.entry_id = ef.entry_id
        WHERE t.country = 'SE' AND ef.race_date >= '2014-01-01' AND ef.race_date < CURRENT_DATE
          AND NOT COALESCE(e.withdrawn, false) AND ef.y_gal IS NOT NULL
    """, conn)
    for c in GAL_FEATURES:
        if c in GAL_CATEGORICAL:
            df[c] = df[c].astype('category')
        else:
            df[c] = pd.to_numeric(df[c], errors='coerce').astype('float32')
    t = pd.to_datetime(df['race_date']).map(lambda d: d.toordinal()).values
    blk = BR.blocks(t, start, cut, K)
    log(f"[xgal] {len(df):,} rows, gallop rate {df['y'].mean():.1%}")

    def fit(m, max_iter=None):
        kw = dict(GAL_GBM)
        kw.update(early_stopping=max_iter is None, validation_fraction=0.1, n_iter_no_change=30)
        if max_iter is not None:
            kw['max_iter'] = max_iter
        g = HistGradientBoostingClassifier(categorical_features='from_dtype', random_state=42, **kw)
        g.fit(df.loc[m, GAL_FEATURES], df.loc[m, 'y'])
        return g

    out = {}
    first = fit(blk < K)
    n_iter = int(first.n_iter_)
    fw = blk == K
    p = first.predict_proba(df.loc[fw, GAL_FEATURES])[:, 1]
    from sklearn.metrics import log_loss, roc_auc_score
    log(f"[xgal] forward test auc {roc_auc_score(df.loc[fw, 'y'], p):.4f} · log-loss "
        f"{log_loss(df.loc[fw, 'y'], p):.4f} · trees {n_iter}")
    out.update(zip(df.loc[fw, 'entry_id'].astype(int), p))
    for k in range(K):
        g = fit((blk < K) & (blk != k), max_iter=n_iter)
        m = blk == k
        out.update(zip(df.loc[m, 'entry_id'].astype(int), g.predict_proba(df.loc[m, GAL_FEATURES])[:, 1]))
        log(f"[xgal] fold {k + 1}/{K}")
        del g
        gc.collect()
    return out


def build(args, log):
    since = dt.date.fromisoformat(args.since)
    lead_from = dt.date.fromisoformat(args.lead_from)
    rows_from = dt.date.fromisoformat(args.rows_from)
    test_from = dt.date.fromisoformat(args.test_from)
    start, cut, K = rows_from.toordinal(), test_from.toordinal(), args.folds
    OUT.mkdir(parents=True, exist_ok=True)
    conn = get_connection()
    try:
        cur = conn.cursor(name='lab_market')
        cur.itersize = 20000
        mp = XO.MarketPass(today=dt.date.today(), bt_start=rows_from, snap_from=lead_from)
        mp.run(cur, since)
        cur.close()
        conn.commit()
        rows, snap = mp.rows, mp.snap
        log(f"market pass: {len(rows):,} rows")
        del mp
        gc.collect()

        cur = conn.cursor(name='lab_lead')
        cur.itersize = 20000
        lp = XL.LeadPass(today=dt.date.today(), train_from=lead_from, ext=snap, other_from=rows_from,
                         front=True)
        lp.run(cur, since)
        cur.close()
        conn.commit()
        lead_df, other_df, front_df = XL.frame(lp.train), XL.frame(lp.other), XL.frame(lp.front)
        log(f"lead pass: {len(lead_df):,} early-labelled rows, {len(front_df):,} front-labelled rows")
        del lp, snap
        gc.collect()

        every = pd.concat([lead_df[lead_df['t'] >= start], other_df], ignore_index=True)
        rest = every[~every['entry_id'].isin(set(front_df['entry_id']))].copy()
        front_oof, front_rep = {}, {}
        for method in ('auto', 'volt'):
            _, metrics, counts, oof = BR.lead_stage(front_df, rest, method, start, cut, K, False, log)
            front_oof.update(zip(oof['entry_id'].astype(int), oof['p'].astype(float)))
            front_rep[method] = {'race': metrics['test']['race'], 'auc': metrics['test']['roc_auc'],
                                 'importances': metrics['importances'][:15], **counts}
        shape_frames = [lead_df[lead_df['t'] >= start], other_df]
        # did the early leader keep it? (front label next to the early one, per entry)
        front_y = dict(zip(front_df['entry_id'].astype(int), front_df['y'].astype(float)))
        early_y = dict(zip(lead_df['entry_id'].astype(int), lead_df['y'].astype(float)))

        rids, ts, eids, X, mask, q, won = XO.tensors(rows)
        del rows
        gc.collect()
        blk = BR.blocks(ts, start, cut, K)
        shape = XS.shape_columns(eids, mask, shape_frames)
        del lead_df, other_df, front_df, every, rest, shape_frames
        gc.collect()

        with conn.cursor() as c2:
            lead = {**_backtest(c2, LIVE['lead_auto']), **_backtest(c2, LIVE['lead_volt'])}
            P = _table(eids, mask, _backtest(c2, LIVE['xodds']))
            W = _table(eids, mask, _backtest(c2, LIVE['xwin']))
            tf_cols = ', '.join(TRAINER)
            c2.execute(f"SELECT entry_id, {tf_cols} FROM trainer_form_exp WHERE entry_id = ANY(%s)",
                       (eids[mask].astype(int).tolist(),))
            tf = {int(r[0]): r[1:] for r in c2.fetchall()}
            c2.execute("""SELECT entry_id, COALESCE(finish_order, placement), COALESCE(galopp, false),
                                 COALESCE(disqualified, false)
                          FROM entry WHERE entry_id = ANY(%s)""", (eids[mask].astype(int).tolist(),))
            lab = {int(e): (fo, g, d) for e, fo, g, d in c2.fetchall()}
        log(f"backtests: xodds {np.isfinite(P[mask]).mean():.1%} of runners, "
            f"xlead {np.mean([int(e) in lead for e in eids[mask]]):.1%}, trainer {len(tf) / mask.sum():.1%}")
        L = XS.lead_columns(eids, mask, lead)
        Fr = XS.lead_columns(eids, mask, front_oof)
        lead_cov = _table(eids, mask, {e: 1.0 for e in lead}, 0.0)
        front_cov = _table(eids, mask, {e: 1.0 for e in front_oof}, 0.0)
        TF = np.full(mask.shape + (len(TRAINER),), np.nan, dtype=np.float32)
        idx = np.nonzero(mask)
        for i, j in zip(*idx):
            r = tf.get(int(eids[i, j]))
            if r is not None:
                TF[i, j] = [np.nan if v is None else float(v) for v in r]
        finish = np.zeros(mask.shape, dtype=np.int16)
        gal = np.zeros(mask.shape, dtype=bool)
        dq = np.zeros(mask.shape, dtype=bool)
        for i, j in zip(*idx):
            fo, g, d = lab.get(int(eids[i, j]), (None, False, False))
            finish[i, j] = fo or 0
            gal[i, j], dq[i, j] = g, d
        y_front = _table(eids, mask, front_y)
        y_early = _table(eids, mask, early_y)

        G = _table(eids, mask, _xgal_oof(conn, start, cut, K, log))
        np.savez_compressed(
            CACHE, rids=rids, ts=ts, eids=eids, X=X.astype(np.float32), mask=mask, q=q, won=won, blk=blk,
            L=L, Fr=Fr, lead_cov=lead_cov, front_cov=front_cov, shape=shape.astype(np.float32),
            P=P, W=W, G=G, TF=TF, finish=finish, gal=gal, dq=dq, y_front=y_front, y_early=y_early)
        (OUT / 'front_report.json').write_text(json.dumps(front_rep, indent=1, default=float))
        log(f"cache -> {CACHE} ({CACHE.stat().st_size / 1e6:.0f} MB)")
    finally:
        conn.close()


# ── experiments ──────────────────────────────────────────────────────────

def load():
    z = np.load(CACHE)
    return {k: z[k] for k in z.files}


def rel_cols(a, mask):
    """[a, a - race mean, a - race max] for one per-runner value."""
    n = np.maximum(mask.sum(axis=1, keepdims=True), 1)
    a0 = np.where(mask, a, 0.0)
    return np.stack([a0, a0 - a0.sum(axis=1, keepdims=True) / n,
                     a0 - np.where(mask, a, -np.inf).max(axis=1, keepdims=True)], axis=-1)


def winner_ll(p, won):
    return float(-np.log(np.clip((p * won).sum(axis=1), 1e-12, 1)).mean())


def blend(parts, won, mask, tr, te):
    """Conditional logit on log shares, fitted on `tr`, winner log-loss on `te`."""
    Z = np.stack([np.where(mask, np.log(np.clip(p, 1e-9, 1)), 0.0) for p in parts], axis=-1)
    b = XO.fit_soft_logit(Z[tr], mask[tr], won[tr], l2=0.01)
    return winner_ll(XO.softmax(Z[te] @ b, mask[te]), won[te]), [round(float(x), 3) for x in b]


def softmax_boost(D, mask, won, offset, tr, va, te, log, rounds=60, inner=None):
    """Race-softmax boosting from an offset (log shares): every round fits a
    small tree ensemble to the Newton step of the winner log-likelihood.
    Stops when the validation races stop improving."""
    from sklearn.ensemble import HistGradientBoostingRegressor
    inner = inner or dict(max_iter=15, learning_rate=0.1, max_leaf_nodes=15, min_samples_leaf=400,
                          l2_regularization=2.0)
    S = np.where(mask, offset, 0.0).astype(float)
    best, best_r, stale, models = np.inf, 0, 0, []
    Dtr, mtr = D[tr][mask[tr]], mask[tr]
    for r in range(rounds):
        p = XO.softmax(S[tr], mtr)
        g = (won[tr] - p)[mtr]
        h = np.clip(p * (1 - p), 1e-3, None)[mtr]
        reg = HistGradientBoostingRegressor(random_state=r, **inner)
        reg.fit(Dtr, g / h, sample_weight=h)
        models.append(reg)
        for m in (tr, va, te):
            sub = S[m]
            add = np.zeros(sub.shape)
            add[mask[m]] = reg.predict(D[m][mask[m]])
            S[m] = sub + add
        v = winner_ll(XO.softmax(S[va], mask[va]), won[va])
        if v < best - 1e-5:
            best, best_r, stale = v, r, 0
            S_best = S.copy()
        else:
            stale += 1
            if stale >= 6:
                break
    log(f"    softmax boost: {best_r + 1} rounds, validation {best:.4f}")
    return XO.softmax(S_best, mask)


LGB_EDGE = dict(learning_rate=0.03, num_leaves=15, min_data_in_leaf=400, lambda_l2=2.0,
                feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, num_threads=4, verbose=-1)


def lgbm_softmax(D, mask, won, offset, tr, va, te, log, params=None, rounds=3000):
    """LightGBM with the race-softmax winner likelihood as its objective,
    starting from an offset (log shares): the same model as softmax_boost,
    grown one tree at a time with early stopping on the validation races."""
    import lightgbm as lgb

    def flat(m):
        n = mask[m].sum(axis=1)
        starts = np.concatenate([[0], np.cumsum(n)[:-1]])
        return (D[m][mask[m]], won[m][mask[m]], offset[m][mask[m]], starts, n)

    def probs(z, starts, n):
        z = z - np.repeat(np.maximum.reduceat(z, starts), n)
        e = np.exp(z)
        return e / np.repeat(np.add.reduceat(e, starts), n)

    Xt, yt, ot, st, nt = flat(tr)
    Xv, yv, ov, sv, nv = flat(va)

    def objective(preds, data):
        p = probs(ot + preds, st, nt)
        return p - yt, np.maximum(p * (1 - p), 1e-6)

    def metric(preds, data):
        p = probs(ov + preds, sv, nv)
        return 'winner_ll', float(-np.log(np.clip(p[yv == 1], 1e-12, 1)).mean()), False

    ds = lgb.Dataset(Xt, label=yt, free_raw_data=False)
    dv = lgb.Dataset(Xv, label=yv, reference=ds)
    m = lgb.train({**(params or LGB_EDGE), 'objective': objective, 'metric': 'None'}, ds, num_boost_round=rounds,
                  valid_sets=[dv], feval=metric, callbacks=[lgb.early_stopping(100, verbose=False)])
    log(f"    lightgbm softmax: {m.best_iteration} trees, validation {m.best_score['valid_0']['winner_ll']:.4f}")
    out = np.zeros(mask.shape)
    Xe, _, oe, se, ne = flat(te)
    sub = np.zeros(mask[te].shape)
    sub[mask[te]] = probs(oe + m.predict(Xe, num_iteration=m.best_iteration), se, ne)
    out[te] = sub
    return out, m


def experiments(names, log):
    from sklearn.ensemble import HistGradientBoostingClassifier
    z = load()
    mask, won, q, blk = z['mask'], z['won'].astype(float), z['q'], z['blk']
    K = int(blk.max())
    X = z['X'].astype(float)
    has = (won.sum(axis=1) == 1) & np.isfinite(np.where(mask, z['P'], 0)).all(axis=1)
    tr, te = (blk >= 0) & (blk < K) & has, (blk == K) & has
    # second half of the forward test, for blends fitted on its first half
    te_idx = np.nonzero(te)[0]
    te1 = np.zeros_like(te)
    te1[te_idx[: len(te_idx) // 2]] = True
    te2 = te & ~te1
    P = np.where(mask, np.nan_to_num(z['P'], nan=0.0), 0.0)
    L, Fr, shape = z['L'], z['Fr'], z['shape'].astype(float)
    Xa = np.concatenate([X, L], axis=-1)
    log(f"cache: {int(mask.sum()):,} runners, {int(tr.sum()):,} train races, {int(te.sum()):,} test races")

    res = {'market': winner_ll(q[te], won[te]), 'stored_xwin': winner_ll(z['W'][te], won[te]),
           'xodds': winner_ll(P[te], won[te])}
    res['market_2nd_half'] = winner_ll(q[te2], won[te2])
    log(f"market {res['market']:.4f} · stored xwin {res['stored_xwin']:.4f} · xodds {res['xodds']:.4f}")

    preds = {}

    def fit_xwin(D, label, train=tr, params=None):
        kw = dict(BR.XWIN_GBM, **(params or {}))
        kw.update(early_stopping=True, validation_fraction=0.12, n_iter_no_change=60)
        t0 = time.time()
        m = HistGradientBoostingClassifier(random_state=42, **kw)
        m.fit(D[train][mask[train]].astype(np.float32), won[train][mask[train]].astype(int))
        Wp = np.zeros(mask.shape)
        Wp[te] = XS.predict_xwin({'model': m}, D[te].astype(np.float32), mask[te])
        ll = winner_ll(Wp[te], won[te])
        bl, coef = blend([q, Wp], won, mask, te1, te2)
        out = {'winner_ll': round(ll, 4), 'vs_market': round(ll - res['market'], 4),
               'blend_2nd_half': round(bl, 4), 'blend_gain': round(res['market_2nd_half'] - bl, 4),
               'blend_coef': coef, 'trees': int(m.n_iter_), 'inputs': int(D.shape[-1]),
               'secs': round(time.time() - t0)}
        log(f"  {label:22s} {out}")
        res[label] = out
        preds[label] = Wp
        return Wp

    base = XS.xwin_design(Xa, mask, P, shape)
    nx = len(XO.FEATURES)
    i_xo = [nx + 2 + i for i in range(len(XS.XWIN_COLS))]
    i_lead = [nx, nx + 1]
    i_shape = list(range(base.shape[-1] - len(XS.SHAPE), base.shape[-1]))
    keep = lambda D, drop: D[..., [i for i in range(D.shape[-1]) if i not in set(drop)]]  # noqa: E731
    gal_cols = rel_cols(np.log(np.clip(np.nan_to_num(z['G'], nan=0.15), 1e-4, 1)), mask)
    front_cols = np.concatenate([Fr, rel_cols(np.log(np.clip(Fr[..., 0], 1e-6, 1)), mask)[..., 1:]], axis=-1)
    TF = z['TF'].astype(float)
    S1 = XO.stage1_shares(X, mask)

    want = set(names) if names else None

    def on(name):
        return want is None or name in want

    if on('base'):
        fit_xwin(base, 'base')
    if on('no_xodds'):
        fit_xwin(keep(base, i_xo), 'no_xodds')
    if on('rating_share'):
        fit_xwin(XS.xwin_design(Xa, mask, S1, shape), 'rating_share')
    if on('logit_xodds'):
        Pl = np.zeros(mask.shape)
        for f in range(K + 1):
            fit_m = (blk >= 0) & (blk < K) & (blk != f) if f < K else (blk >= 0) & (blk < K)
            mu, sd = XO.standardize(Xa[fit_m], mask[fit_m])
            beta = XO.fit_soft_logit((Xa[fit_m] - mu) / sd, mask[fit_m], q[fit_m], l2=BR.XODDS_L2)
            idx = blk == f
            Pl[idx] = XO.softmax(((Xa[idx] - mu) / sd) @ beta, mask[idx])
        res['logit_xodds_alone'] = round(winner_ll(Pl[te], won[te]), 4)
        fit_xwin(XS.xwin_design(Xa, mask, Pl, shape), 'logit_xodds')
    if on('no_xlead'):
        fit_xwin(keep(base, i_lead + i_shape), 'no_xlead')
    if on('xfront'):
        fit_xwin(np.concatenate([base, front_cols], axis=-1), 'xfront')
    if on('xgal'):
        fit_xwin(np.concatenate([base, gal_cols], axis=-1), 'xgal')
    if on('trainer'):
        cov = np.isfinite(TF[..., 0]) | ~mask
        tcov = te & cov.all(axis=1)
        res['trainer_cover_test'] = round(float(tcov.sum() / te.sum()), 3)
        Wt = fit_xwin(np.concatenate([base, TF], axis=-1), 'trainer')
        Wb = preds['base'] if 'base' in preds else fit_xwin(base, 'base')
        res['trainer_on_covered'] = {
            'trainer': round(winner_ll(Wt[tcov], won[tcov]), 4),
            'base': round(winner_ll(Wb[tcov], won[tcov]), 4),
            'market': round(winner_ll(q[tcov], won[tcov]), 4),
        }
        log(f"  trainer on covered races: {res['trainer_on_covered']}")
    if on('trainer_clean'):
        # Only the windows over earlier days: the last-N windows of
        # trainer_form_exp count the trainer's other starts of the same day.
        clean = [TRAINER.index(c) for c in TRAINER_DAY]
        cov = np.isfinite(TF[..., 0]) | ~mask
        tcov = te & cov.all(axis=1)
        Wt = fit_xwin(np.concatenate([base, TF[..., clean]], axis=-1), 'trainer_clean')
        Wb = preds['base'] if 'base' in preds else fit_xwin(base, 'base')
        res['trainer_clean_on_covered'] = {
            'trainer_clean': round(winner_ll(Wt[tcov], won[tcov]), 4),
            'base': round(winner_ll(Wb[tcov], won[tcov]), 4),
            'market': round(winner_ll(q[tcov], won[tcov]), 4),
        }
        log(f"  trainer (earlier days only) on covered races: {res['trainer_clean_on_covered']}")
    if on('all_new'):
        fit_xwin(np.concatenate([base, front_cols, gal_cols], axis=-1), 'all_new')
    if on('bigger_trees'):
        fit_xwin(base, 'bigger_trees', params=dict(max_leaf_nodes=31, min_samples_leaf=200,
                                                    learning_rate=0.05))
    if on('softmax'):
        has_va = np.zeros_like(tr)
        tr_idx = np.nonzero(tr)[0]
        has_va[tr_idx[int(len(tr_idx) * 0.88):]] = True
        tr2 = tr & ~has_va
        for label, D, off in (('softmax_from_xodds', np.concatenate([base, front_cols, gal_cols], axis=-1),
                               np.log(np.clip(P, 1e-9, 1))),):
            t0 = time.time()
            Wp = softmax_boost(D.astype(np.float32), mask, won, off, tr2, has_va, te, log)
            ll = winner_ll(Wp[te], won[te])
            bl, coef = blend([q, Wp], won, mask, te1, te2)
            res[label] = {'winner_ll': round(ll, 4), 'vs_market': round(ll - res['market'], 4),
                          'blend_2nd_half': round(bl, 4), 'blend_gain': round(res['market_2nd_half'] - bl, 4),
                          'blend_coef': coef, 'secs': round(time.time() - t0)}
            log(f"  {label:22s} {res[label]}")
    if on('resid_market'):
        # The final model as a correction to the market: race-softmax trees
        # from log market shares, with and without xodds among the inputs.
        va = np.zeros_like(tr)
        tr_idx = np.nonzero(tr)[0]
        va[tr_idx[int(len(tr_idx) * 0.88):]] = True
        tr2 = tr & ~va
        lq = np.log(np.clip(q, 1e-9, 1))
        full = np.concatenate([base, front_cols, gal_cols], axis=-1)
        rating = np.concatenate([XS.xwin_design(Xa, mask, S1, shape), front_cols, gal_cols], axis=-1)
        for label, D in (('resid_market', full), ('resid_market_rating', rating)):
            t0 = time.time()
            Wp = softmax_boost(D.astype(np.float32), mask, won, lq, tr2, va, te, log)
            preds[label] = Wp
            res[label] = {'winner_ll': round(winner_ll(Wp[te], won[te]), 4),
                          'vs_market': round(winner_ll(Wp[te], won[te]) - res['market'], 4),
                          '2nd_half': round(winner_ll(Wp[te2], won[te2]), 4),
                          'gain_2nd_half': round(res['market_2nd_half'] - winner_ll(Wp[te2], won[te2]), 4),
                          'secs': round(time.time() - t0)}
            log(f"  {label:22s} {res[label]}")
    if on('lgbm_edge'):
        va = np.zeros_like(tr)
        tr_idx = np.nonzero(tr)[0]
        va[tr_idx[int(len(tr_idx) * 0.88):]] = True
        tr2 = tr & ~va
        lq = np.log(np.clip(q, 1e-9, 1))
        full = np.concatenate([base, front_cols, gal_cols], axis=-1).astype(np.float32)
        t0 = time.time()
        Wp, _ = lgbm_softmax(full, mask, won, lq, tr2, va, te, log)
        preds['lgbm_edge'] = Wp
        res['lgbm_edge'] = {'winner_ll': round(winner_ll(Wp[te], won[te]), 4),
                            'vs_market': round(winner_ll(Wp[te], won[te]) - res['market'], 4),
                            '2nd_half': round(winner_ll(Wp[te2], won[te2]), 4),
                            'gain_2nd_half': round(res['market_2nd_half'] - winner_ll(Wp[te2], won[te2]), 4),
                            'secs': round(time.time() - t0)}
        log(f"  {'lgbm_edge':22s} {res['lgbm_edge']}")
    if on('blend3'):
        for label in ('base', 'rating_share'):
            if label not in preds:
                D = base if label == 'base' else XS.xwin_design(Xa, mask, S1, shape)
                fit_xwin(D, label)
        bl, coef = blend([q, preds['base'], preds['rating_share']], won, mask, te1, te2)
        res['blend3'] = {'blend_2nd_half': round(bl, 4), 'blend_gain': round(res['market_2nd_half'] - bl, 4),
                         'coef [market, xwin, xwin on rating]': coef}
        log(f"  blend3 {res['blend3']}")
    if on('market_tests'):
        # Does xlead / xfront / xgal say anything the market doesn't? Winner
        # logits on [log market, extra] fitted on the out-of-fold years.
        lq = np.where(mask, np.log(np.clip(q, 1e-9, 1)), 0.0)
        out = {}
        for label, extra in (('market', None), ('+xlead', np.log(np.clip(L[..., 0], 1e-6, 1))),
                             ('+xfront', np.log(np.clip(Fr[..., 0], 1e-6, 1))),
                             ('+xgal', gal_cols[..., 0]),
                             ('+xodds', np.log(np.clip(P, 1e-9, 1)))):
            Z = lq[..., None] if extra is None else np.stack([lq, np.where(mask, extra, 0.0)], axis=-1)
            b = XO.fit_soft_logit(Z[tr], mask[tr], won[tr], l2=0.01)
            out[label] = {'winner_ll': round(winner_ll(XO.softmax(Z[te] @ b, mask[te]), won[te]), 4),
                          'coef': [round(float(x), 3) for x in b]}
        res['market_tests'] = out
        log(f"  market tests: {out}")
    return res


def oof(log):
    """Out-of-fold win chances of xwin (as built) and of the market
    correction, for every block: each block's from models fitted on the
    other blocks before the forward test; the forward test's from all of
    them. -> out/race_lab/oof.npz (entry ids, block, xwin, edge)."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    z = load()
    mask, won, q, blk, eids = z['mask'], z['won'].astype(float), z['q'], z['blk'], z['eids']
    K = int(blk.max())
    X = z['X'].astype(float)
    P = np.where(mask, np.nan_to_num(z['P'], nan=0.0), 0.0)
    has = (won.sum(axis=1) == 1) & np.isfinite(np.where(mask, z['P'], 0)).all(axis=1)
    Xa = np.concatenate([X, z['L']], axis=-1)
    base = XS.xwin_design(Xa, mask, P, z['shape'].astype(float))
    Fr = z['Fr']
    front_cols = np.concatenate([Fr, rel_cols(np.log(np.clip(Fr[..., 0], 1e-6, 1)), mask)[..., 1:]], axis=-1)
    gal_cols = rel_cols(np.log(np.clip(np.nan_to_num(z['G'], nan=0.15), 1e-4, 1)), mask)
    full = np.concatenate([base, front_cols, gal_cols], axis=-1).astype(np.float32)
    lq = np.log(np.clip(q, 1e-9, 1))
    Wx, We = np.zeros(mask.shape), np.zeros(mask.shape)
    for f in list(range(K)) + [K]:
        fit = (blk >= 0) & (blk < K) & has & (blk != f)
        out = blk == f
        kw = dict(BR.XWIN_GBM, early_stopping=True, validation_fraction=0.12, n_iter_no_change=60)
        m = HistGradientBoostingClassifier(random_state=42, **kw)
        m.fit(base[fit][mask[fit]].astype(np.float32), won[fit][mask[fit]].astype(int))
        Wx[out] = XS.predict_xwin({'model': m}, base[out].astype(np.float32), mask[out])
        idx = np.nonzero(fit)[0]
        va = np.zeros_like(fit)
        va[idx[int(len(idx) * 0.88):]] = True
        tr2 = fit & ~va
        Wf = softmax_boost(full, mask, won, lq, tr2, va, out, log)
        We[out] = Wf[out]
        log(f"[oof] block {f}: xwin {winner_ll(Wx[out & has], won[out & has]):.4f} · edge "
            f"{winner_ll(We[out & has], won[out & has]):.4f} · market {winner_ll(q[out & has], won[out & has]):.4f}")
    np.savez_compressed(OUT / 'oof.npz', eids=eids[mask], blk=np.broadcast_to(blk[:, None], mask.shape)[mask],
                        xwin=Wx[mask], edge=We[mask])
    log(f"-> {OUT / 'oof.npz'}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('cmd', choices=['build', 'exp', 'oof'])
    ap.add_argument('names', nargs='*')
    ap.add_argument('--since', default='2012-01-01')
    ap.add_argument('--lead-from', default='2014-01-01')
    ap.add_argument('--rows-from', default='2021-01-01')
    ap.add_argument('--test-from', default='2025-07-01')
    ap.add_argument('--folds', type=int, default=4)
    args = ap.parse_args()
    log = _log_factory()
    if args.cmd == 'build':
        build(args, log)
    elif args.cmd == 'oof':
        oof(log)
    else:
        res = experiments(args.names, log)
        path = OUT / f"exp_{dt.datetime.now():%Y%m%d_%H%M}.json"
        path.write_text(json.dumps(res, indent=1, default=float))
        log(f"results -> {path}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
