"""Build the race models in one go: rating -> xlead, xfront (auto, volt), xgal
-> xodds -> xwin -> xedge.

1. The rating pass (core/xodds.py) streams races since --since: the price
   rating reads Nordic win pools, the results rating (core/results_pass.py)
   finishing orders in every country. It yields stage-2 rows for Swedish
   priced races from --rows-from, and a strength snapshot of every Swedish
   starter from --lead-from.
2. The xlead pass (core/xlead.py) reads that snapshot as extra inputs. Its
   features train two labels: xlead (in front after 100–150 m) and xfront
   (in front at 700 m, read off the comment's timeline). xgal is the chance
   of a gallop, from entry_features.
3. Stacking (core/xstack.py). The rows from --rows-from up to --test-from
   are cut into --folds time blocks. Each block's xlead, xfront, xgal and
   xodds values come from models fitted without that block, so the next
   model up only ever learns from out-of-fold inputs. The forward test
   (--test-from on) is predicted by models fitted on everything before it,
   end to end.
4. xwin and xedge are race-softmax trees (LightGBM, the winner's
   likelihood within its race as the objective) on the same inputs, grown
   from a starting point: xwin from the xodds share, xedge from the
   market's closing shares. Both keep out-of-fold chances for every block.
5. The rating pass is scored as a model of its own (its strength sum per
   race), with diagnostics of the data under it: evidence, horses raced
   abroad, the last comment, the market's reaction to the last race.
6. Every model is refit on all rows and registered in ml_model (targets
   'rating', 'y_lead' ×2, 'y_front' ×2, 'xgal', 'xodds', 'xwin', 'xedge');
   scripts/score_upcoming chains them. Their out-of-sample predictions go
   to ml_backtest and are settled against the stored ATG pools
   (scripts/eval_pools.py).

Run under the ML venv:
    .venv-ml/bin/python -m scripts.build_race_models
    .venv-ml/bin/python -m scripts.build_race_models --dry-run --ablation
"""
from __future__ import annotations

import argparse
import datetime as dt
import gc
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from psycopg2.extras import Json  # noqa: E402

from core import xlead as XL  # noqa: E402
from core import xodds as XO  # noqa: E402
from core import xstack as XS  # noqa: E402
from core.db import get_connection  # noqa: E402
from scripts.train_model import pick_model_name  # noqa: E402

ARTIFACT_DIR = _ROOT / "web" / "ml_artifacts"
LEAD_GBM = dict(learning_rate=0.05, max_iter=800, max_leaf_nodes=31, min_samples_leaf=100,
                l2_regularization=1.0)
XODDS_L2 = 1.0
XODDS_GBM = dict(learning_rate=0.05, max_iter=1500, max_leaf_nodes=31, min_samples_leaf=200)
XWIN_GBM = dict(learning_rate=0.03, max_iter=2000, max_leaf_nodes=15, min_samples_leaf=400,
                l2_regularization=2.0)
# xwin / xedge: LightGBM on the race-softmax objective.
CHAIN_LGB = dict(learning_rate=0.03, num_leaves=15, min_data_in_leaf=400, lambda_l2=2.0,
                 feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, num_threads=4,
                 seed=42, verbose=-1)
# xgal: the y_gal model's inputs, from entry_features.
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
# The first xodds version's inputs: everything before the comment / record features.
XODDS_BASE = XO.FEATURES.index('rec_rel')
# What each part of the rating pass's output stands for, for grouped importances.
PASS_GROUPS = {
    'rating strength': ['mu_h', 'mu_d', 'mu_t', 'ctx', 'sd_h'],
    'evidence & rest': ['debut', 'log_obs', 'no_last', 'log_days', 'break90', 'young_days', 'age3', 'age4'],
    'results vs the market': ['fm', 'g_last', 'won_last', 'placed_last', 'gal_last', 'dq_last'],
    'times & margins': ['rec_rel', 'rec_none', 'mg_avg', 'mg_last', 'gal_rate'],
    'last comment': ['tr_death', 'tr_wide', 'tr_disturbed', 'tr_trouble', 'tr_finish', 'tr_strength'],
    'equipment & earnings': ['bare', 'bare_delta', 'am', 'am_delta', 'log_earn', 'earn_ps'],
    'stable / driver form': ['t_form', 'd_form'],
    'results rating': XO.R_FEATURES,
}
assert sorted(sum(PASS_GROUPS.values(), [])) == sorted(XO.FEATURES)
XODDS_GROUPS = {**PASS_GROUPS, 'xlead': XS.LEAD_COLS}
XWIN_GROUPS = {'xodds share': XS.XWIN_COLS, **PASS_GROUPS, 'results rating': XO.R_FEATURES + XS.RS_COLS,
               'xlead': XS.LEAD_COLS, 'race shape': XS.SHAPE}
CHAIN_GROUPS = {**XWIN_GROUPS, 'xfront': XS.FRONT_COLS, 'xgal': XS.GAL_COLS}
assert sorted(sum(CHAIN_GROUPS.values(), [])) == sorted(XS.CHAIN_FEATURES)


def blocks(ts, start, cut, k):
    """Block per row: -1 before `start` (always trained on), 0..k-1 the
    out-of-fold blocks, k the forward test."""
    edges = np.linspace(start, cut, k + 1).round().astype(int)
    b = np.full(len(ts), -1)
    for i in range(k):
        b[(ts >= edges[i]) & (ts < edges[i + 1])] = i
    b[ts >= cut] = k
    return b


# ── metrics ──────────────────────────────────────────────────────────────

def race_metrics(df, col):
    """Leader log-loss / top-1 over races, from per-race normalised shares."""
    p = df[col].clip(1e-6, 1)
    lead = df[df['y'] == 1]
    top = df.loc[df.groupby('race_id')[col].idxmax()]
    return {'races': int(df['race_id'].nunique()),
            'logloss': round(float(-np.log(p[lead.index]).mean()), 4),
            'top1': round(float(top['y'].mean()), 4),
            'p_leader': round(float(p[lead.index].mean()), 4)}


def binary_metrics(p, y):
    from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
    base = float(y.mean())
    d = pd.DataFrame({'p': p, 'y': y})
    d['bin'] = pd.qcut(d['p'].rank(method='first'), 10, labels=False)
    cal = d.groupby('bin').agg(pred=('p', 'mean'), actual=('y', 'mean'), n=('y', 'size')).reset_index()
    return {
        'n': int(len(y)), 'base_rate': round(base, 4),
        'roc_auc': round(float(roc_auc_score(y, p)), 4),
        'pr_auc': round(float(average_precision_score(y, p)), 4),
        'log_loss': round(float(log_loss(y, np.clip(p, 1e-6, 1 - 1e-6))), 4),
        'brier': round(float(brier_score_loss(y, p)), 4),
        'calibration': [{'decile': int(r.bin) + 1, 'pred': round(float(r.pred), 4),
                         'actual': round(float(r.actual), 4), 'n': int(r.n)} for r in cal.itertuples()],
        'top_decile_lift': round(float(cal.iloc[-1]['actual'] / base), 3) if base else None,
    }


def _ce(q, p, mask):
    return float(-np.sum(np.where(mask, q * np.log(np.clip(p, 1e-12, 1)), 0.0)) / mask.shape[0])


def win_report(p, won, mask, q=None):
    """Winner log-loss / top-1 over races with one winner; agreement with the market."""
    has = won.sum(axis=1) == 1
    pw = (p * won).sum(axis=1)[has]
    pick = np.where(mask, p, -1).argmax(axis=1)
    out = {'winner_logloss': round(float(-np.log(np.clip(pw, 1e-12, 1)).mean()), 4),
           'top1': round(float(won[np.arange(len(pick)), pick][has].mean()), 4),
           'p_winner': round(float(pw.mean()), 4), 'races': int(has.sum())}
    if q is not None:
        out['cross_entropy_vs_market'] = round(_ce(q, p, mask), 4)
        if p is not q:
            lp, lq = np.log(np.clip(p[mask], 1e-9, 1)), np.log(np.clip(q[mask], 1e-9, 1))
            out['corr_log_odds'] = round(float(np.corrcoef(lp, lq)[0, 1]), 4)
            same = pick == np.where(mask, q, -1).argmax(axis=1)
            out['same_favourite'] = round(float(same.mean()), 4)
    return out


def calibration(p, won, mask, bins=10):
    """Predicted vs actual win rate by decile of the predicted chance."""
    d = pd.DataFrame({'p': p[mask], 'y': won[mask]})
    d['bin'] = pd.qcut(d['p'].rank(method='first'), bins, labels=False)
    cal = d.groupby('bin').agg(pred=('p', 'mean'), actual=('y', 'mean'), n=('y', 'size')).reset_index()
    return [{'decile': int(r.bin) + 1, 'pred': round(float(r.pred), 4),
             'actual': round(float(r.actual), 4), 'n': int(r.n)} for r in cal.itertuples()]


# ── xlead ────────────────────────────────────────────────────────────────

def lead_fit(X, y, *, max_iter=None):
    from sklearn.ensemble import HistGradientBoostingClassifier
    kw = dict(LEAD_GBM)
    if max_iter is not None:
        kw.update(max_iter=max_iter, early_stopping=False)
    else:
        kw.update(early_stopping=True, validation_fraction=0.12, n_iter_no_change=30)
    model = HistGradientBoostingClassifier(categorical_features='from_dtype', random_state=42, **kw)
    model.fit(X, y)
    return model


def lead_stage(df, od, method, start, cut, K, ablation, log, name='xlead'):
    """Forward test, out-of-fold shares for the stack, final refit. Also
    xfront's, on its own labels: the same features, a later moment."""
    from sklearn.inspection import permutation_importance
    F = XL.ALL_FEATURES
    m = method == 'auto'
    df = df[df['auto'] == m].copy()
    od = od[od['auto'] == m].copy()
    cats = {'track': sorted(set(df['track'].dropna()) | set(od['track'].dropna()))}
    XL.set_categories(df, cats)
    XL.set_categories(od, cats)
    df['blk'] = blocks(df['t'].values, start, cut, K)
    od['blk'] = blocks(od['t'].values, start, cut, K)
    train, test = df[df['blk'] < K], df[df['blk'] == K].copy()
    log(f"[{name} {method}] train {len(train):,} rows ({train['race_id'].nunique():,} races) · "
        f"test {len(test):,} ({test['race_id'].nunique():,}) · unlabelled {len(od):,}")

    model = lead_fit(train[F], train['y'].astype(int))
    n_iter = int(model.n_iter_)
    test['raw'] = model.predict_proba(test[F])[:, 1]
    test['p'] = XL.normalize(test, 'raw')
    test['p_base'] = XL.normalize(test, 'base')
    test['p_unif'] = 1.0 / test.groupby('race_id')['y'].transform('size')
    rep = {'model': race_metrics(test, 'p'), 'post_baseline': race_metrics(test, 'p_base'),
           'uniform': race_metrics(test, 'p_unif')}
    if ablation:
        first = lead_fit(train[XL.BASE_FEATURES], train['y'].astype(int))
        test['raw1'] = first.predict_proba(test[XL.BASE_FEATURES])[:, 1]
        test['p1'] = XL.normalize(test, 'raw1')
        rep['first_version'] = race_metrics(test, 'p1')
    log(f"[{name} {method}] forward test: " + " | ".join(f"{k} {v['logloss']}/{v['top1']:.1%}"
                                                       for k, v in rep.items()) + f" · trees {n_iter}")

    rng = np.random.RandomState(42)
    idx = rng.choice(len(test), size=min(20000, len(test)), replace=False)
    perm = permutation_importance(model, test[F].iloc[idx], test['y'].iloc[idx].astype(int),
                                  scoring='roc_auc', n_repeats=3, random_state=42, n_jobs=-1)
    imps = sorted([{'feature': f, 'importance': round(float(v), 5)}
                   for f, v in zip(F, perm.importances_mean)], key=lambda d: -d['importance'])
    log(f"[{name} {method}] top features: {[d['feature'] for d in imps[:12]]}")

    parts = []
    for k in range(K):
        fold = lead_fit(train.loc[train['blk'] != k, F], train.loc[train['blk'] != k, 'y'].astype(int),
                        max_iter=n_iter)
        for frame in (df, od):
            sub = frame[frame['blk'] == k]
            if len(sub):
                parts.append(pd.DataFrame({'race_id': sub['race_id'].values, 'entry_id': sub['entry_id'].values,
                                           'blk': k, 'raw': fold.predict_proba(sub[F])[:, 1]}))
    for frame in (df, od):
        sub = frame[frame['blk'] == K]
        if len(sub):
            parts.append(pd.DataFrame({'race_id': sub['race_id'].values, 'entry_id': sub['entry_id'].values,
                                       'blk': K, 'raw': model.predict_proba(sub[F])[:, 1]}))
    oof = pd.concat(parts, ignore_index=True)
    oof['p'] = XL.normalize(oof, 'raw')
    log(f"[{name} {method}] out-of-fold shares: {len(oof):,} starters")

    final = lead_fit(df[F], df['y'].astype(int), max_iter=n_iter)
    metrics = {'test': {**binary_metrics(test['p'].values, test['y'].values), 'race': rep},
               'importances': imps, 'n_iter': n_iter}
    counts = {'n_total': int(len(df)), 'n_train': int(len(train)), 'n_test': int(len(test)),
              'races_train': int(train['race_id'].nunique()), 'races_test': int(test['race_id'].nunique()),
              'years': [dt.date.fromordinal(int(df['t'].min())).year,
                        dt.date.fromordinal(int(df['t'].max())).year]}
    bundle = {'model': final, 'features': F, 'categories': cats, 'method': method}
    return bundle, metrics, counts, oof[['entry_id', 'blk', 'p']]


# ── xgal ─────────────────────────────────────────────────────────────────

def gal_rows(conn, log):
    """Every finished Swedish start since 2014 with a gallop label, in the
    xgal model's inputs (streamed, so the raw rows never all sit in memory)."""
    cols = ', '.join(f'ef.{c}' for c in GAL_FEATURES)
    cur = conn.cursor(name='race_models_gal')
    cur.itersize = 100000
    cur.execute(f"""
        SELECT ef.entry_id, ef.race_date, ef.y_gal::int, {cols}
        FROM entry_features ef
        JOIN track t ON t.track_id = ef.track_id
        JOIN entry e ON e.entry_id = ef.entry_id
        WHERE t.country = 'SE' AND ef.race_date >= '2014-01-01' AND ef.race_date < CURRENT_DATE
          AND NOT COALESCE(e.withdrawn, false) AND ef.y_gal IS NOT NULL
    """)
    parts = []
    while True:
        rows = cur.fetchmany(100000)
        if not rows:
            break
        part = pd.DataFrame(rows, columns=['entry_id', 'race_date', 'y'] + GAL_FEATURES)
        for c in GAL_FEATURES:
            if c not in GAL_CATEGORICAL:
                part[c] = pd.to_numeric(part[c], errors='coerce').astype('float32')
        parts.append(part)
    cur.close()
    conn.commit()
    df = pd.concat(parts, ignore_index=True)
    cats = {c: sorted(df[c].dropna().unique().tolist()) for c in GAL_CATEGORICAL}
    XS.gal_frame(df, GAL_FEATURES, cats)
    log(f"[xgal] {len(df):,} rows, gallop rate {df['y'].mean():.1%}")
    return df, cats


def gal_stage(conn, start, cut, K, log):
    """xgal: forward test, out-of-fold chances from `start` on, final refit."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    df, cats = gal_rows(conn, log)
    t = np.array([d.toordinal() for d in df['race_date']])
    blk = blocks(t, start, cut, K)

    def fit(m, max_iter=None):
        kw = dict(GAL_GBM)
        if max_iter is None:
            kw.update(early_stopping=True, validation_fraction=0.1, n_iter_no_change=30)
        else:
            kw.update(max_iter=max_iter, early_stopping=False)
        g = HistGradientBoostingClassifier(categorical_features='from_dtype', random_state=42, **kw)
        g.fit(df.loc[m, GAL_FEATURES], df.loc[m, 'y'])
        return g

    first = fit(blk < K)
    n_iter = int(first.n_iter_)
    fw = blk == K
    p = first.predict_proba(df.loc[fw, GAL_FEATURES])[:, 1]
    test = binary_metrics(p, df.loc[fw, 'y'].values)
    log(f"[xgal] forward test auc {test['roc_auc']:.4f} · log-loss {test['log_loss']:.4f} · trees {n_iter}")
    oof = dict(zip(df.loc[fw, 'entry_id'].astype(int), p.astype(float)))
    folds = dict.fromkeys(oof, K)
    for k in range(K):
        g = fit((blk < K) & (blk != k), max_iter=n_iter)
        m = blk == k
        ids = df.loc[m, 'entry_id'].astype(int).tolist()
        oof.update(zip(ids, g.predict_proba(df.loc[m, GAL_FEATURES])[:, 1].astype(float)))
        folds.update(dict.fromkeys(ids, k))
        log(f"[xgal] fold {k + 1}/{K}: {int(m.sum()):,} starts out of fold")
        del g
        gc.collect()
    final = fit(np.ones(len(df), bool), max_iter=n_iter)
    counts = {'n_total': int(len(df)), 'n_train': int((blk < K).sum()), 'n_test': int(fw.sum()),
              'years': [int(df['race_date'].min().year), int(df['race_date'].max().year)]}
    bundle = {'model': final, 'features': GAL_FEATURES, 'categories': cats}
    del df
    gc.collect()
    return bundle, {'test': test, 'n_iter': n_iter}, counts, oof, folds


# ── xodds ────────────────────────────────────────────────────────────────

def xodds_fit(X, mask, q, *, max_iter=None):
    """Logit, then residual trees on the logit's leftovers in log q."""
    from sklearn.ensemble import HistGradientBoostingRegressor
    mu, sd = XO.standardize(X, mask)
    art = {'mu': mu, 'sd': sd, 'beta': XO.fit_soft_logit((X - mu) / sd, mask, q, l2=XODDS_L2)}
    S = XO.logit_scores(art, X)
    kw = dict(XODDS_GBM)
    if max_iter is not None:
        kw.update(max_iter=max_iter, early_stopping=False)
    else:
        kw.update(early_stopping=True, validation_fraction=0.1, n_iter_no_change=30)
    gbm = HistGradientBoostingRegressor(random_state=42, **kw)
    gbm.fit(XO.design(S, X, mask)[mask], XO.residual_target(S, mask, q)[mask],
            sample_weight=np.sqrt(q[mask]))
    art['gbm'] = gbm
    return art


def xodds_stage(X, L, mask, q, won, blk, K, ablation, log):
    Xa = np.concatenate([X, L], axis=-1)
    tr, te = blk < K, blk == K
    log(f"[xodds] train {int(mask[tr].sum()):,} runners ({int(tr.sum()):,} races) · "
        f"test {int(mask[te].sum()):,} ({int(te.sum()):,})")
    art = xodds_fit(Xa[tr], mask[tr], q[tr])
    n_iter = int(art['gbm'].n_iter_)
    mt, qt, wt = mask[te], q[te], won[te]
    P_te = XO.predict(art, Xa[te], mt)
    k = {f: i for i, f in enumerate(XO.FEATURES)}
    Xt = X[te]
    s1 = XO.softmax(Xt[..., k['mu_h']] + Xt[..., k['mu_d']] + Xt[..., k['mu_t']] + Xt[..., k['ctx']], mt)
    unif = np.where(mt, 1.0, 0.0)
    unif /= unif.sum(axis=1, keepdims=True)
    rep = {'market': win_report(qt, wt, mt, qt), 'xodds': win_report(P_te, wt, mt, qt),
           'logit_only': win_report(XO.softmax(XO.logit_scores(art, Xa[te]), mt), wt, mt, qt),
           'stage1': win_report(s1, wt, mt, qt), 'uniform': win_report(unif, wt, mt, qt)}
    ent = _ce(qt, qt, mt)
    cu = rep['uniform']['cross_entropy_vs_market']

    def closed(p):
        return round((cu - _ce(qt, p, mt)) / (cu - ent), 4)

    rep['gap_closed'] = closed(P_te)
    rep['market_entropy'] = round(ent, 4)
    groups = xodds_group_importance(art, Xa[te], mt, qt, XODDS_GROUPS, XO.FEATURES + XS.LEAD_COLS)
    log("[xodds] input groups (cross-entropy vs market added when shuffled): "
        + ", ".join(f"{g['group']} {g['importance']:.5f} ({g['share']:.0%})" for g in groups))
    if ablation:
        a1 = xodds_fit(X[tr][..., :XODDS_BASE], mask[tr], q[tr])
        p1 = XO.predict(a1, Xt[..., :XODDS_BASE], mt)
        a2 = xodds_fit(X[tr], mask[tr], q[tr])
        p2 = XO.predict(a2, Xt, mt)
        rep['ablation'] = {
            'first_version': {**win_report(p1, wt, mt, qt), 'gap_closed': closed(p1)},
            'no_xlead': {**win_report(p2, wt, mt, qt), 'gap_closed': closed(p2)},
            'with_xlead': {**rep['xodds'], 'gap_closed': rep['gap_closed']},
        }
        del a1, a2
    for name, v in rep.items():
        log(f"  {name:12s} {v}")
    log(f"[xodds] closes {rep['gap_closed']:.1%} of the uniform → market gap · trees {n_iter}")
    coefs = sorted(zip(XO.FEATURES + XS.LEAD_COLS, art['beta']), key=lambda t: -abs(t[1]))
    log("  logit coefficients (standardised): " + ", ".join(f"{f} {b:+.3f}" for f, b in coefs[:14]))

    P = np.zeros(mask.shape)
    P[te] = P_te
    for f in range(K):
        idx = blk == f
        a = xodds_fit(Xa[tr & ~idx], mask[tr & ~idx], q[tr & ~idx], max_iter=n_iter)
        P[idx] = XO.predict(a, Xa[idx], mask[idx])
        log(f"[xodds] fold {f + 1}/{K}: {int(idx.sum()):,} races out of fold")
        del a
        gc.collect()
    final = xodds_fit(Xa, mask, q, max_iter=n_iter)
    metrics = {'test': rep, 'n_iter': n_iter, 'groups': groups,
               'coefficients': [{'feature': f, 'coef': round(float(b), 4)} for f, b in coefs]}
    return final, metrics, Xa, P


def xodds_group_importance(art, Xa, mask, q, groups, features, repeats=2):
    """Cross-entropy against the market added when a group of inputs is
    shuffled together across runners."""
    col = {f: i for i, f in enumerate(features)}
    base = _ce(q, XO.predict(art, Xa, mask), mask)
    flat = Xa[mask]
    rng = np.random.RandomState(42)
    out = []
    for name, cols in groups.items():
        ix = [col[c] for c in cols]
        loss = []
        for _ in range(repeats):
            shuffled = flat.copy()
            shuffled[:, ix] = flat[rng.permutation(len(flat))][:, ix]
            Xp = Xa.copy()
            Xp[mask] = shuffled
            loss.append(_ce(q, XO.predict(art, Xp, mask), mask) - base)
            del Xp, shuffled
        out.append({'group': name, 'inputs': cols, 'importance': round(float(np.mean(loss)), 5)})
    tot = sum(max(g['importance'], 0.0) for g in out) or 1.0
    for g in out:
        g['share'] = round(max(g['importance'], 0.0) / tot, 4)
    return out


# ── rating pass ──────────────────────────────────────────────────────────

def rating_stage(conn, Xa, mask, q, won, eids, blk, K, P, W, E, stage1, log):
    """The rating pass's own shares (horse + driver + stable + post) scored
    like a model, plus diagnostics of the data under it, all on the forward
    test next to the market, xodds, xwin and xedge."""
    S1 = XO.stage1_shares(Xa, mask)
    Rs = XO.results_shares(Xa, mask)
    fw = blk == K
    has = won.sum(axis=1) == 1
    te = fw & has
    mt, qt, wt = mask[te], q[te], won[te]
    unif = np.where(mt, 1.0, 0.0)
    unif /= unif.sum(axis=1, keepdims=True)
    rep = {'market': win_report(qt, wt, mt, qt), 'rating': win_report(S1[te], wt, mt, qt),
           'results': win_report(Rs[te], wt, mt, qt),
           'xodds': win_report(P[te], wt, mt, qt), 'xwin': win_report(W[te], wt, mt, qt),
           'xedge': win_report(E[te], wt, mt, qt), 'uniform': win_report(unif, wt, mt, qt)}
    mk, rt, un = (rep[k]['winner_logloss'] for k in ('market', 'rating', 'uniform'))
    rep['gap_closed'] = round((un - rt) / (un - mk), 4)
    rep['results_gap_closed'] = round((un - rep['results']['winner_logloss']) / (un - mk), 4)
    cu, ent = _ce(qt, unif, mt), _ce(qt, qt, mt)
    rep['market_gap_closed'] = round((cu - _ce(qt, S1[te], mt)) / (cu - ent), 4)
    rep['calibration'] = calibration(S1[te], wt, mt)
    log(f"[rating] forward test: winner log-loss {rt} (market {mk}, uniform {un}) · closes "
        f"{rep['gap_closed']:.1%} of the winner gap, {rep['market_gap_closed']:.1%} of the gap to the market")
    rep['results_adds'] = results_adds(Xa, mask, won, blk, K, {'market': q, 'rating': S1, 'xodds': P}, log)
    diag = rating_diagnostics(conn, Xa[te], mt, qt, wt, eids[te],
                              {'market': qt, 'rating': S1[te], 'results': Rs[te], 'xodds': P[te], 'xwin': W[te],
                               'xedge': E[te]}, log)
    return {'stage1': stage1, 'test': rep, 'diagnostics': diag}, S1


def results_adds(Xa, mask, won, blk, K, bases, log):
    """What the results rating adds on top of each base: winner logits
    [log base] and [log base, results strength], fitted on the out-of-fold
    years, winner log-loss on the forward test."""
    has = won.sum(axis=1) == 1
    tr, te = (blk >= 0) & (blk < K) & has, (blk == K) & has
    rs = np.where(mask, XO.results_strength(Xa), 0.0)
    out = []
    for name, p in bases.items():
        lp = np.where(mask, np.log(np.clip(p, 1e-9, 1)), 0.0)
        row = {'base': name}
        for key, Z in (('alone', lp[..., None]), ('with_results', np.stack([lp, rs], axis=-1))):
            b = XO.fit_soft_logit(Z[tr], mask[tr], won[tr], l2=0.1)
            pt = XO.softmax(Z[te] @ b, mask[te])
            row[key] = round(float(-np.log(np.clip((pt * won[te]).sum(axis=1), 1e-12, 1)).mean()), 4)
            row['coef_' + key] = [round(float(x), 3) for x in b]
        row['gain'] = round(row['alone'] - row['with_results'], 4)
        out.append(row)
    log("[rating] results rating adds (winner log-loss, forward test): "
        + ", ".join(f"{r['base']} {r['alone']} -> {r['with_results']}" for r in out))
    return out


def _ae_rows(buckets, y, probs):
    """Per bucket: runners, winners, and actual / expected wins per model."""
    out = []
    for label, m in buckets:
        if not m.any():
            continue
        row = {'bucket': label, 'n': int(m.sum()), 'wins': int(y[m].sum())}
        for name, p in probs.items():
            row['ae_' + name] = round(float(y[m].sum() / max(p[m].sum(), 1e-9)), 3)
        out.append(row)
    return out


def rating_diagnostics(conn, X, mask, q, won, eids, models, log):
    k = {f: i for i, f in enumerate(XO.FEATURES)}
    col = lambda f: X[..., k[f]][mask]  # noqa: E731
    y = won[mask]
    probs = {name: p[mask] for name, p in models.items()}
    E = eids[mask].astype(int)
    obs = np.rint(np.expm1(col('log_obs')))
    sd = col('sd_h')
    out = {'runners': int(mask.sum()), 'races': int(mask.shape[0])}

    edges = [(0, 0, 'never priced'), (1, 2, '1–2 prices'), (3, 5, '3–5'), (6, 15, '6–15'),
             (16, 40, '16–40'), (41, 10 ** 6, '41+')]
    eb = [(lab, (obs >= a) & (obs <= b)) for a, b, lab in edges]
    eb = [b for b in eb if b[1].any()]
    out['evidence'] = _ae_rows(eb, y, probs)
    for row, (_, m) in zip(out['evidence'], eb):
        row['sd'] = round(float(sd[m].mean()), 3)

    with conn.cursor() as cur:
        cur.execute("""
            SELECT x.entry_id, COUNT(p.entry_id),
                   COUNT(p.entry_id) FILTER (WHERE t.country IS NULL OR t.country <> ALL(%s))
            FROM entry x
            LEFT JOIN entry p ON p.horse_id = x.horse_id AND p.race_date < x.race_date
                             AND NOT COALESCE(p.withdrawn, false)
            LEFT JOIN race r ON r.race_id = p.race_id
            LEFT JOIN track t ON t.track_id = r.track_id
            WHERE x.entry_id = ANY(%s)
            GROUP BY x.entry_id
        """, (list(XO.COUNTRIES), E.tolist()))
        prior = {int(e): (int(n), int(f)) for e, n, f in cur.fetchall()}
    n_all = np.array([prior.get(e, (0, 0))[0] for e in E])
    n_abroad = np.array([prior.get(e, (0, 0))[1] for e in E])
    share = np.where(n_all > 0, n_abroad / np.maximum(n_all, 1), 0.0)
    fb = [('no earlier start', n_all == 0),
          ('Nordic starts only', (n_all > 0) & (n_abroad == 0)),
          ('some abroad (< 50%)', (n_abroad > 0) & (share < 0.5)),
          ('mostly abroad (≥ 50%)', (share >= 0.5) & (n_abroad < n_all)),
          ('only abroad', (n_all > 0) & (n_abroad == n_all))]
    fb = [b for b in fb if b[1].any()]
    out['abroad'] = _ae_rows(fb, y, probs)
    for row, (_, m) in zip(out['abroad'], fb):
        row['priced_median'] = int(np.median(obs[m]))
        row['starts_median'] = int(np.median(n_all[m]))
    log(f"[rating] abroad: " + ", ".join(f"{r['bucket']} {r['n']:,} (A/E rating {r['ae_rating']}, "
                                         f"market {r['ae_market']})" for r in out['abroad']))

    # The market's reaction to the last race: strength by the rating's rank
    # in the field (set by earlier prices), last race against its own odds.
    S = np.where(mask, models['rating'], -1.0)
    order = np.argsort(-S, axis=1)
    rank = np.empty_like(order)
    np.put_along_axis(rank, order, np.arange(S.shape[1])[None, :].repeat(S.shape[0], 0), axis=1)
    rank = rank[mask] + 1
    g, ran = col('g_last'), col('no_last') == 0
    qs = np.quantile(g[ran], [0.2, 0.4, 0.6, 0.8])
    gb = np.digitize(g, qs)
    g_names = ['far below its odds', 'below its odds', 'about its odds', 'above its odds', 'far above its odds']
    cells = []
    for sname, sm in (('rated 1st', rank == 1), ('rated 2nd–3rd', (rank >= 2) & (rank <= 3)),
                      ('rated 4th+', rank >= 4)):
        for b, gname in enumerate(g_names):
            for row in _ae_rows([(gname, ran & sm & (gb == b))], y, probs):
                cells.append({'strength': sname, **row})
    out['last_race'] = {'edges': [round(float(x), 3) for x in qs], 'cells': cells}

    tr = ['tr_death', 'tr_wide', 'tr_disturbed', 'tr_trouble', 'tr_finish', 'tr_strength']
    T = np.stack([col(f) for f in tr], axis=1)[ran]
    out['last_comment'] = {
        'with_last_start': int(ran.sum()),
        'any_signal': round(float((np.abs(T) > 0).any(axis=1).mean()), 4),
        'flags': {f[3:]: round(float((np.abs(T[:, i]) > 0).mean()), 4) for i, f in enumerate(tr)},
    }
    with conn.cursor() as cur:
        cur.execute("""
            SELECT t.country, EXTRACT(year FROM r.race_date)::int, COUNT(*), COUNT(ec.comment)
            FROM entry e
            JOIN race r ON r.race_id = e.race_id
            JOIN track t ON t.track_id = r.track_id
            LEFT JOIN entry_comment ec ON ec.entry_id = e.entry_id
            WHERE t.country = ANY(%s) AND r.race_date >= '2019-01-01' AND r.race_date < CURRENT_DATE
              AND NOT COALESCE(e.withdrawn, false)
            GROUP BY 1, 2 ORDER BY 1, 2
        """, (list(XO.COUNTRIES),))
        out['comment_coverage'] = [{'country': c, 'year': yr, 'starts': int(n),
                                    'share': round(int(m) / int(n), 4) if n else None}
                                   for c, yr, n, m in cur.fetchall()]
    log(f"[rating] last comment gives a signal for {out['last_comment']['any_signal']:.0%} of runners "
        f"with a last start")
    return out


# ── xwin / xedge ─────────────────────────────────────────────────────────

def winner_ll(p, won):
    return float(-np.log(np.clip((p * won).sum(axis=1), 1e-12, 1)).mean())


def _flat(D, mask, won, offset, m):
    """The races in m as flat runner arrays, with each race's first index and size."""
    n = mask[m].sum(axis=1)
    return (np.asarray(D[m][mask[m]], dtype=np.float32), won[m][mask[m]].astype(float),
            offset[m][mask[m]], np.concatenate([[0], np.cumsum(n)[:-1]]), n)


def _race_probs(z, starts, n):
    z = z - np.repeat(np.maximum.reduceat(z, starts), n)
    e = np.exp(z)
    return e / np.repeat(np.add.reduceat(e, starts), n)


def softmax_fit(D, mask, won, offset, fit, *, trees=None):
    """LightGBM grown on the race-softmax winner likelihood from `offset`
    (log shares): per runner the gradient is p − won, p its race's softmax.
    Without `trees` the last 12% of the `fit` races stop it early.
    Returns (booster, trees)."""
    import lightgbm as lgb
    if trees is None:
        idx = np.nonzero(fit)[0]
        va = np.zeros_like(fit)
        va[idx[int(len(idx) * 0.88):]] = True
        tr = fit & ~va
    else:
        tr, va = fit, None
    Xt, yt, ot, st, nt = _flat(D, mask, won, offset, tr)

    def objective(preds, data):
        p = _race_probs(ot + preds, st, nt)
        return p - yt, np.maximum(p * (1 - p), 1e-6)

    ds = lgb.Dataset(Xt, label=yt, feature_name=list(XS.CHAIN_FEATURES), free_raw_data=False)
    params = {**CHAIN_LGB, 'objective': objective, 'metric': 'None'}
    if va is None:
        return lgb.train(params, ds, num_boost_round=trees), trees
    Xv, yv, ov, sv, nv = _flat(D, mask, won, offset, va)

    def metric(preds, data):
        p = _race_probs(ov + preds, sv, nv)
        return 'winner_ll', float(-np.log(np.clip(p[yv == 1], 1e-12, 1)).mean()), False

    dv = lgb.Dataset(Xv, label=yv, reference=ds)
    model = lgb.train(params, ds, num_boost_round=3000, valid_sets=[dv], feval=metric,
                      callbacks=[lgb.early_stopping(100, verbose=False)])
    return model, int(model.best_iteration)


def _shares(booster, D, mask, offset, m, trees=None):
    """softmax(offset + trees) over the races in m."""
    F = np.zeros(mask[m].shape)
    F[mask[m]] = booster.predict(np.asarray(D[m][mask[m]], dtype=np.float32), num_iteration=trees)
    return XO.softmax(np.where(mask[m], offset[m], 0.0) + F, mask[m])


def softmax_groups(booster, D, mask, won, offset, groups, repeats=2):
    """Winner log-loss added when a group of inputs is shuffled together
    across runners; the starting point (offset) stays as it was."""
    col = {f: i for i, f in enumerate(XS.CHAIN_FEATURES)}
    flat = np.asarray(D[mask], dtype=np.float32)
    rng = np.random.RandomState(42)

    def loss(F):
        G = np.zeros(mask.shape)
        G[mask] = F
        return winner_ll(XO.softmax(np.where(mask, offset, 0.0) + G, mask), won)

    base = loss(booster.predict(flat))
    out = []
    for name, cols in groups.items():
        ix = [col[c] for c in cols]
        d = []
        for _ in range(repeats):
            sh = flat.copy()
            sh[:, ix] = flat[rng.permutation(len(flat))][:, ix]
            d.append(loss(booster.predict(sh)) - base)
            del sh
        out.append({'group': name, 'inputs': cols, 'importance': round(float(np.mean(d)), 5)})
    tot = sum(max(g['importance'], 0.0) for g in out) or 1.0
    for g in out:
        g['share'] = round(max(g['importance'], 0.0) / tot, 4)
    return out


def gain_importances(booster):
    """Each input's share of the trees' total split gain."""
    g = booster.feature_importance('gain')
    tot = float(g.sum()) or 1.0
    return sorted([{'feature': f, 'importance': round(float(v) / tot, 5)}
                   for f, v in zip(XS.CHAIN_FEATURES, g)], key=lambda d: -d['importance'])


def chain_stage(name, D, mask, offset, won, blk, K, log):
    """Forward test from trees fitted before it, out-of-fold shares for every
    earlier block, final refit on everything. Returns the stored bundle, the
    shares [R, N] of every block, the forward-test trees and their count."""
    import lightgbm as lgb
    has = won.sum(axis=1) == 1
    tr, te = (blk >= 0) & (blk < K) & has, (blk == K) & has
    log(f"[{name}] train {int(mask[tr].sum()):,} runners ({int(tr.sum()):,} races) · "
        f"test {int(mask[te].sum()):,} ({int(te.sum()):,})")
    model, trees = softmax_fit(D, mask, won, offset, tr)
    out = np.zeros(mask.shape)
    fw = blk == K
    out[fw] = _shares(model, D, mask, offset, fw, trees)
    log(f"[{name}] forward test winner log-loss {winner_ll(out[te], won[te]):.4f} · trees {trees}")
    for f in range(K):
        m, _ = softmax_fit(D, mask, won, offset, tr & (blk != f), trees=trees)
        out[blk == f] = _shares(m, D, mask, offset, blk == f)
        log(f"[{name}] fold {f + 1}/{K}: {int((blk == f).sum()):,} races out of fold")
        del m
        gc.collect()
    final, _ = softmax_fit(D, mask, won, offset, has & (blk >= 0), trees=trees)
    counts = {'n_total': int(mask[has].sum()), 'races_total': int(has.sum()),
              'n_train': int(mask[tr].sum()), 'n_test': int(mask[te].sum()),
              'races_train': int(tr.sum()), 'races_test': int(te.sum())}
    bundle = {'booster': lgb.Booster(model_str=final.model_to_string()), 'features': XS.CHAIN_FEATURES,
              'trees': trees, 'target': name}
    return bundle, out, model, trees, counts


def _uniform(mask):
    u = np.where(mask, 1.0, 0.0)
    return u / u.sum(axis=1, keepdims=True)


def xwin_fit(X, y, *, max_iter=None):
    """The earlier xwin: a classifier per runner, normalised per race
    (kept for the --ablation comparison)."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    kw = dict(XWIN_GBM)
    if max_iter is not None:
        kw.update(max_iter=max_iter, early_stopping=False)
    else:
        kw.update(early_stopping=True, validation_fraction=0.12, n_iter_no_change=60)
    model = HistGradientBoostingClassifier(random_state=42, **kw)
    model.fit(X, y)
    return model


def xwin_stage(D, mask, P, q, won, blk, K, ablation, log):
    """xwin: who wins, grown from the xodds share."""
    lp = np.log(np.clip(P, 1e-9, 1))
    bundle, W, model, trees, counts = chain_stage('xwin', D, mask, lp, won, blk, K, log)
    has = won.sum(axis=1) == 1
    tr, te = (blk >= 0) & (blk < K) & has, (blk == K) & has
    mt, wt, qt = mask[te], won[te], q[te]
    rep = {'market': win_report(qt, wt, mt, qt), 'xwin': win_report(W[te], wt, mt, qt),
           'xodds': win_report(P[te], wt, mt, qt), 'uniform': win_report(_uniform(mt), wt, mt, qt)}
    if ablation:
        c = xwin_fit(D[tr][mask[tr]], won[tr][mask[tr]].astype(int))
        rep['classifier'] = win_report(XS.predict_xwin({'model': c}, D[te], mt), wt, mt, qt)
        del c
    for k, v in rep.items():
        log(f"  {k:10s} {v}")
    mk, xo, xw, un = (rep[k]['winner_logloss'] for k in ('market', 'xodds', 'xwin', 'uniform'))
    rep['gap_closed'] = round((un - xw) / (un - mk), 4)
    rep['calibration'] = calibration(W[te], wt, mt)
    log(f"[xwin] closes {rep['gap_closed']:.1%} of the uniform → market winner log-loss gap "
        f"(xodds {(un - xo) / (un - mk):.1%})")
    groups = softmax_groups(model, D[te], mt, wt, lp[te], CHAIN_GROUPS)
    log("[xwin] input groups (winner log-loss added when shuffled): "
        + ", ".join(f"{g['group']} {g['importance']:.5f} ({g['share']:.0%})" for g in groups))
    imps = gain_importances(bundle['booster'])
    log(f"[xwin] top inputs by gain: {[d['feature'] for d in imps[:12]]}")
    metrics = {'test': rep, 'importances': imps, 'importance_kind': 'gain', 'groups': groups, 'n_iter': trees}
    return bundle, metrics, counts, W


def xedge_stage(D, mask, q, won, blk, K, ts, W, log):
    """xedge: where the closing odds are wrong, grown from the market's shares."""
    lq = np.log(np.clip(q, 1e-9, 1))
    bundle, E, model, trees, counts = chain_stage('xedge', D, mask, lq, won, blk, K, log)
    has = won.sum(axis=1) == 1
    tr, te = (blk >= 0) & (blk < K) & has, (blk == K) & has
    mt, wt, qt = mask[te], won[te], q[te]
    # The market against its own favourite-longshot bias: its log shares
    # times one power, fitted on the out-of-fold years.
    Z = np.where(mask, lq, 0.0)[..., None]
    b = XO.fit_soft_logit(Z[tr], mask[tr], won[tr], l2=0.01)
    Mr = XO.softmax(Z[te] @ b, mt)
    rep = {'market': win_report(qt, wt, mt, qt), 'xedge': win_report(E[te], wt, mt, qt),
           'market_recal': {**win_report(Mr, wt, mt, qt), 'power': round(float(b[0]), 3)},
           'xwin': win_report(W[te], wt, mt, qt), 'uniform': win_report(_uniform(mt), wt, mt, qt)}
    for k, v in rep.items():
        log(f"  {k:12s} {v}")
    rep['gain'] = round(rep['market']['winner_logloss'] - rep['xedge']['winner_logloss'], 4)
    rep['calibration'] = calibration(E[te], wt, mt)
    rep['blocks'] = []
    for f in range(K + 1):
        m = (blk == f) & has
        if m.any():
            day = lambda x: dt.date.fromordinal(int(x)).isoformat()  # noqa: E731
            row = {'block': f, 'forward': f == K, 'from': day(ts[m].min()), 'to': day(ts[m].max()),
                   'races': int(m.sum()), 'market': round(winner_ll(q[m], won[m]), 4),
                   'xedge': round(winner_ll(E[m], won[m]), 4)}
            row['gain'] = round(row['market'] - row['xedge'], 4)
            rep['blocks'].append(row)
    log("[xedge] gain on the market per block: "
        + ", ".join(f"{r['from'][:7]}…{r['to'][:7]} {r['gain']:+.4f}" for r in rep['blocks']))
    groups = softmax_groups(model, D[te], mt, wt, lq[te], CHAIN_GROUPS)
    log("[xedge] input groups (winner log-loss added when shuffled): "
        + ", ".join(f"{g['group']} {g['importance']:.5f} ({g['share']:.0%})" for g in groups))
    imps = gain_importances(bundle['booster'])
    log(f"[xedge] top inputs by gain: {[d['feature'] for d in imps[:12]]}")
    metrics = {'test': rep, 'importances': imps, 'importance_kind': 'gain', 'groups': groups, 'n_iter': trees}
    return bundle, metrics, counts, E


# ── registry ─────────────────────────────────────────────────────────────

def register(conn, target, algo, stem, bundle, slice_def, metrics, log):
    import joblib
    name = pick_model_name(conn)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    artifact = ARTIFACT_DIR / f"{stem}_{name}.joblib"
    joblib.dump(bundle, artifact)
    with conn.cursor() as cur:
        cur.execute("""
            INSERT INTO ml_model (name, scope, track_id, track_name, target, algo,
                                  slice_def, metrics, artifact_path)
            VALUES (%s, 'general', NULL, NULL, %s, %s, %s, %s, %s)
            RETURNING model_id
        """, (name, target, algo, Json(slice_def), Json(metrics), str(artifact.relative_to(_ROOT))))
        mid = cur.fetchone()[0]
    conn.commit()
    log(f"registered ml_model #{mid} '{name}' ({target}) -> {artifact.name}")
    return mid


def store_backtest(conn, model_id, target, method, eids, folds, probs, log):
    """A model's out-of-sample predictions into ml_backtest. Older models of
    the same target lose theirs: their pool results stay in their metrics."""
    import io
    d = pd.DataFrame({'e': np.asarray(eids, dtype=np.int64), 'f': np.asarray(folds, dtype=int),
                      'p': np.asarray(probs, dtype=float)}).drop_duplicates('e', keep='last')
    buf = io.StringIO()
    d.assign(m=model_id)[['m', 'e', 'f', 'p']].to_csv(buf, sep='\t', header=False, index=False,
                                                      float_format='%.6g')
    buf.seek(0)
    with conn.cursor() as cur:
        cur.execute("""
            DELETE FROM ml_backtest WHERE model_id IN (
                SELECT model_id FROM ml_model
                WHERE target = %s AND model_id <> %s
                  AND (%s::text IS NULL OR slice_def->>'method' = %s))
        """, (target, model_id, method, method))
        cur.copy_expert("COPY ml_backtest (model_id, entry_id, fold, prob) FROM STDIN", buf)
    conn.commit()
    log(f"ml_backtest #{model_id} ({target}{' ' + method if method else ''}): {len(d):,} starters")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--since', default='2012-01-01', help='history both passes read from')
    ap.add_argument('--lead-from', default='2014-01-01', help='first race that trains xlead')
    ap.add_argument('--rows-from', default='2021-01-01', help='first race in the xodds / xwin rows')
    ap.add_argument('--test-from', default='2025-07-01', help='forward test from here on')
    ap.add_argument('--folds', type=int, default=4, help='out-of-fold time blocks before the test')
    ap.add_argument('--ablation', action='store_true', help='also test the first versions\' inputs')
    ap.add_argument('--dry-run', action='store_true', help='evaluate only, register nothing')
    args = ap.parse_args()
    t0 = time.time()

    def log(msg):
        print(f"[{time.time() - t0:6.0f}s] {msg}", flush=True)

    since = dt.date.fromisoformat(args.since)
    lead_from = dt.date.fromisoformat(args.lead_from)
    rows_from = dt.date.fromisoformat(args.rows_from)
    test_from = dt.date.fromisoformat(args.test_from)
    start, cut, K = rows_from.toordinal(), test_from.toordinal(), args.folds

    conn = get_connection()
    try:
        cur = conn.cursor(name='race_models_market')
        cur.itersize = 20000
        mp = XO.MarketPass(today=dt.date.today(), bt_start=rows_from, snap_from=lead_from)
        mp.run(cur, since)
        cur.close()
        conn.commit()
        ce, nr, ent = mp.fit
        payout = round(mp.payout[0] / mp.payout[1], 4)
        log(f"market pass: {mp.priced:,} priced races · {len(mp.rows):,} stage-2 rows · "
            f"{len(mp.snap):,} snapshots · stage 1 {ce / nr:.4f} vs entropy {ent / nr:.4f} · "
            f"results rating {mp.rp.races:,} races, {len(mp.rp.H):,} horses")
        rows, snap, params, rparams = mp.rows, mp.snap, mp.p, mp.rp.p
        stage1 = {'cross_entropy': round(ce / nr, 4), 'entropy': round(ent / nr, 4), 'races': nr}
        del mp
        gc.collect()

        cur = conn.cursor(name='race_models_lead')
        cur.itersize = 20000
        lp = XL.LeadPass(today=dt.date.today(), train_from=lead_from, ext=snap, other_from=rows_from,
                         front=True)
        lp.run(cur, since)
        cur.close()
        conn.commit()
        log(f"lead pass: {lp.races:,} races, {lp.usable:,} with one clear early leader · "
            f"{len(lp.train):,} labelled rows · {len(lp.other):,} unlabelled · "
            f"{len(lp.front):,} rows with one starter in front at {XL.FRONT_M} m")
        lead_df, other_df, front_df = XL.frame(lp.train), XL.frame(lp.other), XL.frame(lp.front)
        del lp, snap
        gc.collect()

        lead_out, lead_oof = {}, {}
        for method in ('auto', 'volt'):
            bundle, metrics, counts, oof = lead_stage(lead_df, other_df, method, start, cut, K,
                                                      args.ablation, log)
            lead_out[method] = (bundle, metrics, counts, oof)
            lead_oof.update(zip(oof['entry_id'].astype(int), oof['p'].astype(float)))
        # xfront: the same features; every other starter of the stack's rows is unlabelled for it.
        every = pd.concat([lead_df[lead_df['t'] >= start], other_df], ignore_index=True)
        rest = every[~every['entry_id'].isin(set(front_df['entry_id']))]
        front_out, front_oof = {}, {}
        for method in ('auto', 'volt'):
            bundle, metrics, counts, oof = lead_stage(front_df, rest, method, start, cut, K, False, log,
                                                      name='xfront')
            front_out[method] = (bundle, metrics, counts, oof)
            front_oof.update(zip(oof['entry_id'].astype(int), oof['p'].astype(float)))
        shape_frames = [lead_df[lead_df['t'] >= start], other_df]
        del every, rest, front_df
        gc.collect()
        gal_bundle, gal_metrics, gal_counts, gal_oof, gal_folds = gal_stage(conn, start, cut, K, log)

        rids, ts, eids, X, mask, q, won = XO.tensors(rows)
        del rows
        gc.collect()
        blk = blocks(ts, start, cut, K)
        L = XS.lead_columns(eids, mask, lead_oof)
        xodds_art, xodds_metrics, Xa, P = xodds_stage(X, L, mask, q, won, blk, K, args.ablation, log)
        del X
        gc.collect()

        shape = XS.shape_columns(eids, mask, shape_frames)
        del lead_df, other_df, shape_frames
        D = XS.chain_design(Xa, mask, P, shape, XS.front_columns(eids, mask, front_oof),
                            XS.gal_columns(eids, mask, gal_oof))
        del shape
        gc.collect()
        log(f"chain design: {D.shape[-1]} inputs per runner ({D.nbytes / 1e6:.0f} MB)")
        xwin_bundle, xwin_metrics, xwin_counts, W = xwin_stage(D, mask, P, q, won, blk, K, args.ablation, log)
        xedge_bundle, xedge_metrics, xedge_counts, E = xedge_stage(D, mask, q, won, blk, K, ts, W, log)
        del D
        gc.collect()
        rating_metrics, S1 = rating_stage(conn, Xa, mask, q, won, eids, blk, K, P, W, E, stage1, log)

        if args.dry_run:
            log('done (dry run)')
            return 0

        stack = {'folds': K, 'rows_from': rows_from.isoformat(), 'test_since': test_from.isoformat()}
        n_te = int(mask[blk == K].sum())
        mids = {}
        mids['rating'] = register(conn, 'rating', 'MarketKalman', 'rating', {
            'params': params, 'results_params': rparams, 'features': ['mu_h', 'mu_d', 'mu_t', 'ctx'],
            'payout': payout,
        }, {
            'label': 'rating pass (horse + driver + stable + post, from win-pool prices)',
            'country': 'SE', 'history': list(XO.COUNTRIES), 'test_since': test_from.isoformat(),
            'starters_only': True, 'features': XO.FEATURES, 'n_features': len(XO.FEATURES),
            'hyperparams': {**params, 'results': rparams}, 'stack': {**stack, 'inputs': []},
            'note': 'prob = softmax of the pass\'s strength sum per race; outputs feed xlead, xfront, xodds, '
                    'xwin and xedge',
            'n_total': int(mask.sum()), 'races_total': int(mask.shape[0]),
            'n_train': int(mask.sum()) - n_te, 'n_test': n_te,
        }, rating_metrics, log)
        for method, (bundle, metrics, counts, _) in lead_out.items():
            mids[method] = register(conn, 'y_lead', 'HistGradientBoosting', f"y_lead_{method}", bundle, {
                'label': 'xlead (first 100–150 m)', 'method': method, 'country': 'SE',
                'test_since': test_from.isoformat(), 'starters_only': True,
                'features': XL.ALL_FEATURES, 'n_features': len(XL.ALL_FEATURES),
                'hyperparams': LEAD_GBM, 'refit_on_all': True, 'stack': stack,
                'note': 'label: the comment names exactly one early leader; shares normalised per race',
                **counts,
            }, metrics, log)
        for method, (bundle, metrics, counts, _) in front_out.items():
            mids['front_' + method] = register(conn, 'y_front', 'HistGradientBoosting', f"y_front_{method}", bundle, {
                'label': f'xfront (in front at {XL.FRONT_M} m)', 'method': method, 'country': 'SE',
                'test_since': test_from.isoformat(), 'starters_only': True,
                'features': XL.ALL_FEATURES, 'n_features': len(XL.ALL_FEATURES),
                'hyperparams': LEAD_GBM, 'refit_on_all': True, 'stack': stack,
                'note': f'label: the comment\'s timeline puts exactly one starter in front at {XL.FRONT_M} m '
                        '(two claims: the later takeover); shares normalised per race',
                **counts,
            }, metrics, log)
        mids['xgal'] = register(conn, 'xgal', 'HistGradientBoosting', 'xgal', gal_bundle, {
            'label': 'xgal (chance of a gallop, input to xwin / xedge)', 'country': 'SE',
            'test_since': test_from.isoformat(), 'source_table': 'entry_features',
            'features': GAL_FEATURES, 'n_features': len(GAL_FEATURES), 'categorical': GAL_CATEGORICAL,
            'hyperparams': GAL_GBM, 'refit_on_all': True, 'stack': stack,
            'note': 'label: galloped (entry_features.y_gal); its out-of-fold chances feed xwin and xedge',
            **gal_counts,
        }, gal_metrics, log)
        xodds_art.update(features=XO.FEATURES + XS.LEAD_COLS, params=params, results_params=rparams,
                         payout=payout)
        mids['xodds'] = register(conn, 'xodds', 'MarketKalman+Logit+GBM', 'xodds', xodds_art, {
            'label': 'xodds (market rating + logit + trees)', 'country': 'SE', 'history': list(XO.COUNTRIES),
            'test_since': test_from.isoformat(), 'starters_only': True,
            'features': XO.FEATURES + XS.LEAD_COLS, 'n_features': len(XO.FEATURES) + len(XS.LEAD_COLS),
            'hyperparams': {**params, 'l2': XODDS_L2, 'gbm': XODDS_GBM}, 'refit_on_all': True,
            'payout': payout, 'stack': {**stack, 'inputs': ['y_lead']},
            'note': 'prob = share of the win pool; odds = payout / prob',
            'n_total': int(mask.sum()), 'races_total': int(mask.shape[0]),
            'n_train': int(mask.sum()) - n_te, 'n_test': n_te,
        }, {**xodds_metrics, 'stage1': stage1}, log)
        chain_inputs = ['xodds', 'y_lead', 'y_front', 'xgal']
        mids['xwin'] = register(conn, 'xwin', 'LightGBM race-softmax', 'xwin', xwin_bundle, {
            'label': 'xwin (race-softmax trees from xodds)', 'country': 'SE',
            'test_since': test_from.isoformat(), 'starters_only': True,
            'features': XS.CHAIN_FEATURES, 'n_features': len(XS.CHAIN_FEATURES), 'offset': 'xodds',
            'hyperparams': CHAIN_LGB, 'refit_on_all': True, 'stack': {**stack, 'inputs': chain_inputs},
            'note': 'objective: the winner\'s likelihood within its race; the trees start from the log xodds share',
            **xwin_counts,
        }, xwin_metrics, log)
        mids['xedge'] = register(conn, 'xedge', 'LightGBM race-softmax', 'xedge', xedge_bundle, {
            'label': 'xedge (corrects the closing odds)', 'country': 'SE',
            'test_since': test_from.isoformat(), 'starters_only': True,
            'features': XS.CHAIN_FEATURES, 'n_features': len(XS.CHAIN_FEATURES), 'offset': 'market',
            'hyperparams': CHAIN_LGB, 'refit_on_all': True, 'stack': {**stack, 'inputs': chain_inputs},
            'note': 'prob = market share × tilt, renormalised per race. The trees start from the log market share '
                    'and never see the race\'s odds, so the live value stored is the tilt alone',
            **xedge_counts,
        }, xedge_metrics, log)
        log(f"registered: {mids}")

        fold = np.broadcast_to(blk[:, None], mask.shape)[mask]
        store_backtest(conn, mids['rating'], 'rating', None, eids[mask], fold, S1[mask], log)
        store_backtest(conn, mids['xodds'], 'xodds', None, eids[mask], fold, P[mask], log)
        store_backtest(conn, mids['xwin'], 'xwin', None, eids[mask], fold, W[mask], log)
        store_backtest(conn, mids['xedge'], 'xedge', None, eids[mask], fold, E[mask], log)
        for method, (_, _, _, oof) in lead_out.items():
            store_backtest(conn, mids[method], 'y_lead', method, oof['entry_id'], oof['blk'], oof['p'], log)
        for method, (_, _, _, oof) in front_out.items():
            store_backtest(conn, mids['front_' + method], 'y_front', method, oof['entry_id'], oof['blk'],
                           oof['p'], log)
        store_backtest(conn, mids['xgal'], 'xgal', None, list(gal_oof), [gal_folds[e] for e in gal_oof],
                       list(gal_oof.values()), log)

        from scripts.eval_pools import score_models
        with conn.cursor() as cur:
            cur.execute("SELECT model_id, target, name FROM ml_model WHERE model_id = ANY(%s)",
                        ([mids['rating'], mids['xodds'], mids['xwin'], mids['xedge']],))
            score_models(conn, cur.fetchall(), log=log)
    finally:
        conn.close()
    log('done')
    return 0


if __name__ == '__main__':
    sys.exit(main())
