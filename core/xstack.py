"""How the race models feed each other: xlead / xfront / xgal -> xodds -> xwin, xedge.

xlead and xfront (core/xlead.py) read the xodds rating pass's strength
snapshot; xodds stage 2 (core/xodds.py) takes each starter's xlead share as
two extra inputs. xwin and xedge share one design: the pass features, xlead,
xfront, xgal, the xodds share and where it sits in the field, and the race
shape. Both are race-softmax trees grown from a starting point: xwin from
the xodds share (who wins, before any odds exist), xedge from the market's
own shares (where the closing odds are wrong). xedge's trees never see the
race's odds, so a race's correction is a fixed tilt per starter, stored
alone and multiplied into whatever odds are on the board.

A stacked input is only honest if it was predicted by a model that never saw
that race: in training every xlead / xfront / xgal / xodds value fed upward
is out-of-fold (scripts/build_race_models.py); when scoring, the upstream
models never saw the upcoming race anyway.
"""
from __future__ import annotations

import datetime as dt

import numpy as np

from core import xlead as XL
from core import xodds as XO

LEAD_COLS = ['xlead', 'xlead_n']
# Race shape from the xlead pass (not in the market pass's own features).
SHAPE = ['post', 'front', 'till', 'distance', 'n_front', 'speed', 'speed_rank', 'h_lead_rate']
XWIN_COLS = ['xo_logp', 'xo_rel', 'xo_gap', 'xo_rank']
# Where the results rating sits in the field, beside where xodds puts the horse.
RS_COLS = ['rs_rel', 'rs_gap']
XWIN_FEATURES = XO.FEATURES + LEAD_COLS + XWIN_COLS + RS_COLS + SHAPE
FRONT_COLS = ['xfront', 'xfront_n', 'xfront_rel', 'xfront_gap']
GAL_COLS = ['xgal', 'xgal_rel', 'xgal_gap']
CHAIN_FEATURES = XWIN_FEATURES + FRONT_COLS + GAL_COLS
# Gallop chance of a starter without one (about the base rate).
GAL_FILL = 0.15


def predict_lead(df, bundles):
    """Per-race xlead shares for a frame of pass rows: entry id -> share.
    `bundles` maps 'auto' / 'volt' to a stored model bundle."""
    out = {}
    for method, b in bundles.items():
        if b is None:
            continue
        sub = df[df['auto'] == (method == 'auto')].copy()
        if sub.empty:
            continue
        XL.set_categories(sub, b['categories'])
        sub['raw'] = b['model'].predict_proba(sub[b['features']])[:, 1]
        sub['p'] = XL.normalize(sub, 'raw')
        out.update(zip(sub['entry_id'].astype(int), sub['p'].astype(float)))
    return out


def _lookup(eids, mask, table, fill=np.nan):
    v = np.full(mask.shape, fill, dtype=float)
    idx = np.nonzero(mask)
    v[idx] = [table.get(int(e), fill) for e in eids[idx]]
    return v


def lead_columns(eids, mask, lead):
    """[R, N, 2]: xlead share and share × field size. Starters the lead pass
    didn't see (abroad, tiny fields) get an even share."""
    n = mask.sum(axis=1, keepdims=True).astype(float)
    L = _lookup(eids, mask, lead)
    L = np.where(mask & np.isnan(L), 1.0 / np.maximum(n, 1), L)
    L = np.where(mask, L, 0.0)
    L /= np.maximum(L.sum(axis=1, keepdims=True), 1e-12)
    return np.stack([L, L * n], axis=-1)


def shape_columns(eids, mask, frames):
    """[R, N, len(SHAPE)] from xlead pass frames (NaN where missing)."""
    out = np.full(mask.shape + (len(SHAPE),), np.nan)
    tables = [{} for _ in SHAPE]
    for df in frames:
        if df is None or df.empty:
            continue
        ids = df['entry_id'].astype(int).tolist()
        for k, c in enumerate(SHAPE):
            tables[k].update(zip(ids, df[c].astype(float).tolist()))
    for k in range(len(SHAPE)):
        out[..., k] = _lookup(eids, mask, tables[k])
    return out


def xwin_design(Xa, mask, P, shape):
    """xodds stage-2 inputs + where the xodds share and the results rating
    sit in their field + race shape."""
    n = mask.sum(axis=1, keepdims=True)

    def rel_gap(a):
        a = np.where(mask, a, 0.0)
        return (a - a.sum(axis=1, keepdims=True) / np.maximum(n, 1),
                a - np.where(mask, a, -np.inf).max(axis=1, keepdims=True))

    lp = np.where(mask, np.log(np.clip(P, 1e-9, 1)), 0.0)
    rel, gap = rel_gap(lp)
    order = np.argsort(-np.where(mask, P, -1.0), axis=1)
    rank = np.empty_like(order)
    np.put_along_axis(rank, order, np.arange(P.shape[1])[None, :].repeat(P.shape[0], 0), axis=1)
    rank_pct = rank / np.maximum(n - 1, 1)
    rs_rel, rs_gap = rel_gap(XO.results_strength(Xa))
    extra = np.stack([lp, rel, gap, rank_pct, rs_rel, rs_gap], axis=-1)
    return np.concatenate([Xa, extra, shape], axis=-1)


def _rel(a, mask):
    """A per-runner value against its field's mean and its best."""
    n = np.maximum(mask.sum(axis=1, keepdims=True), 1)
    a0 = np.where(mask, a, 0.0)
    return a0 - a0.sum(axis=1, keepdims=True) / n, a0 - np.where(mask, a, -np.inf).max(axis=1, keepdims=True)


def front_columns(eids, mask, front):
    """[R, N, 4]: xfront share, share × field size, and its log share
    against the field's mean and best."""
    F = lead_columns(eids, mask, front)
    rel, gap = _rel(np.log(np.clip(F[..., 0], 1e-6, 1)), mask)
    return np.concatenate([F, np.stack([rel, gap], axis=-1)], axis=-1)


def gal_columns(eids, mask, gal):
    """[R, N, 3]: log gallop chance, against the field's mean and its max."""
    lg = np.log(np.clip(np.nan_to_num(_lookup(eids, mask, gal), nan=GAL_FILL), 1e-4, 1))
    rel, gap = _rel(lg, mask)
    return np.stack([np.where(mask, lg, 0.0), rel, gap], axis=-1)


def chain_design(Xa, mask, P, shape, Fr, G):
    """The xwin / xedge inputs, in CHAIN_FEATURES order."""
    return np.concatenate([xwin_design(Xa, mask, P, shape), Fr, G], axis=-1).astype(np.float32)


def predict_xwin(art, D, mask):
    """Per-race win probabilities (padded slots 0)."""
    raw = np.zeros(mask.shape)
    raw[mask] = art['model'].predict_proba(D[mask])[:, 1]
    raw = np.where(mask, raw, 0.0)
    return raw / np.maximum(raw.sum(axis=1, keepdims=True), 1e-12)


def softmax_scores(art, D, mask):
    """A race-softmax model's trees per runner (padded slots 0)."""
    F = np.zeros(mask.shape)
    if mask.any():
        F[mask] = art['booster'].predict(np.asarray(D[mask], dtype=np.float32))
    return F


def predict_softmax(art, D, mask, offset):
    """Per-race shares of softmax(offset + trees)."""
    return XO.softmax(np.where(mask, offset, 0.0) + softmax_scores(art, D, mask), mask)


def tilted(tilt, q):
    """xedge's chances from its stored tilts and the market's shares of one
    race (both per runner, any scale)."""
    p = np.asarray(tilt, float) * np.asarray(q, float)
    return p / max(p.sum(), 1e-12)


def predict_gal(conn, art, eids):
    """Gallop chance per entry id from entry_features, with the xgal model's
    inputs (scripts/build_race_models.py gal_stage)."""
    import pandas as pd
    eids = [int(e) for e in eids]
    if not eids:
        return {}
    cols = ', '.join(f'ef.{c}' for c in art['features'])
    with conn.cursor() as cur:
        cur.execute(f"SELECT ef.entry_id, {cols} FROM entry_features ef WHERE ef.entry_id = ANY(%s)", (eids,))
        rows = cur.fetchall()
    if not rows:
        return {}
    df = gal_frame(pd.DataFrame(rows, columns=['entry_id'] + art['features']), art['features'],
                   art['categories'])
    p = art['model'].predict_proba(df[art['features']])[:, 1]
    return dict(zip(df['entry_id'].astype(int).tolist(), p.astype(float).tolist()))


def gal_frame(df, features, categories):
    """entry_features rows as the xgal model reads them: categories pinned
    to the training sets, everything else float."""
    import pandas as pd
    for c in features:
        if c in categories:
            df[c] = pd.Categorical(df[c], categories=categories[c])
        else:
            df[c] = pd.to_numeric(df[c], errors='coerce').astype('float32')
    return df


def score(conn, *, today: dt.date, lead_bundles: dict, xodds_art: dict | None,
          xwin_art: dict | None, front_bundles: dict | None = None, gal_art: dict | None = None,
          xedge_art: dict | None = None, since: dt.date = dt.date(2012, 1, 1), log=print):
    """Run both passes up to `today` and chain the models over the upcoming
    starters. Returns {'y_lead': {eid: p}, 'y_front', 'xgal', 'xodds', 'xwin',
    'xedge' (tilts: multiply by the market's shares and renormalise),
    'rating' (the rating pass's own shares), 'lead_method': {eid: 'auto' | 'volt'}}."""
    import time
    t0 = time.time()
    params = xodds_art['params'] if xodds_art else None
    results = xodds_art.get('results_params') if xodds_art else None
    mp = XO.MarketPass(today=today, params=params, results=results, keep_rows=False, snap_from=today)
    cur = conn.cursor(name='xstack_market')
    cur.itersize = 20000
    mp.run(cur, since)
    cur.close()
    conn.commit()
    log(f"market pass: {mp.priced:,} priced races, {len(mp.upcoming):,} upcoming starters "
        f"({time.time() - t0:.0f}s)")

    lp = XL.LeadPass(today=today, collect=False, ext=mp.snap)
    cur = conn.cursor(name='xstack_lead')
    cur.itersize = 20000
    lp.run(cur, since)
    cur.close()
    conn.commit()
    lead_df = XL.frame(lp.upcoming)
    lead = predict_lead(lead_df, lead_bundles) if not lead_df.empty else {}
    front = predict_lead(lead_df, front_bundles or {}) if not lead_df.empty else {}
    log(f"xlead / xfront: {len(lead):,} / {len(front):,} starters ({time.time() - t0:.0f}s)")
    out = {'y_lead': lead, 'y_front': front, 'xgal': {}, 'xodds': {}, 'xwin': {}, 'xedge': {}, 'rating': {},
           'lead_method': {int(e): ('auto' if a else 'volt')
                           for e, a in zip(lead_df['entry_id'], lead_df['auto'])}}
    if not mp.upcoming:
        return out
    _, _, eids, X, mask, _, _ = XO.tensors(mp.upcoming, with_target=False)
    ids = eids[mask].astype(int).tolist()
    S1 = XO.stage1_shares(X, mask)
    out['rating'] = dict(zip(ids, S1[mask].astype(float).tolist()))
    if xodds_art is None:
        return out

    Xa = np.concatenate([X, lead_columns(eids, mask, lead)], axis=-1)
    if len(xodds_art['mu']) != Xa.shape[-1]:
        log(f"xodds: model expects {len(xodds_art['mu'])} inputs, pass gives {Xa.shape[-1]} — rebuild it")
        return out
    P = XO.predict(xodds_art, Xa, mask)
    out['xodds'] = dict(zip(ids, P[mask].astype(float).tolist()))
    shape = shape_columns(eids, mask, [lead_df])
    if gal_art is not None:
        out['xgal'] = predict_gal(conn, gal_art, ids)
    chained = [a for a in (xwin_art, xedge_art) if a is not None and 'booster' in a]
    if chained:
        D = chain_design(Xa, mask, P, shape, front_columns(eids, mask, front), gal_columns(eids, mask, out['xgal']))
        for a in chained:
            if a.get('features') and len(a['features']) != D.shape[-1]:
                log(f"{a.get('target', 'model')}: expects {len(a['features'])} inputs, the chain gives "
                    f"{D.shape[-1]} — rebuild it")
                return out
    if xwin_art is not None:
        if 'booster' in xwin_art:
            W = predict_softmax(xwin_art, D, mask, np.log(np.clip(P, 1e-9, 1)))
        else:
            W = predict_xwin(xwin_art, xwin_design(Xa, mask, P, shape), mask)
        out['xwin'] = dict(zip(ids, W[mask].astype(float).tolist()))
    if xedge_art is not None and 'booster' in xedge_art:
        T = XO.softmax(softmax_scores(xedge_art, D, mask), mask)
        out['xedge'] = dict(zip(ids, T[mask].astype(float).tolist()))
    log(f"xgal / xodds / xwin / xedge: {len(out['xgal']):,} / {len(out['xodds']):,} / {len(out['xwin']):,} / "
        f"{len(out['xedge']):,} starters ({time.time() - t0:.0f}s)")
    return out
