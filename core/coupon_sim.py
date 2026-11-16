"""Spelteknik for the marking pools (V4 … V86): bankers, coverage, weights,
multipliers and whether to play at all, on top of a model's win chances.

A coupon is held in product form. In leg i, pick j gets a weight w_ij (its
share of the leg's rows; a leg's weights sum to one) and the coupon carries
U stake units, so the row (j1 … jL) carries U·∏ w_iji. A flat system is
w = 1/n_i with U = ∏ n_i rows; a banker is a leg with one pick; the game
page's weighted coupon is any other w (rows bought several times or a
fraction of a unit; the builder rounds that to whole units, see
core/reduce_system.py); playing the same coupon m times is U·m.

Our units on rows with k legs right are U·[x^k] ∏_i (1 − a_i + a_i x), a_i
being our weight on leg i's winner, so a game settles in a few vector ops:

  realised  per tier, the public's systems and the net pot as ATG paid out,
            plus our units and our stake (as core/pools.py)
  expected  winners drawn from the model's chances; the public's rows with k
            right ≈ κ·N·[x^k] ∏(1 − s_i + s_i x), N = turnover / row price and
            s_i the public's share of leg i's winner. Against the real
            number of winning systems the share product holds up at
            r ≈ 0.98–0.99 with slope ≈ 1; κ is the median ratio per game type
            and tier.

`menu` lays out the coupons a game can be played with; `CouponEnv` is the
gym-style environment over those menus that a betting agent learns in.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field

import numpy as np

from core.reduce_system import top_pool

MAX_PICKS = 10
MC = 384
MULTS = (1, 2, 3, 5, 10)
BANKER_RULES = ('p', 'margin', 'value', 'ratio')
MAX_BANKERS = {'V4': 1, 'V5': 2, 'V64': 2, 'V65': 2}
COVERS = (0.5, 0.6, 0.7, 0.8, 0.9)
WEIGHTS = (('flat', 1.0), ('p', 1.0), ('value', 1.0), ('p', 0.5), ('value', 0.5))
# Median ratio of real winning systems to κ = 1, per game type and legs
# missed (2025-07 → 2026-11); V64 / V65 count their V6 top-pool units too.
KAPPA = {('V64', 0): 1.46, ('V64', 1): 0.69, ('V64', 2): 0.70, ('V65', 0): 1.29, ('V65', 1): 0.76,
         ('GS75', 0): 1.12, ('GS75', 1): 0.88, ('GS75', 2): 0.90, ('V75', 0): 0.88, ('V5', 1): 0.84}


@dataclass
class Leg:
    numbers: np.ndarray
    p: np.ndarray        # the source's win chances
    q: np.ndarray        # the market's (closing win odds)
    s: np.ndarray        # the public's share of the pool's rows
    win: np.ndarray      # bool per number (two on a dead heat)


@dataclass
class Game:
    gid: str
    gtype: str
    date: dt.datetime
    legs: list
    price: float
    rtp: float
    T: float
    tiers: np.ndarray    # legs right that pay, e.g. [6, 7, 8]
    net: np.ndarray      # net pot per tier as paid out (systems × payout), before us
    sys: np.ndarray      # the public's winning systems per tier
    void: np.ndarray     # tier rolled over (payout under the minimum)
    pot: np.ndarray      # tier pots
    kappa: np.ndarray
    top: dict | None = None
    feats: dict = field(default_factory=dict)


@dataclass(frozen=True)
class Item:
    bankers: int
    rule: str
    cover: float
    weight: str
    units: float         # stake units against the flat system's rows
    top: bool = False

    @property
    def label(self):
        b = f"{self.bankers} banker{'s' if self.bankers != 1 else ''} ({self.rule})" if self.bankers else 'no banker'
        w = 'flat' if self.weight == 'flat' else f"weighted {self.weight}" + (
            f" ×{self.units:g}" if self.units != 1 else '')
        return f"{b} · cover {self.cover:.0%} · {w}" + (' · top pool' if self.top else '')


# ── games ────────────────────────────────────────────────────────────────

def build_games(st, raw_games, probs: dict, types=None) -> list:
    """Finished marking games where every leg has the source's chances, the
    public's shares and a winner."""
    from core.pools import _race_frame, row_price
    d = _race_frame(st, probs)
    by_race = {r: g for r, g in d.groupby('race')}
    out = []
    for g in raw_games:
        gt = g['type']
        if types and gt not in types:
            continue
        if len(g['winners']) != len(g['races']) or not all(g['winners']) or not g['payouts']:
            continue
        legs = []
        for i, rid in enumerate(g['races']):
            f = by_race.get(rid)
            sh = g['shares'][i]
            if f is None or not sh:
                legs = None
                break
            nums = f['number'].values.astype(int)
            pub = np.array([(sh.get(n) or (0, 0))[0] or 0.0 for n in nums], float) + 1e-4
            win = np.array([n in g['winners'][i] for n in nums])
            if not win.any():
                legs = None
                break
            legs.append(Leg(nums, f['p'].values.astype(float), f['q'].values.astype(float), pub / pub.sum(), win))
        if not legs:
            continue
        L = len(legs)
        tiers = np.array(sorted(g['payouts']), dtype=int)
        info = [g['payouts'][t] or {} for t in tiers]
        sys_n = np.array([float(x.get('systems') or 0) for x in info])
        pot = np.array([float(g['pots'].get(int(t), g['turnover'] * g['rtp'])) for t in tiers])
        net = np.where(sys_n > 0, np.array([(x.get('payout') or 0) / 100 for x in info]) * sys_n, pot)
        void = np.array([bool(x.get('jackpot')) and s > 0 for x, s in zip(info, sys_n)])
        kappa = np.array([KAPPA.get((gt, L - int(t)), 1.0) for t in tiers])
        out.append(Game(g['id'], gt, g['date'], legs, row_price(gt, g['date']), g['rtp'], g['turnover'], tiers, net,
                        sys_n, void, pot, kappa, top_pool(gt)))
    out.sort(key=lambda x: (x.date, x.gid))
    return out


# ── coupons ──────────────────────────────────────────────────────────────

def banker_legs(legs, b, rule):
    """The b legs to bank on, by the source favourite's chance ('p'), its
    margin to the second ('margin'), how underbet it is among favourites with
    p ≥ 0.35 ('value'), or how underbet it is whatever its chance ('ratio')."""
    if b <= 0:
        return []
    top = [np.sort(lg.p)[::-1] for lg in legs]
    p1 = np.array([t[0] for t in top])
    p2 = np.array([t[1] if len(t) > 1 else 0.0 for t in top])
    s1 = np.array([lg.s[np.argmax(lg.p)] for lg in legs])
    if rule == 'p':
        score = p1
    elif rule == 'margin':
        score = p1 - p2
    elif rule == 'ratio':
        score = p1 / s1
    elif rule == 'value':
        score = np.where(p1 >= 0.35, 10 + p1 / s1, p1)
    else:
        raise ValueError(rule)
    return [int(i) for i in np.argsort(-score, kind='stable')[:b]]


def coupon(game, item):
    """Per-leg weights over the leg's numbers (0 = not picked) and the units U."""
    banks = set(banker_legs(game.legs, item.bankers, item.rule))
    W, rows = [], 1
    for i, lg in enumerate(game.legs):
        order = np.argsort(-lg.p, kind='stable')
        if i in banks:
            k = 1
        else:
            k = int(np.searchsorted(np.cumsum(lg.p[order]), item.cover - 1e-12) + 1)
            k = max(1, min(k, MAX_PICKS, len(order)))
        pick = order[:k]
        w = np.zeros(len(lg.p))
        if item.weight == 'flat':
            w[pick] = 1.0
        elif item.weight == 'p':
            w[pick] = lg.p[pick]
        elif item.weight == 'value':
            w[pick] = lg.p[pick] ** 2 / lg.s[pick]
        else:
            raise ValueError(item.weight)
        W.append(w / w.sum())
        rows *= k
    return W, item.units * rows, rows


def poly(a):
    """Coefficients of ∏_i (1 − a_i + a_i x) over the last axis: [..., L+1]."""
    out = np.ones(a.shape[:-1] + (1,))
    for i in range(a.shape[-1]):
        ai = a[..., i:i + 1]
        z = np.zeros_like(ai)
        out = np.concatenate([out * (1 - ai), z], axis=-1) + np.concatenate([z, out * ai], axis=-1)
    return out


def _returns(game, o, c1, mults, top, net, sys_n, void):
    """Return per multiplier: o, net and sys_n are [..., T] (our units at
    multiplier 1, net pot, the public's systems), void is [T]; out [..., M]."""
    m = np.asarray(mults, float)
    if top:
        tp = game.top
        k = int(np.nonzero(game.tiers == tp['tier'])[0][0])
        u = o[..., k, None] * m * tp['factor']
        pot = net[..., k, None] + m * c1 * game.rtp
        r = np.where(u > 0, pot * u / np.maximum(sys_n[..., k, None] + u, 1e-12), 0.0)
        return np.where(void[k], 0.0, r)
    u = o[..., None, :] * m[:, None]                          # [..., M, T]
    pot = net[..., None, :] + m[:, None] * c1 * (game.pot / max(game.T, 1.0))
    r = np.where(u > 0, pot * u / np.maximum(sys_n[..., None, :] + u, 1e-12), 0.0)
    return np.where(void, 0.0, r).sum(axis=-1)


def settle(game, W, U, mults=MULTS, top=False):
    """Realised (cost per multiplier, return per multiplier)."""
    a = np.array([w[lg.win].sum() for w, lg in zip(W, game.legs)])
    o = U * poly(a[None, :])[0][game.tiers]
    c1 = U * game.price
    ret = _returns(game, o, c1, mults, top, game.net, game.sys, game.void)
    return c1 * np.asarray(mults, float), ret


class Sampler:
    """Winners drawn from the source's chances, shared by every coupon of a
    game (common random numbers), with the public's rows per tier."""

    def __init__(self, game, n=MC, seed=0, belief=None):
        rng = np.random.default_rng(seed)
        L = len(game.legs)
        self.idx = np.empty((n, L), dtype=int)
        sw = np.empty((n, L))
        for i, lg in enumerate(game.legs):
            p = lg.p if belief is None else belief[i]
            j = np.minimum(np.searchsorted(np.cumsum(p), rng.random(n) * p.sum()), len(p) - 1)
            self.idx[:, i] = j
            sw[:, i] = lg.s[j]
        N = game.T / game.price
        self.pub = game.kappa * N * poly(sw)[:, game.tiers]      # [n, T]
        self.game = game

    def expected(self, W, U, mults=MULTS, top=False):
        """(mean return per multiplier, P(any payout), mean of return²)."""
        g = self.game
        a = np.stack([W[i][self.idx[:, i]] for i in range(len(W))], axis=1)
        o = U * poly(a)[:, g.tiers]                                  # [n, T]
        c1 = U * g.price
        net = np.broadcast_to(g.pot, o.shape)
        r = _returns(g, o, c1, mults, top, net, self.pub, np.zeros(len(g.tiers), bool))
        hit = (o[:, -1] > 0) if top else (o > 0).any(axis=1)
        return r.mean(axis=0), float(hit.mean()), (r ** 2).mean(axis=0)

    def expected_many(self, Ws, Us, mults=MULTS, top=False):
        """expected() for K coupons of the same kind (all top-pool or none)
        in one pass: (mean return [K, M], P(any payout) [K])."""
        g = self.game
        L = len(g.legs)
        a = np.stack([np.stack([W[i][self.idx[:, i]] for i in range(L)], axis=1) for W in Ws])  # [K, n, L]
        U = np.asarray(Us, float)[:, None, None]
        o = U * poly(a)[..., g.tiers]                                                         # [K, n, T]
        c1 = (U * g.price)[..., None] if not top else U * g.price                             # broadcasts in _returns
        net = np.broadcast_to(g.pot, o.shape)
        r = _returns(g, o, c1, mults, top, net, self.pub, np.zeros(len(g.tiers), bool))      # [K, n, M]
        hit = (o[..., -1] > 0) if top else (o > 0).any(axis=-1)
        return r.mean(axis=1), hit.mean(axis=1)


def items_for(gtype):
    out = []
    nb = MAX_BANKERS.get(gtype, 3)
    for b in range(nb + 1):
        for rule in (BANKER_RULES if b else ('-',)):
            for c in COVERS:
                for w, u in WEIGHTS:
                    out.append(Item(b, rule, c, w, u))
                if top_pool(gtype):
                    out.append(Item(b, rule, c, 'flat', 1.0, top=True))
    return out


def menu(game, items=None, n=MC, seed=0):
    """Every coupon of the game's menu, with its cost, expected and realised
    returns per multiplier and its chance of paying anything."""
    items = items or items_for(game.gtype)
    smp = Sampler(game, n=n, seed=seed)
    rows = []
    for it in items:
        W, U, nrows = coupon(game, it)
        cost, ret = settle(game, W, U, top=it.top)
        ev, phit, ev2 = smp.expected(W, U, top=it.top)
        rows.append((it, nrows, cost[0], ev, ev2, phit, ret))
    return rows


# ── environment ──────────────────────────────────────────────────────────

class CouponEnv:
    """A season of marking games in date order. Each step is one game: the
    agent sees the game and its menu (cost, the model's expected return per
    multiplier, chance of any payout) and answers with a menu index and a
    multiplier index, or None to pass. The reward is the log growth of its
    bankroll; the episode ends after the last game or when the bankroll
    can't pay the cheapest coupon.

    Menus come precomputed (scripts/coupon_lab.py) as arrays:
      cost [G, I]      cost at multiplier 1 (NaN where the item doesn't exist)
      ev   [G, I, M]   expected return per multiplier
      phit [G, I]      chance of any payout
      ret  [G, I, M]   realised return per multiplier
    """

    def __init__(self, arrays, games_idx=None, bankroll=10_000.0, mults=MULTS):
        self.A = arrays
        self.idx = np.arange(arrays['cost'].shape[0]) if games_idx is None else np.asarray(games_idx)
        self.B0 = float(bankroll)
        self.mults = np.asarray(mults, float)

    def reset(self):
        self.t = 0
        self.B = self.B0
        self.log = []
        return self._obs()

    def _obs(self):
        if self.t >= len(self.idx):
            return None
        g = self.idx[self.t]
        return {'game': int(g), 'bankroll': self.B, 'cost': self.A['cost'][g], 'ev': self.A['ev'][g],
                'phit': self.A['phit'][g], 'feats': {k: v[g] for k, v in self.A.get('game_feats', {}).items()}}

    def step(self, action):
        g = self.idx[self.t]
        cost = ret = 0.0
        if action is not None:
            i, k = action
            cost = float(self.A['cost'][g, i] * self.mults[k])
            if not np.isfinite(cost) or cost > self.B:
                cost = 0.0
            else:
                ret = float(self.A['ret'][g, i, k])
        before = self.B
        self.B = self.B - cost + ret
        self.log.append((int(g), cost, ret))
        self.t += 1
        done = self.t >= len(self.idx) or self.B < 1.0
        reward = float(np.log(max(self.B, 1e-9) / before))
        return self._obs(), reward, done, {'cost': cost, 'ret': ret}
