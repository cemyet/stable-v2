"""The results rating: how strong a horse is by whom it beat, not by what the
market thought.

Every race with a result, in every country stored (France has full fields,
so a horse raced there meets the Nordic horses that cross over), is read as
a ranking under a Plackett–Luce model: the winner beat the whole field,
second beat the rest, third the rest after that. Each stage is a softmax of

    s_i = θ_horse + δ_driver + post / tillägg effect

so θ sits on the scale of log win odds, the same scale as the market
rating. Only the top three placings are read, the second at 0.7 and the
third at 0.49 of the winner's weight: lower places are mostly noise once a
driver eases up. Galloped, disqualified and unplaced runners lose to every
placed one. A loss counts in full even when the comment gives an excuse
(start trouble, death seat, wide, no room): tested, discounting it fits
worse.

One Newton step per race (diagonal, Glicko-style) moves θ and δ in
proportion to their variances; `lam` scales the evidence of one race.
θ's variance grows with days off, faster for young horses, so old results
fade. A horse never rated comes in at the running mean of newcomers in the
country of its first start.

core/xodds.py's MarketPass owns one: it reads each runner's pre-race state
(`pre`) into its own features before the race updates it (`race`), so the
build and live scoring see the same values.
"""

from __future__ import annotations

import math
from array import array

from core.xodds import (QUALIFIER_RE, S_AGE, S_AUTO, S_CTRY, S_DATE, S_DID, S_DIST, S_EID,
                        S_HID, S_METHOD, S_POST, S_PTXT, S_RID, S_TILL, finish_scores)

NORDIC = frozenset(('SE', 'NO', 'DK', 'FI'))

# Standalone stream, for tuning (MarketPass feeds `race` from its own).
STREAM_SQL = """
    SELECT r.race_id, r.race_date, r.track_id, t.country, r.start_method,
           e.entry_id, e.horse_id, e.driver_id, e.post, e.distance, e.tillagg, e.auto,
           e.odds, e.placement, e.placement_text, COALESCE(e.galopp, false),
           COALESCE(e.disqualified, false), e.time_seconds, e.age,
           e.trainer_id, e.shoe_code, e.sulky, e.prize_kr, NULL::text
    FROM race r
    JOIN track t ON t.track_id = r.track_id
    JOIN entry e ON e.race_id = r.race_id
    WHERE r.race_date >= %s
      AND NOT COALESCE(e.withdrawn, false) AND e.horse_id IS NOT NULL
    ORDER BY r.race_date, r.race_id, e.entry_id
"""

DEFAULTS = {
    'lam': 3.0,        # weight of one race's evidence
    'top': 3,          # placings read per race
    'decay': 0.7,      # placing k counts decay^(k-1)
    'p0': 0.6,         # prior variance of a horse never rated
    'q_day': 0.0012,   # θ variance added per day without a rated race
    'young': 2.0,      # q_day multiplier at age ≤ 3 (4 years: half of that)
    'p_max': 1.0,
    'pd0': 0.05,       # driver prior variance
    'qd_day': 0.00002,
    'eta': 0.002,      # learning rate of the post / tillägg effects
}


class _Horse:
    __slots__ = ('mu', 'P', 't', 'n', 'na')

    def __init__(self, mu0, p0):
        self.mu = mu0
        self.P = p0
        self.t = None          # day of the last rated race
        self.n = 0             # rated races
        self.na = 0            # of them outside the Nordics


def _till(rows):
    dists = [r[S_DIST] for r in rows]
    base_d = min((x for x in dists if x), default=None)
    return [max(0, (x - base_d) if (x and base_d) else (r[S_TILL] or 0)) for x, r in zip(dists, rows)]


def _auto(r):
    m = r[S_METHOD]
    return m == 'A' if m in ('A', 'V') else bool(r[S_AUTO])


class ResultsPass:
    """Races go in date order through `race`. Standalone (`run`), Swedish
    races on or after `collect_from` leave their runners' pre-race features
    in `rows` (race id, day, entry id, won, features)."""

    def __init__(self, *, params: dict | None = None, collect_from: int | None = None):
        self.p = {**DEFAULTS, **(params or {})}
        self.collect_t = collect_from
        self.H: dict[int, _Horse] = {}
        self.D: dict[int, list] = {}      # driver -> [mu, P, day]
        self.post: dict = {}              # (country, auto, post, front) -> effect
        self.till: dict = {}              # (auto, tillägg / 20) -> effect
        self.mu0: dict = {}               # country of first start -> [mean, n]
        self.rows: list = []
        self.races = 0

    def run(self, cur, since):
        cur.execute(STREAM_SQL, (since,))
        batch, rid = [], None
        for row in cur:
            if row[S_RID] != rid:
                if batch:
                    self._collect(batch)
                    self.race(batch)
                batch, rid = [], row[S_RID]
            batch.append(row)
        if batch:
            self._collect(batch)
            self.race(batch)

    # ── state as of a day ────────────────────────────────────────────────
    def _horse(self, hid, t, age, ctry):
        h = self.H.get(hid)
        if h is None:
            m = self.mu0.get(ctry)
            h = self.H[hid] = _Horse(m[0] if m else 0.0, self.p['p0'])
            return h, h.P
        if h.t is None:
            return h, h.P
        q = self.p['q_day']
        if age is not None and age <= 3:
            q *= self.p['young']
        elif age == 4:
            q *= 0.5 * (1 + self.p['young'])
        return h, min(self.p['p_max'], h.P + q * max(0, t - h.t))

    def _driver(self, did, t):
        if did is None:
            return None, 0.0
        d = self.D.get(did)
        if d is None:
            d = self.D[did] = [0.0, self.p['pd0'], t]
            return d, d[1]
        return d, d[1] + self.p['qd_day'] * max(0, t - d[2])

    def _ctx(self, ctry, r, till):
        k = (_auto(r), min(r[S_POST] or 99, 15), till == 0)
        return self.post.get((ctry,) + k, 0.0) + self.till.get((k[0], min(till // 20, 5)), 0.0)

    def pre(self, r, t, ctry, till):
        """One runner's features before the race, in xodds.R_FEATURES order."""
        h, ph = self._horse(r[S_HID], t, r[S_AGE], ctry)
        dv, _ = self._driver(r[S_DID], t)
        n = h.n
        return (h.mu, dv[0] if dv else 0.0, self._ctx(ctry, r, till), math.sqrt(ph), math.log1p(n),
                h.na / n if n else 0.0, 1.0 if n == 0 else 0.0)

    def _collect(self, rows):
        t = rows[0][S_DATE].toordinal()
        if self.collect_t is None or t < self.collect_t or rows[0][S_CTRY] != 'SE':
            return
        sc = finish_scores(rows)
        till = _till(rows)
        for i, r in enumerate(rows):
            self.rows.append((r[S_RID], t, r[S_EID], 1.0 if sc[i] == 1.0 else 0.0,
                              array('d', self.pre(r, t, 'SE', till[i]))))

    # ── one race ─────────────────────────────────────────────────────────
    def race(self, rows):
        n = len(rows)
        if n < 2:
            return
        ptxt = [(r[S_PTXT] or '').strip() for r in rows]
        if sum(1 for p in ptxt if QUALIFIER_RE.match(p)) * 2 >= n:
            return
        sc = finish_scores(rows)
        ranked = sorted((s, i) for i, s in enumerate(sc) if s is not None)
        if not ranked:
            return
        self.races += 1
        t = rows[0][S_DATE].toordinal()
        ctry = rows[0][S_CTRY] or '?'
        P = self.p
        till = _till(rows)
        hs, Ph, ds, Pd, s = [], [], [], [], []
        for i, r in enumerate(rows):
            h, ph = self._horse(r[S_HID], t, r[S_AGE], ctry)
            dv, pd = self._driver(r[S_DID], t)
            hs.append(h)
            Ph.append(ph)
            ds.append(dv)
            Pd.append(pd)
            s.append(h.mu + (dv[0] if dv else 0.0) + self._ctx(ctry, r, till[i]))
        mx = max(s)
        ex = [math.exp(x - mx) for x in s]

        # Plackett–Luce over the placings read: each runner's gradient and curvature.
        g = [0.0] * n
        c = [0.0] * n
        alive = [True] * n
        tot = sum(ex)
        wk = 1.0
        for _, i in ranked[:P['top']]:
            if tot <= 0:
                break
            for j in range(n):
                if alive[j]:
                    p = ex[j] / tot
                    g[j] -= wk * p
                    c[j] += wk * p * (1.0 - p)
            g[i] += wk
            alive[i] = False
            tot -= ex[i]
            wk *= P['decay']

        lam = P['lam']
        abroad = ctry not in NORDIC
        for i in range(n):
            h = hs[i]
            S = Ph[i] + Pd[i]
            prec = 1.0 / S + lam * c[i]
            delta = lam * g[i] / prec
            shrink = S - 1.0 / prec
            kh = Ph[i] / S
            first = h.n == 0
            h.mu += kh * delta
            h.P = max(1e-4, Ph[i] - kh * kh * shrink)
            h.t = t
            h.n += 1
            h.na += 1 if abroad else 0
            if first:
                m = self.mu0.setdefault(ctry, [0.0, 0])
                m[1] += 1
                m[0] += (h.mu - m[0]) / min(m[1], 5000)
            dv = ds[i]
            if dv is not None:
                kd = Pd[i] / S
                dv[0] += kd * delta
                dv[1] = max(1e-4, Pd[i] - kd * kd * shrink)
                dv[2] = t
        if n >= 6 and len(ranked) >= 3:
            for i, r in enumerate(rows):
                k = (_auto(r), min(r[S_POST] or 99, 15), till[i] == 0)
                self.post[(ctry,) + k] = self.post.get((ctry,) + k, 0.0) + P['eta'] * g[i]
                kt = (k[0], min(till[i] // 20, 5))
                self.till[kt] = self.till.get(kt, 0.0) + P['eta'] * g[i]
