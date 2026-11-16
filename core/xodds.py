"""xodds — the winner odds a starter "should" have, from how the market has
priced it against the horses it has met.

Stage 1, the market rating. Every past Nordic race with a full win pool says
how the market ranked those horses against each other: the implied share
q = (1/odds) / Σ(1/odds) gives z = log q, and within a race only the
differences matter. Each horse carries a strength θ (mean and variance);
the driver carries an effect δ; posts and tillägg carry context effects.
A race is read as

    z_i = θ_horse + δ_driver + τ_trainer + post/track/tillägg effect + race level + noise

and a Kalman update (race level marginalised out, so only the ordering
inside the race informs) moves every runner at once. Because each horse's
θ sits on the one scale shared by everything it has met, A-vs-C and B-vs-C
put A against B without them ever meeting, through as many links as the
graph has. Every price counts alike (`r_long` can make longshots count less;
tuned, it is off). θ's variance grows with days off (faster for young horses), so old prices fade
and a horse back from a break or still developing can move a lot on its
next price.

Stage 2, the odds. A soft-target multinomial logit maps the stage-1 pieces
plus what the market reacts to between races (the last result against its
odds, form against the market, days off, age, shoes, sulky, earnings, and
the results rating of core/results_pass.py, which reads races abroad) onto
the next race's market shares; gradient-boosted trees then fit what the
logit leaves in log q. xodds = payout / p, with payout the win pool's
typical return.
"""

from __future__ import annotations

import datetime as dt
import math
import re
from array import array

from core.ratings.comments import parse as parse_comment

LN2 = math.log(2)
COUNTRIES = ('SE', 'NO', 'DK', 'FI')

# Every country: the results rating reads races abroad; the market pass skips them.
STREAM_SQL = """
    SELECT r.race_id, r.race_date, r.track_id, t.country, r.start_method,
           e.entry_id, e.horse_id, e.driver_id, e.post, e.distance, e.tillagg, e.auto,
           e.odds, e.placement, e.placement_text, COALESCE(e.galopp, false),
           COALESCE(e.disqualified, false), e.time_seconds, e.age,
           e.trainer_id, e.shoe_code, e.sulky, e.prize_kr, ec.comment
    FROM race r
    JOIN track t ON t.track_id = r.track_id
    JOIN entry e ON e.race_id = r.race_id
    LEFT JOIN entry_comment ec ON ec.entry_id = e.entry_id
    WHERE r.race_date >= %s
      AND NOT COALESCE(e.withdrawn, false) AND e.horse_id IS NOT NULL
    ORDER BY r.race_date, r.race_id, e.entry_id
"""
(S_RID, S_DATE, S_TRACK, S_CTRY, S_METHOD, S_EID, S_HID, S_DID, S_POST, S_DIST, S_TILL,
 S_AUTO, S_ODDS, S_PLC, S_PTXT, S_GAL, S_DQ, S_TIME, S_AGE, S_TID, S_SHOE, S_SULKY,
 S_PRIZE, S_COMMENT) = range(24)

BARE_ENDS = {'1': 2.0, '2': 1.0, '3': 1.0, '4': 0.0}

QUALIFIER_RE = re.compile(
    r"^(gdk|ejg|ejp|gd|gk|egk|egdk|gkd|gdj|gdb|gdek|gdgk|gdl|ddk|frdk|erj|ejk|"
    r"ejgk|ejgd|ej|EJ|EJG|EJP|GDK|Gdk)[0-9]?$|^[12]?[pP][0-9]?$")

DEFAULTS = {
    'r0': 0.09,        # market noise (variance, log-share units) for a horse at a fair share
    'r_long': 0.0,     # extra noise per unit of log(1 / (q·n)) below a fair share
    'p0': 0.3,         # prior variance of a horse never priced
    'q_day': 0.0012,   # θ variance added per day without a price
    'young': 2.0,      # q_day multiplier at age ≤ 3 (4 years: half of that)
    'p_max': 0.8,      # θ variance never grows past this
    'pd0': 0.03,       # driver prior variance
    'qd_day': 0.00002,
    'pt0': 0.02,       # trainer prior variance
    'qt_day': 0.00002,
    'eta': 0.002,      # learning rate of the post / tillägg effects
    'eta_t': 0.004,    # learning rate of the track-specific post deviations
    'tau_f': 120,      # half-life (days) of form against the market
    'tau_p': 45,       # half-life (days) of a stable's / driver's recent form
    'tau_m': 180,      # half-life (days) of margins behind the winner
}

# The results rating (core/results_pass.py), in the order of ResultsPass.pre.
R_FEATURES = ['r_h', 'r_d', 'r_ctx', 'r_sd', 'r_log_n', 'r_abroad', 'r_new']
FEATURES = [
    'mu_h', 'mu_d', 'mu_t', 'ctx', 'sd_h', 'debut', 'log_obs',
    'fm', 'g_last', 'won_last', 'placed_last', 'gal_last', 'dq_last', 'no_last',
    'log_days', 'break90', 'young_days', 'age3', 'age4',
    'bare', 'bare_delta', 'am', 'am_delta', 'log_earn', 'earn_ps',
    # what the program and the last comment show
    'rec_rel', 'rec_none', 'mg_avg', 'mg_last', 'gal_rate',
    'tr_death', 'tr_wide', 'tr_disturbed', 'tr_trouble', 'tr_finish', 'tr_strength',
    't_form', 'd_form',
    *R_FEATURES,
]
EXTRA = ['s_rel', 's_gap', 'n', 'mu_h_rel', 'mu_t_rel', 'r_rel']

# Record lanes: auto / volt × short (< 1800 m), middle, long (≥ 2400 m).
def _rec_slot(auto, dist):
    band = 0 if dist < 1800 else 1 if dist < 2400 else 2
    return band * 2 + (1 if auto else 0)


class _Horse:
    __slots__ = ('mu', 'P', 't', 'obs', 'fs', 'fw', 'ft', 'g', 'won', 'placed', 'gal', 'dq',
                 'last', 'starts', 'bare', 'am', 'earn', 'rec', 'mg', 'mg_last', 'gr',
                 'trip')

    def __init__(self, mu0, p0):
        self.mu = mu0
        self.P = p0
        self.t = None          # day of the last price
        self.obs = 0
        self.fs = self.fw = 0.0
        self.ft = None
        self.g = None          # last race against its odds (-1..1)
        self.won = self.placed = self.gal = self.dq = 0.0
        self.last = None       # day of the last start with a result
        self.starts = 0
        self.bare = None       # barefoot ends at the last start
        self.am = None         # american sulky at the last start
        self.earn = 0.0
        self.rec = None        # best km time per record lane (_rec_slot)
        self.mg = None         # margin behind the winner, s/km: [sum, weight, day]
        self.mg_last = None
        self.gr = [0.0, 0.0]   # galopp: (count, starts), each start counted once
        self.trip = None       # last start's comment: (death, wide, disturbed, trouble, finish, strength)


def kalman(m, P, d, w):
    """Posterior means and variances of the runner scores s ~ N(m, P) after
    seeing d = s + race level + noise (precisions w), with a flat prior on
    the race level so only differences inside the race count."""
    n = len(m)
    W = sum(w)
    dbar = sum(w[i] * d[i] for i in range(n)) / W
    a = [1.0 / P[i] + w[i] for i in range(n)]
    b = [m[i] / P[i] + w[i] * (d[i] - dbar) for i in range(n)]
    s2 = sum(w[i] * w[i] / a[i] for i in range(n)) / W
    denom = 1.0 - s2
    c = sum(w[i] * b[i] / a[i] for i in range(n)) / W / denom
    mean = [b[i] / a[i] + (w[i] / a[i]) * c for i in range(n)]
    var = [1.0 / a[i] + (w[i] / a[i]) ** 2 / W / denom for i in range(n)]
    return mean, var


def finish_scores(rows):
    """Placing, then unplaced finishers by time; galopp / DQ / no time: None."""
    sc = []
    for r in rows:
        p = r[S_PLC]
        if p is not None and p > 0 and not r[S_DQ]:
            sc.append(float(p))
        elif (r[S_PTXT] or '').strip() == '0' and r[S_TIME] and not r[S_GAL] and not r[S_DQ]:
            sc.append(100.0 + float(r[S_TIME]))
        else:
            sc.append(None)
    return sc


def vs_market(sc, q):
    """Per runner: pairwise results against the market's pairwise odds."""
    n = len(sc)
    g = [0.0] * n
    W = [0.0] * n
    for i in range(n):
        for j in range(i + 1, n):
            si, sj = sc[i], sc[j]
            if si is None and sj is None:
                continue
            if si is None:
                S, w = 0.0, 0.5
            elif sj is None:
                S, w = 1.0, 0.5
            else:
                S, w = (1.0 if si < sj else 0.5 if si == sj else 0.0), 1.0
            e = q[i] / (q[i] + q[j])
            g[i] += w * (S - e)
            g[j] -= w * (S - e)
            W[i] += w
            W[j] += w
    return [g[i] / W[i] if W[i] else 0.0 for i in range(n)]


class MarketPass:
    """Streams Nordic races in date order, rating horses off their prices.

    Priced races on or after `bt_start` in `collect` countries yield stage-2
    rows (features, market shares, winner); races without results on or
    after `today` yield feature rows for scoring. Swedish starters from
    `snap_from` on leave their pre-race strength in `snap` (entry id ->
    (horse + driver + trainer, driver)) for the xlead pass. The results
    rating (`rp`, tuned by `results`) reads every race, abroad too."""

    def __init__(self, *, today: dt.date, bt_start: dt.date | None = None,
                 collect=('SE',), params: dict | None = None, keep_rows: bool = True,
                 snap_from: dt.date | None = None, results: dict | None = None):
        from core.results_pass import ResultsPass
        self.rp = ResultsPass(params=results)
        self.p = {**DEFAULTS, **(params or {})}
        self.today = today
        self.bt_t = bt_start.toordinal() if bt_start else None
        self.collect = set(collect or ())
        self.keep_rows = keep_rows
        self.snap_t = snap_from.toordinal() if snap_from else None
        self.snap: dict[int, tuple] = {}
        self.H: dict[int, _Horse] = {}
        # driver / trainer -> [mu, P, day, form sum, form weight, form day]
        self.D: dict[int, list] = {}
        self.T: dict[int, list] = {}
        self.post: dict = {}              # (country, auto, post, front) -> effect
        self.track: dict = {}             # (track, auto, post, front) -> deviation
        self.till: dict = {}              # (auto, tillägg / 20) -> effect
        self.mu0 = 0.0                    # where never-priced horses come in
        self.mu0_n = 0
        self.rows: list = []
        self.upcoming: list = []
        self.payout = [0.0, 0]            # Σ 1/Σ(1/odds), races, over collected races
        self.fit = [0.0, 0, 0.0]          # Σ H(q, stage 1), races, Σ H(q, q)
        self.races = 0
        self.priced = 0

    def run(self, cur, since: dt.date, countries=COUNTRIES):
        cur.execute(STREAM_SQL, (since,))
        countries = set(countries)
        batch = []
        rid = None
        for row in cur:
            if row[S_RID] != rid:
                if batch:
                    if batch[0][S_CTRY] in countries:
                        self._race(batch)
                    self.rp.race(batch)
                batch = []
                rid = row[S_RID]
            batch.append(row)
        if batch:
            if batch[0][S_CTRY] in countries:
                self._race(batch)
            self.rp.race(batch)

    # ── state as of a day ────────────────────────────────────────────────
    def _horse(self, hid, t, age):
        h = self.H.get(hid)
        if h is None:
            h = self.H[hid] = _Horse(self.mu0, self.p['p0'])
            return h, h.P
        if h.t is None:
            return h, h.P
        q = self.p['q_day']
        if age is not None and age <= 3:
            q *= self.p['young']
        elif age == 4:
            q *= 0.5 * (1 + self.p['young'])
        return h, min(self.p['p_max'], h.P + q * max(0, t - h.t))

    @staticmethod
    def _person(table, pid, t, p0, q_day):
        if pid is None:
            return None, 0.0
        d = table.get(pid)
        if d is None:
            d = table[pid] = [0.0, p0, t, 0.0, 0.0, t]
            return d, d[1]
        return d, d[1] + q_day * max(0, t - d[2])

    def _form(self, pv, t):
        """A driver's / stable's recent results against their odds."""
        if pv is None or pv[4] <= 0:
            return 0.0
        f = math.exp(-(t - pv[5]) * LN2 / self.p['tau_p'])
        return pv[3] * f / (pv[4] * f + 3.0)

    def _ctx(self, ctry, track, auto, post, front, till):
        k = (auto, post, front)
        return (self.post.get((ctry,) + k, 0.0) + self.track.get((track,) + k, 0.0)
                + self.till.get((auto, min(till // 20, 5)), 0.0))

    # ── one race ─────────────────────────────────────────────────────────
    def _race(self, rows):
        d = rows[0][S_DATE]
        t = d.toordinal()
        ptxt = [(r[S_PTXT] or '').strip() for r in rows]
        has_result = any(ptxt) or any(r[S_PLC] is not None for r in rows)
        upcoming = not has_result and d >= self.today
        if not has_result and not upcoming:
            return
        if has_result and sum(1 for p in ptxt if QUALIFIER_RE.match(p)) * 2 >= len(rows):
            return
        n = len(rows)
        if n < 2:
            return
        self.races += 1
        ctry = rows[0][S_CTRY]
        track = rows[0][S_TRACK]
        method = rows[0][S_METHOD]
        auto = method == 'A' if method in ('A', 'V') else bool(rows[0][S_AUTO])
        dists = [r[S_DIST] for r in rows]
        base_d = min((x for x in dists if x), default=None)
        till = [max(0, (x - base_d) if (x and base_d) else (r[S_TILL] or 0)) for x, r in zip(dists, rows)]
        posts = [min(r[S_POST] or 99, 15) for r in rows]

        P = self.p
        hs, Ph, ds, Pd, ts, Pt, ctx = [], [], [], [], [], [], []
        for i, r in enumerate(rows):
            h, ph = self._horse(r[S_HID], t, r[S_AGE])
            dv, pd = self._person(self.D, r[S_DID], t, P['pd0'], P['qd_day'])
            tv, pt = self._person(self.T, r[S_TID], t, P['pt0'], P['qt_day'])
            hs.append(h)
            Ph.append(ph)
            ds.append(dv)
            Pd.append(pd)
            ts.append(tv)
            Pt.append(pt)
            ctx.append(self._ctx(ctry, track, auto, posts[i], till[i] == 0, till[i]))

        odds = [float(r[S_ODDS]) if r[S_ODDS] is not None else None for r in rows]
        valid = [o is not None and 1.0 < o < 1000 for o in odds]
        priced = has_result and n >= 5 and sum(valid) == n
        q = None
        if priced:
            inv = [1.0 / o for o in odds]
            tot = sum(inv)
            q = [x / tot for x in inv]

        in_bt = priced and self.bt_t is not None and t >= self.bt_t and ctry in self.collect
        m = [hs[i].mu + (ds[i][0] if ds[i] else 0.0) + (ts[i][0] if ts[i] else 0.0)
             for i in range(n)]
        pred = [m[i] + ctx[i] for i in range(n)]
        if self.snap_t is not None and ctry == 'SE' and t >= self.snap_t:
            for i in range(n):
                self.snap[rows[i][S_EID]] = (m[i], ds[i][0] if ds[i] else 0.0)
        if upcoming or (in_bt and self.keep_rows):
            recs = []
            for i, r in enumerate(rows):
                rc = hs[i].rec
                recs.append(rc[_rec_slot(auto, r[S_DIST] or base_d or 2140)] if rc else None)
            known = sorted(x for x in recs if x is not None)
            med = known[len(known) // 2] if known else None
            feats = []
            for i in range(n):
                f = self._features(hs[i], Ph[i], ds[i], ts[i], ctx[i], rows[i], auto, t,
                                   (recs[i] - med) if (recs[i] is not None and med is not None) else None)
                f.extend(self.rp.pre(rows[i], t, ctry, till[i]))
                feats.append(f)
        if upcoming:
            for i in range(n):
                self.upcoming.append((rows[i][S_RID], t, rows[i][S_EID], feats[i]))
        elif in_bt:
            if self.keep_rows:
                sc = finish_scores(rows)
                for i in range(n):
                    self.rows.append((rows[i][S_RID], t, rows[i][S_EID], q[i],
                                      1.0 if sc[i] == 1.0 else 0.0, feats[i]))
            self.payout[0] += 1.0 / sum(1.0 / o for o in odds)
            self.payout[1] += 1
            mx = max(pred)
            ex = [math.exp(x - mx) for x in pred]
            Z = sum(ex)
            self.fit[0] -= sum(q[i] * math.log(ex[i] / Z) for i in range(n))
            self.fit[2] -= sum(x * math.log(x) for x in q)
            self.fit[1] += 1

        if priced:
            self.priced += 1
            self._update(rows, hs, Ph, ds, Pd, ts, Pt, ctx, q, m, pred, ctry, track, auto,
                         posts, till, t)
        if has_result:
            self._results(rows, hs, ds, ts, q, t, auto)

    def _features(self, h, ph, dv, tv, ctx, r, auto, t, rec_rel):
        age = r[S_AGE]
        days = (t - h.last) if h.last is not None else None
        days_p = (t - h.t) if h.t is not None else None
        fm = 0.0
        if h.fw > 0:
            f = math.exp(-(t - h.ft) * LN2 / self.p['tau_f'])
            fm = (h.fs * f) / (h.fw * f + 1.0)
        young = age is not None and age <= 4
        bare = BARE_ENDS.get((r[S_SHOE] or '').strip())
        am = 1.0 if r[S_SULKY] == 'AM' else 0.0 if r[S_SULKY] else None
        mg_avg = 1.5
        if h.mg is not None:
            f = math.exp(-(t - h.mg[2]) * LN2 / self.p['tau_m'])
            mg_avg = (h.mg[0] * f + 1.5) / (h.mg[1] * f + 1.0)
        trip = h.trip
        return array('d', (
            h.mu, dv[0] if dv else 0.0, tv[0] if tv else 0.0, ctx, math.sqrt(ph),
            1.0 if h.obs == 0 else 0.0, math.log1p(h.obs),
            fm,
            h.g if h.g is not None else 0.0, h.won, h.placed, h.gal, h.dq,
            1.0 if h.last is None else 0.0,
            math.log1p(days) if days is not None else 0.0,
            1.0 if days is not None and days >= 90 else 0.0,
            math.log1p(days_p) if (young and days_p is not None) else 0.0,
            1.0 if age is not None and age <= 3 else 0.0,
            1.0 if age == 4 else 0.0,
            bare or 0.0,
            (bare - h.bare) if (bare is not None and h.bare is not None) else 0.0,
            am or 0.0,
            (am - h.am) if (am is not None and h.am is not None) else 0.0,
            math.log1p(h.earn / 1000.0),
            math.log1p(h.earn / 1000.0 / h.starts) if h.starts else 0.0,
            max(-4.0, min(4.0, rec_rel)) if rec_rel is not None else 0.0,
            1.0 if rec_rel is None else 0.0,
            mg_avg,
            h.mg_last if h.mg_last is not None else mg_avg,
            (h.gr[0] + 0.36) / (h.gr[1] + 3.0),
            *(trip if trip else (0.0,) * 6),
            self._form(tv, t),
            self._form(dv, t),
        ))

    def _update(self, rows, hs, Ph, ds, Pd, ts, Pt, ctx, q, m, pred, ctry, track, auto,
                posts, till, t):
        n = len(rows)
        P = self.p
        z = [math.log(x) for x in q]
        dd = [z[i] - ctx[i] for i in range(n)]
        Ps = [Ph[i] + Pd[i] + Pt[i] for i in range(n)]
        w = [1.0 / (P['r0'] * (1.0 + P['r_long'] * max(0.0, -math.log(q[i] * n)))) for i in range(n)]
        W = sum(w)
        zb = sum(w[i] * z[i] for i in range(n)) / W
        pb = sum(w[i] * pred[i] for i in range(n)) / W
        innov = [(z[i] - zb) - (pred[i] - pb) for i in range(n)]
        mean, var = kalman(m, Ps, dd, w)
        for i in range(n):
            h = hs[i]
            delta = mean[i] - m[i]
            shrink = Ps[i] - var[i]
            kh = Ph[i] / Ps[i]
            first = h.obs == 0
            h.mu += kh * delta
            h.P = max(1e-4, Ph[i] - kh * kh * shrink)
            h.t = t
            h.obs += 1
            if first:
                self.mu0_n += 1
                self.mu0 += (h.mu - self.mu0) / min(self.mu0_n, 5000)
            for pv, pp in ((ds[i], Pd[i]), (ts[i], Pt[i])):
                if pv is not None:
                    k = pp / Ps[i]
                    pv[0] += k * delta
                    pv[1] = max(1e-4, pp - k * k * shrink)
                    pv[2] = t
        # Context effects learn from what the ratings did not expect.
        for i in range(n):
            e = innov[i]
            k = (auto, posts[i], till[i] == 0)
            kp = (ctry,) + k
            self.post[kp] = self.post.get(kp, 0.0) + P['eta'] * e
            kt = (track,) + k
            self.track[kt] = self.track.get(kt, 0.0) + P['eta_t'] * e
            kl = (auto, min(till[i] // 20, 5))
            self.till[kl] = self.till.get(kl, 0.0) + P['eta'] * e

    def _results(self, rows, hs, ds, ts, q, t, auto):
        n = len(rows)
        sc = finish_scores(rows)
        g = vs_market(sc, q) if q is not None else None
        tau = self.p['tau_f']
        decay_p = LN2 / self.p['tau_p']
        decay_m = LN2 / self.p['tau_m']
        win_time = next((float(r[S_TIME]) for r, s in zip(rows, sc)
                         if s == 1.0 and r[S_TIME]), None)
        for i, r in enumerate(rows):
            h = hs[i]
            if g is not None:
                if h.fw > 0:
                    f = math.exp(-(t - h.ft) * LN2 / tau)
                    h.fs *= f
                    h.fw *= f
                h.fs += g[i]
                h.fw += 1.0
                h.ft = t
                h.g = g[i]
                for pv in (ds[i], ts[i]):
                    if pv is not None:
                        f = math.exp(-(t - pv[5]) * decay_p)
                        pv[3] = pv[3] * f + g[i]
                        pv[4] = pv[4] * f + 1.0
                        pv[5] = t
            else:
                h.g = None
            km = float(r[S_TIME]) if r[S_TIME] and not r[S_DQ] and 60.0 < float(r[S_TIME]) < 110.0 else None
            h.mg_last = None
            if km is not None and win_time is not None:
                mg = min(5.0, max(0.0, km - win_time))
                if h.mg is None:
                    h.mg = [0.0, 0.0, t]
                f = math.exp(-(t - h.mg[2]) * decay_m)
                h.mg[0] = h.mg[0] * f + mg
                h.mg[1] = h.mg[1] * f + 1.0
                h.mg[2] = t
                h.mg_last = mg
            if km is not None and not r[S_GAL] and sc[i] is not None:
                if h.rec is None:
                    h.rec = [None] * 6
                k = _rec_slot(auto, r[S_DIST] or 2140)
                if h.rec[k] is None or km < h.rec[k]:
                    h.rec[k] = km
            h.gr[0] += 1.0 if r[S_GAL] else 0.0
            h.gr[1] += 1.0
            tr = parse_comment(r[S_COMMENT], field=n, distance=r[S_DIST], galopp=bool(r[S_GAL]))
            h.trip = None if tr is None else (
                float(tr.death), float(tr.wide), float(tr.disturbed), float(tr.start_trouble),
                tr.finish or 0.0, tr.strength or 0.0)
            s = sc[i]
            h.won = 1.0 if s == 1.0 else 0.0
            h.placed = 1.0 if s is not None and s <= 3.0 else 0.0
            h.gal = 1.0 if r[S_GAL] else 0.0
            h.dq = 1.0 if r[S_DQ] else 0.0
            h.last = t
            h.starts += 1
            bare = BARE_ENDS.get((r[S_SHOE] or '').strip())
            if bare is not None:
                h.bare = bare
            if r[S_SULKY]:
                h.am = 1.0 if r[S_SULKY] == 'AM' else 0.0
            if r[S_PRIZE]:
                h.earn += float(r[S_PRIZE])


# ── stage 2 ──────────────────────────────────────────────────────────────

def tensors(rows, with_target=True):
    """Pass rows -> (race ids, entry ids, X[R,N,K], mask[R,N], q[R,N], won[R,N])."""
    import numpy as np
    races: dict = {}
    for r in rows:
        races.setdefault(r[0], []).append(r)
    R = len(races)
    N = max((len(v) for v in races.values()), default=1)
    K = len(rows[0][-1]) if rows else len(FEATURES)
    X = np.zeros((R, N, K))
    mask = np.zeros((R, N), dtype=bool)
    q = np.zeros((R, N))
    won = np.zeros((R, N))
    rids, eids, ts = [], np.zeros((R, N), dtype=np.int64), np.zeros(R, dtype=np.int64)
    for ri, (rid, rs) in enumerate(races.items()):
        rids.append(rid)
        ts[ri] = rs[0][1]
        for j, r in enumerate(rs):
            X[ri, j] = r[-1]
            mask[ri, j] = True
            eids[ri, j] = r[2]
            if with_target:
                q[ri, j] = r[3]
                won[ri, j] = r[4]
    return rids, ts, eids, X, mask, q, won


def softmax(S, mask):
    import numpy as np
    S = np.where(mask, S, -np.inf)
    S = S - S.max(axis=1, keepdims=True)
    E = np.where(mask, np.exp(S), 0.0)
    return E / E.sum(axis=1, keepdims=True)


def stage1_shares(X, mask):
    """The rating pass's own win-pool shares: horse + driver + stable
    strength plus the post / track / tillägg effect, softmaxed per race."""
    k = {f: i for i, f in enumerate(FEATURES)}
    return softmax(X[..., k['mu_h']] + X[..., k['mu_d']] + X[..., k['mu_t']] + X[..., k['ctx']], mask)


def results_strength(X):
    """The results rating's horse + driver + post sum (log win odds)."""
    k = {f: i for i, f in enumerate(FEATURES)}
    return X[..., k['r_h']] + X[..., k['r_d']] + X[..., k['r_ctx']]


def results_shares(X, mask):
    """The results rating's own win chances, softmaxed per race."""
    return softmax(results_strength(X), mask)


def standardize(X, mask):
    """Feature means / scales over real runners (kept with the model)."""
    import numpy as np
    flat = X[mask]
    mu = flat.mean(axis=0)
    sd = flat.std(axis=0)
    sd[sd < 1e-9] = 1.0
    return mu, sd


def fit_soft_logit(X, mask, q, l2=1.0, iters=25, log=None):
    """Multinomial logit over each race's runners fitted to the market's
    shares q (cross-entropy), by Newton steps."""
    import numpy as np
    K = X.shape[2]
    beta = np.zeros(K)
    n_races = X.shape[0]
    Xf = X.reshape(-1, K)
    for it in range(iters):
        p = softmax(X @ beta, mask)
        xm = np.einsum('rn,rnk->rk', p, X)
        g = Xf.T @ (q - p).reshape(-1) - l2 * beta
        H = (Xf * p.reshape(-1, 1)).T @ Xf - xm.T @ xm + l2 * np.eye(K)
        step = np.linalg.solve(H, g)
        beta += step
        if log:
            ce = -np.sum(np.where(mask, q * np.log(np.clip(p, 1e-12, 1)), 0.0)) / n_races
            log(f"[xodds] newton {it + 1}: cross-entropy/race {ce:.4f}")
        if np.max(np.abs(step)) < 1e-6:
            break
    return beta


def design(S, X, mask):
    """Logit score + features + where the runner sits in its field: the
    inputs of the residual trees."""
    import numpy as np
    n = mask.sum(axis=1, keepdims=True)

    def rel(a):
        return a - np.where(mask, a, 0.0).sum(axis=1, keepdims=True) / n

    smax = np.where(mask, S, -np.inf).max(axis=1, keepdims=True)
    k = {f: i for i, f in enumerate(FEATURES)}
    extra = np.stack([rel(S), S - smax, np.broadcast_to(n, S.shape).astype(float),
                      rel(X[..., k['mu_h']]), rel(X[..., k['mu_t']]), rel(results_strength(X))], axis=-1)
    return np.concatenate([X, extra], axis=-1)


def logit_scores(art, X):
    return ((X - art['mu']) / art['sd']) @ art['beta']


def residual_target(S, mask, q):
    """log q - log p, centred on the market-weighted mean of each race."""
    import numpy as np
    y = np.log(np.clip(q, 1e-9, 1)) - np.log(np.clip(softmax(S, mask), 1e-9, 1))
    return y - np.where(mask, y * q, 0.0).sum(axis=1, keepdims=True)


def predict(art, X, mask):
    """Win-pool shares per runner (padded slots 0)."""
    import numpy as np
    S = logit_scores(art, X)
    gbm = art.get('gbm')
    if gbm is not None:
        adj = np.zeros_like(S)
        adj[mask] = gbm.predict(design(S, X, mask)[mask])
        S = S + adj
    return softmax(S, mask)
