"""One chronological pass over every result since 2010.

For each race it (1) reads every runner's pre-race state, (2) updates the
horse / driver / start-post Elo from the finishing order, (3) folds the
start's trip, verdict, time and xLabs facts into the horse's running
evidence, and (4) once inside the snapshot window, writes the horse's
post-race sub-ratings, scaled per breed against the horses active that
month. The as-of rating for a date D is the horse's last snapshot before D.

Evidence is recency weighted without per-start decay passes: an observation
at day t carries weight exp((t - t0) / tau), so a running sum and weight sum
give the decayed mean at any later time, and weight_sum * exp(-(T - t0)/tau)
is the effective number of observations at T.
"""

from __future__ import annotations

import datetime as dt
import math
import re
from array import array
from collections import deque

from core.ratings import keys as K
from core.ratings.comments import parse

LN2 = math.log(2)
T0 = dt.date(2010, 1, 1).toordinal()
TAU_A = 365 / LN2        # abilities and class evidence: one-year half-life
TAU_F = 60 / LN2         # recent form
TAU_T = 548 / LN2        # trainer A/E
TAU_S = 45 / LN2         # stable form
WINDOW = 730             # best-of lists look back two years

ELO0 = 1500.0
K_HORSE = 40.0
K_DRIVER = 8.0
K_POST = 2.0
K_START = 32.0

QUALIFIER_RE = re.compile(
    r"^(gdk|ejg|ejp|gd|gk|egk|egdk|gkd|gdj|gdb|gdek|gdgk|gdl|ddk|frdk|erj|ejk|"
    r"ejgk|ejgd|ej|EJ|EJG|EJP|GDK|Gdk)[0-9]?$|^[12]?[pP][0-9]?$")
NON_FINISH_GALLOP = {'DA', 'd', 'g'}

# Running-evidence slots: (sum, weight) pairs in Horse.acc.
(A_EARLY, A_GAINS, A_FVERD, A_DEATH, A_WIDE, A_FADE, A_TEMPO, A_LONG, A_G, A_G2,
 A_SVERD, A_GALLOP, A_GAIT, A_RATE, A_CLOSE, A_HOLD, A_FIGHT, A_ACCEL, A_TESTED,
 A_PRIZE, A_RECENT) = range(21)
N_ACC = 21
FORM_SLOTS = {A_RECENT}

ROW = ('entry_id', 'race_id', 'race_date', 'horse_id', 'driver_id', 'trainer_id',
       'post', 'auto', 'distance', 'tillagg', 'placement', 'ptext', 'galopp', 'dq',
       'time', 'odds', 'prize', 'f200', 'l200', 'b100', 'intervals', 'comment',
       'breed', 'dob', 'track_id')
(R_EID, R_RID, R_DATE, R_HID, R_DID, R_TID, R_POST, R_AUTO, R_DIST, R_TILL, R_PLC,
 R_PTXT, R_GAL, R_DQ, R_TIME, R_ODDS, R_PRIZE, R_F200, R_L200, R_B100, R_INT,
 R_COMMENT, R_BREED, R_DOB, R_TRACK) = range(len(ROW))

FRESH_BUCKETS = (7, 14, 21, 35, 60, 90, 180, 365, 10 ** 6)


def fresh_bucket(days: int | None) -> int:
    if days is None:
        return len(FRESH_BUCKETS)
    for i, hi in enumerate(FRESH_BUCKETS):
        if days <= hi:
            return i
    return len(FRESH_BUCKETS) - 1


def dist_bucket(d: int | None) -> int:
    d = d or 2140
    return 0 if d < 1800 else 1 if d < 2400 else 2 if d < 2900 else 3


class Horse:
    __slots__ = ('R', 'n', 'last', 'last_entry', 'breed', 'votes', 'by', 'hist', 'm3', 'peak',
                 'yp', 'yn', 'sR', 'sn', 'acc', 'best')

    def __init__(self, breed, by):
        self.R = ELO0
        self.n = 0
        self.last = None
        self.last_entry = None
        # Most foreign horses carry no breed code; races are one breed, so
        # theirs is read off the coded horses they meet (votes: [V, K]).
        self.breed = breed if breed in ('V', 'K') else 'V'
        self.votes = None if breed in ('V', 'K') else [0, 0]
        self.by = by
        self.hist = deque(maxlen=6)
        self.m3 = deque(maxlen=3)
        self.peak = None
        self.yp = None          # best Elo through age 4, for the sire index
        self.yn = 0
        self.sR = ELO0
        self.sn = 0
        self.acc = array('d', bytes(16 * N_ACC))
        self.best = None


def _mean(acc, slot):
    w = acc[2 * slot + 1]
    return acc[2 * slot] / w if w > 0 else None


def _best_of(lst, t):
    vals = [v for (d, v) in lst if d >= t - WINDOW]
    if not vals:
        return None, 0
    vals.sort(reverse=True)
    if len(vals) >= 3:
        return (vals[0] + vals[1]) / 2, len(vals)
    return vals[0], len(vals)


def _median_recent(lst, t, k=8):
    vals = [v for (d, v) in lst if d >= t - WINDOW][-k:]
    if not vals:
        return None, 0
    s = sorted(vals)
    m = len(s) // 2
    return (s[m] if len(s) % 2 else (s[m - 1] + s[m]) / 2), len(vals)


def _robust(vals):
    """(median, spread) — IQR-based unless the distribution is mostly tied."""
    n = len(vals)
    s = sorted(vals)
    med = s[n // 2]
    iqr = (s[(3 * n) // 4] - s[n // 4]) / 1.349
    mu = sum(s) / n
    sd = math.sqrt(sum((x - mu) ** 2 for x in s) / max(1, n - 1))
    spread = iqr if iqr > 0.3 * sd else sd
    return med, (spread if spread > 1e-9 else 1.0)


def _clip(z, lim=4.0):
    return -lim if z < -lim else lim if z > lim else z


def _disp(z):
    v = int(round(K.SCALE_MID + K.SCALE_SD * z))
    return K.SCALE_LO if v < K.SCALE_LO else K.SCALE_HI if v > K.SCALE_HI else v


class Engine:
    def __init__(self, *, snap_start: dt.date, bt_start: dt.date, km_off, track_off,
                 post_off, log=print):
        self.snap_t = snap_start.toordinal()
        self.bt_t = bt_start.toordinal()
        self.km_off = km_off            # (breed, auto, dist bucket) -> s/km
        self.track_off = track_off      # track_id -> s/km
        self.post_off = post_off        # (auto, post) -> ms/km on the first 200 m
        self.log = log
        self.H: dict[int, Horse] = {}
        self.D: dict[int, list] = {}    # driver: [R, n, last]
        self.T: dict[int, list] = {}    # trainer: [A, E, As, Es, last]
        self.post_b: dict = {}
        self.till_b: dict = {}
        self.spost_b: dict = {}
        self.exp_early: dict = {}       # (auto, post) -> [sum pct, n]
        self.sector_m: dict = {}        # (breed, metres) -> [sum pace, n]
        self.fresh: list = [[0.0, 0] for _ in range(len(FRESH_BUCKETS) + 1)]
        self.pair: dict = {}            # (horse, driver) -> [sum resid, n]
        self.params: dict = {}          # breed -> census params
        self.pparams: dict | None = None
        self.month = None
        self.seeded = False
        self.last_snap: dict = {}       # horse -> (hv, hc, starts, t)
        self.snap_rows: list = []
        self.person_rows: list = []
        self.bt: list = []              # backtest races
        self.races = 0
        self.entries = 0

    # ── state access ──────────────────────────────────────────────────────
    def horse(self, r) -> Horse:
        h = self.H.get(r[R_HID])
        if h is None:
            dob = r[R_DOB]
            h = Horse(r[R_BREED], dob.year if dob else None)
            self.H[r[R_HID]] = h
        return h

    def _push_best(self, h, name, t, v):
        if h.best is None:
            h.best = {}
        lst = h.best.get(name)
        if lst is None:
            lst = h.best[name] = []
        lst.append((t, v))
        while lst and lst[0][0] < t - WINDOW:
            lst.pop(0)
        if len(lst) > 12:
            lst.pop(0)

    # ── raw sub values ────────────────────────────────────────────────────
    def horse_raw(self, h: Horse, t: int):
        """Raw value (higher is better) and evidence count per HORSE_KEYS."""
        a = h.acc
        ia = math.exp(-(t - T0) / TAU_A)
        if_ = math.exp(-(t - T0) / TAU_F)

        def slot(s):
            w = a[2 * s + 1]
            if w <= 0:
                return None, 0.0
            return a[2 * s] / w, w * (if_ if s in FORM_SLOTS else ia)

        best = h.best or {}
        out = {}
        for name in ('gate', 'top', 'closing', 'sectors', 'record'):
            lst = best.get(name)
            out[name] = _best_of(lst, t) if lst else (None, 0)
        lst = best.get('cruise')
        out['cruise'] = _median_recent(lst, t) if lst else (None, 0)
        out['early'] = slot(A_EARLY)
        out['duels'] = (h.sR, h.sn) if h.sn else (None, 0)
        out['accel'] = slot(A_ACCEL)
        out['gains'] = slot(A_GAINS)
        out['fverdict'] = slot(A_FVERD)
        out['death'] = slot(A_DEATH)
        out['wide'] = slot(A_WIDE)
        out['fade'] = slot(A_FADE)
        out['tempo'] = slot(A_TEMPO)
        lg, ln = slot(A_LONG)
        g, gn = slot(A_G)
        out['distance'] = ((lg - g), ln) if lg is not None and g is not None else (None, 0)
        out['sverdict'] = slot(A_SVERD)
        gal, galn = slot(A_GALLOP)
        out['gallop'] = (-gal, galn) if gal is not None else (None, 0)
        gt, gtn = slot(A_GAIT)
        out['gait'] = (gt, gtn)
        out['rate'] = slot(A_RATE)
        g2, _ = slot(A_G2)
        if g is not None and gn >= 1:
            out['consistency'] = (-math.sqrt(max(0.0, g2 - g * g)), gn)
        else:
            out['consistency'] = (None, 0)
        out['close'] = slot(A_CLOSE)
        out['holding'] = slot(A_HOLD)
        out['fight'] = slot(A_FIGHT)
        out['elo'] = (h.R, h.n) if h.n else (None, 0)
        out['peak'] = (h.peak, h.n) if h.peak is not None else (None, 0)
        out['tested'] = slot(A_TESTED)
        out['prize'] = slot(A_PRIZE)
        if len(h.hist) >= 2:
            out['trend'] = (h.hist[-1] - h.hist[0], len(h.hist) - 1)
        else:
            out['trend'] = (None, 0)
        out['recent'] = slot(A_RECENT)
        raw = []
        ns = []
        for k in K.HORSE_KEYS:
            v, n = out[k]
            if v is None or n <= 0:
                raw.append(None)
                ns.append(0.0)
            else:
                raw.append(v)
                ns.append(n)
        return raw, ns

    # ── census: per-breed scaling against horses active this month ────────
    def census(self, t: int):
        act = {'V': [], 'K': []}
        for h in self.H.values():
            if h.last is not None and h.last >= t - 365 and h.n:
                act[h.breed].append(h)
        nk = len(K.HORSE_KEYS)
        kk = [K.SUB_BY_KEY[k].k for k in K.HORSE_KEYS]
        ie = K.HORSE_INDEX['elo']
        for b, hs in act.items():
            if len(hs) < 200:
                continue
            rows = [self.horse_raw(h, t) for h in hs]
            elos = [h.R for h in hs if h.n >= 6]
            e_mu, e_sd = _robust(elos) if len(elos) > 50 else (ELO0, 150.0)
            subs = []
            for k in range(nk):
                good = [r[0][k] for r in rows if r[0][k] is not None and r[1][k] >= kk[k]]
                if len(good) < 40:
                    good = [r[0][k] for r in rows if r[0][k] is not None]
                if len(good) < 20:
                    subs.append((0.0, 1.0, 0.0, 0))
                    continue
                mu, sd = _robust(good)
                rho = 0.0
                if k != ie:
                    xs, ys = [], []
                    for (raw, ns), h in zip(rows, hs):
                        if raw[k] is not None and ns[k] >= kk[k] and h.n >= 6:
                            xs.append(_clip((raw[k] - mu) / sd))
                            ys.append(_clip((h.R - e_mu) / e_sd))
                    if len(xs) >= 40:
                        mx = sum(xs) / len(xs)
                        my = sum(ys) / len(ys)
                        cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
                        vx = sum((x - mx) ** 2 for x in xs)
                        vy = sum((y - my) ** 2 for y in ys)
                        if vx > 0 and vy > 0:
                            rho = max(-0.9, min(0.9, cov / math.sqrt(vx * vy)))
                subs.append((mu, sd, rho, len(good)))
            self.params[b] = {'elo': (e_mu, e_sd), 'subs': subs, 'active': len(hs), 't': t}
        self._person_census(t)

    def _decayed_trainer(self, s, t):
        dl = t - s[4]
        f = math.exp(-dl / TAU_T)
        fs = math.exp(-dl / TAU_S)
        return s[0] * f, s[1] * f, s[2] * fs, s[3] * fs

    def _person_census(self, t: int):
        tr = []
        for tid, s in self.T.items():
            if s[4] >= t - 365:
                A, E, As, Es = self._decayed_trainer(s, t)
                tr.append((tid, A, E, As, Es))
        dr = [(did, s) for did, s in self.D.items() if s[2] >= t - 365 and s[1] > 0]
        P = {}
        long_ = [math.log((A + 1) / (E + 1)) for _, A, E, _, _ in tr if E >= 15]
        short = [math.log((As + .5) / (Es + .5)) for _, _, _, As, Es in tr if Es >= 3]
        drv = [s[0] for _, s in dr if s[1] >= 50]
        P['trainer'] = _robust(long_) if len(long_) > 30 else (0.0, 0.3)
        P['stable'] = _robust(short) if len(short) > 30 else (0.0, 0.5)
        P['driver'] = _robust(drv) if len(drv) > 30 else (0.0, 60.0)
        self.pparams = P
        if t < self.snap_t:
            return
        month = dt.date.fromordinal(t).replace(day=1)
        for tid, A, E, As, Es in tr:
            v, c, raw = self._trainer_value(A, E)
            v2, c2, raw2 = self._stable_value(As, Es)
            self.person_rows.append((tid, 'T', month, v, c, raw, v2, c2, raw2, round(E, 2)))
        for did, s in dr:
            v, c, raw = self._driver_value(s)
            self.person_rows.append((did, 'D', month, v, c, raw, None, None, None, s[1]))

    def _trainer_value(self, A, E):
        mu, sd = self.pparams['trainer']
        raw = math.log((A + 1) / (E + 1))
        c = E / (E + K.SUB_BY_KEY['trainer'].k)
        return _disp(c * _clip((raw - mu) / sd)), int(round(100 * c)), round((A + 1) / (E + 1), 3)

    def _stable_value(self, As, Es):
        mu, sd = self.pparams['stable']
        raw = math.log((As + .5) / (Es + .5))
        c = Es / (Es + K.SUB_BY_KEY['stable'].k)
        return _disp(c * _clip((raw - mu) / sd)), int(round(100 * c)), round((As + .5) / (Es + .5), 3)

    def _driver_value(self, s):
        mu, sd = self.pparams['driver']
        c = s[1] / (s[1] + K.SUB_BY_KEY['driver'].k)
        return _disp(c * _clip((s[0] - mu) / sd)), int(round(100 * c)), round(s[0], 1)

    def person_now(self, tid, did, t):
        """(trainer v,c, stable v,c, driver v,c) as of t, before the race."""
        out = [0, 0, 0, 0, 0, 0]
        if self.pparams is None:
            return out
        s = self.T.get(tid)
        if s is not None:
            A, E, As, Es = self._decayed_trainer(s, t)
            out[0], out[1], _ = self._trainer_value(A, E)
            out[2], out[3], _ = self._stable_value(As, Es)
        d = self.D.get(did)
        if d is not None and d[1] > 0:
            out[4], out[5], _ = self._driver_value(d)
        return out

    # ── snapshot: scaled sub values for one horse ─────────────────────────
    def snapshot(self, h: Horse, t: int):
        P = self.params.get(h.breed) or self.params.get('V')
        raw, ns = self.horse_raw(h, t)
        e_mu, e_sd = P['elo']
        ce = h.n / (h.n + 6)
        ze = _clip((h.R - e_mu) / e_sd) * ce
        hv = []
        hc = []
        for i, k in enumerate(K.HORSE_KEYS):
            x = raw[i]
            if x is None:
                hv.append(0)
                hc.append(0)
                continue
            mu, sd, rho, _ = P['subs'][i]
            n = ns[i]
            c = n / (n + K.SUB_BY_KEY[k].k)
            z = c * _clip((x - mu) / sd) + (1 - c) * rho * ze
            hv.append(_disp(z))
            hc.append(max(1, int(round(100 * c))))
        return tuple(hv), tuple(hc), raw

    # ── the race ──────────────────────────────────────────────────────────
    def race(self, rows):
        t = rows[0][R_DATE].toordinal()
        n_q = sum(1 for r in rows if r[R_PTXT] and QUALIFIER_RE.match(r[R_PTXT].strip()))
        if n_q * 2 >= len(rows):
            return
        if not any(r[R_PLC] is not None for r in rows):
            return
        rows = [r for r in rows if not (r[R_PTXT] and QUALIFIER_RE.match(r[R_PTXT].strip()))]
        n = len(rows)
        if n == 0:
            return

        month = (rows[0][R_DATE].year, rows[0][R_DATE].month)
        if month != self.month and t >= self.snap_t - 31:
            self.month = month
            self.census(t)
        if not self.seeded and t >= self.snap_t:
            self.seeded = True
            self._seed_snapshots(t)
        self.races += 1
        self.entries += n

        hs = [self.horse(r) for r in rows]
        if any(h.votes is not None for h in hs):
            nv = sum(1 for h in hs if h.votes is None and h.breed == 'V')
            nk = sum(1 for h in hs if h.votes is None and h.breed == 'K')
            if nv or nk:
                for h in hs:
                    if h.votes is not None:
                        h.votes[0 if nv >= nk else 1] += 1
                        h.breed = 'V' if h.votes[0] >= h.votes[1] else 'K'
        # finishing score: placing, then unplaced finishers by time, then the rest tied
        sc = []
        for r in rows:
            p = r[R_PLC]
            if p is not None and p > 0:
                sc.append(float(p))
            elif (r[R_PTXT] or '').strip() == '0' and r[R_TIME] and not r[R_GAL] and not r[R_DQ]:
                sc.append(100.0 + r[R_TIME])
            else:
                sc.append(None)
        fin = sorted(s for s in sc if s is not None)
        fin_pct = []
        for s in sc:
            if s is None or n < 2:
                fin_pct.append(None)
            else:
                lo = fin.index(s)
                hi = lo + fin.count(s) - 1
                fin_pct.append(((lo + hi) / 2) / (n - 1))

        auto = [bool(r[R_AUTO]) for r in rows]
        posts = [min(r[R_POST] or 0, 15) for r in rows]
        short = [(r[R_DIST] or 2140) < 1800 for r in rows]
        pk = [(auto[i], posts[i], short[i]) for i in range(n)]
        tk = [min((rows[i][R_TILL] or 0) // 20, 4) for i in range(n)]
        dr = [self.D.get(r[R_DID]) for r in rows]
        R_pre = [h.R for h in hs]
        T = [R_pre[i] + (dr[i][0] if dr[i] else 0.0) + self.post_b.get(pk[i], 0.0)
             + self.till_b.get(tk[i], 0.0) for i in range(n)]
        Tm = max(T)
        Q = [10 ** ((x - Tm) / 400) for x in T]
        sQ = sum(Q)
        pwin = [q / sQ for q in Q]

        if t >= self.bt_t:
            self._record_backtest(rows, hs, pwin, t)

        g = [0.0] * n
        W = [0.0] * n
        for i in range(n):
            si = sc[i]
            Qi = Q[i]
            for j in range(i + 1, n):
                sj = sc[j]
                if si is None and sj is None:
                    continue
                if si is None:
                    S, w = 0.0, 0.5
                elif sj is None:
                    S, w = 1.0, 0.5
                else:
                    S = 1.0 if si < sj else 0.5 if si == sj else 0.0
                    w = 1.0
                d = w * (S - Qi / (Qi + Q[j]))
                g[i] += d
                g[j] -= d
                W[i] += w
                W[j] += w
        resid = [g[i] / W[i] if W[i] > 0 else 0.0 for i in range(n)]

        if n >= 2:
            inv = 1.0 / (n - 1)
            for i in range(n):
                h = hs[i]
                h.R += K_HORSE * (1 + 2 * math.exp(-h.n / 4)) * g[i] * inv
                d = dr[i]
                did = rows[i][R_DID]
                if did is not None:
                    if d is None:
                        d = self.D[did] = [0.0, 0, t]
                    d[0] += K_DRIVER * (1 + 2 * math.exp(-d[1] / 20)) * g[i] * inv
                self.post_b[pk[i]] = self.post_b.get(pk[i], 0.0) + K_POST * g[i] * inv
                self.till_b[tk[i]] = self.till_b.get(tk[i], 0.0) + K_POST * g[i] * inv
            for i in range(n):
                did = rows[i][R_DID]
                if did is not None:
                    d = self.D[did]
                    d[1] += 1
                    d[2] = t

        trips = [parse(r[R_COMMENT], field=n, distance=r[R_DIST], galopp=bool(r[R_GAL]))
                 if r[R_COMMENT] else None for r in rows]
        self._start_duels(rows, hs, trips, auto, posts, n)
        self._close_finishes(rows, hs, trips, t)

        wa = math.exp((t - T0) / TAU_A)
        wf = math.exp((t - T0) / TAU_F)
        sum_R = sum(R_pre)
        max_prize = max((r[R_PRIZE] or 0) for r in rows)
        complete = n >= 6
        for i in range(n):
            r = rows[i]
            h = hs[i]
            tr = trips[i]
            a = h.acc
            gal = bool(r[R_GAL]) or (r[R_PTXT] or '').strip() == 'DA'
            x = resid[i]

            def add(slot, v, w=wa):
                a[2 * slot] += w * v
                a[2 * slot + 1] += w

            if n >= 2:
                add(A_G, x)
                add(A_G2, x * x)
                add(A_RECENT, x, wf)
                if (r[R_DIST] or 0) >= 2500:
                    add(A_LONG, x)
                if n >= 5:
                    add(A_TESTED, (sum_R - R_pre[i]) / (n - 1))
            if max_prize > 0 and n >= 5:
                add(A_PRIZE, math.log10(max_prize))
            add(A_GALLOP, 1.0 if gal and not (tr and tr.disturbed) else 0.0)

            if tr is not None:
                add(A_FVERD, tr.finish or 0.0)
                add(A_SVERD, tr.strength or 0.0)
                add(A_FIGHT, tr.heart or 0.0)
                add(A_RATE, tr.rate or 0.0)
                add(A_GAIT, tr.gait or 0.0)
                if tr.early_rank is not None and n >= 5:
                    pct = (tr.early_rank - 1) / (n - 1)
                    ek = (auto[i], posts[i])
                    m = self.exp_early.get(ek)
                    if m is None:
                        m = self.exp_early[ek] = [0.0, 0]
                    if m[1] >= 30:
                        add(A_EARLY, m[0] / m[1] - pct)
                    m[0] += pct
                    m[1] += 1
                    if fin_pct[i] is not None and pct >= 0.25:
                        add(A_GAINS, pct - fin_pct[i])
                if n >= 2 and not gal:
                    if tr.death:
                        add(A_DEATH, x)
                    if tr.wide:
                        add(A_WIDE, x)
                    if tr.led:
                        add(A_TEMPO, x + (0.15 if tr.hard_tempo else 0.0)
                            - (0.15 if tr.soft_tempo else 0.0))
                if tr.led_through and n >= 2:
                    add(A_HOLD, (1.0 if r[R_PLC] == 1 else 0.0) - pwin[i])
                if tr.sector is not None:
                    pace, metres = tr.sector
                    key = (h.breed, metres)
                    m = self.sector_m.get(key)
                    if m is None:
                        m = self.sector_m[key] = [0.0, 0]
                    base = self.sector_m.get((h.breed, 500))
                    if m[1] >= 30 and base and base[1] >= 30:
                        self._push_best(h, 'sectors', t, -(pace - (m[0] / m[1] - base[0] / base[1])))
                    m[0] += pace
                    m[1] += 1

            tm = r[R_TIME]
            if tm and 55 < tm < 130 and not gal and not r[R_DQ] and sc[i] is not None:
                adj = (tm - self.km_off.get((h.breed, auto[i], dist_bucket(r[R_DIST])), 0.0)
                       - self.track_off.get(r[R_TRACK], 0.0))
                self._push_best(h, 'record', t, -adj)
                self._push_best(h, 'cruise', t, -adj)
            self._xlabs(h, r, tr, auto[i], posts[i], gal, t, add)

            h.n += 1
            h.hist.append(h.R)
            h.m3.append(h.R)
            if len(h.m3) == 3:
                med = sorted(h.m3)[1]
                if h.peak is None or med > h.peak:
                    h.peak = med
            if h.by and rows[0][R_DATE].year - h.by <= 4:
                h.yp = h.R if h.yp is None else max(h.yp, h.R)
                h.yn += 1
            days = (t - h.last) if h.last is not None else None
            if t >= self.snap_t - 365 * 8:
                fb = self.fresh[fresh_bucket(days)]
                fb[0] += x
                fb[1] += 1
            h.last = t
            h.last_entry = r[R_EID]

            if t >= self.bt_t - 365 * 3 and r[R_DID] is not None and n >= 2:
                pkey = (r[R_HID], r[R_DID])
                pv = self.pair.get(pkey)
                if pv is None:
                    pv = self.pair[pkey] = [0.0, 0]
                pv[0] += x
                pv[1] += 1

            tid = r[R_TID]
            if tid is not None and complete:
                s = self.T.get(tid)
                if s is None:
                    s = self.T[tid] = [0.0, 0.0, 0.0, 0.0, t]
                if s[4] != t:
                    dl = t - s[4]
                    f = math.exp(-dl / TAU_T)
                    fs = math.exp(-dl / TAU_S)
                    s[0] *= f
                    s[1] *= f
                    s[2] *= fs
                    s[3] *= fs
                    s[4] = t
                won = 1.0 if r[R_PLC] == 1 else 0.0
                s[0] += won
                s[1] += pwin[i]
                s[2] += won
                s[3] += pwin[i]

            if t >= self.snap_t:
                hv, hc, raw = self.snapshot(h, t)
                self.snap_rows.append((r[R_EID], r[R_HID], rows[0][R_DATE], round(h.R, 1), h.n,
                                       hv, hc, raw, round(x, 4), round(pwin[i], 4), h.breed))
                self.last_snap[r[R_HID]] = (hv, hc, h.n, t)

    def _seed_snapshots(self, t):
        """Horses whose last start predates the window still need an as-of row."""
        for hid, h in self.H.items():
            if h.last is not None and h.last >= t - 3 * 365 and h.n and h.last_entry:
                hv, hc, raw = self.snapshot(h, h.last)
                self.snap_rows.append((h.last_entry, hid, dt.date.fromordinal(h.last),
                                       round(h.R, 1), h.n, hv, hc, raw, None, None, h.breed))
                self.last_snap[hid] = (hv, hc, h.n, h.last)

    def _xlabs(self, h, r, tr, auto, post, gal, t, add):
        f200 = r[R_F200]
        if f200 and 55000 < f200 < 95000 and not (tr and tr.start_trouble):
            off = self.post_off.get((auto, post), self.post_off.get((auto, None), 0.0))
            self._push_best(h, 'gate', t, -(f200 - off) / 1000)
        b100 = r[R_B100]
        if b100 and 50000 < b100 < 90000:
            self._push_best(h, 'top', t, -b100 / 1000)
        l200 = r[R_L200]
        if l200 and 55000 < l200 < 100000 and not gal:
            self._push_best(h, 'closing', t, -l200 / 1000)
        iv = r[R_INT]
        if not iv or gal:
            return
        try:
            secs = [float(s['duration']) for s in sorted(iv, key=lambda s: float(s['start']))]
        except (TypeError, KeyError, ValueError):
            return
        if len(secs) < 10 or any(not (45000 < s < 120000) for s in secs):
            return
        p2 = [(secs[k] + secs[k + 1]) / 2 for k in range(len(secs) - 1)]
        kick = max(p2[k - 2] - p2[k] for k in range(5, len(p2)))
        add(A_ACCEL, kick / 1000)
        if l200 and 55000 < l200 < 100000 and len(secs) >= 12:
            mid = secs[5:len(secs) - 4]
            if mid:
                add(A_FADE, (sum(mid) / len(mid) - l200) / 1000)

    def _start_duels(self, rows, hs, trips, auto, posts, n):
        idx = [i for i in range(n) if trips[i] is not None and trips[i].early_rank is not None]
        if len(idx) < 4:
            return
        m = len(idx)
        T = [hs[i].sR + self.spost_b.get((auto[i], posts[i]), 0.0) for i in idx]
        Tm = max(T)
        Q = [10 ** ((x - Tm) / 400) for x in T]
        g = [0.0] * m
        for a in range(m):
            ra = trips[idx[a]].early_rank
            for b in range(a + 1, m):
                rb = trips[idx[b]].early_rank
                S = 1.0 if ra < rb else 0.5 if ra == rb else 0.0
                d = S - Q[a] / (Q[a] + Q[b])
                g[a] += d
                g[b] -= d
        inv = 1.0 / (m - 1)
        for a in range(m):
            i = idx[a]
            h = hs[i]
            h.sR += K_START * (1 + 2 * math.exp(-h.sn / 4)) * g[a] * inv
            h.sn += 1
            key = (auto[i], posts[i])
            self.spost_b[key] = self.spost_b.get(key, 0.0) + K_POST * g[a] * inv

    def _close_finishes(self, rows, hs, trips, t):
        wa = math.exp((t - T0) / TAU_A)
        placed = sorted((r[R_PLC], i) for i, r in enumerate(rows)
                        if r[R_PLC] is not None and r[R_PLC] > 0 and r[R_TIME])
        seen = set()
        for (pa, a), (pb, b) in zip(placed, placed[1:]):
            if pb > pa and abs(rows[a][R_TIME] - rows[b][R_TIME]) < 0.05:
                for i, v in ((a, 1.0), (b, -1.0)):
                    acc = hs[i].acc
                    acc[2 * A_CLOSE] += wa * v
                    acc[2 * A_CLOSE + 1] += wa
                    seen.add(i)
        for i, r in enumerate(rows):
            tr = trips[i]
            if i in seen or tr is None or not tr.close or r[R_PLC] not in (1, 2):
                continue
            acc = hs[i].acc
            acc[2 * A_CLOSE] += wa * (1.0 if r[R_PLC] == 1 else -1.0)
            acc[2 * A_CLOSE + 1] += wa

    def _record_backtest(self, rows, hs, pwin, t):
        runners = []
        for i, r in enumerate(rows):
            h = hs[i]
            snap = self.last_snap.get(r[R_HID])
            person = self.person_now(r[R_TID], r[R_DID], t)
            days = (t - h.last) if h.last is not None else None
            pv = self.pair.get((r[R_HID], r[R_DID]))
            pair = (pv[0] / pv[1] - (_mean(h.acc, A_G) or 0.0), pv[1]) if pv and pv[1] else (0.0, 0)
            odds = float(r[R_ODDS]) if r[R_ODDS] else 0.0
            runners.append((r[R_HID], 1 if r[R_PLC] == 1 else 0, odds, pwin[i], snap,
                            person, days, pair, h.breed, h.n))
        self.bt.append((t, runners))
