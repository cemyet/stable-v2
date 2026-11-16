"""xlead — the chance a starter is in front after the first 100–150 m.

The label comes from the race comment's opening clause ("Ledn", "Tog ledn",
"Släppte ledn e 400", "Ledn e 100"); a race only teaches when exactly one
starter reads as the early leader and most of the field has a comment, so a
missing comment can't hide the real leader. Swedish tracks only: abroad the
car wings, volt set-ups and comment coverage differ.

Features come from one chronological pass over every Swedish start since
2012. Each race is read before it updates anything, so every value is as of
the morning of the race:

  posts      how often each post leads, by start method and front/back group,
             globally and on this track (shrunk toward the global rate)
  horse      how often it led against what its posts promised, how far
             forward it gets early against its posts, start trouble, fast
             starts, galopp
  driver     how often the driver leads against what the posts promised,
             and whether the horse changed driver since its last start
  field      the same early speed for everyone else in the race: who is
             faster, and whether the faster ones start inside or outside
  market     (optional `ext`, from the xodds rating pass) how strong the
             horse and its driver are priced against this field: the best
             horse and the top driver often get to choose

Recency: evidence decays with a one-year half-life (drivers two years).
"""

from __future__ import annotations

import datetime as dt
import math
import re
from array import array

from core.ratings.comments import opening

LN2 = math.log(2)
T0 = dt.date(2012, 1, 1).toordinal()
TAU_H = 365 / LN2
TAU_D = 730 / LN2

# Opening phrases that put a horse in front. "Ledn 1450 kv" (took over with
# 1450 m left) is a later lead; "ledn e N" counts when N is within the early
# phase, and "släppte ledn e N" when the horse held it past that phase.
EARLY_M = 150
_PRE = (r'(?:tog|höll|fick|snabb(?:ast)?\s+till|mkt\s+snabb\s+till|flög\s+till|'
        r'körde\s+(?:tufft\s+)?om(?:\s+till)?|rask\s+till|vass\s+till|laddade\s+till|'
        r'snabb\s+start\s+till)')
_RX_LEAD = re.compile(rf'^(?:{_PRE}\s+)?ledn(?:ingen)?(?:\b|(?=eget))'
                      r'(?:\s+e\s+(\d{2,4})|\s+(\d{2,4})\s*kv)?')
_RX_DROP = re.compile(r'^(?:släppte|släpp|tappade|tapp|lämnade)\s+ledn(?:ingen)?\b'
                      r'(?:\s+e\s+(\d{2,4}))?')
_RX_GAVE_UP = re.compile(r'^(?:rygg led|släppte|ner i rygg|ner inv|utv led|ner)\b.*?\be\s+(\d{2,4})')
_RX_OPENER = re.compile(
    r'^(?:(?:bra|fin|snabb|snabbast|kvick|rask|mkt snabb|lugn|ok)\s+(?:start|ut|iväg|ur volten)|'
    r'snabbast|ensam på start|öppnade (?:bra|snabbt)|laddade)\.?$')


def lead_label(comment: str | None) -> int | None:
    """1 if the comment puts the horse in front through the early phase,
    0 if it doesn't, None without a usable comment."""
    if not comment:
        return None
    text = comment.strip().lower()
    if not text or text.startswith('struken'):
        return None
    clauses = [c.strip() for c in text.split(',')]
    heads = clauses[:2] if _RX_OPENER.match(clauses[0]) else clauses[:1]
    for i, h in enumerate(heads):
        m = _RX_LEAD.match(h)
        if m:
            if m.group(2):
                return 0
            if m.group(1):
                return 1 if int(m.group(1)) <= EARLY_M else 0
            nxt = clauses[i + 1] if i + 1 < len(clauses) else ''
            g = _RX_GAVE_UP.match(nxt)
            if g and int(g.group(1)) <= EARLY_M:
                return 0
            return 1
        m = _RX_DROP.match(h)
        if m:
            return 1 if not m.group(1) or int(m.group(1)) > EARLY_M else 0
    return 0


# xfront: who is in front once the early scramble has settled. By 700 m the
# lead has changed hands in about a third of the races, and the share of
# changes stays flat from there to 1000 m.
FRONT_M = 700
_RX_POS = re.compile(r'(?:^|\s)(?:gratis)?ledn(?:ingen)?\b|\bled\b|\binv\b|-inv|-utv|\butv\b|spår|'
                     r'\brygg\b|\bsist\b|\bkö')
_RX_IN_FRONT = re.compile(r'(?:^|\s)(?:gratis)?ledn(?:ingen)?\b')
_RX_NOT_FRONT = re.compile(r'\b(?:släppte|släpp|tappade|tapp|lämnade|utmanade|provade|försökte|'
                           r'för ledn|mot ledn)\b')
_RX_AT = re.compile(r'\be\s+(\d{2,4})\b')
_RX_LEFT = re.compile(r'\b(\d{2,4})\s*kv\b')


def trip(comment: str | None, dist: int | None) -> list:
    """The positions a comment describes, in order: [(metres from the
    start, in front?)]. A clause without a distance marker opens the race
    if it comes first, and is undated (None) after that."""
    if not comment:
        return []
    text = comment.strip().lower()
    if not text or text.startswith('struken'):
        return []
    out = []
    for i, c in enumerate(x.strip() for x in text.split(',')):
        if not c or 'inledning' in c or not _RX_POS.search(c):
            continue
        at, left = _RX_AT.search(c), _RX_LEFT.search(c)
        if at:
            d = int(at.group(1))
        elif left and dist:
            d = dist - int(left.group(1))
        elif 'varvet kv' in c and dist:
            d = dist - 1000
        else:
            d = 0 if not out else None
        out.append((d, bool(_RX_IN_FRONT.search(c)) and not _RX_NOT_FRONT.search(c)))
    return out


def front_since(events: list, x: int = FRONT_M) -> int | None:
    """Where the horse took the lead it still holds at `x` metres; None if it
    isn't in front there or the comment can't place it."""
    state, since = None, None
    for d, lead in events:
        if d is None:
            state = None
            continue
        if d > x:
            break
        state, since = lead, d
    return since if state else None


def front_labels(comments, dists, x: int = FRONT_M) -> list | None:
    """1 for the starter in front at `x`, 0 for the rest; None when the
    comments don't name exactly one (two claims resolve to the later
    takeover) or too few starters have one."""
    n = len(comments)
    if sum(1 for c in comments if c and c.strip()) < 0.8 * n:
        return None
    base = max((d or 0) for d in dists) or None
    since = [front_since(trip(c, d or base), x) for c, d in zip(comments, dists)]
    claims = [s for s in since if s is not None]
    if not claims:
        return None
    top = max(claims)
    if sum(1 for s in since if s == top) != 1:
        return None
    return [1 if s == top else 0 for s in since]


QUALIFIER_RE = re.compile(
    r"^(gdk|ejg|ejp|gd|gk|egk|egdk|gkd|gdj|gdb|gdek|gdgk|gdl|ddk|frdk|erj|ejk|"
    r"ejgk|ejgd|ej|EJ|EJG|EJP|GDK|Gdk)[0-9]?$|^[12]?[pP][0-9]?$")

STREAM_SQL = """
    SELECT r.race_id, r.race_date, r.track_id, r.start_method,
           e.entry_id, e.horse_id, e.driver_id, e.post, e.distance, e.tillagg,
           e.auto, COALESCE(e.galopp, false), e.placement_text, e.age, ec.comment
    FROM race r
    JOIN track t ON t.track_id = r.track_id
    JOIN entry e ON e.race_id = r.race_id
    LEFT JOIN entry_comment ec ON ec.entry_id = e.entry_id
    WHERE t.country = 'SE' AND r.race_date >= %s
      AND NOT COALESCE(e.withdrawn, false) AND e.horse_id IS NOT NULL
    ORDER BY r.race_date, r.race_id, e.entry_id
"""
(S_RID, S_DATE, S_TRACK, S_METHOD, S_EID, S_HID, S_DID, S_POST, S_DIST, S_TILL,
 S_AUTO, S_GAL, S_PTXT, S_AGE, S_COMMENT) = range(15)

FEATURES = [
    'n', 'n_front', 'post', 'front', 'till', 'distance', 'age', 'days_off',
    'exp_post', 'exp_track', 'base',
    'h_labels', 'h_lead_rate', 'h_lead_dev', 'h_lead_dev_m', 'h_early_dev',
    'h_trouble', 'h_fast', 'h_gal', 'h_last_lead',
    'd_lead_dev', 'd_labels',
    'speed', 'speed_rank', 'speed_gap', 'max_speed_inside', 'max_speed_outside',
    'n_faster_inside', 'n_faster',
    'drv_change', 'h_last_early',
    'mk_rel', 'mk_rank', 'mk_gap', 'mk_inside', 'mk_drv',
]
CATEGORICAL = ['track']
ALL_FEATURES = FEATURES + CATEGORICAL
# The first version's inputs (kept for the ablation in build_race_models).
BASE_FEATURES = FEATURES[:FEATURES.index('drv_change')] + CATEGORICAL


class _Horse:
    __slots__ = ('dev', 'devw', 'devm', 'lead', 'leadw', 'early', 'earlyw', 'trb', 'fst',
                 'cw', 'gal', 'galw', 'last_lead', 'labels', 'last', 'last_did', 'last_early')

    def __init__(self):
        self.dev = self.devw = 0.0
        self.devm = {True: [0.0, 0.0], False: [0.0, 0.0]}
        self.lead = self.leadw = 0.0
        self.early = self.earlyw = 0.0
        self.trb = self.fst = self.cw = 0.0
        self.gal = self.galw = 0.0
        self.last_lead = None
        self.labels = 0
        self.last = None
        self.last_did = None
        self.last_early = None   # early position at the last start, 0 = front .. 1 = last


def _shrunk(s, w, now_w, k):
    """Decayed mean of (s, w) pulled toward 0 by k effective observations."""
    return (s / now_w) / (w / now_w + k) if w > 0 else 0.0


def _rate(s, w):
    return s / w if w > 0 else None


def _nan(v):
    return math.nan if v is None else float(v)


class LeadPass:
    """Streams Swedish races in date order. Labelled races yield training
    rows (if `collect`); finished races without a clear leader from
    `other_from` on yield unlabelled rows (`other`); races without results
    on or after `today` yield feature rows for scoring. `ext` maps entry id
    -> (strength, driver) as priced by the xodds rating pass."""

    def __init__(self, *, today: dt.date, collect: bool = True, train_from: dt.date | None = None,
                 ext: dict | None = None, other_from: dt.date | None = None, front: bool = False):
        self.today = today
        self.collect = collect
        self.ext = ext or {}
        self.other_t = other_from.toordinal() if other_from else None
        self.other: list = []
        self.want_front = front
        self.front: list = []       # xfront training rows: who is in front at FRONT_M
        self.train_t = (train_from or dt.date(2014, 1, 1)).toordinal()
        self.P: dict = {}           # (auto, post, front) -> [leads, n]
        self.TP: dict = {}          # (track, auto, post, front) -> [leads, n]
        self.EP: dict = {}          # (auto, post, front) -> [sum early pct, n]
        self.H: dict[int, _Horse] = {}
        self.D: dict[int, list] = {}  # driver -> [dev, w, labels]
        self.train: list = []
        self.upcoming: list = []
        self.races = 0
        self.usable = 0

    def run(self, cur, since: dt.date):
        cur.execute(STREAM_SQL, (since,))
        batch = []
        rid = None
        for row in cur:
            if row[S_RID] != rid:
                if batch:
                    self._race(batch)
                batch = []
                rid = row[S_RID]
            batch.append(row)
        if batch:
            self._race(batch)

    # ── expectations ─────────────────────────────────────────────────────
    def _exp_post(self, key, n):
        s = self.P.get(key)
        prior = 1.0 / max(n, 2)
        if not s:
            return prior
        return (s[0] + 20 * prior) / (s[1] + 20)

    def _exp_track(self, tkey, ep):
        s = self.TP.get(tkey)
        if not s:
            return ep
        return (s[0] + 40 * ep) / (s[1] + 40)

    # ── one race ─────────────────────────────────────────────────────────
    def _race(self, rows):
        d = rows[0][S_DATE]
        t = d.toordinal()
        ptxt = [(r[S_PTXT] or '').strip() for r in rows]
        has_result = any(ptxt)
        upcoming = not has_result and d >= self.today
        if not has_result and not upcoming:
            return
        if has_result and sum(1 for p in ptxt if QUALIFIER_RE.match(p)) * 2 >= len(rows):
            return
        n = len(rows)
        if n < 4:
            return
        method = rows[0][S_METHOD]
        auto = method == 'A' if method in ('A', 'V') else bool(rows[0][S_AUTO])
        track = rows[0][S_TRACK]
        dists = [r[S_DIST] for r in rows]
        base_d = min((x for x in dists if x), default=None)
        till = [((x - base_d) if (x and base_d) else (r[S_TILL] or 0)) for x, r in zip(dists, rows)]
        front = [tl <= 0 for tl in till]
        n_front = sum(front)
        posts = [min(r[S_POST] or 99, 15) for r in rows]
        keys = [(auto, posts[i], front[i]) for i in range(n)]

        ep = [self._exp_post(k, n) for k in keys]
        et = [self._exp_track((track,) + k, ep[i]) for i, k in enumerate(keys)]
        tot = sum(et) or 1.0
        base = [x / tot for x in et]

        wh = math.exp((t - T0) / TAU_H)
        wd = math.exp((t - T0) / TAU_D)
        hs = []
        speed = []
        for i, r in enumerate(rows):
            h = self.H.get(r[S_HID])
            if h is None:
                h = self.H[r[S_HID]] = _Horse()
            hs.append(h)
            dv = self.D.get(r[S_DID])
            sp = _shrunk(h.dev, h.devw, wh, 3.0)
            if dv:
                sp += 0.5 * _shrunk(dv[0], dv[1], wd, 20.0)
            sp += 0.5 * _shrunk(h.early, h.earlyw, wh, 3.0)
            speed.append(sp)

        want_other = self.other_t is not None and t >= self.other_t
        if (self.collect and t >= self.train_t) or upcoming or want_other:
            feats = self._features(rows, hs, speed, auto, till, front, n_front, posts,
                                   ep, et, base, t, wh, wd)
        else:
            feats = None

        labels = None
        if has_result:
            labels = [lead_label(r[S_COMMENT]) for r in rows]
            n_comm = sum(1 for x in labels if x is not None)
            leaders = sum(1 for x in labels if x == 1)
            usable = leaders == 1 and n_comm >= 0.8 * n
            self.races += 1
            if usable:
                self.usable += 1
                ys = [1 if x == 1 else 0 for x in labels]
                if feats is not None and self.collect and t >= self.train_t:
                    for i in range(n):
                        self.train.append((rows[i][S_RID], t, auto, ys[i], rows[i][S_EID], track, feats[i]))
                self._learn_lead(rows, hs, keys, track, base, ys, auto, wh, wd)
            elif want_other:
                for i in range(n):
                    self.other.append((rows[i][S_RID], t, auto, None, rows[i][S_EID], track, feats[i]))
            if self.want_front and feats is not None and self.collect and t >= self.train_t:
                fl = front_labels([r[S_COMMENT] for r in rows], dists)
                if fl is not None:
                    for i in range(n):
                        self.front.append((rows[i][S_RID], t, auto, fl[i], rows[i][S_EID], track, feats[i]))
            self._learn_trip(rows, hs, keys, n, wh)
        else:
            for i in range(n):
                self.upcoming.append((rows[i][S_RID], t, auto, None, rows[i][S_EID], track, feats[i]))

    def _features(self, rows, hs, speed, auto, till, front, n_front, posts,
                  ep, et, base, t, wh, wd):
        """One float row per runner, in FEATURES order (NaN = unknown)."""
        n = len(rows)
        order = sorted(range(n), key=lambda i: -speed[i])
        rank = [0] * n
        for k, i in enumerate(order):
            rank[i] = k + 1
        mk = [self.ext.get(r[S_EID]) for r in rows]
        known = [x for x in mk if x is not None]
        mk_mean = sum(x[0] for x in known) / len(known) if known else 0.0
        drv_mean = sum(x[1] for x in known) / len(known) if known else 0.0
        out = []
        for i, r in enumerate(rows):
            h = hs[i]
            dv = self.D.get(r[S_DID])
            others = [speed[j] for j in range(n) if j != i and front[j] == front[i]]
            inside = [speed[j] for j in range(n) if j != i and front[j] == front[i] and posts[j] < posts[i]]
            outside = [speed[j] for j in range(n) if j != i and front[j] == front[i] and posts[j] > posts[i]]
            dm = h.devm[auto]
            if mk[i] is not None and len(known) >= 2:
                s = mk[i][0]
                mk_o = [mk[j][0] for j in range(n) if j != i and mk[j] is not None and front[j] == front[i]]
                mk_in = [mk[j][0] for j in range(n) if j != i and mk[j] is not None
                         and front[j] == front[i] and posts[j] < posts[i]]
                mkf = (s - mk_mean, sum(1 for x in known if x[0] > s) / (len(known) - 1),
                       (s - max(mk_o)) if mk_o else 0.0,
                       (max(mk_in) - s) if mk_in else None,
                       mk[i][1] - drv_mean)
            else:
                mkf = (None,) * 5
            out.append(array('d', (_nan(v) for v in (
                n, n_front, posts[i], int(front[i]), till[i], r[S_DIST], r[S_AGE],
                (t - h.last) if h.last is not None else None,
                ep[i], et[i], base[i],
                h.labels,
                _rate(h.lead, h.leadw),
                _shrunk(h.dev, h.devw, wh, 3.0) if h.devw else None,
                _shrunk(dm[0], dm[1], wh, 3.0) if dm[1] else None,
                _shrunk(h.early, h.earlyw, wh, 3.0) if h.earlyw else None,
                _rate(h.trb, h.cw),
                _rate(h.fst, h.cw),
                _rate(h.gal, h.galw),
                h.last_lead,
                _shrunk(dv[0], dv[1], wd, 20.0) if dv and dv[1] else None,
                dv[2] if dv else 0,
                speed[i], rank[i],
                speed[i] - max(others) if others else 0.0,
                max(inside) if inside else None,
                max(outside) if outside else None,
                sum(1 for x in inside if x > speed[i]),
                sum(1 for x in others if x > speed[i]),
                (1.0 if r[S_DID] != h.last_did else 0.0) if (h.last_did and r[S_DID]) else None,
                h.last_early,
                *mkf,
            ))))
        return out

    # ── learning from a finished race ────────────────────────────────────
    def _learn_lead(self, rows, hs, keys, track, base, ys, auto, wh, wd):
        for i, r in enumerate(rows):
            y = ys[i]
            k = keys[i]
            s = self.P.get(k)
            if s is None:
                s = self.P[k] = [0, 0]
            s[0] += y
            s[1] += 1
            tk = (track,) + k
            s = self.TP.get(tk)
            if s is None:
                s = self.TP[tk] = [0, 0]
            s[0] += y
            s[1] += 1
            h = hs[i]
            x = y - base[i]
            h.dev += wh * x
            h.devw += wh
            dm = h.devm[auto]
            dm[0] += wh * x
            dm[1] += wh
            h.lead += wh * y
            h.leadw += wh
            h.labels += 1
            h.last_lead = y
            did = r[S_DID]
            if did is not None:
                dv = self.D.get(did)
                if dv is None:
                    dv = self.D[did] = [0.0, 0.0, 0]
                dv[0] += wd * x
                dv[1] += wd
                dv[2] += 1

    def _learn_trip(self, rows, hs, keys, n, wh):
        t = rows[0][S_DATE].toordinal()
        for i, r in enumerate(rows):
            h = hs[i]
            c = r[S_COMMENT]
            h.last_early = None
            if c:
                text = c.strip().lower()
                if text and not text.startswith('struken'):
                    rank, _led, trouble, fast = opening(text.split(','), n)
                    h.trb += wh * trouble
                    h.fst += wh * fast
                    h.cw += wh
                    if rank is not None:
                        h.last_early = (rank - 1) / (n - 1)
                    if rank is not None and n >= 5:
                        pct = (rank - 1) / (n - 1)
                        e = self.EP.get(keys[i])
                        if e is None:
                            e = self.EP[keys[i]] = [0.0, 0]
                        if e[1] >= 30:
                            h.early += wh * (e[0] / e[1] - pct)
                            h.earlyw += wh
                        e[0] += pct
                        e[1] += 1
            h.gal += wh * (1.0 if r[S_GAL] else 0.0)
            h.galw += wh
            h.last = t
            if r[S_DID]:
                h.last_did = r[S_DID]


def frame(rows):
    """Pass rows (training or upcoming) -> DataFrame: keys, y, features."""
    import numpy as np
    import pandas as pd
    if not rows:
        return pd.DataFrame(columns=['race_id', 't', 'auto', 'y', 'entry_id'] + ALL_FEATURES)
    df = pd.DataFrame(np.array([f for *_, f in rows], dtype='float64'), columns=FEATURES)
    df.insert(0, 'race_id', [r[0] for r in rows])
    df.insert(1, 't', [r[1] for r in rows])
    df.insert(2, 'auto', [bool(r[2]) for r in rows])
    df.insert(3, 'y', [r[3] for r in rows])
    df.insert(4, 'entry_id', [r[4] for r in rows])
    df['track'] = [r[5] for r in rows]
    return df


def set_categories(df, categories: dict):
    """Pin categorical columns to the category sets seen in training."""
    import pandas as pd
    for c in CATEGORICAL:
        cats = categories.get(c)
        df[c] = pd.Categorical(df[c], categories=cats) if cats is not None else df[c].astype('category')
    return df


def normalize(df, col='p'):
    """Per-race shares: exactly one horse leads."""
    return df[col] / df.groupby('race_id')[col].transform('sum')
