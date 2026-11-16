"""Roll sub-ratings up into the ten mains and the overall.

Missing is not zero: a sub-rating with no evidence drops out of its main's
average instead of counting as a low value, and a main with nothing behind
it drops out of the overall the same way. Each value is weighted by its
confidence, so thin evidence counts for little rather than pulling the
average toward the middle a second time (the snapshot already shrank it).
"""

from __future__ import annotations

import math

from core.ratings import keys as K

SUBS_OF = {m.key: [s for s in K.SUBS if s.main == m.key] for m in K.MAINS}

# Freshness: mean Elo residual per days-since-last-start bucket, relative to
# all starts, is about ±0.03; this maps that onto the display scale.
FRESH_SCALE = 0.03
PAIR_SCALE = 0.12


def fresh_value(table, bucket: int):
    if not table or bucket >= len(table):
        return None, 0.0
    tot = sum(s for s, n in table)
    cnt = sum(n for s, n in table)
    s, n = table[bucket]
    if not n or not cnt:
        return None, 0.0
    z = (s / n - tot / cnt) / FRESH_SCALE
    return _disp(z), 0.8


def pair_value(diff: float, n: int):
    if not n:
        return None, 0.0
    c = n / (n + K.SUB_BY_KEY['pairing'].k)
    return _disp(c * max(-4.0, min(4.0, diff / PAIR_SCALE))), c


def _disp(z):
    v = int(round(K.SCALE_MID + K.SCALE_SD * z))
    return max(K.SCALE_LO, min(K.SCALE_HI, v))


def rollup(values: dict, weights: dict, starts: int):
    """values: sub key -> (v, c) with v on the display scale (or None) and c
    in [0, 1]. Returns (mains, overall): mains maps key -> (v, c) and the
    overall is (v, c), both with v None when there is nothing to go on."""
    mains = {}
    for m in K.MAINS:
        num = den = tot = 0.0
        for s in SUBS_OF[m.key]:
            tot += s.weight
            v, c = values.get(s.key, (None, 0.0))
            if v is None or c <= 0:
                continue
            num += s.weight * c * v
            den += s.weight * c
        mains[m.key] = (num / den, den / tot) if den > 0 else (None, 0.0)
    num = den = tot = 0.0
    for key, w in weights.items():
        if key == 'pedigree':
            w *= K.PEDIGREE_K / (starts + K.PEDIGREE_K)
        tot += w
        v, c = mains.get(key, (None, 0.0))
        if v is None or c <= 0:
            continue
        num += w * c * v
        den += w * c
    overall = (num / den, den / tot) if den > 0 else (None, 0.0)
    return mains, overall


def display_overall(raw: float | None, conf: float, norm: dict | None):
    """Overall on the per-breed 65 + 10·z scale, with a ± that grows as the
    evidence behind it thins."""
    if raw is None:
        return None, None
    if norm:
        z = (raw - norm['mu']) / norm['sd']
    else:
        z = (raw - K.SCALE_MID) / K.SCALE_SD
    return _disp(z), int(round(K.SCALE_SD * math.sqrt(max(0.0, 1.0 - conf))))


def horse_values(hv, hc, raw=None):
    """Snapshot arrays -> {key: (v, c)}; 0 in hv means no value."""
    out = {}
    for i, k in enumerate(K.HORSE_KEYS):
        v = hv[i] if hv and i < len(hv) else 0
        c = (hc[i] if hc and i < len(hc) else 0) / 100.0
        out[k] = (v if v else None, c if v else 0.0)
    return out
