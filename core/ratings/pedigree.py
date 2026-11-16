"""Pedigree sub-ratings: own index, sire, dam and damsire.

Two sources per ancestor, each standardised within the foal's own crop
(breed × birth year) because both drift strongly over time:

* the BLUP breeding index from the stud book ("72 0,75" = index, reliability);
* foal merit: each foal's best Elo through age 4, as a z-score among its
  crop. Young sires are therefore compared with foals of the same age, not
  with established sires' four-year-old earnings. Swedish-registered and
  foreign foals are scored within their own pool, since we hold only part
  of the foreign racing, and only a Swedish-registered foal of five or older
  that never started counts as a low merit: a foreign one may simply race
  where we don't look.

The horse itself is left out of its own sire, dam and damsire indexes.
"""

from __future__ import annotations

import datetime as dt
import math
import re
from collections import defaultdict

_BLUP_RE = re.compile(r'^\s*(-?\d+(?:[.,]\d+)?)\s+(\d(?:[.,]\d+)?)')
UNRACED_Z = -1.0
K_SIRE = 8.0
K_DAM = 3.0
K_DAMSIRE = 10.0
FIRST_CROP = 2007


def _blup(text):
    if not text:
        return None
    m = _BLUP_RE.match(text)
    if not m:
        return None
    v = float(m.group(1).replace(',', '.'))
    rel = float(m.group(2).replace(',', '.'))
    if rel > 1:
        rel = rel / 100.0
    return v, max(0.0, min(1.0, rel))


def _ms(vals):
    n = len(vals)
    if n < 5:
        return None
    mu = sum(vals) / n
    sd = math.sqrt(sum((x - mu) ** 2 for x in vals) / (n - 1))
    return mu, (sd if sd > 1e-6 else 1.0)


def _clip(z):
    return max(-4.0, min(4.0, z))


def _combine(parts):
    """parts: [(z, c)] -> confidence-weighted z and the combined confidence."""
    parts = [(z, c) for z, c in parts if z is not None and c > 0]
    if not parts:
        return None, 0.0
    den = sum(c for _, c in parts)
    z = sum(z * c for z, c in parts) / den
    miss = 1.0
    for _, c in parts:
        miss *= (1 - c)
    return z, 1 - miss


def build(cur, horses_state, *, today: dt.date, log=print):
    """horses_state: horse_id -> object with .yp (best Elo through age 4),
    .yn (starts through age 4). Returns horse_id -> (v[4], c[4], info)."""
    cur.execute("""
        SELECT horse_id, breed_code, EXTRACT(YEAR FROM date_of_birth)::int,
               sire_id, dam_id, breed_index, st_id IS NOT NULL
        FROM horse
        WHERE date_of_birth IS NOT NULL AND date_of_birth >= '1975-01-01'
    """)
    H = {}
    swedish = set()
    for hid, breed, by, sire, dam, bi, se in cur.fetchall():
        if breed not in ('V', 'K'):
            st = horses_state.get(hid)
            breed = st.breed if st is not None else 'V'
        H[hid] = (breed, by, sire, dam, _blup(bi))
        if se:
            swedish.add(hid)
    log(f"[pedigree] {len(H):,} horses with a birth date, {len(swedish):,} Swedish-registered")

    # foal merit, z within crop and pool
    crop_yp = defaultdict(list)
    for hid, (b, by, *_rest) in H.items():
        st = horses_state.get(hid)
        if st is not None and st.yn >= 3 and st.yp is not None and by >= FIRST_CROP:
            crop_yp[(b, by, hid in swedish)].append(st.yp)
    crop_ms = {k: _ms(v) for k, v in crop_yp.items()}
    merit = {}
    for hid, (b, by, *_rest) in H.items():
        if by is None or by < FIRST_CROP:
            continue
        st = horses_state.get(hid)
        ms = crop_ms.get((b, by, hid in swedish))
        if st is not None and st.yn >= 3 and st.yp is not None and ms:
            merit[hid] = _clip((st.yp - ms[0]) / ms[1])
        elif hid in swedish and (st is None or st.yn == 0) and today.year - by >= 5:
            merit[hid] = UNRACED_Z

    foals_of = defaultdict(list)
    dam_foals = defaultdict(list)
    for hid, (b, by, sire, dam, _bl) in H.items():
        z = merit.get(hid)
        if z is None:
            continue
        if sire:
            foals_of[sire].append((hid, z))
        if dam:
            dam_foals[dam].append((hid, z))
    sire_sum = {s: (sum(z for _, z in fs), len(fs)) for s, fs in foals_of.items()}

    def sire_index(s, exclude=None):
        tot = sire_sum.get(s)
        if not tot:
            return None, 0, 0.0
        sm, n = tot
        if exclude is not None and exclude in merit and any(h == exclude for h, _ in foals_of[s]):
            sm -= merit[exclude]
            n -= 1
        if n <= 0:
            return None, 0, 0.0
        return sm / (n + K_SIRE), n, n / (n + K_SIRE)

    # broodmare-sire index: grand-foals through daughters
    bms = defaultdict(list)
    for hid, (b, by, sire, dam, _bl) in H.items():
        z = merit.get(hid)
        if z is None or not dam:
            continue
        d = H.get(dam)
        if d and d[2]:
            bms[d[2]].append((hid, z))

    # BLUP z of each ancestor, standardised among the foals of the same crop
    crop_own = defaultdict(list)
    crop_sire = defaultdict(list)
    crop_dam = defaultdict(list)
    crop_dsire = defaultdict(list)
    for hid, (b, by, sire, dam, bl) in H.items():
        if by is None:
            continue
        key = (b, by)
        if bl:
            crop_own[key].append(bl[0])
        s = H.get(sire) if sire else None
        if s and s[4]:
            crop_sire[key].append(s[4][0])
        d = H.get(dam) if dam else None
        if d and d[4]:
            crop_dam[key].append(d[4][0])
        if d and d[2]:
            ds = H.get(d[2])
            if ds and ds[4]:
                crop_dsire[key].append(ds[4][0])
    ms_own = {k: _ms(v) for k, v in crop_own.items()}
    ms_sire = {k: _ms(v) for k, v in crop_sire.items()}
    ms_dam = {k: _ms(v) for k, v in crop_dam.items()}
    ms_dsire = {k: _ms(v) for k, v in crop_dsire.items()}

    def bz(bl, ms):
        if not bl or not ms:
            return None, 0.0
        return _clip((bl[0] - ms[0]) / ms[1]), bl[1]

    out = {}
    min_by = today.year - 20
    for hid, (b, by, sire, dam, bl) in H.items():
        if by is None or by < min_by:
            continue
        key = (b, by)
        info = {}
        own = bz(bl, ms_own.get(key))
        if bl:
            info['own'] = {'blup': bl[0], 'rel': bl[1]}

        s = H.get(sire) if sire else None
        s_bz = bz(s[4] if s else None, ms_sire.get(key))
        s_idx, s_n, s_c = sire_index(sire, hid) if sire else (None, 0, 0.0)
        sire_v = _combine([s_bz, (s_idx, s_c)])
        if s:
            info['sire'] = {'blup': s[4][0] if s[4] else None, 'foals': s_n,
                            'idx': round(s_idx, 2) if s_idx is not None else None}

        d = H.get(dam) if dam else None
        d_bz = bz(d[4] if d else None, ms_dam.get(key))
        prod = []
        for fh, fz in dam_foals.get(dam, ()) if dam else ():
            if fh == hid:
                continue
            fs = H[fh][2]
            si = sire_index(fs, fh)[0] if fs else None
            prod.append(fz - 0.5 * (si or 0.0))
        d_prod = (sum(prod) / (len(prod) + K_DAM), len(prod) / (len(prod) + K_DAM)) if prod else (None, 0.0)
        d_own = (merit[dam], 0.4) if dam in merit and merit[dam] != UNRACED_Z else (None, 0.0)
        dam_v = _combine([d_bz, d_prod, d_own])
        if d:
            info['dam'] = {'blup': d[4][0] if d[4] else None, 'foals': len(prod)}

        ds_id = d[2] if d else None
        ds = H.get(ds_id) if ds_id else None
        ds_bz = bz(ds[4] if ds else None, ms_dsire.get(key))
        gf = [z for fh, z in bms.get(ds_id, ()) if fh != hid] if ds_id else []
        ds_idx = (sum(gf) / (len(gf) + K_DAMSIRE), len(gf) / (len(gf) + K_DAMSIRE)) if gf else (None, 0.0)
        dsire_v = _combine([ds_bz, ds_idx])
        if ds:
            info['damsire'] = {'blup': ds[4][0] if ds[4] else None, 'grandfoals': len(gf)}

        parts = [own, sire_v, dam_v, dsire_v]
        if all(z is None for z, _ in parts):
            continue
        vs = []
        cs = []
        for z, c in parts:
            if z is None or c <= 0:
                vs.append(0)
                cs.append(0)
            else:
                vs.append(max(1, min(99, int(round(65 + 10 * z)))))
                cs.append(max(1, int(round(100 * c))))
        out[hid] = (vs, cs, info)
    log(f"[pedigree] {len(out):,} horses rated; {len(foals_of):,} sires, {len(bms):,} damsires indexed")
    return out
