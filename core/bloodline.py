"""Where a horse's genes come from, by the country its ancestors were bred in.

A flag says little about blood: Viking Kronos was bred in Italy from two
American parents, and the Swedish share in Maharajah is gone within a few
generations. So no flag closer in than the horizon counts. Every line is
traced HORIZON generations back, and each ancestor there carries
1 / 2**HORIZON of the genes, credited to the country it was bred in.

Six generations is where the shares stop moving: Swedish- and Italian-bred
lines have dissolved into the American and French stock they were built
from, while a seventh mostly adds holes as old pedigrees thin out. A line
whose pedigree ends before the horizon is unknown — the last horse on record
is no proxy for its ancestry.
"""

from __future__ import annotations

HORIZON = 6

# Canadian-bred standardbreds are the same North American stock.
GROUPS = {
    'FR': 'french',
    'US': 'american',
    'CA': 'american',
    'IT': 'italian',
    'SE': 'swedish',
    'NO': 'norwegian',
}

# 'XU' is TravSport's code for an unrecorded country.
_ANCESTORS_SQL = """
WITH RECURSIVE anc(horse_id, depth) AS (
    SELECT %(horse_id)s, 0
    UNION
    SELECT p.parent_id, a.depth + 1
    FROM anc a
    JOIN horse h ON h.horse_id = a.horse_id
    CROSS JOIN LATERAL (VALUES (h.sire_id), (h.dam_id)) p(parent_id)
    WHERE a.depth < %(depth)s AND p.parent_id IS NOT NULL
)
SELECT h.horse_id, h.sire_id, h.dam_id,
       COALESCE(NULLIF(NULLIF(h.bred_country, ''), 'XU'),
                NULLIF(NULLIF(h.birth_country, ''), 'XU')) AS country
FROM horse h
WHERE h.horse_id IN (SELECT horse_id FROM anc)
"""


def origin(cur, horse_id: int, horizon: int = HORIZON) -> dict | None:
    """Gene shares by origin group, or None when the horse is not on record.

    `shares` sums to 1 over the groups plus 'other' and 'unknown';
    `other` breaks the 'other' share down by country.
    """
    cur.execute(_ANCESTORS_SQL, {'horse_id': horse_id, 'depth': horizon})
    horses = {hid: (sire, dam, country) for hid, sire, dam, country in cur.fetchall()}
    if horse_id not in horses:
        return None

    by_country: dict[str, float] = {}
    unknown = 0.0

    def walk(hid: int, gen: int, share: float) -> None:
        nonlocal unknown
        sire, dam, country = horses[hid]
        if gen == horizon:
            if country:
                by_country[country] = by_country.get(country, 0.0) + share
            else:
                unknown += share
            return
        for parent in (sire, dam):
            if parent in horses:
                walk(parent, gen + 1, share / 2)
            else:
                unknown += share / 2

    walk(horse_id, 0, 1.0)

    shares = {group: 0.0 for group in dict.fromkeys(GROUPS.values())}
    shares['other'] = 0.0
    other: dict[str, float] = {}
    for country, share in by_country.items():
        group = GROUPS.get(country)
        if group:
            shares[group] += share
        else:
            shares['other'] += share
            other[country] = share
    shares['unknown'] = unknown
    return {
        'horizon': horizon,
        'shares': {k: round(v, 4) for k, v in shares.items()},
        'other': {k: round(v, 4) for k, v in sorted(other.items(), key=lambda kv: -kv[1])},
    }


# Inbreeding is Wright's coefficient over the whole known pedigree. That is
# what TravSport stores as inbreed_coefficient — recomputed here it matches
# theirs to the hundredth wherever our pedigree is complete, and Breedly's
# runs within half a point of both. Missing ancestors only ever remove terms,
# so every figure is a lower bound and the larger of TravSport's and ours is
# the better one.
_PEDIGREE_SQL = """
WITH RECURSIVE anc(horse_id) AS (
    SELECT %(horse_id)s
    UNION
    SELECT p.parent_id
    FROM anc a
    JOIN horse h ON h.horse_id = a.horse_id
    CROSS JOIN LATERAL (VALUES (h.sire_id), (h.dam_id)) p(parent_id)
    WHERE p.parent_id IS NOT NULL
)
SELECT h.horse_id, h.sire_id, h.dam_id, h.inbreed_coefficient, h.breed_code
FROM horse h
WHERE h.horse_id IN (SELECT horse_id FROM anc)
"""

# Below 95% of ancestors known through ten generations our figure misses
# whole founder lines and reads several points low.
COMPLETE_GENS = 10
COMPLETE_SHARE = 0.95

_TYPICAL_SQL = """
SELECT percentile_cont(ARRAY[0.1, 0.5, 0.9]) WITHIN GROUP (ORDER BY v)
FROM (SELECT replace(inbreed_coefficient, ',', '.')::float AS v
      FROM horse
      WHERE breed_code = %(breed)s AND date_of_birth >= %(born)s
        AND inbreed_coefficient ~ '^[0-9]+,[0-9]+$' AND inbreed_coefficient <> '0,00') t
"""
_typical: dict[str, list[float] | None] = {}


def _parse_pct(raw: str | None) -> float | None:
    try:
        return float(raw.replace(',', '.')) if raw else None
    except ValueError:
        return None


def _ranks(ped: dict[int, list]) -> dict[int, int]:
    """Generations above the oldest founder, dropping any edge that closes a loop."""
    rank: dict[int, int] = {}
    for root in ped:
        if root in rank:
            continue
        stack, on_path = [(root, 0)], {root}
        while stack:
            hid, i = stack[-1]
            parents = ped[hid]
            if i < 2:
                stack[-1] = (hid, i + 1)
                p = parents[i]
                if p is None or p in rank:
                    continue
                if p in on_path:
                    parents[i] = None
                    continue
                on_path.add(p)
                stack.append((p, 0))
                continue
            stack.pop()
            on_path.discard(hid)
            rank[hid] = 1 + max((rank[p] for p in parents if p is not None), default=-1)
    return rank


def _wright(ped: dict[int, list], horse_id: int) -> float:
    rank = _ranks(ped)
    kin_memo: dict[tuple[int, int], float] = {}
    f_memo: dict[int, float] = {}

    def f(a: int) -> float:
        if a not in f_memo:
            s, d = ped[a]
            f_memo[a] = kin(s, d) if s is not None and d is not None else 0.0
        return f_memo[a]

    def kin(a: int | None, b: int | None) -> float:
        if a is None or b is None:
            return 0.0
        if a == b:
            return 0.5 * (1 + f(a))
        key = (a, b) if a < b else (b, a)
        if key not in kin_memo:
            if rank[a] < rank[b]:
                a, b = b, a
            s, d = ped[a]
            kin_memo[key] = 0.5 * (kin(s, b) + kin(d, b))
        return kin_memo[key]

    return f(horse_id)


def _known_share(ped: dict[int, list], horse_id: int, gens: int) -> float:
    frontier, total = [horse_id], 0.0
    for g in range(1, gens + 1):
        frontier = [p for hid in frontier for p in ped[hid] if p is not None]
        total += len(frontier) / 2 ** g
    return total / gens


def typical(cur, breed: str | None) -> list[float] | None:
    """10th, 50th and 90th percentile of TravSport's figure for modern horses of the breed."""
    breed = breed or 'V'
    if breed not in _typical:
        cur.execute(_TYPICAL_SQL, {'breed': breed, 'born': '2012-01-01'})
        row = cur.fetchone()
        _typical[breed] = [round(v, 2) for v in row[0]] if row and row[0] else None
    return _typical[breed]


def inbreeding(cur, horse_id: int) -> dict | None:
    """Coefficient of inbreeding in percent and the breed's spread."""
    cur.execute(_PEDIGREE_SQL, {'horse_id': horse_id})
    rows = cur.fetchall()
    ped = {hid: [sire, dam] for hid, sire, dam, *_ in rows}
    if horse_id not in ped:
        return None
    for parents in ped.values():
        parents[:] = [p if p in ped else None for p in parents]
    breeds = {hid: breed for hid, *_, breed in rows if breed}
    sire, dam = ped[horse_id]
    # Norwegian records carry no breed code; the parents' tells trotter from coldblood.
    breed = breeds.get(horse_id) or breeds.get(sire) or breeds.get(dam)
    travsport = _parse_pct(next(r[3] for r in rows if r[0] == horse_id))
    if sire is None or dam is None:
        return {'coi': travsport, 'exact': travsport is not None, 'source': 'travsport' if travsport else None,
                'typical': typical(cur, breed), 'breed': breed}

    computed = round(_wright(ped, horse_id) * 100, 2)
    complete = _known_share(ped, horse_id, COMPLETE_GENS) >= COMPLETE_SHARE
    # A zero from TravSport is what a pedigree that runs dry yields as well,
    # so it is no proof of an outcross.
    use_ts = travsport is not None and travsport > 0 and travsport >= computed
    return {
        'coi': travsport if use_ts else computed,
        'exact': use_ts or complete,
        'source': 'travsport' if use_ts else 'pedigree',
        'typical': typical(cur, breed),
        'breed': breed,
    }
