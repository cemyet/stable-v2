"""Turn a weighted / reduced coupon into something ATG will accept.

The game page lets the user give every pick a *weight* (its share of the
coupon's rows) and optionally a *cap* below that weight, which throws the
difference away and makes the coupon cheaper.  What comes out is a per-leg,
per-horse coverage in percent.  ATG cannot be handed that directly: its own
reduction tool only speaks in cross-leg conditions (ABC letters, points,
anchor horses, payout intervals), none of which can express arbitrary
per-horse shares.

What ATG *does* accept is a file — "Filinlämning" on
https://www.atg.se/spel/reducerat — containing plain mathematical coupons.
So instead of approximating the user's system with ATG's conditions, we
realise it exactly as a set of sub-coupons and write the file.

    coverage %  ->  stake units per horse  ->  rows  ->  boxes  ->  XML

STAKE UNITS, NOT DISTINCT ROWS
------------------------------
A weighted system cannot always be built from distinct rows.  Two legs with
two picks each is four rows; asking for a 75/25 split in leg one means the
favourite has to sit on three of them, but it can only pair with two
different opponents.  The missing row is bought a second time instead.  That
is what weighting *is* — the same total stake, piled unevenly — and ATG
supports it natively through a coupon's `betmultiplier`.

So the unit of account here is a *stake unit* (one row at the game's line
price), and a row may carry several.  With that, the allocation is always
feasible: any non-negative integer matrix with the right margins will do.

Weights and distinct rows really are exclusive, by the way, not just
awkward to combine: the rows of a system that never repeats one form a
downward-closed set, and on a three-leg grid only four of the twenty-seven
plausible weightings can be the margins of such a set.  Refusing to stack
would mean quietly rewriting the user's weights instead.

WHICH rows get the units
------------------------
The margins fix how much each horse gets but not which horses share a row.
The weights do say it, though, read as a product: a row is worth its
horses' shares multiplied together, so a row of two favourites is meant to
carry more than a favourite with a longshot, in proportion.  Aimed levers
reshape that product wherever they fire (Stage 2).  Either way every row
has an intended stake, a fraction of a unit or many units, and the coupon
is that stake rounded to whole units once, for the whole coupon.

The rounding keeps the stake *monotone*: swap a horse on a played row for
one the plan rates higher and you land on a row carrying at least as much.
Without that, a coupon can pay out on the longshot and not on the favourite
standing next to it.  The price is that a horse's margin can land a few
units off its exact share — `verify()` holds it to half a percentage point
of the coupon.

An earlier dealer laid one unit on as many rows as it could before
stacking any.  That kept the margins exact but bent the shape: the stack
all went on the corner of the grid, the rows of favourites only, at a
hundred times the depth of their neighbours.

Nothing here talks to Flask or the database; `web/app.py` wraps it.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field
from datetime import date as _date, datetime as _datetime
from typing import Iterable, Iterator, Sequence

# ---------------------------------------------------------------------------
# ATG file betting
# ---------------------------------------------------------------------------
# The newest schema ATG publishes is 1.8.4:
#     https://www.atg.se/services/schemas/filebet/1.8.4/atg_filebetting.xsd
# The validator behind the upload is ahead of it — 1.8.6 adds `v85Coupon`
# (V85 replaced V75 in October 2025) and a coupon-level `trackcode`, so that
# several V85 rounds can run on the same day.  1.8.6 is not published as an
# XSD; the element/attribute set below is taken from the generated bindings
# in the open-source HPT client, which submits V85 files successfully:
#     https://github.com/Hospodaren/HPTClient  (atg_filebetting_1_8_6.cs)
# The `schemaversion` attribute is a fixed string and stayed at "ver 1.8".
ATG_SCHEMA_VERSION = "ATG File Betting XSD ver 1.8"
ATG_SCHEMA_URL = (
    "https://www.atg.se/services/schemas/filebet/1.8.4/atg_filebetting.xsd")
ATG_UPLOAD_URL = "https://www.atg.se/spel/reducerat"

# Coupon element per game type: element name, how many <leg> children it must
# carry, and whether it takes a coupon-level `trackcode`.
#
# Only the leg-based V-pools are here.  V3, DD and LD are also reducible on
# ATG, but the schema gives them a flat shape — `marks1`/`marks2` attributes
# instead of <leg> elements, and DD/LD price by `stake` rather than
# `betmultiplier` — so they cannot share this writer.  They are not in the
# game page's pricing table either, so nothing can reach the file route with
# one.
#
# The 7-leg games and V64/V65 predate multi-track days and leave `trackcode`
# out; V85 reintroduced it (1.8.6) because several V85 rounds can share a day.
ATG_COUPON_ELEMENT = {
    'V86':  ('v86Coupon',  8, True),
    'V85':  ('v85Coupon',  8, True),
    'V75':  ('v75Coupon',  7, False),
    'GS75': ('gs75Coupon', 7, False),
    'V65':  ('v65Coupon',  6, False),
    'V64':  ('v64Coupon',  6, False),
    'V5':   ('v5Coupon',   5, True),
    'V4':   ('v4Coupon',   4, True),
}

# `marks` is a fixed-width string of "0"/"1" by starting position.  Most legs
# are 15 wide (legType); V4 uses the 20-wide legType20.
ATG_MARKS_WIDTH = {'V4': 20}
ATG_MARKS_DEFAULT = 15

# betmultiplier is capped at 100, and the schema documents the values ATG
# actually accepts per coupon type.  A row that wants seven units goes in as
# a 5 and a 2.  The six-leg games and V4 take 3 and 4 but not 20.
ATG_MULTIPLIERS = (100, 50, 20, 10, 5, 2, 1)
ATG_MULTIPLIERS_BY_GAME = {
    'V64': (100, 50, 10, 5, 4, 3, 2, 1),
    'V65': (100, 50, 10, 5, 4, 3, 2, 1),
    'V4':  (100, 50, 10, 5, 4, 3, 2, 1),
}

# Top-pool play.  A coupon carrying the attribute competes for the top tier
# alone, and its stake counts `factor` times there: the inverse of that
# tier's share of the pot, so V64's 40% six-right pool pays 2.5x and V65's
# 50% pays 2x.  Fewer right pays nothing at all.  The attribute sits on each
# coupon element, so a file can play some of its coupons this way and not
# others.  V75's v7 and V86's v8 have been fixed to false since 1.8.1, and
# V85 has no such attribute.
ATG_TOP_POOL = {
    'V64':  ('v6', 6, 2.5),
    'V65':  ('v6', 6, 2.0),
    'GS75': ('v7', 7, 2.5),
}


def multipliers_for(game_type: str) -> tuple[int, ...]:
    return ATG_MULTIPLIERS_BY_GAME.get(str(game_type or '').upper(),
                                       ATG_MULTIPLIERS)


def top_pool(game_type: str) -> dict | None:
    """The game's top-pool option — `{'label', 'tier', 'factor'}` — if any."""
    spec = ATG_TOP_POOL.get(str(game_type or '').upper())
    if not spec:
        return None
    attr, tier, factor = spec
    return {'label': attr.upper(), 'tier': tier, 'factor': factor}

# `couponid` is a 1..9999 sequence number, and ATG rejects files with more
# systems than the cap below.  It also treats a file made of one-row coupons
# as robot play, so merging rows into boxes is not optional.
ATG_MAX_SYSTEMS = 5000
ATG_MAX_COUPON_ID = 9999

# Guard rails for the allocation. A weighted system has to be walked row by
# row, which is a few seconds at the row cap and rises with it — and a coupon
# that big is unsubmittable and unaffordable anyway (500k rows is 125,000 kr
# at the cheapest line price, against ATG's 5000-system file limit).
_MAX_UNITS = 2_000_000
_MAX_ROWS = 500_000


# ---------------------------------------------------------------------------
# Input
# ---------------------------------------------------------------------------
@dataclass
class Leg:
    """One leg of the plan, as the game page drew it.

    `coverage` is percent of the leg's unreduced whole, so it sums to 100
    when nothing is moved, to less when a pick is reduced, and to more when
    one is levered.  `reserves` are program numbers of horses *not* on the
    coupon, best first.
    """
    leg: int
    race_id: str
    picks: list[int]
    coverage: dict[int, float]
    reserves: list[int] = field(default_factory=list)
    starters: int = 0
    # The race's own strongest-first order. Ties in coverage are broken by
    # it, and a cascade tapers its outsiders down it.
    order: list[int] | None = None

    @property
    def retention(self) -> float:
        """Rows this leg buys against its untouched size.

        Above 1 is legal: the extra stake lands as depth on rows the
        levered pick already sits on, which is what a stake unit is for.
        """
        total = sum(self.coverage.get(n, 0.0) for n in self.picks)
        if total <= 0:
            return 1.0
        return max(0.0, total / 100.0)

    def shares(self) -> dict[int, float]:
        """Each pick's slice of the leg, as fractions adding up to 1.

        `coverage` is a share of the leg's *unreduced* whole, so it adds up
        to less than 100 once a pick is reduced and to more once one is
        levered.  Either way the stake the leg ends up buying splits
        between the picks in these proportions.
        """
        total = sum(max(0.0, self.coverage.get(n, 0.0)) for n in self.picks)
        if total <= 0:
            return {n: 1.0 / len(self.picks) for n in self.picks}
        return {n: max(0.0, self.coverage.get(n, 0.0)) / total
                for n in self.picks}

    def ranked(self) -> list[int]:
        """Picks strongest first — the order everything else works in."""
        if self.order:
            seat = {n: i for i, n in enumerate(self.order)}
            return sorted(self.picks, key=lambda n: (seat.get(n, len(seat)), n))
        return sorted(self.picks, key=lambda n: (-self.coverage.get(n, 0.0), n))


@dataclass
class Lever:
    """Extra stake anchored on one horse, and the company it keeps.

    Levering a pick past its weight buys a second, smaller coupon that has
    that horse alone in its leg.  `amount` is how far past the weight it
    went, in the same percent-of-the-leg units as `coverage`.

    `targets` says how that sub-coupon should spread over the *other* legs —
    `{leg number: {horse: share}}`, shares unnormalised.  A leg left out
    means "the same shape as the rest of the coupon", which is what the
    plain proportional allocation did before any of this existed.  Aiming a
    lever moves stake between rows without changing how many are bought, so
    targeting never costs anything.
    """
    leg: int
    horse: int
    amount: float
    targets: dict[int, dict[int, float]] = field(default_factory=dict)
    # Cascade. The race opens to the outsiders the target names: half the
    # component stays on the picks and the rest tapers down the outsiders
    # by rank. One race per lever.
    cascade_targets: frozenset[int] = field(default_factory=frozenset)
    # Aimed at the pool. Every reading this lever fires in is played for the
    # top tier alone, on coupons of their own; the rest of the system pays
    # on every tier as usual.
    pool: bool = False


@dataclass
class Plan:
    game_type: str
    legs: list[Leg]
    line_price: float = 0.0
    game_date: str = ''
    track_code: int | None = None
    track_name: str = ''
    levers: list[Lever] = field(default_factory=list)
    # The whole system played for the top tier alone (V6 / V7).
    top_only: bool = False
    # The whole system bought at a multiple of what its coverage prices:
    # 0.85 takes 15% off every horse's stake, 3 triples it. It is the same
    # plan as cutting or levering a banker by that much, and is rounded the
    # same way: the whole coupon at once, so a lever reaches rows that were
    # too thin to play at 100% as well as deepening the ones that were.
    scale: float = 1.0

    @property
    def full_rows(self) -> int:
        rows = 1
        for leg in self.legs:
            rows *= max(1, len(leg.picks))
        return rows

    @property
    def retention(self) -> float:
        r = self.scale if self.scale > 0 else 1.0
        for leg in self.legs:
            r *= leg.retention
        return r

    @property
    def rows(self) -> int:
        value = self.full_rows * self.retention
        nearest_half = round(value * 2) / 2
        # Saved coverage is rounded to four decimals; rebuilding an exact
        # half can therefore yield 22.50001. Snap only that serialization
        # noise away before Python's even-half row rounding.
        if abs(value - nearest_half) < 0.0001:
            value = nearest_half
        return max(1, round(value))

    def effective_coverage(self) -> dict[int, dict[int, float]]:
        """Per-leg coverage once aimed levers have moved the stake around.

        A leg keeps its total — aiming is free — but a leg some lever is
        pointed at will show that target ahead of where the user's own
        weight left it.  That drift is the whole point of aiming, so this,
        not the nominal `coverage`, is what the UI should draw and what
        `verify` should hold the built system to.
        """
        out = {leg.leg: {n: max(0.0, leg.coverage.get(n, 0.0)) for n in leg.picks}
               for leg in self.legs}
        if not _live_levers(self):
            return out

        stake = intended_stake(self)
        for leg, margin in zip(self.legs, _margins(stake, len(self.legs))):
            total = _leg_total(leg)
            spread = sum(margin.values())
            if total <= 0 or spread <= 0:
                continue
            names = list(leg.picks) + sorted(
                (n for n in margin if n not in leg.picks),
                key=lambda n: -margin[n])
            out[leg.leg] = {n: margin.get(n, 0.0) / spread * total
                            for n in names}
        return out


class ReduceError(ValueError):
    """The plan cannot be turned into a submittable system."""


# ---------------------------------------------------------------------------
# Stage 1 — coverage percentages to whole stake units
# ---------------------------------------------------------------------------
def integer_targets(leg: Leg, units: int) -> dict[int, int]:
    """Split `units` stake units across a leg's picks by coverage.

    Largest remainder, so the parts add up to `units` exactly.  A pick with
    any coverage at all is guaranteed at least one unit — a horse on the
    coupon that ends up on no row is a rounding artefact, not a choice, so
    we take that unit off the largest holder instead.
    """
    picks = leg.ranked()
    if not picks:
        return {}
    if units < len(picks):
        raise ReduceError(
            f'leg {leg.leg} has {len(picks)} picks but the system is only '
            f'{units} rows — every pick needs at least one row')

    total = sum(max(0.0, leg.coverage.get(n, 0.0)) for n in picks)
    if total <= 0:
        raise ReduceError(f'leg {leg.leg} has no coverage')

    raw = {n: max(0.0, leg.coverage.get(n, 0.0)) * units / total for n in picks}
    out = {n: int(math.floor(v)) for n, v in raw.items()}

    # Ties on the remainder go to the stronger pick rather than the lower
    # program number: the leftover unit is a row, and a row is worth more on
    # the horse the user rated higher.
    seat = {n: i for i, n in enumerate(picks)}
    short = units - sum(out.values())
    for n in sorted(picks, key=lambda n: (-(raw[n] - math.floor(raw[n])), seat[n])):
        if short <= 0:
            break
        out[n] += 1
        short -= 1
    # Ties can leave a unit or two over; hand them to the strongest picks.
    i = 0
    while short > 0:
        out[picks[i % len(picks)]] += 1
        short -= 1
        i += 1

    # Nobody is allowed to sit on zero rows.
    for n in picks:
        if out[n] > 0:
            continue
        donor = max(picks, key=lambda m: (out[m], -m))
        if out[donor] <= 1:
            raise ReduceError(
                f'leg {leg.leg} cannot give every pick a row at '
                f'{units} rows')
        out[donor] -= 1
        out[n] = 1
    return out


# ---------------------------------------------------------------------------
# Boxes
# ---------------------------------------------------------------------------
@dataclass
class Box:
    """A mathematical coupon: one set of horses per leg, played `multiplier`
    times.  Covers `prod(len(sets))` rows.  `top_only` plays it for the top
    tier alone."""
    sets: tuple[tuple[int, ...], ...]
    multiplier: int = 1
    top_only: bool = False

    @property
    def rows(self) -> int:
        n = 1
        for s in self.sets:
            n *= len(s)
        return n

    @property
    def units(self) -> int:
        return self.rows * self.multiplier


# ---------------------------------------------------------------------------
# Stage 2 — the stake each row is meant to carry
# ---------------------------------------------------------------------------
# Coverage alone says how much of a leg a horse gets, never who it gets it
# with.  An aimed lever says the second thing: "when 12 wins leg three, put
# my stake in leg four on 3".
#
# Coverage reads apart into a base, where nothing is levered, plus one
# component per lever, and multiplied out across the legs those add back up
# to exactly the coverage drawn.  Aiming reshapes the legs a lever points at
# whenever its component is the one in play.  A row's intended stake is a sum
# over the ways its own horses can be read — each levered horse as its base
# share or as its lever — of the product of the coverages.  Only levers on
# the row's own horses take part, so however many levers the coupon carries,
# a row is never more than 2^legs readings, and an unaimed coupon is just the
# product of its shares.

# A cascade family worth less than this many units in total is not walked:
# every row in it would round to nothing.
_FAMILY_FLOOR = 0.01


def _live_levers(plan: Plan) -> list[Lever]:
    seats = {leg.leg: leg for leg in plan.legs}
    return [lever for lever in plan.levers
            if lever.amount > 0 and lever.leg in seats
            and lever.horse in seats[lever.leg].picks]


def _leg_total(leg: Leg) -> float:
    return sum(max(0.0, leg.coverage.get(n, 0.0)) for n in leg.picks)


def _base_coverage(leg: Leg, levers: Sequence[Lever]) -> dict[int, float]:
    """The leg with every lever's amount on it taken back out."""
    base = {n: max(0.0, leg.coverage.get(n, 0.0)) for n in leg.picks}
    for lever in levers:
        if lever.leg == leg.leg:
            base[lever.horse] = max(0.0, base[lever.horse] - lever.amount)
    return base


def _averaged(base: dict[int, float],
              views: Sequence[dict[int, float]]) -> dict[int, float]:
    """Base coverage reshaped by the levers pointed at a leg.

    Every lever in play has its own view of the leg. Their average is the
    shape, scaled to the base total so the piece keeps its size. Multiplying
    them would sharpen it instead: seven levers with the same view would
    give that view to the seventh power.
    """
    total = sum(base.values())
    normed = []
    for view in views:
        spread = sum(max(0.0, v) for v in view.values())
        if spread > 0:
            normed.append({n: v / spread for n, v in view.items() if v > 0})
    if total <= 0 or not normed:
        return dict(base)
    out: dict[int, float] = {}
    for view in normed:
        for n, v in view.items():
            out[n] = out.get(n, 0.0) + v * total / len(normed)
    return out


def _cascade_fan(leg: Leg, total: float,
                 outsiders: Sequence[int]) -> dict[int, float]:
    """Half on the picks by their coverage, half down the outsiders by rank."""
    weight = {n: max(0.0, leg.coverage.get(n, 0.0)) for n in leg.picks}
    spread = sum(weight.values())
    half = total * 0.5
    out = {n: half * (weight[n] / spread if spread > 0 else 1 / len(weight))
           for n in leg.picks}
    count = len(outsiders)
    for rank, n in enumerate(outsiders):
        out[n] = total * (count - rank) / (count * (count + 1))
    return out


class _Stake:
    """Everything the intended stake is built from, read once per plan."""

    def __init__(self, plan: Plan):
        self.legs = plan.legs
        self.seat = {leg.leg: i for i, leg in enumerate(plan.legs)}
        self.levers = _live_levers(plan)
        self.pool = frozenset(k for k, lever in enumerate(self.levers)
                              if lever.pool)
        self.totals = [_leg_total(leg) for leg in plan.legs]
        self.base = [_base_coverage(leg, self.levers) for leg in plan.legs]
        picks = [set(leg.picks) for leg in plan.legs]

        self.lever_on: dict[tuple[int, int], list[int]] = {}
        self.aims_into = [set() for _ in plan.legs]
        named: list[dict[int, float]] = [dict() for _ in plan.legs]
        self.opens: dict[int, int] = {}
        for k, lever in enumerate(self.levers):
            own = self.seat[lever.leg]
            self.lever_on.setdefault((own, lever.horse), []).append(k)
            opened = []
            for leg_no, view in lever.targets.items():
                i = self.seat.get(leg_no)
                if i is None or i == own or not view:
                    continue
                self.aims_into[i].add(k)
                if leg_no not in lever.cascade_targets:
                    continue
                outside = {n: v for n, v in view.items()
                           if v > 0 and n not in picks[i]}
                if not outside:
                    continue
                opened.append(i)
                for n, v in outside.items():
                    named[i][n] = max(named[i].get(n, 0.0), v)
            if len(opened) > 1:
                raise ReduceError(
                    f'leg {lever.leg} horse {lever.horse} cascades into '
                    f'{len(opened)} races — a lever can open one race')
            if opened:
                self.opens[k] = opened[0]

        # Outsiders best first. The fan needs the race's rank to taper by;
        # without one the cascade's own shares are used as they came.
        self.outsiders: list[list[int]] = []
        self.ranked_fan: list[bool] = []
        for i, leg in enumerate(plan.legs):
            seat = {n: r for r, n in enumerate(leg.order or [])}
            ranked = bool(named[i]) and all(n in seat for n in named[i])
            key = ((lambda n: seat[n]) if ranked
                   else (lambda n, d=named[i]: (-d[n], n)))
            self.outsiders.append(sorted(named[i], key=key))
            self.ranked_fan.append(ranked)
        self._shapes: dict[tuple[int, frozenset[int]], dict[int, float]] = {}

    def shape(self, i: int, active: frozenset[int]) -> dict[int, float]:
        """Leg i's coverage while the levers in `active` are in play."""
        key = (i, active)
        got = self._shapes.get(key)
        if got is not None:
            return got
        leg = self.legs[i]
        openers = sorted(k for k in active if self.opens.get(k) == i)
        if openers and self.ranked_fan[i]:
            named = set()
            for k in openers:
                named.update(n for n, v in self.levers[k].targets[leg.leg].items()
                             if v > 0)
            got = _cascade_fan(leg, sum(self.base[i].values()),
                               [n for n in self.outsiders[i] if n in named])
        elif openers:
            got = _averaged(self.base[i], [self.levers[k].targets[leg.leg]
                                           for k in openers])
        else:
            picks = self.base[i]
            got = _averaged(self.base[i], [
                {n: v for n, v in self.levers[k].targets[leg.leg].items()
                 if n in picks}
                for k in sorted(active)])
        self._shapes[key] = got
        return got

    def families(self, want: int) -> Iterator[list[list[int]]]:
        """Horse sets per leg whose products cover every row in play.

        The coupon's picks are one family. A cascade opens its race to
        outsiders only on rows its lever's horse is on, so each further
        family pins some cascading levers' horses and puts nothing but
        outsiders in the races they open. Families overlap; the caller
        keeps the first sighting of a row.
        """
        picks = [leg.ranked() for leg in self.legs]
        openers = sorted(self.opens)

        def walk(start, sets, used, mass):
            yield sets
            for j in range(start, len(openers)):
                k = openers[j]
                lever = self.levers[k]
                own, opened = self.seat[lever.leg], self.opens[k]
                if own in used or opened in used:
                    continue
                share = lever.amount / self.totals[own] if self.totals[own] else 0
                more = mass * share * (0.5 if self.ranked_fan[opened] else 1.0)
                if want * more < _FAMILY_FLOOR:
                    continue
                nxt = list(sets)
                nxt[own] = [lever.horse]
                nxt[opened] = self.outsiders[opened]
                yield from walk(j + 1, nxt, used | {own, opened}, more)

        yield from walk(0, picks, frozenset(), 1.0)

    def value(self, row: tuple[int, ...]) -> tuple[float, float]:
        """The row's stake before scaling — every reading of it, summed —
        and the part of it read through a pool lever."""
        levered = [(i, self.lever_on[(i, h)]) for i, h in enumerate(row)
                   if (i, h) in self.lever_on]
        if not levered:
            v = 1.0
            for i, h in enumerate(row):
                c = self.base[i].get(h, 0.0)
                if c <= 0:
                    return 0.0, 0.0
                v *= c
            return v, 0.0

        total = 0.0
        pooled = 0.0
        options = [(None,) + tuple(ks) for _, ks in levered]
        for choice in itertools.product(*options):
            fired = {i: k for (i, _), k in zip(levered, choice) if k is not None}
            active = frozenset(fired.values())
            v = 1.0
            for i, h in enumerate(row):
                k = fired.get(i)
                if k is not None:
                    v *= self.levers[k].amount
                    continue
                into = active & self.aims_into[i]
                c = (self.shape(i, frozenset(into)) if into
                     else self.base[i]).get(h, 0.0)
                if c <= 0:
                    v = 0.0
                    break
                v *= c
            total += v
            if v and active & self.pool:
                pooled += v
        return total, pooled


def intended_stake(plan: Plan,
                   units: int | None = None) -> dict[tuple[int, ...], float]:
    """Stake units every row is meant to carry, adding up to the system size.

    Cached on the plan: a plan is not edited once it is being built.
    """
    want = int(units if units is not None else plan.rows)
    return _stake_parts(plan, want)[0]


def pooled_stake(plan: Plan,
                 units: int | None = None) -> dict[tuple[int, ...], float]:
    """The part of every row's intended stake that pool levers put there.

    Already inside `intended_stake`; rows without any are left out.
    """
    want = int(units if units is not None else plan.rows)
    return _stake_parts(plan, want)[1]


def _stake_parts(plan: Plan, want: int) -> tuple[dict[tuple[int, ...], float],
                                                 dict[tuple[int, ...], float]]:
    cache = plan.__dict__.setdefault('_intended', {})
    if want in cache:
        return cache[want]

    if plan.full_rows > _MAX_ROWS:
        raise ReduceError(
            f'a weighted system over {plan.full_rows} rows is too big to '
            f'build')
    stake = _Stake(plan)
    raw: dict[tuple[int, ...], float] = {}
    raw_pooled: dict[tuple[int, ...], float] = {}
    for sets in stake.families(want):
        for row in itertools.product(*sets):
            if row in raw:
                continue
            v, pooled = stake.value(row)
            if v > 0:
                raw[row] = v
                if pooled > 0:
                    raw_pooled[row] = pooled
            if len(raw) > _MAX_ROWS:
                raise ReduceError(
                    f'the cascades open more than {_MAX_ROWS} rows — too big '
                    f'to build')

    size = math.prod(t for t in stake.totals if t > 0)
    scale = want / size if size > 0 else 0.0
    out = {row: v * scale for row, v in raw.items()}
    top = {row: v * scale for row, v in raw_pooled.items()}
    # A pruned family takes a sliver of stake with it. Scale the rest back
    # up so the system is still the size the coupon sphere shows.
    spread = sum(out.values())
    if spread > 0 and abs(spread - want) > 1e-9 * want:
        out = {row: v * want / spread for row, v in out.items()}
        top = {row: v * want / spread for row, v in top.items()}
    cache[want] = (out, top)
    return cache[want]


def _seat_of(leg: Leg) -> dict[int, int]:
    """Rank of every horse a leg can hold: picks first, then the order."""
    seat = {n: r for r, n in enumerate(leg.ranked())}
    for n in leg.order or []:
        seat.setdefault(n, len(seat))
    return seat


def _margins(stake: dict[tuple[int, ...], float],
             width: int) -> list[dict[int, float]]:
    out: list[dict[int, float]] = [dict() for _ in range(width)]
    for row, v in stake.items():
        for i, h in enumerate(row):
            out[i][h] = out[i].get(h, 0.0) + v
    return out


def whole_margins(plan: Plan, stake: dict[tuple[int, ...], float],
                  want: int, seat: bool = True) -> list[dict[int, int]]:
    """Per leg, the stake units each horse should end up with.

    Largest remainder, ties to the stronger horse. With `seat`, a coupon
    pick with any stake at all keeps at least one row: a horse on the coupon
    that ends up on none is a rounding artefact, not a choice.
    """
    out = []
    for leg, margin in zip(plan.legs, _margins(stake, len(plan.legs))):
        rank = _seat_of(leg)
        clean = {h: round(v, 9) for h, v in margin.items()}
        whole = {h: int(math.floor(v)) for h, v in clean.items()}
        short = want - sum(whole.values())
        ranked = sorted(clean, key=lambda h: (-(clean[h] - whole[h]),
                                              rank.get(h, len(rank)), h))
        for h in ranked[:max(0, short)]:
            whole[h] += 1
        for n in (leg.picks if seat else ()):
            if whole.get(n, 0) > 0 or clean.get(n, 0.0) <= 0:
                continue
            donor = max(whole, key=lambda h: (whole[h], -rank.get(h, len(rank))))
            if whole[donor] <= 1:
                raise ReduceError(
                    f'leg {leg.leg} cannot give every pick a row at '
                    f'{want} rows')
            whole[donor] -= 1
            whole[n] = 1
        out.append(whole)
    return out


def _neighbours(row: tuple[int, ...],
                horses: list[list[int]]) -> Iterator[tuple[int, ...]]:
    for i, hs in enumerate(horses):
        for h in hs:
            if h != row[i]:
                yield row[:i] + (h,) + row[i + 1:]


# Rounding passes. The multipliers settle within a few dozen rounds; past a
# stretch without improvement they are only trading one horse's unit for
# another's, and the step is halved.
_FIT_ROUNDS = 120
_FIT_PATIENCE = 6
_FIT_STEP = 0.7
_FIT_MIN_STEP = 0.08
_EDGE_SCANS = 200
# Both passes walk every row, so a big coupon gets fewer of them. Coupons
# past _EXACT_ROWS stop polishing once the margins are close; they are
# already far past what ATG accepts in one file.
_FIT_WORK = 6_000_000
_EDGE_WORK = 1_000_000
_EXACT_ROWS = 50_000
# Two stakes this close are the same stake told apart by float noise.
_SAME = 1e-9


def round_stake(plan: Plan, stake: dict[tuple[int, ...], float],
                want: int, seat: bool = True) -> dict[tuple[int, ...], int]:
    """Whole stake units per row, one threshold for the whole coupon.

    Every row gets the floor of its stake and the rows with the largest
    remainders get one more, so a row never rounds below a row it beats.
    That alone leaves the margins off, so each horse carries a multiplier,
    fitted until the margins land; horses the plan cannot tell apart share
    theirs, which keeps their rows level. Pairs the multipliers flipped are
    swapped back, and what is still off is moved a unit at a time, only
    where no row would end up below one it beats.

    `seat` holds every coupon pick to a row. A pool lever's part of the
    stake lives on its own horse's rows alone and is rounded without it.
    """
    targets = whole_margins(plan, stake, want, seat)
    width = len(plan.legs)
    seats = [_seat_of(leg) for leg in plan.legs]
    rows = sorted(stake, key=lambda r: [seats[i].get(h, len(seats[i]))
                                        for i, h in enumerate(r)])
    base = [stake[r] for r in rows]
    horses = [sorted(t, key=lambda h, s=seats[i]: s.get(h, len(s)))
              for i, t in enumerate(targets)]

    # Horses by position, rows as one column of positions per leg, and every
    # horse pointed at the first horse it is tied with.
    exact = _margins(stake, width)
    cols = []
    alias: list[list[int]] = []
    for i, hs in enumerate(horses):
        pos = {h: k for k, h in enumerate(hs)}
        cols.append([pos[r[i]] for r in rows])
        first: dict[float, int] = {}
        alias.append([first.setdefault(round(exact[i].get(h, 0.0) / want, 12), k)
                      for k, h in enumerate(hs)])
    goal = [[targets[i][h] for h in hs] for i, hs in enumerate(horses)]
    alpha = [[1.0] * len(hs) for hs in horses]
    close = max(width, want // 2000)

    best: tuple[int, list[int], list[list[float]]] | None = None
    stale = 0
    step = _FIT_STEP
    for _ in range(max(12, min(_FIT_ROUNDS, _FIT_WORK // len(rows)))):
        xs = base
        for i in range(width):
            factor = [alpha[i][a] for a in alias[i]]
            xs = [v * f for v, f in zip(xs, map(factor.__getitem__, cols[i]))]
        scale = want / sum(xs)
        xs = [v * scale for v in xs]
        units = [int(v) for v in xs]
        frac = [v - n for v, n in zip(xs, units)]
        extra = want - sum(units)
        # Stable, and the rows are in rank order, so a tie goes to the
        # stronger row.
        for k in sorted(range(len(rows)), key=frac.__getitem__,
                        reverse=True)[:extra]:
            units[k] += 1

        got = []
        for i in range(width):
            sums = [0] * len(horses[i])
            for k, n in zip(cols[i], units):
                sums[k] += n
            got.append(sums)
        err = sum(abs(g - t) for gs, ts in zip(got, goal)
                  for g, t in zip(gs, ts))
        if best is None or err < best[0]:
            best = (err, units, [list(a) for a in alpha])
            stale = 0
        else:
            stale += 1
        if err <= close:
            break
        # A skewed coupon makes the full step overshoot and circle. Go back
        # to the best fit so far and take smaller steps from there.
        if stale >= _FIT_PATIENCE:
            step *= 0.5
            if step < _FIT_MIN_STEP:
                break
            alpha = [list(a) for a in best[2]]
            stale = 0

        for i in range(width):
            pooled_t: dict[int, int] = {}
            pooled_g: dict[int, int] = {}
            for k, a in enumerate(alias[i]):
                pooled_t[a] = pooled_t.get(a, 0) + goal[i][k]
                pooled_g[a] = pooled_g.get(a, 0) + got[i][k]
            for a, t in pooled_t.items():
                g = pooled_g[a]
                if t == 0:
                    alpha[i][a] *= 0.5
                elif g == 0:
                    alpha[i][a] *= 1.5
                else:
                    alpha[i][a] *= (t / g) ** step

    whole = {r: n for r, n in zip(rows, best[1])}
    if _live_levers(plan) or not _keeps_order(exact, horses, alias, best[2]):
        _mend_order(whole, stake, horses)
    enough = 0 if len(rows) <= _EXACT_ROWS else close
    _settle_edges(whole, stake, rows, horses, targets, enough)
    # A thin coupon with levers can be left with nothing but near ties to
    # trade, every row on one unit or none. Those may swap to land the
    # margins; a row is still never below one it clearly beats.
    if _worst_miss(whole, targets) > math.ceil(want * MARGIN_TOLERANCE):
        _settle_edges(whole, stake, rows, horses, targets, enough, loose=True)
    if seat:
        _seat_every_pick(plan, whole, stake, horses)
    return {r: n for r, n in whole.items() if n > 0}


# Two neighbouring rows the plan means to carry within a quarter and within
# a unit of each other are a near tie: whole units cannot tell them apart,
# so the rounding may give the unit to either.
_NEAR_TIE = 0.25


def outranks(a: float, b: float) -> bool:
    """Whether a row meant to carry `a` must carry at least as much as `b`."""
    return a > b * (1 + _NEAR_TIE) or a > b + 1


def _worst_miss(units: dict[tuple[int, ...], int],
                targets: list[dict[int, int]]) -> int:
    got: list[dict[int, int]] = [dict() for _ in targets]
    for r, n in units.items():
        for i, h in enumerate(r):
            got[i][h] = got[i].get(h, 0) + n
    return max((abs(got[i].get(h, 0) - t) for i, ts in enumerate(targets)
                for h, t in ts.items()), default=0)


def _keeps_order(exact: list[dict[int, float]], horses: list[list[int]],
                 alias: list[list[int]], alpha: list[list[float]]) -> bool:
    """Whether the multipliers left every leg's horses in the plan's order.

    Without levers a row's stake is a product of shares, so if no horse's
    multiplier lifts it past a stronger one, rounding on the fitted stake
    cannot put a row below one it beats.
    """
    for margin, hs, names, scale in zip(exact, horses, alias, alpha):
        fitted = sorted(((margin.get(h, 0.0), margin.get(h, 0.0) * scale[names[k]])
                         for k, h in enumerate(hs)), reverse=True)
        if any(b[1] > a[1] * (1 + _SAME)
               for a, b in zip(fitted, fitted[1:]) if a[0] > b[0]):
            return False
    return True


def _mend_order(units: dict[tuple[int, ...], int],
                stake: dict[tuple[int, ...], float],
                horses: list[list[int]]) -> None:
    """Swap the stake of neighbours the fit left in the wrong order.

    The multipliers are per horse, but with levers in play which of two
    horses the plan rates higher can depend on the rest of the row. A
    multiplier that suits one context then flips a close pair in another.
    Every swap removes an inversion against the plan's own order, so this
    ends; it costs a little margin, which the edge pass wins back.
    """
    by_stake = sorted(stake, key=lambda r: -stake[r])
    while True:
        swapped = False
        for r in by_stake:
            n = units.get(r, 0)
            if not n:
                continue
            for alt in _neighbours(r, horses):
                if units.get(alt, 0) < n and outranks(stake.get(alt, 0.0), stake[r]):
                    units[r], units[alt] = units.get(alt, 0), n
                    n = units[r]
                    swapped = True
        if not swapped:
            return


def _settle_edges(units: dict[tuple[int, ...], int],
                  fitted: dict[tuple[int, ...], float],
                  rows: list[tuple[int, ...]], horses: list[list[int]],
                  targets: list[dict[int, int]], enough: int = 0,
                  loose: bool = False) -> None:
    """Move single units toward the margins without breaking the order.

    A row may gain a unit only if every row that beats it already carries
    more, and give one up only if every row it beats already carries less.
    `loose` lets near ties trade places. Stops once the margins are `enough`
    units off in total.
    """
    need = [dict(t) for t in targets]
    for r, n in units.items():
        for i, h in enumerate(r):
            need[i][h] -= n

    def err():
        return sum(abs(v) for d in need for v in d.values())

    if loose:
        def beats(a, b):
            return outranks(a, b)
    else:
        def beats(a, b):
            return a > b * (1 + _SAME)

    def can_add(a):
        k = units.get(a, 0) + 1
        mine = fitted[a]
        return all(not (units.get(nb, 0) < k and beats(fitted.get(nb, 0.0), mine))
                   for nb in _neighbours(a, horses))

    def can_take(b):
        k = units.get(b, 0) - 1
        if k < 0:
            return False
        mine = fitted[b]
        return all(not (nb in fitted and units.get(nb, 0) > k
                        and beats(mine, fitted[nb]))
                   for nb in _neighbours(b, horses))

    # A leg with one horse is on every row, so an added unit and a taken
    # one always cancel there; counting it would make every pair look
    # worse than it is.
    open_legs = [i for i, hs in enumerate(horses) if len(hs) > 1]

    def gain(r, d):
        return sum(abs(need[i][r[i]] - d) - abs(need[i][r[i]])
                   for i in open_legs)

    # Every leg's misses add up to nothing, so a horse over its share can
    # always hand a unit to one under it in the same leg. Doing it between
    # rows that differ in that leg alone leaves every other leg as it was.
    holding: list[dict[int, list[tuple[int, ...]]]] = [dict() for _ in horses]
    for r in rows:
        for i, h in enumerate(r):
            holding[i].setdefault(h, []).append(r)

    def hand_over() -> int:
        moved = 0
        for i, hs in enumerate(horses):
            for h in hs:
                if need[i][h] >= 0:
                    continue
                for b in sorted((r for r in holding[i][h] if units.get(r, 0)),
                                key=lambda r: fitted[r]):
                    if need[i][h] >= 0:
                        break
                    for g in hs:
                        if need[i][g] <= 0:
                            continue
                        a = b[:i] + (g,) + b[i + 1:]
                        if a not in fitted or not can_take(b):
                            continue
                        units[b] -= 1
                        if can_add(a):
                            units[a] = units.get(a, 0) + 1
                            need[i][h] += 1
                            need[i][g] -= 1
                            moved += 1
                            break
                        units[b] += 1
        return moved

    for _ in range(max(2, min(_EDGE_SCANS, _EDGE_WORK // max(1, len(rows))))):
        before = err()
        if before <= enough:
            break
        if hand_over():
            continue
        # Only a pair that gains overall is worth checking, and the order
        # check is the expensive part, so it runs on those alone.
        up = [(gain(r, 1), r) for r in rows]
        down = [(gain(r, -1), r) for r in rows if units.get(r, 0) > 0]
        if not up or not down:
            break
        low_up = min(g for g, _ in up)
        low_down = min(g for g, _ in down)
        adds = sorted(p for p in up if p[0] + low_down < 0)
        takes = sorted(p for p in down if p[0] + low_up < 0)
        touched: set[tuple[int, ...]] = set()
        moved = ai = ti = 0
        while ai < len(adds) and ti < len(takes):
            ga, a = adds[ai]
            gt, b = takes[ti]
            if ga + gt >= 0:
                break
            if a in touched or not can_add(a):
                ai += 1
                continue
            if b in touched or b == a or not can_take(b):
                ti += 1
                continue
            d = gain(a, 1)
            for i, h in enumerate(a):
                need[i][h] -= 1
            if d + gain(b, -1) >= 0:
                for i, h in enumerate(a):
                    need[i][h] += 1
                ai += 1
                continue
            for i, h in enumerate(b):
                need[i][h] += 1
            units[a] = units.get(a, 0) + 1
            units[b] -= 1
            touched.update((a, b))
            touched.update(_neighbours(a, horses))
            touched.update(_neighbours(b, horses))
            moved += 1
            ai += 1
            ti += 1
        if not moved or err() >= before:
            break


def _seat_every_pick(plan: Plan, units: dict[tuple[int, ...], int],
                     fitted: dict[tuple[int, ...], float],
                     horses: list[list[int]]) -> None:
    """A coupon pick the rounding left on no row gets its best row back.

    The best row is the strongest one that also seats other stranded picks,
    so a thin coupon is not walked into a corner one pick at a time. The
    unit comes off the deepest row, which keeps every horse on it. When
    every row is down to one unit, the weakest row whose horses all sit on
    another row gives way instead.
    """
    held: list[dict[int, int]] = [dict.fromkeys(leg.picks, 0)
                                  for leg in plan.legs]
    for r, n in units.items():
        for j, h in enumerate(r):
            if n and h in held[j]:
                held[j][h] += n

    def stranded(r: tuple[int, ...]) -> int:
        return sum(1 for j, h in enumerate(r) if held[j].get(h, 1) == 0)

    for i, leg in enumerate(plan.legs):
        for horse in leg.picks:
            if held[i][horse]:
                continue
            mine = [r for r in fitted if r[i] == horse]
            if not mine:
                continue
            mine.sort(key=lambda r: (stranded(r), fitted[r]), reverse=True)
            give = mine[0]
            donor = max(units, key=lambda r: (units[r], -fitted[r]))
            if units[donor] < 2:
                for give in mine[:64]:
                    donor = _spare_row(plan, units, fitted, give, held)
                    if donor is not None:
                        break
                else:
                    donor = None
            if donor is None:
                raise ReduceError(
                    f'leg {leg.leg} cannot give every pick a row')
            units[donor] -= 1
            units[give] = units.get(give, 0) + 1
            for j, h in enumerate(donor):
                if h in held[j]:
                    held[j][h] -= 1
            for j, h in enumerate(give):
                if h in held[j]:
                    held[j][h] += 1


def _spare_row(plan: Plan, units: dict[tuple[int, ...], int],
               fitted: dict[tuple[int, ...], float],
               keep: tuple[int, ...],
               held: list[dict[int, int]]) -> tuple[int, ...] | None:
    """The weakest one-unit row no pick depends on once `keep` has its unit,
    rank breaking ties."""
    seats = [_seat_of(leg) for leg in plan.legs]
    spare = [r for r, n in units.items()
             if n == 1 and r != keep
             and all(h not in held[j] or held[j][h] + (keep[j] == h) > 1
                     for j, h in enumerate(r))]
    if not spare:
        return None
    return min(spare, key=lambda r: (
        fitted.get(r, 0.0),
        [-seats[j].get(h, len(seats[j])) for j, h in enumerate(r)]))


def _split_stake(plan: Plan, want: int) -> tuple[dict[tuple[int, ...], float],
                                                 dict[tuple[int, ...], float]]:
    """The intended stake as the part paying on every tier and the part its
    pool levers play for the top tier. A coupon played wholly for the top
    tier has one part and keeps it first."""
    stake = intended_stake(plan, want)
    pooled = {} if plan.top_only else pooled_stake(plan, want)
    if not pooled:
        return stake, {}
    rest = {row: v - pooled.get(row, 0.0) for row, v in stake.items()}
    return {row: v for row, v in rest.items() if v > 1e-9}, pooled


def allocate(plan: Plan, units: int | None = None) -> list[Box]:
    """Spread the system's stake over rows and return them as boxes."""
    if not plan.legs:
        raise ReduceError('plan has no legs')
    for leg in plan.legs:
        if not leg.picks:
            raise ReduceError(f'leg {leg.leg} has no picks')

    want = int(units if units is not None else plan.rows)
    if want <= 0:
        raise ReduceError('system has no rows')
    if want > _MAX_UNITS:
        raise ReduceError(f'system is too large to build ({want} rows)')

    levers = _live_levers(plan)
    if ((plan.top_only or any(lever.pool for lever in levers))
            and not top_pool(plan.game_type)):
        raise ReduceError(
            f'{plan.game_type} cannot be played for the top pool alone')

    # An unweighted system at full size is every row played once, which is a
    # single coupon — worth taking straight rather than walking the grid to
    # rediscover it, since that grid can run to millions of rows.
    targets = [integer_targets(leg, want) for leg in plan.legs]
    if (not levers and want == plan.full_rows
            and all(len(set(t.values())) == 1 for t in targets)):
        return [Box(tuple(tuple(sorted(leg.picks)) for leg in plan.legs),
                    top_only=plan.top_only)]

    return _round_boxes(plan, want)


def _round_boxes(plan: Plan, want: int) -> list[Box]:
    rest, pooled = _split_stake(plan, want)
    if not pooled:
        whole = round_stake(plan, rest, want)
        return [Box(tuple((h,) for h in row), n, plan.top_only)
                for row, n in whole.items()]

    # Pool levers make the system two stakes, rounded apart so that each
    # keeps its own order: a top-pool unit is not worth an ordinary one.
    top_units = round(sum(pooled.values()))
    boxes = [Box(tuple((h,) for h in row), n)
             for row, n in round_stake(plan, rest,
                                       want - top_units).items()]
    if top_units > 0:
        boxes += [Box(tuple((h,) for h in row), n, True)
                  for row, n in round_stake(plan, pooled, top_units,
                                            seat=False).items()]
    return boxes


def merge_boxes(boxes: Sequence[Box]) -> list[Box]:
    """Fold boxes that differ in exactly one leg back together.

    The allocation splits one horse at a time, so it hands back a lot of
    boxes that are really slices of the same coupon.  Finding the smallest
    cover is NP-hard; repeatedly unioning along a single leg is the standard
    approximation and gets the count down far enough to submit.
    """
    if not boxes:
        return []

    current = list(boxes)
    legs = len(current[0].sets)

    changed = True
    while changed:
        changed = False
        for leg in range(legs):
            buckets: dict[tuple, list[Box]] = {}
            for box in current:
                key = (box.multiplier, box.top_only,
                       box.sets[:leg], box.sets[leg + 1:])
                buckets.setdefault(key, []).append(box)

            merged: list[Box] = []
            for (mult, top, head, tail), group in buckets.items():
                if len(group) == 1:
                    merged.append(group[0])
                    continue
                union: set[int] = set()
                for box in group:
                    union |= set(box.sets[leg])
                merged.append(
                    Box(head + (tuple(sorted(union)),) + tail, mult, top))
                changed = True
            current = merged

    current.sort(key=lambda b: (b.top_only, -b.units, b.sets))
    return current


def split_multipliers(boxes: Sequence[Box], game_type: str = '') -> list[Box]:
    """Rewrite multipliers ATG will not accept as a sum of ones it will."""
    allowed = multipliers_for(game_type)
    out: list[Box] = []
    for box in boxes:
        left = box.multiplier
        if left in allowed:
            out.append(box)
            continue
        for step in allowed:
            while left >= step:
                out.append(Box(box.sets, step, box.top_only))
                left -= step
        if left:  # unreachable: 1 is in every table
            out.append(Box(box.sets, left, box.top_only))
    return out


def solve(plan: Plan) -> list[Box]:
    """Coverage in, submittable boxes out."""
    return split_multipliers(merge_boxes(allocate(plan)), plan.game_type)


# ---------------------------------------------------------------------------
# Stage 3 — check the answer
# ---------------------------------------------------------------------------
@dataclass
class Verification:
    ok: bool
    units: int
    expected_units: int
    boxes: int
    warnings: list[str]
    # leg -> horse -> {'target': units, 'actual': units, 'coverage': pct}
    coverage: dict[int, dict[int, dict[str, float]]]
    # How many different combinations the stake actually lands on, the most
    # any single row carries, and rows left behind one they beat.
    distinct_rows: int = 0
    deepest_row: int = 1
    gaps: int = 0


def row_units(boxes: Sequence[Box]) -> dict[tuple[int, ...], int] | None:
    """Stake on every row the boxes cover, or None when there are too many."""
    out: dict[tuple[int, ...], int] = {}
    for box in boxes:
        for row in itertools.product(*box.sets):
            out[row] = out.get(row, 0) + box.multiplier
            if len(out) > _MAX_ROWS:
                return None
    return out


def monotonicity_gaps(plan: Plan, units: dict[tuple[int, ...], int],
                      stake: dict[tuple[int, ...], float] | None = None,
                      *, strict: bool = False) -> list[tuple]:
    """Rows carrying less stake than a row they beat.

    A row that differs in one horse and that the plan means to carry more
    must never come back with less.  Every gap this finds is a system that
    pays out on the longshot and not on the favourite standing next to it.
    Without aims "means to carry more" is just the higher weight; with them
    it is the intended stake, which an aim deliberately reorders.  Near ties
    only count when `strict`.
    """
    if stake is None:
        stake = intended_stake(plan, sum(units.values()))
    horses = [sorted(m) for m in _margins(stake, len(plan.legs))]
    out = []
    for row, n in units.items():
        mine = stake.get(row, 0.0)
        for i, h in enumerate(row):
            for g in horses[i]:
                if g == h:
                    continue
                alt = row[:i] + (g,) + row[i + 1:]
                if units.get(alt, 0) >= n:
                    continue
                theirs = stake.get(alt, 0.0)
                if (theirs > mine * (1 + _SAME) if strict
                        else outranks(theirs, mine)):
                    out.append((plan.legs[i].leg, g, h, alt))
    return out


# How far a horse's built share may sit from the plan's, as a fraction of
# the coupon. Rounding the whole coupon at once keeps every row at or above
# the rows it beats, and this is what that costs the margins.
MARGIN_TOLERANCE = 0.005


def verify(plan: Plan, boxes: Sequence[Box]) -> Verification:
    """Recompute everything from the boxes and compare with the plan."""
    expected = plan.rows
    warnings: list[str] = []

    actual_units = sum(b.units for b in boxes)
    per_leg: list[dict[int, int]] = [dict() for _ in plan.legs]
    for box in boxes:
        for i, horses in enumerate(box.sets):
            # Every horse in a leg's set sits on the box's rows divided
            # evenly among that leg's horses.
            share = box.units // len(horses)
            for h in horses:
                per_leg[i][h] = per_leg[i].get(h, 0) + share

    coverage: dict[int, dict[int, dict[str, float]]] = {}
    ok = actual_units == expected
    if not ok:
        warnings.append(
            f'built {actual_units} rows, expected {expected}')

    # Margins are checked against what was built, not what was asked for:
    # when the weights force the system smaller, the shares still have to
    # be right at the size it came out.
    stake = intended_stake(plan, actual_units) if actual_units > 0 else {}
    # Pool levers round as two stakes, so each is held to its own margins
    # and the slack is the two parts' slack added up.
    rest, pooled = (_split_stake(plan, actual_units) if stake else ({}, {}))
    top_units = sum(b.units for b in boxes if b.top_only) if pooled else 0
    parts = [(rest, actual_units - top_units, True)]
    if pooled:
        parts.append((pooled, top_units, False))
    targets = [{} for _ in plan.legs]
    slack = 0
    for part, part_units, seat in parts:
        if not part or part_units <= 0:
            continue
        slack += max(1, math.ceil(part_units * MARGIN_TOLERANCE))
        for i, whole in enumerate(whole_margins(plan, part, part_units, seat)):
            for h, n in whole.items():
                targets[i][h] = targets[i].get(h, 0) + n
    slack = max(1, slack)
    exact = _margins(stake, len(plan.legs))
    for i, leg in enumerate(plan.legs):
        seat = _seat_of(leg)
        rows: dict[int, dict[str, float]] = {}
        for horse in sorted(set(targets[i]) | set(per_leg[i]),
                            key=lambda h: (seat.get(h, len(seat)), h)):
            target = targets[i].get(horse, 0)
            got = per_leg[i].get(horse, 0)
            rows[horse] = {
                'target': target,
                'actual': got,
                'coverage': round(got / actual_units * 100, 2) if actual_units else 0.0,
                'wanted': round(exact[i].get(horse, 0.0) / actual_units * 100, 2)
                if actual_units else 0.0,
            }
            if abs(got - target) > slack:
                ok = False
                warnings.append(
                    f'leg {leg.leg} horse {horse}: {got} rows, wanted {target}')
        coverage[leg.leg] = rows

    # Breadth and shape. Neither can fail a build — piling stake on the
    # favourites' rows is what a weighted coupon is for — but both are worth
    # saying out loud, because a coupon that spans 2268 rows and covers 1173
    # of them does not look like one from the outside.
    units = row_units(boxes)
    if units is not None:
        distinct = len(units)
        deepest = max(units.values(), default=0)
        gaps = []
        if pooled:
            for part, top in ((rest, False), (pooled, True)):
                mine = row_units([b for b in boxes if b.top_only == top]) or {}
                if part and mine:
                    gaps += monotonicity_gaps(plan, mine, part)
        elif stake:
            gaps = monotonicity_gaps(plan, units, stake)
        if gaps:
            leg, strong, weak, _ = gaps[0]
            warnings.append(
                f'{len(gaps)} stronger-swap comparisons carry less stake'
                f' — leg {leg} horse {strong} behind {weak}')
    else:
        distinct, deepest, gaps = 0, 0, []

    if len(boxes) > ATG_MAX_SYSTEMS:
        ok = False
        warnings.append(
            f'{len(boxes)} coupons — ATG accepts at most {ATG_MAX_SYSTEMS} '
            f'per file')

    return Verification(ok=ok, units=actual_units, expected_units=expected,
                        boxes=len(boxes), warnings=warnings, coverage=coverage,
                        distinct_rows=distinct, deepest_row=deepest,
                        gaps=len(gaps))


# ---------------------------------------------------------------------------
# Stage 4 — the ATG file
# ---------------------------------------------------------------------------
def crc16(data: bytes) -> int:
    """CRC-16/ARC — poly 0x8005, init 0x0000, reflected.

    ATG's file-betting spec points at Princeton's `CRC16.java` for the four
    hex digits on the file name (`java CRC16 123456789` prints `bb3d`).
    CCITT-FALSE is a different polynomial and is what made ATG warn that the
    file had been edited, even when it had not.
    """
    crc = 0x0000
    for byte in data:
        crc ^= byte
        for _ in range(8):
            crc = (crc >> 1) ^ 0xA001 if crc & 1 else crc >> 1
    return crc & 0xFFFF


def atg_filename(game_type: str, game_date: str, xml: str) -> str:
    """`stable_V85_2026-08-19_1A2B.xml` — checksum last, before the suffix."""
    stamp = crc16(xml.encode('utf-8'))
    safe_date = (game_date or _date.today().isoformat()).replace('/', '-')
    return f'stable_{game_type.upper()}_{safe_date}_{stamp:04X}.xml'


def marks_string(horses: Iterable[int], width: int) -> str:
    """Program numbers to ATG's "0"/"1" string, position 1 leftmost."""
    marks = ['0'] * width
    for n in horses:
        if not 1 <= n <= width:
            raise ReduceError(
                f'horse number {n} does not fit ATG\'s {width}-runner leg')
        marks[n - 1] = '1'
    return ''.join(marks)


def atg_xml(plan: Plan, boxes: Sequence[Box], *, product: str = 'Stable',
            company: str = 'Stable', version: str = '1') -> str:
    """The Filinlämning document for a solved system."""
    game_type = plan.game_type.upper()
    spec = ATG_COUPON_ELEMENT.get(game_type)
    if not spec:
        raise ReduceError(f'{game_type} cannot be submitted as a file')
    element, legs_required, needs_track = spec
    if len(plan.legs) != legs_required:
        raise ReduceError(
            f'{game_type} needs {legs_required} legs, plan has {len(plan.legs)}')
    if not boxes:
        raise ReduceError('nothing to submit')
    if len(boxes) > ATG_MAX_COUPON_ID:
        raise ReduceError(
            f'{len(boxes)} coupons — a file holds at most {ATG_MAX_SYSTEMS}')

    width = ATG_MARKS_WIDTH.get(game_type, ATG_MARKS_DEFAULT)
    for leg in plan.legs:
        if leg.starters and leg.starters > width:
            raise ReduceError(
                f'leg {leg.leg} has {leg.starters} starters; ATG\'s file '
                f'format for {game_type} only holds {width}')

    game_date = plan.game_date or _date.today().isoformat()
    now = _datetime.now()
    head = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<issuer xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"\n'
        f'        xsi:noNamespaceSchemaLocation="{ATG_SCHEMA_URL}"\n'
        f'        company="{_esc(company)}" product="{_esc(product)}"'
        f' version="{_esc(version)}"\n'
        f'        createddate="{now.date().isoformat()}"'
        f' createdtime="{now.strftime("%H:%M:%S")}"\n'
        f'        schemaversion="{ATG_SCHEMA_VERSION}">\n'
        '  <betcoupons>\n')

    track_attr = ''
    if needs_track:
        if plan.track_code is None:
            raise ReduceError(f'{game_type} coupons need a track code')
        track_attr = f' trackcode="{int(plan.track_code)}"'

    allowed = multipliers_for(game_type)
    pool = ATG_TOP_POOL.get(game_type)
    body: list[str] = []
    for i, box in enumerate(boxes, start=1):
        if box.multiplier not in allowed:
            raise ReduceError(
                f'{game_type} does not take betmultiplier {box.multiplier}')
        top_attr = ''
        if box.top_only:
            if not pool:
                raise ReduceError(
                    f'{game_type} cannot be played for the top pool alone')
            top_attr = f' {pool[0]}="true"'
        body.append(
            f'    <{element} couponid="{i}" date="{game_date}"{track_attr}'
            f' betmultiplier="{box.multiplier}"{top_attr}>\n')
        for leg_i, leg in enumerate(plan.legs):
            marks = marks_string(box.sets[leg_i], width)
            reserves = ''.join(
                f' r{k}="{n}"' for k, n in
                enumerate(_box_reserves(leg, box.sets[leg_i]), start=1))
            body.append(
                f'      <leg legno="{leg.leg}" marks="{marks}"{reserves}/>\n')
        body.append(f'    </{element}>\n')

    return head + ''.join(body) + '  </betcoupons>\n</issuer>\n'


def _box_reserves(leg: Leg, marked: Sequence[int]) -> list[int]:
    """The leg's reserves for one coupon.

    A cascade can mark a horse the leg holds in reserve, and a marked horse
    cannot stand in for a scratched one. The next unmarked horse down the
    rank takes its place, as many as the user asked for.
    """
    want = min(2, len(leg.reserves))
    out: list[int] = []
    for n in list(leg.reserves) + list(leg.order or []):
        if len(out) == want:
            break
        if n not in marked and n not in leg.picks and n not in out:
            out.append(n)
    return out


def _esc(value: str) -> str:
    return (str(value).replace('&', '&amp;').replace('<', '&lt;')
            .replace('>', '&gt;').replace('"', '&quot;'))


# ---------------------------------------------------------------------------
# Stage 5 — the same system said in ATG's own language
# ---------------------------------------------------------------------------
# A fallback for when the user would rather click through ATG's reduction
# tool than upload a file.  ABC letters and points are cross-leg conditions,
# so they can only bracket the system, never reproduce it — the recipe below
# is derived from the rows we actually built and is therefore guaranteed to
# keep all of them, plus some extra.  We report how many extra.
_LETTERS = 'ABCD'


def letter_groups(leg: Leg) -> dict[int, int]:
    """Horse -> letter index, bucketing picks that share a coverage.

    ATG's tool goes up to D, so anything past the fourth group joins it.
    """
    out: dict[int, int] = {}
    index = -1
    last: float | None = None
    for horse in leg.ranked():
        cov = leg.coverage.get(horse, 0.0)
        if last is None or abs(last - cov) >= 0.5:
            index += 1
            last = cov
        out[horse] = min(index, len(_LETTERS) - 1)
    return out


def _profile_counts(legs_letters: list[list[int]]) -> dict[tuple[int, ...], int]:
    """How many full-system rows fall in each letter profile."""
    dist: dict[tuple[int, ...], int] = {(0,) * len(_LETTERS): 1}
    for letters in legs_letters:
        nxt: dict[tuple[int, ...], int] = {}
        for profile, count in dist.items():
            for letter in letters:
                key = list(profile)
                key[letter] += 1
                k = tuple(key)
                nxt[k] = nxt.get(k, 0) + count
        dist = nxt
    return dist


def abc_recipe(plan: Plan, target_rows: int) -> dict:
    """Conditions to type into ATG's own tool, for the file-shy.

    ABC letters and points are conditions on whole rows, so they cannot
    reproduce per-horse shares — they can only land near the same row count
    with roughly the same shape.  Both variants below are searched for the
    setting that comes closest to `target_rows` without going over, and the
    caller is expected to say out loud that this is the approximate route.
    """
    letters_by_leg = [letter_groups(leg) for leg in plan.legs]
    legs_letters = [[letters_by_leg[i][h] for h in leg.ranked()]
                    for i, leg in enumerate(plan.legs)]
    dist = sorted(_profile_counts(legs_letters).items())
    if not dist:
        return {'letters': [], 'conditions': [], 'rows': 0, 'points': None}

    legs = len(plan.legs)
    full = sum(count for _, count in dist)

    # Points: the letter's position plus one, and a ceiling on the row's
    # total. One number, and the closest thing ATG has to our own ordering.
    by_points: dict[int, int] = {}
    for profile, count in dist:
        score = sum((i + 1) * n for i, n in enumerate(profile))
        by_points[score] = by_points.get(score, 0) + count
    running = 0
    points_pick = (max(by_points), full)
    for score in sorted(by_points):
        running += by_points[score]
        points_pick = (score, running)
        if running >= target_rows:
            break

    # ABC: the three conditions people actually use — at least this many
    # A-winners, at most this many C- and D-winners.
    best = None
    for min_a in range(0, legs + 1):
        for max_c in range(0, legs + 1):
            for max_d in range(0, legs + 1):
                kept = 0
                for profile, count in dist:
                    if (profile[0] >= min_a and profile[2] <= max_c
                            and profile[3] <= max_d):
                        kept += count
                if kept <= 0:
                    continue
                # Closest to target, and among equals the loosest set of
                # conditions, so we do not ask for fiddling that buys little.
                score = (abs(kept - target_rows), kept < target_rows,
                         min_a, -max_c, -max_d)
                if best is None or score < best[0]:
                    best = (score, min_a, max_c, max_d, kept)

    conditions = []
    kept = full
    if best is not None:
        _, min_a, max_c, max_d, kept = best
        if min_a > 0:
            conditions.append({'letter': 'A', 'min': min_a, 'max': legs})
        if max_c < legs:
            conditions.append({'letter': 'C', 'min': 0, 'max': max_c})
        if max_d < legs:
            conditions.append({'letter': 'D', 'min': 0, 'max': max_d})

    letters = []
    for i, leg in enumerate(plan.legs):
        letters.append({
            'leg': leg.leg,
            'horses': [{'number': h,
                        'letter': _LETTERS[letters_by_leg[i][h]],
                        'points': letters_by_leg[i][h] + 1,
                        'coverage': round(leg.coverage.get(h, 0.0), 1)}
                       for h in leg.ranked()],
        })

    return {
        'letters': letters,
        'conditions': conditions,
        'rows': kept,
        'points': {'max': points_pick[0], 'rows': points_pick[1]},
    }


# ---------------------------------------------------------------------------
# Putting it together
# ---------------------------------------------------------------------------
def build(plan: Plan) -> dict:
    """Solve, check and describe a plan — the shape the web layer returns."""
    boxes = solve(plan)
    check = verify(plan, boxes)
    price = plan.line_price or 0.0
    effective = plan.effective_coverage()

    return {
        'gameType': plan.game_type,
        'gameDate': plan.game_date,
        'trackName': plan.track_name,
        'trackCode': plan.track_code,
        'fullRows': plan.full_rows,
        'rows': check.units,
        # Weighting buys some rows more than once, so the stake spans fewer
        # combinations than it has rows. Worth showing: it is the difference
        # between what the coupon costs and what it actually covers.
        'distinctRows': check.distinct_rows,
        'deepestRow': check.deepest_row,
        'retention': round(plan.retention, 6),
        'scale': plan.scale,
        'removedRows': max(0, plan.full_rows - check.units),
        'addedRows': max(0, check.units - plan.full_rows),
        'linePrice': price,
        'cost': round(check.units * price, 2),
        'fullCost': round(plan.full_rows * price, 2),
        'coupons': [{
            'legs': [{'leg': plan.legs[i].leg, 'numbers': list(s)}
                     for i, s in enumerate(box.sets)],
            'rows': box.rows,
            'multiplier': box.multiplier,
            'units': box.units,
            'topOnly': box.top_only,
        } for box in boxes],
        'couponCount': len(boxes),
        'topPool': top_pool(plan.game_type),
        'topOnly': plan.top_only,
        'topRows': sum(box.units for box in boxes if box.top_only),
        'levers': [{'leg': lv.leg, 'horse': lv.horse,
                    'amount': round(lv.amount, 2),
                    'pool': lv.pool,
                    'targets': {str(k): {str(h): round(s, 4)
                                         for h, s in v.items()}
                                for k, v in lv.targets.items()}}
                   for lv in plan.levers],
        'legs': [{
            'leg': leg.leg,
            'raceId': leg.race_id,
            'picks': leg.ranked(),
            'coverage': {str(n): round(leg.coverage.get(n, 0.0), 2)
                         for n in leg.ranked()},
            # Where an aimed lever actually left this leg. Equal to
            # `coverage` unless something is pointed at it.
            'effective': {str(n): round(effective.get(leg.leg, {}).get(n, 0.0), 2)
                          for n in leg.ranked()},
            'retention': round(leg.retention, 6),
            'reserves': leg.reserves[:2],
            'horses': {str(n): v for n, v in check.coverage.get(leg.leg, {}).items()},
        } for leg in plan.legs],
        'ok': check.ok,
        'warnings': check.warnings,
        'recipe': abc_recipe(plan, check.units),
    }
