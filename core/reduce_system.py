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

WHICH rows get the units
------------------------
The margins fix how much each horse gets but not which horses share a row.
We allocate proportionally and recursively, leg by leg, which makes the
units land on rows in proportion to the product of their coverages — the
favourites end up together and the longshots stay rare, the same shape
ATG's points reduction produces.

Nothing here talks to Flask or the database; `web/app.py` wraps it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date as _date, datetime as _datetime
from typing import Iterable, Sequence

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
# actually accepts.  A row that wants seven units goes in as a 5 and a 2.
ATG_MULTIPLIERS = (100, 50, 20, 10, 5, 2, 1)

# `couponid` is a 1..9999 sequence number, and ATG rejects files with more
# systems than the cap below.  It also treats a file made of one-row coupons
# as robot play, so merging rows into boxes is not optional.
ATG_MAX_SYSTEMS = 5000
ATG_MAX_COUPON_ID = 9999

# Guard rails for the allocation. A coupon large enough to trip these cannot
# be submitted as a file anyway.
_MAX_UNITS = 2_000_000
_MAX_BOXES = 200_000


# ---------------------------------------------------------------------------
# Input
# ---------------------------------------------------------------------------
@dataclass
class Leg:
    """One leg of the plan, as the game page drew it.

    `coverage` is percent of the leg's unreduced whole, so it sums to 100
    when nothing is reduced and to less when it is.  `reserves` are program
    numbers of horses *not* on the coupon, best first.
    """
    leg: int
    race_id: str
    picks: list[int]
    coverage: dict[int, float]
    reserves: list[int] = field(default_factory=list)
    starters: int = 0

    @property
    def retention(self) -> float:
        total = sum(self.coverage.get(n, 0.0) for n in self.picks)
        if total <= 0:
            return 1.0
        return max(0.0, min(1.0, total / 100.0))

    def ranked(self) -> list[int]:
        """Picks strongest first — the order everything else works in."""
        return sorted(self.picks, key=lambda n: (-self.coverage.get(n, 0.0), n))


@dataclass
class Plan:
    game_type: str
    legs: list[Leg]
    line_price: float = 0.0
    game_date: str = ''
    track_code: int | None = None
    track_name: str = ''

    @property
    def full_rows(self) -> int:
        rows = 1
        for leg in self.legs:
            rows *= max(1, len(leg.picks))
        return rows

    @property
    def retention(self) -> float:
        r = 1.0
        for leg in self.legs:
            r *= leg.retention
        return r

    @property
    def rows(self) -> int:
        return max(1, round(self.full_rows * self.retention))


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

    short = units - sum(out.values())
    for n in sorted(picks, key=lambda n: (-(raw[n] - math.floor(raw[n])), n)):
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


def _split_units(targets: list[int], take: int) -> list[int]:
    """Take exactly `take` units out of `targets`, proportionally.

    Used to hand a slice of every remaining leg down to one branch of the
    allocation, so siblings share the parent's units without drift.
    """
    total = sum(targets)
    if take <= 0:
        return [0] * len(targets)
    if take >= total:
        return list(targets)

    raw = [t * take / total for t in targets]
    out = [min(t, int(math.floor(x))) for t, x in zip(targets, raw)]
    short = take - sum(out)
    if short <= 0:
        return out

    order = sorted(range(len(targets)),
                   key=lambda i: (-(raw[i] - math.floor(raw[i])), -targets[i], i))
    while short > 0:
        moved = False
        for i in order:
            if short <= 0:
                break
            if out[i] < targets[i]:
                out[i] += 1
                short -= 1
                moved = True
        if not moved:  # cannot happen while take <= total, but never spin
            break
    return out


# ---------------------------------------------------------------------------
# Stage 2 — units to boxes
# ---------------------------------------------------------------------------
@dataclass
class Box:
    """A mathematical coupon: one set of horses per leg, played `multiplier`
    times.  Covers `prod(len(sets))` rows."""
    sets: tuple[tuple[int, ...], ...]
    multiplier: int = 1

    @property
    def rows(self) -> int:
        n = 1
        for s in self.sets:
            n *= len(s)
        return n

    @property
    def units(self) -> int:
        return self.rows * self.multiplier


def _uniform_box(order: list[list[int]], deficits: list[list[int]],
                 units: int) -> tuple[tuple[tuple[int, ...], ...], int] | None:
    """Can the rest of the problem be one box?

    It can when every remaining leg spreads its units evenly over the horses
    that still want any, and the resulting box divides the units cleanly.
    This is what collapses an unreduced, unweighted tail into a single
    coupon instead of enumerating its rows.
    """
    sets: list[tuple[int, ...]] = []
    size = 1
    for horses, want in zip(order, deficits):
        live = [h for h, d in zip(horses, want) if d > 0]
        if not live:
            return None
        first = want[horses.index(live[0])]
        for h in live:
            if want[horses.index(h)] != first:
                return None
        sets.append(tuple(sorted(live)))
        size *= len(live)

    if size <= 0 or units % size:
        return None
    return tuple(sets), units // size


def allocate(plan: Plan, units: int | None = None,
             leg_order: Sequence[int] | None = None) -> list[Box]:
    """Spread the system's stake units over rows and return them as boxes.

    Walks the legs in order.  At each leg the units are handed to the picks
    in the amounts Stage 1 worked out, and every *other* leg's remaining
    units are split proportionally so each branch keeps consistent margins.
    Recursing that way concentrates units on rows whose horses are all
    well covered, which is the shape a reduced system should have.

    `leg_order` only changes which leg the recursion splits first, never the
    result's margins — but it changes how well the boxes merge afterwards,
    so `solve` tries a few.  Boxes always come back in the plan's leg order.
    """
    if not plan.legs:
        raise ReduceError('plan has no legs')
    for leg in plan.legs:
        if not leg.picks:
            raise ReduceError(f'leg {leg.leg} has no picks')

    total = int(units if units is not None else plan.rows)
    if total <= 0:
        raise ReduceError('system has no rows')
    if total > _MAX_UNITS:
        raise ReduceError(f'system is too large to build ({total} rows)')

    walk = list(leg_order) if leg_order is not None else list(range(len(plan.legs)))
    order = [plan.legs[i].ranked() for i in walk]
    targets = [integer_targets(plan.legs[i], total) for i in walk]
    root = [[targets[i][h] for h in order[i]] for i in range(len(walk))]

    out: list[Box] = []
    _distribute(order, 0, total, root, (), out)

    # Put the legs back where they belong.
    if leg_order is not None:
        back = [0] * len(walk)
        for slot, leg_i in enumerate(walk):
            back[leg_i] = slot
        out = [Box(tuple(box.sets[back[i]] for i in range(len(walk))),
                   box.multiplier)
               for box in out]
    return out


def _leg_orders(plan: Plan) -> list[list[int]]:
    """Candidate recursion orders, cheapest structure first.

    Splitting the legs that have the fewest distinct coverage levels first
    leaves the varied ones intact deeper down, where they merge back into
    wider boxes; in practice that is worth about a third of the coupon
    count. The rest are here because no single rule wins on every system.
    """
    idx = list(range(len(plan.legs)))

    def levels(i: int) -> int:
        leg = plan.legs[i]
        return len({round(leg.coverage.get(n, 0.0), 1) for n in leg.picks})

    return [
        sorted(idx, key=lambda i: (levels(i), len(plan.legs[i].picks), i)),
        sorted(idx, key=lambda i: (len(plan.legs[i].picks), levels(i), i)),
        sorted(idx, key=lambda i: (-levels(i), -len(plan.legs[i].picks), i)),
        idx,
    ]


def _distribute(order: list[list[int]], pos: int, units: int,
                deficits: list[list[int]], prefix: tuple[tuple[int, ...], ...],
                out: list[Box]) -> None:
    if units <= 0:
        return
    if len(out) > _MAX_BOXES:
        raise ReduceError('system is too fragmented to build')

    uniform = _uniform_box(order[pos:], deficits[pos:], units)
    if uniform is not None:
        sets, mult = uniform
        out.append(Box(prefix + sets, mult))
        return

    horses = order[pos]
    want = deficits[pos]
    rest = deficits[pos + 1:]

    for idx, horse in enumerate(horses):
        take = want[idx]
        if take <= 0:
            continue
        # Every remaining leg gives up the same number of units, so the
        # child's legs all agree on how many rows it is building.
        child: list[list[int]] = []
        for leg_i, vec in enumerate(rest):
            slice_ = _split_units(vec, take)
            child.append(slice_)
            rest[leg_i] = [a - b for a, b in zip(vec, slice_)]
        _distribute(order, pos + 1, take, deficits[:pos + 1] + child,
                    prefix + ((horse,),), out)


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
                key = (box.multiplier,
                       box.sets[:leg], box.sets[leg + 1:])
                buckets.setdefault(key, []).append(box)

            merged: list[Box] = []
            for (mult, head, tail), group in buckets.items():
                if len(group) == 1:
                    merged.append(group[0])
                    continue
                union: set[int] = set()
                for box in group:
                    union |= set(box.sets[leg])
                merged.append(Box(head + (tuple(sorted(union)),) + tail, mult))
                changed = True
            current = merged

    current.sort(key=lambda b: (-b.units, b.sets))
    return current


def split_multipliers(boxes: Sequence[Box]) -> list[Box]:
    """Rewrite multipliers ATG will not accept as a sum of ones it will."""
    out: list[Box] = []
    for box in boxes:
        left = box.multiplier
        if left in ATG_MULTIPLIERS:
            out.append(box)
            continue
        for step in ATG_MULTIPLIERS:
            while left >= step:
                out.append(Box(box.sets, step))
                left -= step
        if left:  # unreachable: 1 is in the table
            out.append(Box(box.sets, left))
    return out


def solve(plan: Plan) -> list[Box]:
    """Coverage in, submittable boxes out — the fewest coupons we can find."""
    best: list[Box] | None = None
    seen: set[tuple[int, ...]] = set()
    for order in _leg_orders(plan):
        key = tuple(order)
        if key in seen:
            continue
        seen.add(key)
        boxes = split_multipliers(merge_boxes(allocate(plan, leg_order=order)))
        if best is None or len(boxes) < len(best):
            best = boxes
    assert best is not None  # _leg_orders always yields at least one order
    return best


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


def verify(plan: Plan, boxes: Sequence[Box]) -> Verification:
    """Recompute the margins from the boxes and compare with the plan."""
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

    for i, leg in enumerate(plan.legs):
        targets = integer_targets(leg, expected)
        rows: dict[int, dict[str, float]] = {}
        for horse in leg.ranked():
            target = targets.get(horse, 0)
            got = per_leg[i].get(horse, 0)
            rows[horse] = {
                'target': target,
                'actual': got,
                'coverage': round(got / actual_units * 100, 2) if actual_units else 0.0,
                'wanted': round(leg.coverage.get(horse, 0.0) / leg.retention, 2)
                if leg.retention else 0.0,
            }
            if got != target:
                ok = False
                warnings.append(
                    f'leg {leg.leg} horse {horse}: {got} rows, wanted {target}')
        coverage[leg.leg] = rows

    if len(boxes) > ATG_MAX_SYSTEMS:
        ok = False
        warnings.append(
            f'{len(boxes)} coupons — ATG accepts at most {ATG_MAX_SYSTEMS} '
            f'per file')

    return Verification(ok=ok, units=actual_units, expected_units=expected,
                        boxes=len(boxes), warnings=warnings, coverage=coverage)


# ---------------------------------------------------------------------------
# Stage 4 — the ATG file
# ---------------------------------------------------------------------------
def crc16(data: bytes) -> int:
    """CRC-16/CCITT-FALSE — poly 0x1021, init 0xFFFF, no reflection.

    ATG's file-betting spec points at the Princeton `CRC16.java` reference
    for this; the four hex digits go on the end of the file name so the
    upload can tell whether the file was edited on the way.
    """
    crc = 0xFFFF
    for byte in data:
        crc ^= byte << 8
        for _ in range(8):
            crc = ((crc << 1) ^ 0x1021) & 0xFFFF if crc & 0x8000 else (crc << 1) & 0xFFFF
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

    body: list[str] = []
    for i, box in enumerate(boxes, start=1):
        body.append(
            f'    <{element} couponid="{i}" date="{game_date}"{track_attr}'
            f' betmultiplier="{box.multiplier}">\n')
        for leg_i, leg in enumerate(plan.legs):
            marks = marks_string(box.sets[leg_i], width)
            reserves = ''
            if leg.reserves:
                reserves += f' r1="{leg.reserves[0]}"'
                if len(leg.reserves) > 1:
                    reserves += f' r2="{leg.reserves[1]}"'
            body.append(
                f'      <leg legno="{leg.leg}" marks="{marks}"{reserves}/>\n')
        body.append(f'    </{element}>\n')

    return head + ''.join(body) + '  </betcoupons>\n</issuer>\n'


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

    return {
        'gameType': plan.game_type,
        'gameDate': plan.game_date,
        'trackName': plan.track_name,
        'trackCode': plan.track_code,
        'fullRows': plan.full_rows,
        'rows': check.units,
        'retention': round(plan.retention, 6),
        'removedRows': max(0, plan.full_rows - check.units),
        'linePrice': price,
        'cost': round(check.units * price, 2),
        'fullCost': round(plan.full_rows * price, 2),
        'coupons': [{
            'legs': [{'leg': plan.legs[i].leg, 'numbers': list(s)}
                     for i, s in enumerate(box.sets)],
            'rows': box.rows,
            'multiplier': box.multiplier,
            'units': box.units,
        } for box in boxes],
        'couponCount': len(boxes),
        'legs': [{
            'leg': leg.leg,
            'raceId': leg.race_id,
            'picks': leg.ranked(),
            'coverage': {str(n): round(leg.coverage.get(n, 0.0), 2)
                         for n in leg.ranked()},
            'retention': round(leg.retention, 6),
            'reserves': leg.reserves[:2],
            'horses': {str(n): v for n, v in check.coverage.get(leg.leg, {}).items()},
        } for leg in plan.legs],
        'ok': check.ok,
        'warnings': check.warnings,
        'recipe': abc_recipe(plan, check.units),
    }
