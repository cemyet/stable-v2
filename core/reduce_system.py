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
We deal the stake out in passes: pass one lays a single unit on as many
rows as the weights reach, strongest row first, and later passes can only
deepen rows pass one already chose — a row it could not afford then is one
it can never afford, because budgets only shrink.  So the system comes out
broad first and stacked second, and the stack lands on the rows the user
rated highest.

Dealing it that way also makes the system very nearly *monotone*: swap any
horse on a played row for one the user rated higher and you should land on
another played row carrying at least as much stake.  Without that, a coupon
can pay out on the longshot and not on the favourite standing next to it,
which is what the old proportional recursion did to roughly a third of its
rows.

A handful can still slip through, because perfect monotonicity is the
downward-closed condition again and so is fenced off by the same result:
strongest-first order means a row is only ever refused after its stronger
neighbour, but if the stronger horse runs out of budget first, its
neighbour can still be affordable later.  `verify()` counts what is left —
in practice single digits on a few thousand rows, against the thousand-plus
of the old scheme.

Nothing here talks to Flask or the database; `web/app.py` wraps it.
"""

from __future__ import annotations

import heapq
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
# actually accepts.  A row that wants seven units goes in as a 5 and a 2.
ATG_MULTIPLIERS = (100, 50, 20, 10, 5, 2, 1)

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


# ---------------------------------------------------------------------------
# Stage 2 — rows to boxes
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


def _row_stream(ranked: list[list[int]],
                logs: list[list[float]]) -> Iterator[tuple[int, ...]]:
    """Every row of the grid, strongest first.

    A row's strength is the sum of its horses' `logs` — the log of the
    product of their shares, once the ordering has been tuned.  Sorting the
    grid would mean building all of it and a plan can span millions of rows,
    so this is a best-first walk instead: start on the favourites, and each
    time a row comes out, offer up the rows one demotion away from it.  Only
    the part of the grid we actually reach is ever materialised.

    Ties break on the rank tuple, which orders a row before anything it
    beats — without that, two rows of equal strength could come out in the
    wrong order and split a system straight down the middle of a tie.
    """
    def strength(state: tuple[int, ...]) -> float:
        return sum(logs[i][j] for i, j in enumerate(state))

    start = tuple(0 for _ in ranked)
    heap = [(-strength(start), start)]
    seen = {start}
    while heap:
        _, state = heapq.heappop(heap)
        yield tuple(ranked[i][j] for i, j in enumerate(state))
        for i, j in enumerate(state):
            if j + 1 >= len(ranked[i]):
                continue
            nxt = state[:i] + (j + 1,) + state[i + 1:]
            if nxt in seen:
                continue
            seen.add(nxt)
            heapq.heappush(heap, (-strength(nxt), nxt))


def deal_units(plan: Plan, units: int) -> dict[tuple[int, ...], int]:
    """Spread `units` stake units over the system's rows, broad side first.

    Pass one walks the grid strongest first and puts a single unit on every
    row all of whose horses still have units left in their budget.  A row it
    turns down there it can never take, since budgets only shrink, so the
    passes after it only ever deepen rows pass one already chose.  That is
    what keeps the coupon wide: nothing is bought twice until everything
    affordable has been bought once.

    A horse's budget is exactly the share the user gave it, so the weights
    come out exact.
    """
    ranked = [leg.ranked() for leg in plan.legs]
    logs = [[math.log(max(leg.shares()[h], 1e-9)) for h in ranked[i]]
            for i, leg in enumerate(plan.legs)]
    budget = [dict(integer_targets(leg, units)) for leg in plan.legs]

    def afford(row: tuple[int, ...]) -> bool:
        return all(budget[i][h] > 0 for i, h in enumerate(row))

    def take(row: tuple[int, ...]) -> None:
        for i, h in enumerate(row):
            budget[i][h] -= 1

    out: dict[tuple[int, ...], int] = {}
    live: list[tuple[int, ...]] = []
    left = units
    for row in _row_stream(ranked, logs):
        if left <= 0:
            break
        if afford(row):
            take(row)
            out[row] = 1
            live.append(row)
            left -= 1

    # Budgets only ever shrink, so a row nobody can afford now is one nobody
    # will afford later — dropping it keeps the sweeps from re-reading a
    # grid that is mostly dead.
    while left > 0:
        took = 0
        keep: list[tuple[int, ...]] = []
        for row in live:
            if not afford(row):
                continue
            keep.append(row)
            if left <= 0:
                continue
            take(row)
            out[row] += 1
            left -= 1
            took += 1
        live = keep
        if took:
            continue
        # No row can carry the rest at the shares asked for. Rather than
        # fail a coupon the user has already paid for, put the remainder on
        # the strongest rows and let `verify` report the drift.
        for row in sorted(out, key=lambda r: -out[r])[:left]:
            out[row] += 1
            left -= 1
        break
    return out


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

    # An unweighted system at full size is every row played once, which is a
    # single coupon — worth taking straight rather than walking the grid to
    # rediscover it, since that grid can run to millions of rows.
    targets = [integer_targets(leg, want) for leg in plan.legs]
    if want == plan.full_rows and all(len(set(t.values())) == 1
                                      for t in targets):
        return [Box(tuple(tuple(sorted(leg.picks)) for leg in plan.legs))]

    if plan.full_rows > _MAX_ROWS:
        raise ReduceError(
            f'a weighted system over {plan.full_rows} rows is too big to '
            f'build')

    return [Box(tuple((h,) for h in row), n)
            for row, n in deal_units(plan, want).items()]


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
    """Coverage in, submittable boxes out."""
    return split_multipliers(merge_boxes(allocate(plan)))


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


def monotonicity_gaps(plan: Plan,
                      units: dict[tuple[int, ...], int]) -> list[tuple]:
    """Rows carrying less stake than a row they beat.

    Swapping a horse for one the user rated *higher* can only make a row
    more wanted, so it must never come back with less stake on it.  Every
    gap this finds is a system that pays out on the longshot and not on the
    favourite standing next to it.
    """
    stronger = []
    for leg in plan.legs:
        share = leg.shares()
        stronger.append({h: [g for g in leg.picks if share[g] > share[h]]
                         for h in leg.picks})
    out = []
    for row, n in units.items():
        for i, h in enumerate(row):
            for g in stronger[i][h]:
                alt = row[:i] + (g,) + row[i + 1:]
                if units.get(alt, 0) < n:
                    out.append((plan.legs[i].leg, g, h, alt))
    return out


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
    for i, leg in enumerate(plan.legs):
        targets = integer_targets(leg, actual_units)
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

    # Breadth and shape. Neither can fail a build — piling stake on the
    # favourites' rows is what a weighted coupon is for — but both are worth
    # saying out loud, because a coupon that spans 2268 rows and covers 1173
    # of them does not look like one from the outside.
    units = row_units(boxes)
    if units is not None:
        distinct = len(units)
        deepest = max(units.values(), default=0)
        gaps = monotonicity_gaps(plan, units)
        if gaps:
            leg, strong, weak, _ = gaps[0]
            warnings.append(
                f'{len(gaps)} rows carry less than a row they beat — leg '
                f'{leg} horse {strong} behind {weak}')
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
        # Weighting buys some rows more than once, so the stake spans fewer
        # combinations than it has rows. Worth showing: it is the difference
        # between what the coupon costs and what it actually covers.
        'distinctRows': check.distinct_rows,
        'deepestRow': check.deepest_row,
        'retention': round(plan.retention, 6),
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
