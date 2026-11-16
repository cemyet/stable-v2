"""
Self-check for the reduced-system solver in `core/reduce_system.py`.

    python3 -m scripts.check_reduce_system

There is no test runner in this repo, so this is a plain script that asserts
its way through the properties that matter and prints what it found. It needs
no database and no network — the vendored XSD is read off disk.

What it checks:

  1. CRC16 against the Princeton CRC16.java reference vector (`123456789` →
     `0xBB3D`), since the file name ATG validates is nothing but that checksum.
  2. A plain unweighted system collapses to a single coupon (the merge is not
     silently giving up).
  3. A weighted + reduced V85 — the shape the game page produces — comes back
     with the row count the coupon sphere shows and per-horse margins within
     half a percentage point of the coverage the user drew.
  4. The generated XML validates against ATG's schema. The vendored copy is
     1.8.4, which has no `v85Coupon`; V85 is written as its structural twin
     `v86Coupon` for the check only. See core/vendor/README.md.
  5. Heavy weighting still balances — the units land on repeated rows via
     `betmultiplier` rather than being quietly dropped.
  6. No row carries less stake than a row it beats, so a coupon never pays
     out on the longshot and not on the favourite standing next to it.
  7. Stake follows the weights in proportion. The old dealer laid one unit
     on as many rows as it could and stacked the rest on the corner of the
     grid, a hundred units deep; every row here sits within a unit or so of
     its share.
  8. A thin coupon with three levers aimed at one race keeps its order and
     its depth. Coupon 178's 4896th-ranked row once carried 17 units for no
     stated reason.
  9. An aim moves its lever's stake onto the horses it names, and only on
     rows the lever's horse is on.
 10. There is no cap on how many levers a coupon can aim. Past 64 readings
     the old builder quietly ignored every aim.
 11. A cascade opens its race to outsiders only on rows its lever's horse is
     on, tapering down the race's rank; a lever can open one race, and two
     levers may open the same one. A coupon that marks one of the leg's
     reserves names the next horse down the rank instead.
 12. Top-pool play. A whole V64 goes in as one `v6="true"` coupon; a pool
     lever's reading goes on V6 coupons of its own, on its horse's rows
     only; V64 multipliers stay within the values its schema accepts; a
     game without a top pool refuses.
 13. The coupon-wide scale. A lever on an even coupon keeps every row it
     had, x3 plays it three times, a weighted, reduced coupon cut or
     levered as a whole still lands its shares, and scaling builds the same
     rows as cutting or levering the banker by as much.
"""

from __future__ import annotations

import math
import re
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from core import reduce_system as rs

XSD_PATH = _ROOT / 'core' / 'vendor' / 'atg_filebetting_1_8_4.xsd'

# The V85 the summary page was built against: eight legs, two of them reduced
# (leg 3's 9 capped at 12, leg 6's 12 capped at 10), the rest weighted.
V85_SPEC = [
    ([3],         {3: 100}),
    ([2, 7],      {2: 70, 7: 30}),
    ([1, 4, 9],   {1: 50, 4: 30, 9: 12}),
    ([5, 6, 8, 11], {5: 40, 6: 25, 8: 20, 11: 15}),
    ([2, 10],     {2: 60, 10: 40}),
    ([1, 3, 12],  {1: 45, 3: 35, 12: 10}),
    ([4, 8],      {4: 50, 8: 50}),
    ([6, 7, 9],   {6: 40, 7: 35, 9: 25}),
]


def make_plan(spec, game_type='V85', line_price=0.5):
    legs = [
        rs.Leg(leg=i, race_id=f'2026-08-19_5_{i + 3}', picks=picks,
               coverage=coverage, reserves=[13, 14], starters=12)
        for i, (picks, coverage) in enumerate(spec, start=1)
    ]
    return rs.Plan(game_type=game_type, legs=legs, line_price=line_price,
                   game_date='2026-08-19', track_code=5, track_name='Solvalla')


def weighted_only(spec):
    """The same opinions with nothing capped — a coupon at its full price."""
    out = []
    for picks, coverage in spec:
        total = sum(coverage.values())
        out.append((picks, {n: v * 100 / total for n, v in coverage.items()}))
    return out


def worst_miss(check):
    return max(abs(fig['actual'] - fig['target'])
               for horses in check.coverage.values()
               for fig in horses.values())


def slack(units):
    return max(1, math.ceil(units * rs.MARGIN_TOLERANCE))


def check_crc():
    got = rs.crc16(b'123456789')
    assert got == 0xBB3D, f'CRC-16/ARC mismatch: {got:#06x}'
    print(f'  crc16("123456789") = {got:#06X}')


def check_single_box():
    even = [(picks, {n: 100 / len(picks) for n in picks})
            for picks, _ in V85_SPEC]
    plan = make_plan(even)
    boxes = rs.solve(plan)
    assert len(boxes) == 1, f'unweighted system split into {len(boxes)} coupons'
    assert boxes[0].units == plan.full_rows
    print(f'  unweighted {plan.full_rows} rows -> 1 coupon')


def check_margins():
    plan = make_plan(V85_SPEC)
    boxes = rs.solve(plan)
    check = rs.verify(plan, boxes)

    assert check.units == plan.rows, (
        f'built {check.units} rows, coupon sphere says {plan.rows}')
    assert check.ok, check.warnings
    miss = worst_miss(check)
    assert miss <= slack(check.units), (miss, check.warnings)

    print(f'  {plan.full_rows} full rows, {plan.rows} after reduction '
          f'({plan.retention:.3f} retention)')
    print(f'  {len(boxes)} coupons, every horse within {miss} of '
          f'{check.units} units of its share')
    return plan, boxes


def check_xml(plan, boxes):
    xml = rs.atg_xml(plan, boxes)
    name = rs.atg_filename(plan.game_type, plan.game_date, xml)
    assert name.endswith('.xml') and len(name.split('_')[-1]) == 8, name

    marked = sum(line.count('marks="') for line in xml.splitlines())
    assert marked == len(boxes) * len(plan.legs)

    try:
        from lxml import etree
    except ImportError:
        print('  lxml missing — skipped schema validation')
        return

    # 1.8.4 has no v85Coupon; v86Coupon is the same shape, so validate as that.
    schema = etree.XMLSchema(etree.parse(str(XSD_PATH)))
    as_v86 = xml.replace('v85Coupon', 'v86Coupon')
    doc = etree.fromstring(as_v86.encode('utf-8'))
    if not schema.validate(doc):
        for err in schema.error_log:
            print(f'    line {err.line}: {err.message}')
        raise AssertionError('generated XML does not validate')

    print(f'  {name} validates against {XSD_PATH.name} (as v86Coupon)')


def check_heavy_weighting():
    # 60/15/10/8/7 in every leg of an unreduced V85: the favourites want far
    # more rows than there are distinct combinations, so the solver has to
    # reach for multipliers.
    spec = [([1, 2, 3, 4, 5], {1: 60, 2: 15, 3: 10, 4: 8, 5: 7})
            for _ in range(8)]
    plan = make_plan(spec)
    boxes = rs.solve(plan)
    check = rs.verify(plan, boxes)

    assert check.units == plan.rows, (check.units, plan.rows)
    assert max(b.multiplier for b in boxes) > 1, 'expected repeated rows'
    assert all(b.multiplier in rs.ATG_MULTIPLIERS for b in boxes)
    # Too big to submit, and it should say so rather than pretend.
    assert any('at most' in w for w in check.warnings), check.warnings
    print(f'  heavy weighting: {check.units} rows balanced across '
          f'{len(boxes)} coupons, flagged as too many to submit')


def check_monotone():
    for label, spec in (('weighted', weighted_only(V85_SPEC)),
                        ('weighted + reduced', V85_SPEC)):
        plan = make_plan(spec)
        boxes = rs.solve(plan)
        check = rs.verify(plan, boxes)
        assert check.gaps == 0, (label, check.warnings)
        strict = rs.monotonicity_gaps(plan, rs.row_units(boxes), strict=True)
        assert not strict, (label, len(strict), strict[0])
        print(f'  {label}: no row behind one it beats, across all '
              f'{check.distinct_rows} rows carrying stake')

    # The reported case in miniature: back the favourite in a leg and the
    # rows you get must be a superset of what backing the longshot gets.
    plan = make_plan(weighted_only(V85_SPEC))
    units = rs.row_units(rs.solve(plan))
    leg, strong, weak = 1, 2, 7            # leg 2 is 70/30 on horses 2 and 7
    on_weak = {row[:leg] + row[leg + 1:]
               for row in units if row[leg] == weak}
    on_strong = {row[:leg] + row[leg + 1:]
                 for row in units if row[leg] == strong}
    missing = on_weak - on_strong
    assert not missing, (
        f'{len(missing)} rows play horse {weak} but not horse {strong}, '
        f'e.g. {sorted(missing)[0]}')
    print(f'  backing horse {strong} ({70}%) covers every rest that horse '
          f'{weak} ({30}%) does, and {len(on_strong) - len(on_weak)} more')


def check_proportional():
    for label, spec in (('weighted', weighted_only(V85_SPEC)),
                        ('weighted + reduced', V85_SPEC)):
        plan = make_plan(spec)
        check = rs.verify(plan, rs.solve(plan))
        units = rs.row_units(rs.solve(plan))
        stake = rs.intended_stake(plan, check.units)
        off = max(abs(units.get(row, 0) - x) for row, x in stake.items())
        corner = max(stake, key=stake.get)
        # A row keeps the floor of its share or rises by one; the margin fit
        # may push a share past the next whole unit, never far.
        assert off < 1.5, (label, off)
        assert check.deepest_row <= math.ceil(stake[corner]) + 1, (
            label, check.deepest_row, stake[corner])
        unplayed = [row for row, x in stake.items()
                    if x >= 1 and units.get(row, 0) == 0]
        assert not unplayed, (label, len(unplayed), unplayed[0])
        print(f'  {label}: every row within {off:.2f} units of its share; '
              f'the corner carries {units[corner]} for a share of '
              f'{stake[corner]:.2f}')


def check_aimed_coupon():
    coverage = [
        {2: 100.0},
        {5: 39.5083, 6: 21.8091, 7: 21.8091},
        {1: 9.2208, 3: 14.2874, 4: 3.426, 6: 10.504, 7: 10.504,
         8: 6.847, 10: 12.0539, 11: 9.2208, 12: 5.0448, 14: 6.847},
        {3: 100.0},
        {1: 17.1033, 5: 17.1033, 6: 7.1428, 9: 7.1428,
         10: 17.1033, 11: 7.1428, 12: 7.1428},
        {2: 14.4957, 4: 25.0, 8: 12.5, 12: 12.5},
        {5: 33.3333, 7: 39.3871, 9: 44.89},
        {6: 50.0, 15: 64.5813},
    ]
    orders = [None, None, [3, 10, 6, 7, 1, 11, 14, 8, 12, 4],
              None, None, None, None, None]
    legs = [
        rs.Leg(leg=i + 1, race_id=f'2026-09-19_15_{i + 5}',
               picks=sorted(fig), coverage=fig, order=orders[i])
        for i, fig in enumerate(coverage)
    ]
    levers = [
        rs.Lever(7, 7, 6.0538, {
            6: {2: 37.4905, 4: 20.2029, 8: 21.1533, 12: 21.1533},
            8: {6: 34.9664, 15: 65.0336},
        }),
        rs.Lever(7, 9, 11.5567, {
            6: {2: 22.971, 4: 34.624, 8: 21.2025, 12: 21.2025},
        }),
        rs.Lever(8, 15, 14.5813, {
            6: {2: 29.3023, 4: 36.0289, 8: 17.3344, 12: 17.3344},
        }),
    ]
    plan = rs.Plan('V85', legs, 0.5, '2026-09-19', 15, 'Färjestad',
                   levers)
    boxes = rs.solve(plan)
    check = rs.verify(plan, boxes)
    units = rs.row_units(boxes)
    stake = rs.intended_stake(plan, check.units)
    reported = (2, 7, 12, 3, 12, 12, 5, 6)

    assert check.units == 2558, check.units
    assert check.gaps == 0, check.warnings
    assert units.get(reported, 0) <= 1, units[reported]
    assert check.deepest_row <= math.ceil(max(stake.values())) + 1, (
        check.deepest_row, max(stake.values()))
    # Every row is on one unit or none, and the three levers all reshape
    # leg 6, so holding the order costs that leg's margins more than the
    # half point. verify says so.
    miss = worst_miss(check)
    assert miss <= check.units * 0.03, (miss, check.warnings)
    if miss > slack(check.units):
        assert not check.ok and any('wanted' in w for w in check.warnings)
    print(f'  former 17-unit sink carries {units.get(reported, 0)}; deepest '
          f'row {check.deepest_row}; worst margin {miss} of {check.units} '
          f'units{" (flagged)" if miss > slack(check.units) else ""}')


def _aim_plan():
    """Horse 1 in leg 1 is levered by 20 and aimed at the outsider of leg 2."""
    rest = {1: 50, 2: 30, 3: 20}
    legs = [
        rs.Leg(1, 'a', [1, 2], {1: 70, 2: 30}),
        rs.Leg(2, 'b', [3, 4, 5], {3: 20, 4: 40, 5: 40}, order=[4, 5, 3]),
    ] + [rs.Leg(i, f'r{i}', [1, 2, 3], dict(rest)) for i in range(3, 7)]
    levers = [rs.Lever(leg=1, horse=1, amount=20, targets={2: {3: 100}})]
    return rs.Plan('V64', legs, 1.0, '2026-08-19', 5, 'Solvalla', levers)


def check_aim_follows_lever():
    plan = _aim_plan()
    # With horse 1 the 50 base reads 20/40/40 and the 20 lever all on 3:
    # (50 x 0.2 + 20) / 70. Without it leg 2 stays as drawn.
    with_1 = (50 * 0.2 + 20) / 70
    shown = plan.effective_coverage()[2]
    assert abs(shown[3] - (0.7 * with_1 + 0.3 * 0.2) * 100) < 1e-6, shown

    boxes = rs.solve(plan)
    check = rs.verify(plan, boxes)
    units = rs.row_units(boxes)
    assert check.ok and check.gaps == 0, check.warnings

    def share_of_3(first):
        on = sum(n for row, n in units.items() if row[0] == first)
        return sum(n for row, n in units.items()
                   if row[0] == first and row[1] == 3) / on

    assert abs(share_of_3(1) - with_1) < 0.02, share_of_3(1)
    assert abs(share_of_3(2) - 0.2) < 0.02, share_of_3(2)
    print(f'  leg 2 horse 3 on {share_of_3(1):.1%} of horse 1 rows '
          f'(aimed, {with_1:.1%} meant) and {share_of_3(2):.1%} of horse 2 '
          f'rows (as drawn)')


def check_no_cap():
    # Eight levers, each aimed at the next leg's outsider: 256 readings on
    # the strongest row.
    legs = [rs.Leg(i, f'r{i}', [1, 2, 3], {1: 50, 2: 30, 3: 20})
            for i in range(1, 9)]
    levers = [rs.Lever(leg=i, horse=1, amount=10,
                       targets={i % 8 + 1: {3: 100}})
              for i in range(1, 9)]
    plan = rs.Plan('V85', legs, 0.5, '2026-08-19', 5, 'Solvalla', levers)
    shown = plan.effective_coverage()
    lifted = min(shown[i][3] for i in range(1, 9))
    assert lifted > 25, shown

    boxes = rs.solve(plan)
    check = rs.verify(plan, boxes)
    assert check.units == plan.rows and check.gaps == 0, check.warnings
    assert check.ok, check.warnings
    print(f'  8 aimed levers: every outsider shown at {lifted:.1f}% '
          f'(drawn 20%), {check.units} units built')


def _cascade_plan(*cascades):
    """A V64 whose leg 2 has four picks out of a ten-horse rank."""
    fan = {n: 1.0 for n in range(1, 11)}
    rest = {1: 50, 2: 30, 3: 20}
    legs = [
        rs.Leg(1, 'a', [1, 2], {1: 60, 2: 40}),
        rs.Leg(2, 'b', [1, 2, 3, 4], {n: 25 for n in range(1, 5)},
               order=list(range(1, 11)), starters=10),
        rs.Leg(3, 'c', [1, 2, 3], {1: 60, 2: 25, 3: 15},
               order=list(range(1, 9)), starters=8),
    ] + [rs.Leg(i, f'r{i}', [1, 2, 3], dict(rest)) for i in range(4, 7)]
    levers = [
        rs.Lever(leg=leg, horse=1, amount=20,
                 targets={t: dict(fan) for t in opens},
                 cascade_targets=frozenset(opens))
        for leg, opens in cascades
    ]
    return rs.Plan('V64', legs, 1.0, '2026-08-19', 5, 'Solvalla', levers)


def check_cascade():
    plan = _cascade_plan((1, [2]))
    boxes = rs.solve(plan)
    check = rs.verify(plan, boxes)
    units = rs.row_units(boxes)
    assert check.ok and check.gaps == 0, check.warnings

    opened = [row for row in units if row[1] > 4]
    assert opened, 'the cascade opened nothing'
    assert all(row[0] == 1 for row in opened), 'outsider without its lever'
    margin = {n: sum(u for row, u in units.items() if row[1] == n)
              for n in range(5, 11)}
    ranked = [margin[n] for n in range(5, 11)]
    assert ranked == sorted(ranked, reverse=True), margin
    # Lever 20 of 100 in leg 1, half of it on the outsiders.
    meant = check.units * 0.2 * 0.5
    assert abs(sum(ranked) - meant) <= slack(check.units) + 1, (ranked, meant)
    print(f'  outsiders 5-10 carry {ranked} ({sum(ranked)} of '
          f'{meant:.0f} meant), all on horse 1 rows')

    # Reserves are drawn from the rank too, so the cascade marks them.
    plan.legs[1].reserves = [5, 6]
    xml = rs.atg_xml(plan, boxes)
    swapped = 0
    for marks, r1, r2 in re.findall(
            r'legno="2" marks="([01]+)" r1="(\d+)" r2="(\d+)"', xml):
        marked = {k + 1 for k, c in enumerate(marks) if c == '1'}
        assert not marked & {int(r1), int(r2)}, (marks, r1, r2)
        swapped += (int(r1), int(r2)) != (5, 6)
    assert swapped, 'no coupon marked a reserve'
    print(f'  {swapped} coupons mark a reserve and take the next horse '
          f'down the rank instead')

    try:
        rs.solve(_cascade_plan((1, [2, 3])))
    except rs.ReduceError as err:
        assert 'a lever can open one race' in str(err), err
        print(f'  two races from one lever: {err}')
    else:
        raise AssertionError('a lever opening two races should not build')

    both = _cascade_plan((1, [2]), (3, [2]))
    boxes = rs.solve(both)
    check = rs.verify(both, boxes)
    units = rs.row_units(boxes)
    assert check.ok and check.gaps == 0, check.warnings
    opened = [row for row in units if row[1] > 4]
    assert opened and all(row[0] == 1 or row[2] == 1 for row in opened)
    print(f'  two levers opening leg 2: {len(opened)} outsider rows, each on '
          f'one of their horses')


def check_averaged_views():
    view = {1: 80, 2: 20}
    got = rs._averaged({1: 50, 2: 50}, [view, {1: 4, 2: 1}])
    assert all(abs(got[n] - view[n]) < 1e-9 for n in view), got
    print('  two levers with the same view give that view, not its square')


def check_track_code():
    plan = make_plan(V85_SPEC[:7], game_type='V75')
    xml = rs.atg_xml(plan, rs.solve(plan))
    assert 'trackcode' not in xml, 'V75 coupons do not carry a track code'

    plan = make_plan(V85_SPEC)
    plan.track_code = None
    try:
        rs.atg_xml(plan, rs.solve(plan))
    except rs.ReduceError:
        print('  trackcode required for V85, omitted for V75')
    else:
        raise AssertionError('V85 without a track code should not build')


def _schema_errors(xml):
    """Schema errors for a 1.8.4 document, or None when lxml is missing."""
    try:
        from lxml import etree
    except ImportError:
        return None
    schema = etree.XMLSchema(etree.parse(str(XSD_PATH)))
    if schema.validate(etree.fromstring(xml.encode('utf-8'))):
        return []
    return [f'line {err.line}: {err.message}' for err in schema.error_log]


def check_top_pool():
    # A whole mathematical V64 played as V6: one coupon, flagged.
    even = [([1, 2, 3], {n: 100 / 3 for n in (1, 2, 3)}) for _ in range(6)]
    plan = make_plan(even, game_type='V64', line_price=1.0)
    plan.top_only = True
    boxes = rs.solve(plan)
    assert len(boxes) == 1 and boxes[0].top_only, boxes
    xml = rs.atg_xml(plan, boxes)
    assert xml.count('v6="true"') == 1, xml
    errors = _schema_errors(xml)
    assert not errors, errors
    print(f'  V64 played as V6: {boxes[0].units} rows on one coupon, '
          f'v6="true"{"" if errors is None else ", validates"}')

    # The six-leg games take 3 and 4 but not 20.
    heavy = [([1, 2, 3, 4], {1: 70, 2: 15, 3: 10, 4: 5}) for _ in range(6)]
    plan = make_plan(heavy, game_type='V64', line_price=1.0)
    boxes = rs.solve(plan)
    allowed = rs.multipliers_for('V64')
    assert all(b.multiplier in allowed for b in boxes), sorted(
        {b.multiplier for b in boxes})
    assert sum(b.units for b in boxes) == plan.rows
    errors = _schema_errors(rs.atg_xml(plan, boxes))
    assert not errors, errors
    print(f'  heavy V64: multipliers {sorted({b.multiplier for b in boxes})}'
          f' — none of them 20')

    # A pool lever: horse 1's levered reading goes on V6 coupons of its own.
    plan = _aim_plan()
    plan.levers[0].pool = True
    boxes = rs.solve(plan)
    check = rs.verify(plan, boxes)
    assert check.units == plan.rows, (check.units, plan.rows)
    assert check.ok and check.gaps == 0, check.warnings
    top = [b for b in boxes if b.top_only]
    assert top and all(b.sets[0] == (1,) for b in top), top
    top_units = sum(b.units for b in top)
    meant = plan.rows * 20 / 100
    assert abs(top_units - meant) <= 1, (top_units, meant)
    xml = rs.atg_xml(plan, boxes)
    assert xml.count('v6="true"') == len(top)
    errors = _schema_errors(xml)
    assert not errors, errors
    print(f'  pool lever: {top_units} of {check.units} units on {len(top)} '
          f'V6 coupons, all on horse 1 ({meant:.0f} meant)')

    # Games without a top pool refuse rather than write a plain coupon.
    plan = make_plan(V85_SPEC)
    plan.top_only = True
    try:
        rs.solve(plan)
    except rs.ReduceError as err:
        print(f'  V85 played for the top pool: {err}')
    else:
        raise AssertionError('V85 has no top pool to play')


def check_coupon_scale():
    # Every row of an even coupon ties, so a lever keeps every row it had
    # and deepens some, and a whole lever is the system played that often.
    even = [(picks, {n: 100 / len(picks) for n in picks})
            for picks, _ in V85_SPEC]
    plain = rs.row_units(rs.allocate(make_plan(even)))
    for scale in (1.5, 3):
        plan = make_plan(even)
        plan.scale = scale
        boxes = rs.solve(plan)
        check = rs.verify(plan, boxes)
        units = rs.row_units(boxes)
        assert check.units == plan.rows and check.ok, check.warnings
        assert all(units.get(r, 0) >= n for r, n in plain.items()), (
            f'x{scale} left a row below what it carries at 100%')
        if scale == 3:
            assert all(units[r] == 3 * n for r, n in plain.items())
        print(f'  even x{scale}: {check.units} rows over {len(units)} of '
              f'{len(plain)} combinations, none below 100%')

    # The weighted, reduced coupon cut and levered as a whole keeps its shares.
    for scale in (0.85, 1.5):
        plan = make_plan(V85_SPEC)
        plan.scale = scale
        check = rs.verify(plan, rs.solve(plan))
        assert check.units == plan.rows, (check.units, plan.rows)
        assert check.ok, check.warnings
        print(f'  weighted + reduced at {scale:.0%}: {check.units} rows, '
              f'every horse within {worst_miss(check)} of its share')

    # Scaling the coupon is the banker cut or levered by as much: the same
    # rows at the same depth, so a lever spreads into the thin rows too.
    for scale in (0.5, 2):
        scaled = make_plan(V85_SPEC)
        scaled.scale = scale
        levered = make_plan(V85_SPEC)
        levered.legs[0].coverage = {3: 100 * scale}
        a = rs.row_units(rs.allocate(scaled))
        b = rs.row_units(rs.allocate(levered))
        assert a == b, f'x{scale} differs from the banker at {scale:.0%}'
        print(f'  x{scale} = banker at {scale:.0%}: {len(a)} rows')


def main() -> int:
    print('crc')
    check_crc()
    print('single box')
    check_single_box()
    print('margins')
    plan, boxes = check_margins()
    print('xml')
    check_xml(plan, boxes)
    print('heavy weighting')
    check_heavy_weighting()
    print('monotone')
    check_monotone()
    print('proportional')
    check_proportional()
    print('aimed coupon')
    check_aimed_coupon()
    print('aim follows lever')
    check_aim_follows_lever()
    print('no lever cap')
    check_no_cap()
    print('cascade')
    check_cascade()
    print('averaged views')
    check_averaged_views()
    print('track code')
    check_track_code()
    print('top pool')
    check_top_pool()
    print('coupon scale')
    check_coupon_scale()
    print('\nall good')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
