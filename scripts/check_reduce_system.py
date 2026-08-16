"""
Self-check for the reduced-system solver in `core/reduce_system.py`.

    python3 -m scripts.check_reduce_system

There is no test runner in this repo, so this is a plain script that asserts
its way through the properties that matter and prints what it found. It needs
no database and no network — the vendored XSD is read off disk.

What it checks:

  1. CRC16 against the CCITT-FALSE reference vector, since the file name ATG
     validates is nothing but that checksum.
  2. A plain unweighted system collapses to a single coupon (the merge is not
     silently giving up).
  3. A weighted + reduced V85 — the shape the game page produces — comes back
     with the row count the coupon sphere shows and per-horse margins that
     match the coverage the user drew, to the row.
  4. The generated XML validates against ATG's schema. The vendored copy is
     1.8.4, which has no `v85Coupon`; V85 is written as its structural twin
     `v86Coupon` for the check only. See core/vendor/README.md.
  5. Heavy weighting still balances — the units land on repeated rows via
     `betmultiplier` rather than being quietly dropped.
"""

from __future__ import annotations

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


def check_crc():
    got = rs.crc16(b'123456789')
    assert got == 0x29B1, f'CRC-16/CCITT-FALSE mismatch: {got:#06x}'
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
    for leg, horses in check.coverage.items():
        for horse, fig in horses.items():
            assert fig['actual'] == fig['target'], (leg, horse, fig)

    print(f'  {plan.full_rows} full rows, {plan.rows} after reduction '
          f'({plan.retention:.3f} retention)')
    print(f'  {len(boxes)} coupons, margins exact in all {len(plan.legs)} legs')
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
    print('track code')
    check_track_code()
    print('\nall good')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
