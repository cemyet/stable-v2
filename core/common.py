"""Shared utilities for stable-v2.

Most pure parsers live in `core.parser` (TravSport HTML/JSON) or in
`etl.matching` (cross-source ID merge). This module is the catch-all for
sport-specific value parsers (km-time, money, placements) that any module
might want.

Verbatim port of v1's etl/common.py — these are pure functions.
"""

from __future__ import annotations

import re
from datetime import date, datetime
from typing import Optional

# ---------------------------------------------------------------------------
# Dates & headings
# ---------------------------------------------------------------------------

_SV_MONTHS = {
    'JANUARI': 1, 'FEBRUARI': 2, 'MARS': 3, 'APRIL': 4, 'MAJ': 5, 'JUNI': 6,
    'JULI': 7, 'AUGUSTI': 8, 'SEPTEMBER': 9, 'OKTOBER': 10, 'NOVEMBER': 11,
    'DECEMBER': 12,
}
_SV_DAYS = {'MÅNDAG', 'TISDAG', 'ONSDAG', 'TORSDAG', 'FREDAG', 'LÖRDAG', 'SÖNDAG'}

_HEADING_RE = re.compile(
    r'^(?P<track>.+?)\s+(?P<day>\S+)\s+(?P<dom>\d{1,2})\s+(?P<mon>\S+)\s+(?P<year>\d{4})\s*$'
)


def parse_heading(heading: Optional[str]) -> tuple[Optional[str], Optional[date]]:
    if not heading:
        return None, None
    m = _HEADING_RE.match(heading.strip())
    if not m:
        return heading, None
    track = m.group('track').strip()
    day = m.group('day').upper()
    if day not in _SV_DAYS:
        return heading, None
    month = _SV_MONTHS.get(m.group('mon').upper())
    if not month:
        return track, None
    try:
        return track, date(int(m.group('year')), month, int(m.group('dom')))
    except ValueError:
        return track, None


def parse_iso_date(s: Optional[str]) -> Optional[date]:
    if not s:
        return None
    try:
        return date.fromisoformat(s[:10])
    except ValueError:
        return None


def parse_iso_datetime(s: Optional[str]) -> Optional[datetime]:
    if not s:
        return None
    try:
        return datetime.fromisoformat(s)
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Money: '6 685 000 kr' -> 6685000
# ---------------------------------------------------------------------------

_MONEY_RE = re.compile(r'[\d\s]+')


def parse_money_kr(s: Optional[object]) -> Optional[int]:
    if s is None:
        return None
    if isinstance(s, (int, float)):
        return int(s)
    s = str(s).replace('\xa0', ' ').strip()
    if not s or s in ('-', '--', '0 kr', '0'):
        return 0 if s in ('0 kr', '0') else None
    m = _MONEY_RE.search(s.replace('.', ' '))
    if not m:
        return None
    digits = m.group(0).replace(' ', '')
    return int(digits) if digits else None


# ---------------------------------------------------------------------------
# Kilometer time
# ---------------------------------------------------------------------------

_KM_TIME_RE = re.compile(
    r'^\*?(?:(?P<min>\d+)[.,])?(?P<sec>\d{1,2}),(?P<tenths>\d)(?P<suffix>[a-z]*)$',
    re.IGNORECASE,
)


def parse_km_time(text: Optional[str]) -> Optional[int]:
    if not text:
        return None
    t = text.strip()
    if not t:
        return None
    m = _KM_TIME_RE.match(t)
    if not m:
        return None
    minutes = int(m.group('min')) if m.group('min') else 1
    seconds = int(m.group('sec'))
    tenths = int(m.group('tenths'))
    return minutes * 1000 + seconds * 10 + tenths


def parse_km_time_seconds(text: Optional[str]) -> Optional[float]:
    encoded = parse_km_time(text)
    if encoded is None:
        return None
    minutes = encoded // 1000
    rest = encoded % 1000
    seconds = rest // 10
    tenths = rest % 10
    return minutes * 60.0 + seconds + tenths / 10.0


# ---------------------------------------------------------------------------
# Placement
# ---------------------------------------------------------------------------

def classify_placement(
    placement_number_str: Optional[str],
    placement_display: Optional[str],
) -> tuple[Optional[int], bool]:
    pd = (placement_display or '').strip().lower()
    disqualified = False
    if pd.startswith('d') or (len(pd) >= 2 and pd[0] == 'r' and pd[1].isdigit()):
        disqualified = True
    try:
        n = int(placement_number_str) if placement_number_str is not None else 0
    except (TypeError, ValueError):
        n = 0
    if n <= 0:
        return None, disqualified
    return n, disqualified


# ---------------------------------------------------------------------------
# startPositionAndDistance: '7/2140' -> (7, 2140)
# ---------------------------------------------------------------------------

_POS_DIST_RE = re.compile(r'^(\d*)/(\d+)$')


def parse_position_distance(s: Optional[str]) -> tuple[Optional[int], Optional[int]]:
    if not s:
        return None, None
    m = _POS_DIST_RE.match(s.strip())
    if not m:
        return None, None
    pos_s, dist_s = m.group(1), m.group(2)
    pos = int(pos_s) if pos_s else None
    dist = int(dist_s) if dist_s else None
    if dist == 0:
        dist = None
    return pos, dist


def derive_start_method_from_time(km_time_text: Optional[str]) -> Optional[str]:
    if not km_time_text:
        return None
    t = km_time_text.strip().lower()
    if not t or t in {'ug', 'uag', 'd', 'da', 'ag', 'g', '-', '-l'}:
        return None
    suffix = re.sub(r'^\*?(?:\d+[.,])?\d{1,2},\d', '', t)
    if 'a' in suffix:
        return 'A'
    return 'V'


def classify_letrot_placement(code: Optional[str]) -> dict:
    """Map a LeTrot ARRIVÉE *rang* code to canonical result flags.

    French trot has no "broke gait but kept its place" concept the way Swedish
    trot does — breaking gait (allure irrégulière) is an automatic *distancement*
    (disqualification). So in the French data the distancé family IS the galopp
    signal, and we set ``galopp`` together with ``disqualified``.

    Returns the subset of ``{'galopp','disqualified','withdrawn'}`` that should
    be True for this code. An empty dict means "normal finisher / unknown" —
    leave the flags at their defaults (we deliberately skip ambiguous codes
    like CV / FM / G* / H* / R* rather than guess).

        'DA'  -> distancé attelé      (gait-break DQ)  -> galopp + disqualified
        'D','D0'..'D9','DM','DAI'     (distancé family) -> galopp + disqualified
        'NP'  -> non partant          (did not start)   -> withdrawn
        'A'   -> arrêté (pulled up)   (DNF)             -> disqualified
        'T'   -> tombé (fell)         (DNF)             -> disqualified
        '1'..'9', '0', '<n>D/H/R'     finisher/unplaced -> {}  (no flags)
    """
    c = (code or "").strip().upper()
    if not c:
        return {}
    if c == "NP":
        return {"withdrawn": True}
    if c in ("A", "T"):
        return {"disqualified": True}
    # Distancé family: 'D', 'DA', 'DM', 'DAI', 'D0'..'D9'. NOT '<digit>D' (a
    # finisher like '4D') — those start with a digit, so the leading-D guard
    # keeps them out.
    if c in ("DA", "DM", "DAI") or re.fullmatch(r"D[0-9]?", c):
        return {"galopp": True, "disqualified": True}
    return {}


def normalize_country(code: Optional[str]) -> Optional[str]:
    if not code:
        return None
    code = code.strip().upper()
    if len(code) == 2 and code.isalpha():
        return code
    return None
