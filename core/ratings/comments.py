"""Read a Swedish race comment ("4-inv, ut 150 kv, hängde med utan fritt.")
into the trip and verdict facts the sub-ratings use.

Positions: "Ledn" is the lead, "Utv led" the death seat, "Rygg led" the
pocket; "N-inv" / "N-utv" is the N:th pair, inside / outside, so rank is
2N-1 / 2N. "X kv" is metres remaining, "e N" is after N metres.
"""

from __future__ import annotations

import re

_ORD = {
    'tvåa': 2, 'trea': 3, 'fyra': 4, 'femma': 5, 'sexa': 6, 'sjua': 7,
    'åtta': 8, 'nia': 9, 'tia': 10, 'elva': 11, 'tolva': 12,
}
_ORD_ALT = '|'.join(_ORD)

_RE_LEAD = re.compile(r'\b(?:ledn|ledningen)\b(?:\s+e\s+(\d{2,4}))?')
_RE_DEATH = re.compile(r'\butv(?:ändigt)?\s+led\b(?:\s+e\s+(\d{2,4}))?')
_RE_POCKET = re.compile(r'\brygg\s+led\b')
_RE_PAIR = re.compile(r'\b(\d{1,2})(?:\s?-\s?\d{1,2})*\s?-\s?(inv|utv)\b')
_RE_ORDINAL = re.compile(
    rf'\b({_ORD_ALT})(?:\s?-\s?(?:{_ORD_ALT}))*(?:\s+(inv|utv|i raden|i kön|i rygg))?\b')
_RE_LAST = re.compile(r'\b(näst\s+)?sist\b')
_RE_NO_COVER = re.compile(r'\b(?:utan|u|ej|inte)\s+rygg\b')
_RE_START_TROUBLE = re.compile(
    r'\b(?:sg|kort sg|startgalopp|tappstart|långsam start|seg start|'
    r'galopp i start|gal i start|gal innan start|gal innan bilen|dålig start|missade starten)\b')
_RE_FAST_START = re.compile(r'\b(?:snabbast ut|snabb (?:start|ut)|snabb ur volten|bra start|'
                            r'öppnade (?:bra|snabbt)|laddade|flög iväg)\b')

_RE_LANE = re.compile(r'\b([3-6])-spår\b[^,]*?(\d{3,4})\s?kv')
_RE_WIDE_WORDS = re.compile(r'\b(?:kördes runt|maxresa|runt hela|hela vägen utv)\b')
_RE_LEFT_FRONT = re.compile(
    r'\b(?:släppte|avlöst|rygg led e|ner i rygg|ner inv|tappade ledn|fick släppa|lämnade ledn)\b')
_RE_LATE_DEATH = re.compile(r'\butv led\s+(\d{3,4})\s?kv')

_RE_SECTOR = re.compile(r'(?<![\d,])(\d{2})(?:,(\d))?/(\d{3,4})\b')

_RE_HARD_TEMPO = re.compile(r'\b(?:hårt|högt|friskt|bra|skarpt)\s+tempo\b|\bhård press\b|\bansatt\b|'
                            r'\bfick inte vara ifred\b|\bpressad\b')
_RE_SOFT_TEMPO = re.compile(r'\b(?:lågt|lugnt|jämnt|svagt|beskedligt)\s+tempo\b|\bfick (?:vara ifred|bestämma)\b|'
                            r'\bdämpade\b|\btempogynnad\b')

_FINISH = (
    (re.compile(r'\b(?:mkt|mycket) bra avsl|\bstarkt? avsl|\bfullt gdk avsl|\bvass\b|\bfin(?:t)? (?:spurt|avsl)|'
                r'\bsköt till\b|\bgick starkt\b|\bfullföljde (?:starkt|mkt bra)|\bbra spurt\b'), 1.0),
    (re.compile(r'\bbra avsl|\bfullföljde bra\b|\bbra upphämtning\b|\bfullf bra\b'), 0.7),
    (re.compile(r'\bgdk avsl|\bok avsl|\bgdk uppl|\bfullföljde gdk\b|\bfullf hyggligt\b|\bhygglig|'
                r'\bhöll (?:sin |si )?farten (?:bra|fullt gdk|gdk)'), 0.35),
    (re.compile(r'\binget extra\b|\binga plus\b|\btog inget\b|\bkunde inte ta (?:ngt|något)\b|'
                r'\bkunde inget\b|\butan spurt\b|\bingen spurt\b'), -0.45),
    (re.compile(r'\bhöll (?:bara )?(?:sin |si )?fart'), -0.15),
)
_STRENGTH = (
    (re.compile(r'\bhöll (?:starkt|bra)\b|\brejäl\b|\borkade bra\b|\bhöll ut\b|\bgick hela vägen\b|\bstark\b(?! i handen)'), 0.7),
    (re.compile(r'\bstum\b|\bbröt ner\b|\btappade (?:mkt|massor|allt|mängder|mycket)\b|\bstort slagen\b|'
                r'\b(?:mkt|helt) (?:trött|slut)\b|(?<!ej )(?<!inte )(?<!inte helt )\btom\b'), -1.0),
    (re.compile(r'\btappade\b|\btrött\b|\bfrånåkt\b|\borkade inte\b|\btröttnade\b|\bsvag\b'), -0.6),
)
_HEART = (
    (re.compile(r'\bhöll undan\b|\bhöll emot\b|\bförsvarade\b|\bkämpade\b|\btapper\b|\bkontrade\b|'
                r'\bförsökte kontra\b|\bvägrade ge sig\b|\bsegervilja'), 1.0),
    (re.compile(r'\bgav upp\b|\bgav sig\b|\bvek\b|\bville inte\b|\bkom inte förbi\b|\blade av\b'), -0.6),
)
_RE_CLOSE = re.compile(r'\bknappt\b|\bnärkamp|\bhårt i mål\b|\bnos\b|\bhuvud\b|\bhals\b')
_RATE = (
    (re.compile(r'\bdämpade\b|\blåg lugn\b|\bfick bestämma\b|\bbestämde\b|\bbekväm\b|\bavspänd\b'), 0.7),
    (re.compile(r'\bivrig\b|\bhet\b|\bdrog hårt\b|\bstark i handen\b|\bsvårstyrd\b|\bbråk|\btog i\b'), -1.0),
)
_GAIT = (
    (re.compile(r'\btapptrav|\borent\b|\bstilen\?|\bbråkig\b|\borolig\b|\bstartförbud\b|\bsvårstyrd\b|'
                r'\bsned\b|\bosäker\b|\btravade orent\b|\bhängde i (?:vänster|höger)|\btveksam stil\b|'
                r'\bdålig stil\b'), -1.0),
    (re.compile(r'\bfin stil\b|\bstilren\b|\btravsäker\b'), 0.5),
)
_RE_DISTURBED = re.compile(r'\b(?:störd|trängd|hindrad|klev på|tramp|kontakt med|på grund av annan|'
                           r'galopp av|fick gal av|trasslade)\b')


class Trip:
    """early_rank: 1 = lead. led: in front early; led_through: and never gave
    it up. death: spent the race in the death seat. wide: 3rd lane or worse
    for 500 m+, or no cover. close: the comment calls the finish tight.
    disturbed: someone else caused the gallop. sector: (s/km, metres) of a
    noted closing sector. Verdicts are scores in [-1, 1] or None."""

    __slots__ = ('early_rank', 'start_trouble', 'fast_start', 'led', 'led_through',
                 'death', 'wide', 'hard_tempo', 'soft_tempo', 'close', 'disturbed',
                 'sector', 'finish', 'strength', 'heart', 'rate', 'gait')

    def __init__(self):
        self.early_rank = None
        self.start_trouble = self.fast_start = False
        self.led = self.led_through = self.death = self.wide = False
        self.hard_tempo = self.soft_tempo = self.close = self.disturbed = False
        self.sector = None
        self.finish = self.strength = self.heart = self.rate = self.gait = None


def _score(text: str, table) -> float | None:
    hit = False
    s = 0.0
    for rx, w in table:
        if rx.search(text):
            s += w
            hit = True
    if not hit:
        return None
    return max(-1.0, min(1.0, s))


def _early_rank(clauses: list[str], field: int) -> tuple[int | None, bool]:
    """(rank, led) from the opening clauses."""
    for c in clauses[:2]:
        c = c.strip()
        m = _RE_LEAD.search(c)
        if m and (m.start() < 12):
            after = int(m.group(1)) if m.group(1) else 0
            return (1 if after <= 700 else None), True
        if _RE_DEATH.search(c):
            return 2, False
        if _RE_POCKET.search(c):
            return 3, False
        m = _RE_PAIR.search(c)
        if m:
            n = int(m.group(1))
            return (2 * n - 1 if m.group(2) == 'inv' else 2 * n), False
        m = _RE_ORDINAL.search(c)
        if m:
            n = _ORD[m.group(1)]
            tail = m.group(2)
            if tail == 'utv':
                return 2 * n, False
            if tail:
                return 2 * n - 1, False
            return n, False
        m = _RE_LAST.search(c)
        if m:
            return max(2, field - (1 if m.group(1) else 0)), False
        if _RE_NO_COVER.search(c):
            return 4, False
    return None, False


def opening(clauses: list[str], field: int) -> tuple[int | None, bool, bool, bool]:
    """(early rank, led, start trouble, fast start) from a lowercased
    comment already split on commas."""
    head = ','.join(clauses[:2])
    trouble = bool(_RE_START_TROUBLE.search(head))
    fast = bool(_RE_FAST_START.search(head))
    rank, led = _early_rank(clauses, field)
    if rank is None and trouble:
        rank = field
    if rank is not None:
        rank = max(1, min(rank, field))
    return rank, led or rank == 1, trouble, fast


def parse(comment: str | None, *, field: int, distance: int | None,
          galopp: bool) -> Trip | None:
    if not comment:
        return None
    text = comment.strip().lower()
    if not text or text.startswith('struken'):
        return None
    t = Trip()
    clauses = text.split(',')
    head = ','.join(clauses[:2])
    t.early_rank, t.led, t.start_trouble, t.fast_start = opening(clauses, field)
    rank = t.early_rank
    rest = ','.join(clauses[1:])
    t.led_through = t.led and not _RE_LEFT_FRONT.search(rest) and not _RE_DEATH.search(rest)

    dist = distance or 2140
    if rank == 2 and not re.search(r'\b(?:rygg|ner i|ner inv)\b', rest):
        t.death = True
    else:
        m = _RE_LATE_DEATH.search(text)
        if m and int(m.group(1)) >= 0.42 * dist:
            t.death = True
    wide = bool(_RE_WIDE_WORDS.search(text)) or bool(_RE_NO_COVER.search(head))
    if not wide:
        for m in _RE_LANE.finditer(text):
            if int(m.group(2)) >= 500:
                wide = True
                break
    t.wide = wide

    if t.led:
        t.hard_tempo = bool(_RE_HARD_TEMPO.search(text))
        t.soft_tempo = bool(_RE_SOFT_TEMPO.search(text)) and not t.hard_tempo
    t.close = bool(_RE_CLOSE.search(text))
    if galopp:
        t.disturbed = bool(_RE_DISTURBED.search(text))

    for m in _RE_SECTOR.finditer(text):
        metres = int(m.group(3))
        if 300 <= metres <= 1000:
            pace = 60 + int(m.group(1)) + (int(m.group(2)) / 10 if m.group(2) else 0)
            if 62 <= pace <= 95:
                t.sector = (pace, metres)
                break

    t.finish = _score(text, _FINISH)
    if not galopp:
        t.strength = _score(text, _STRENGTH)
    t.heart = _score(text, _HEART)
    t.rate = _score(text, _RATE)
    t.gait = _score(text, _GAIT)
    return t
