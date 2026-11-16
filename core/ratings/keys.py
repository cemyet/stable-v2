"""The rating catalog: main ratings, the sub-ratings that make them up, and
how each sub-rating is read. Shared by the batch engine, the API roll-up and
(through the API) the horse card, so a key means the same thing everywhere.

Every sub-rating's raw value is oriented so that higher is better. `k` is the
number of observations at which the evidence counts as much as the prior:
confidence = n / (n + k), and the stored value is pulled toward the breed's
median by (1 - confidence).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Sub:
    key: str
    main: str
    label: str
    k: float
    weight: float = 1.0
    # Where the value lives: 'horse' subs come from the horse's own snapshot,
    # 'person' and 'race' ones are filled at request time.
    scope: str = 'horse'


@dataclass(frozen=True)
class Main:
    key: str
    code: str
    label: str
    group: str          # 'ability' (radar) | 'standing' (bars)


MAINS: tuple[Main, ...] = (
    Main('start', 'STA', 'start', 'ability'),
    Main('speed', 'SPD', 'speed', 'ability'),
    Main('finish', 'FIN', 'finish', 'ability'),
    Main('strength', 'STR', 'strength', 'ability'),
    Main('discipline', 'DIS', 'discipline', 'ability'),
    Main('heart', 'HRT', 'heart', 'ability'),
    Main('class', 'CLS', 'class', 'standing'),
    Main('form', 'FRM', 'form', 'standing'),
    Main('connections', 'CON', 'connections', 'standing'),
    Main('pedigree', 'PED', 'pedigree', 'standing'),
)

SUBS: tuple[Sub, ...] = (
    Sub('gate', 'start', 'gate speed', 1.5, 1.2),
    Sub('early', 'start', 'early position', 4, 1.0),
    Sub('duels', 'start', 'start duels', 5, 1.0),

    Sub('top', 'speed', 'top speed', 1.5, 1.0),
    Sub('accel', 'speed', 'acceleration', 2, 0.8),
    Sub('record', 'speed', 'record', 2, 1.0),
    Sub('cruise', 'speed', 'cruise speed', 1.5, 0.8),

    Sub('closing', 'finish', 'closing speed', 1.5, 1.0),
    Sub('sectors', 'finish', 'closing sectors', 1.5, 0.7),
    Sub('gains', 'finish', 'late gains', 4, 1.0),
    Sub('fverdict', 'finish', 'finish verdicts', 5, 0.8),

    Sub('death', 'strength', 'death-seat results', 3, 1.0),
    Sub('wide', 'strength', 'wide trips', 3, 1.0),
    Sub('fade', 'strength', 'fade resistance', 2, 0.8),
    Sub('tempo', 'strength', 'tempo from the front', 3, 0.7),
    Sub('distance', 'strength', 'distance', 4, 0.8),
    Sub('sverdict', 'strength', 'strength verdicts', 6, 0.5),

    Sub('gallop', 'discipline', 'gallop rate', 6, 1.5),
    Sub('gait', 'discipline', 'gait & temper', 8, 0.6),
    Sub('rate', 'discipline', 'rateability', 4, 0.5),
    Sub('consistency', 'discipline', 'consistency', 6, 1.0),

    Sub('close', 'heart', 'close finishes', 3, 1.0),
    Sub('holding', 'heart', 'holding on', 4, 1.0),
    Sub('fight', 'heart', 'fighting verdicts', 6, 0.7),

    Sub('elo', 'class', 'elo', 6, 1.6),
    Sub('peak', 'class', 'peak', 10, 0.7),
    Sub('tested', 'class', 'tested', 5, 0.7),
    Sub('prize', 'class', 'prize level', 5, 0.6),

    Sub('trend', 'form', 'elo trend', 3, 1.0),
    Sub('recent', 'form', 'recent runs', 3, 1.0),
    Sub('fresh', 'form', 'freshness', 1, 0.5, scope='race'),

    Sub('trainer', 'connections', 'trainer', 15, 1.0, scope='person'),
    Sub('driver', 'connections', 'driver', 50, 1.0, scope='person'),
    Sub('pairing', 'connections', 'pairing', 4, 0.5, scope='race'),
    Sub('stable', 'connections', 'stable form', 3, 0.6, scope='person'),

    Sub('own', 'pedigree', 'own index', 1, 1.4, scope='pedigree'),
    Sub('sire', 'pedigree', 'sire', 1, 1.0, scope='pedigree'),
    Sub('dam', 'pedigree', 'dam', 1, 0.9, scope='pedigree'),
    Sub('damsire', 'pedigree', 'damsire', 1, 0.4, scope='pedigree'),
)

SUB_BY_KEY = {s.key: s for s in SUBS}
MAIN_BY_KEY = {m.key: m for m in MAINS}

# The horse snapshot stores these, in this order, as two SMALLINT[] arrays.
HORSE_KEYS: tuple[str, ...] = tuple(s.key for s in SUBS if s.scope == 'horse')
HORSE_INDEX = {k: i for i, k in enumerate(HORSE_KEYS)}
PEDIGREE_KEYS: tuple[str, ...] = tuple(s.key for s in SUBS if s.scope == 'pedigree')

# Hand-set starting point; the batch refits these on past races and stores
# the result in rating_model, which the API prefers when present.
DEFAULT_WEIGHTS: dict[str, float] = {
    'class': 0.30, 'pedigree': 0.12, 'form': 0.10, 'speed': 0.09,
    'connections': 0.08, 'start': 0.07, 'finish': 0.07, 'strength': 0.07,
    'discipline': 0.06, 'heart': 0.04,
}

# Pedigree is a prior: its weight in the overall is multiplied by
# PEDIGREE_K / (starts + PEDIGREE_K), so it fades as the record grows.
PEDIGREE_K = 10

# Display scale: 65 + 10·z per breed against active horses, clamped.
SCALE_MID = 65.0
SCALE_SD = 10.0
SCALE_LO = 1
SCALE_HI = 99
