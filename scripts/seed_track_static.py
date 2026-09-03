"""Seed static physical attributes for Swedish trotting tracks.

Figures follow Svensk Travsport's current banfakta, cross-checked against
Travronden's per-track fakta boxes (which match ST on the tracks ST still
exposes). Older aggregators (travstugan, Wikipedia infoboxes) still carry
pre-rebuild numbers in a few places and are only cited in notes.

  - https://www.travsport.se/vara-travbanor/…   (Svensk Travsport)
  - https://www.travronden.se/tavling/<bana>    (fakta box)

length_m / home_stretch_m / width_m in metres. width_m is the 1 640 m start
width rounded to the nearest metre (Kalmar uses the 2 140 m figure, which is
the one that makes it Sweden's widest). open_stretches is the count of
inside open-stretch lanes (Åby has 2, Skellefteå 1, others 0); auto_car_wings
is 'angled' (vinklad startbilsvinge) or 'straight' (rak vinge).

Idempotent: re-running just re-UPDATEs by track_id. Adds the columns first via
plain ALTERs (does NOT touch entry_features, so it's safe to run while a feature
backfill holds locks there).

    .venv-ml/bin/python -m scripts.seed_track_static
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from psycopg2.extras import Json

from core.db import get_connection

SOURCES = [
    "travsport.se/vara-travbanor (Svensk Travsport)",
    "travronden.se/tavling/<bana> (fakta box)",
]

_ADD_COLUMNS = """
ALTER TABLE track ADD COLUMN IF NOT EXISTS track_length_m     INTEGER;
ALTER TABLE track ADD COLUMN IF NOT EXISTS home_stretch_m     INTEGER;
ALTER TABLE track ADD COLUMN IF NOT EXISTS num_open_stretches SMALLINT;
ALTER TABLE track ADD COLUMN IF NOT EXISTS track_width_m      INTEGER;
ALTER TABLE track ADD COLUMN IF NOT EXISTS auto_car_wings     TEXT;
ALTER TABLE track ADD COLUMN IF NOT EXISTS surface            TEXT;
ALTER TABLE track ADD COLUMN IF NOT EXISTS shape              TEXT;
ALTER TABLE track ADD COLUMN IF NOT EXISTS opened_year        SMALLINT;
ALTER TABLE track ADD COLUMN IF NOT EXISTS physical_source    JSONB;
"""

# track_id -> (length_m, home_stretch_m, open_stretches, width_m, wings, note)
# wings: 'angled' | 'straight'.  note: per-track caveat (or None).
TRACKS: dict[int, tuple] = {
    64:  (1000, 200, 0, 22, "straight", "home stretch 200 m (ST / Travronden); older aggregators still list 196 m"),  # Solvalla
    5:   (1000, 180, 2, 23, "straight", "double open stretch"),                  # Åby
    29:  (1000, 190, 0, 23, "angled",   None),                                  # Jägersro
    14:  (1000, 200, 0, 21, "straight", None),                                  # Bergsåker
    4:   (1000, 205, 0, 21, "angled",   None),                                  # Romme
    2:   (1000, 182, 0, 22, "straight", None),                                  # Eskilstuna
    13:  (1000, 227, 0, 23, "straight", "longest home stretch on a 1000 m track"),  # Axevalla
    104: (1000, 203, 0, 23, "straight", None),                                  # Halmstad
    30:  (1000, 178, 0, 20, "straight", None),                                  # Gävle
    49:  (1000, 177, 0, 21, "straight", None),                                  # Färjestad
    34:  (1000, 175, 0, 21, "straight", None),                                  # Örebro
    65:  (1000, 207, 0, 26, "straight", "widest Swedish track (~26 m)"),        # Kalmar
    27:  (1000, 177, 0, 24, "straight", None),                                  # Mantorp
    146: (1000, 207, 0, 22, "straight", None),                                  # Bollnäs
    15:  (1000, 200, 0, 22, "straight", None),                                  # Boden
    105: (1000, 218, 0, 21, "straight", None),                                  # Östersund
    119: (1000, 190, 0, 22, "straight", "190 m (Travronden); older ST aggregators listed 211 m"),  # Umåker
    47:  (1000, 187, 0, 20, "angled",   None),                                  # Rättvik
    48:  (1000, 170, 0, 22, "straight", "170 m (ST / Travronden); older aggregators listed 161 m"),  # Hagmyren
    137: (1000, 175, 1, 21, "straight", None),                                  # Skellefteå (open stretch)
    103: (1000, 205, 0, 20, "straight", None),                                  # Årjäng
    26:  (1000, 156, 0, 20, "straight", "shortest home stretch on a 1000 m track"),  # Visby
    25:  (1000, 200, 0, 21, "straight", None),                                  # Dannero
    40:  (1004, 193, 0, 20, "straight", "lap is 1004 m"),                       # Solänget
    24:  (1000, 192, 0, 20, "straight", None),                                  # Lindesberg
    102: (800,  105, 0, 22, "straight", "shortest home stretch in Sweden (800 m track)"),  # Åmål
    106: (1000, 200, 0, 20, "straight", None),                                  # Vaggeryd
    22:  (800,  114, 0, 19, "straight", None),                                  # Arvika
    28:  (1609, 285, 0, 25, "angled",   "only Swedish mile track (1609 m); home stretch extended from 218 m to 285 m in 2021 (longest in Sweden)"),  # Tingsryd
    16:  (1000, 200, 0, 19, "straight", None),                                  # Lycksele
    44:  (800,  140, 0, None, "straight", None),                                # Oviken
    17:  (800,  160, 0, 17, "straight", None),                                  # Hoting
    100: (800,  157, 0, 19, "straight", None),                                  # Karlshamn
}


def main() -> int:
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(_ADD_COLUMNS)
        conn.commit()

        updated = 0
        with conn.cursor() as cur:
            for tid, (length, stretch, opens, width, wings, note) in TRACKS.items():
                src = {"sources": SOURCES, "retrieved": "2026-08-20"}
                if note:
                    src["note"] = note
                cur.execute("""
                    UPDATE track SET
                        track_length_m     = %s,
                        home_stretch_m     = %s,
                        num_open_stretches = %s,
                        track_width_m      = %s,
                        auto_car_wings     = %s,
                        surface            = 'sand',
                        shape              = 'oval',
                        physical_source    = %s
                    WHERE track_id = %s
                """, (length, stretch, opens, width, wings, Json(src), tid))
                updated += cur.rowcount
        conn.commit()
        print(f"seeded static data for {updated} tracks")

        with conn.cursor() as cur:
            cur.execute("""
                SELECT track_id, name, track_length_m, home_stretch_m,
                       num_open_stretches, track_width_m, auto_car_wings
                FROM track
                WHERE track_length_m IS NOT NULL
                ORDER BY track_id
            """)
            for r in cur.fetchall():
                print(r)
    finally:
        conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
