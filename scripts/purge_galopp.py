"""
Purge galopp data from stable_v2.

Galopp tracks/races/entries snuck in via the v1→v2 backfill (v1's atg_track
table includes both trot and gallop). The ETL filter in
`etl.import_st.backfill_*` now prevents new galopp data from being imported,
but pre-existing rows need a one-shot cleanup.

Usage
-----
    python3 -m scripts.purge_galopp           # dry run, prints counts only
    python3 -m scripts.purge_galopp --execute # actually deletes

Order of operations (single transaction when --execute):
    1. Find galopp tracks (track.sport = 'gallop').
    2. Find races linked to those tracks.
    3. DELETE entries on those races.
    4. DELETE the races themselves.
    5. DELETE the galopp tracks.
    6. Report (NOT delete) horses + persons that became orphans.

Horses/persons are NOT deleted because:
  - We can't reliably distinguish a galopp horse with no entries from a
    pure-pedigree trot horse with no entries.
  - The merge tool (planned for /admin) will surface orphans for review.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from core.db import get_connection  # noqa: E402


def _count(cur, sql: str, params: tuple = ()) -> int:
    cur.execute(sql, params)
    return cur.fetchone()[0]


def _galopp_track_ids(cur) -> list[int]:
    cur.execute("SELECT track_id FROM track WHERE sport = 'gallop' ORDER BY track_id")
    return [r[0] for r in cur.fetchall()]


def _galopp_track_summary(cur) -> list[tuple]:
    cur.execute(
        """
        SELECT track_id, name, country, atg_track_id, st_code, primary_source
          FROM track
         WHERE sport = 'gallop'
         ORDER BY country NULLS LAST, name
        """
    )
    return cur.fetchall()


def _race_ids_for_tracks(cur, track_ids: list[int]) -> list[int]:
    if not track_ids:
        return []
    cur.execute(
        "SELECT race_id FROM race WHERE track_id = ANY(%s)",
        (track_ids,),
    )
    return [r[0] for r in cur.fetchall()]


def _orphan_horses_count(cur) -> int:
    """Horses that, after entry deletion, have no entries AND no pedigree role."""
    cur.execute(
        """
        SELECT COUNT(*) FROM horse h
         WHERE NOT EXISTS (SELECT 1 FROM entry e  WHERE e.horse_id = h.horse_id)
           AND NOT EXISTS (SELECT 1 FROM horse c1 WHERE c1.sire_id = h.horse_id)
           AND NOT EXISTS (SELECT 1 FROM horse c2 WHERE c2.dam_id  = h.horse_id)
           AND h.primary_source IN ('st', 'atg')
           AND h.breedly_id IS NULL
           AND h.hvt_id IS NULL
           AND h.letrot_id IS NULL
        """
    )
    return cur.fetchone()[0]


def _orphan_persons_count(cur) -> int:
    """Persons no longer referenced by any entry."""
    cur.execute(
        """
        SELECT COUNT(*) FROM person p
         WHERE NOT EXISTS (
                 SELECT 1 FROM entry e
                  WHERE e.driver_id = p.person_id
                     OR e.trainer_id = p.person_id)
           AND NOT EXISTS (
                 SELECT 1 FROM horse_owner_history h
                  WHERE h.owner_id = p.person_id)
           AND NOT EXISTS (
                 SELECT 1 FROM horse_trainer_history h
                  WHERE h.trainer_id = p.person_id)
           AND p.primary_source IN ('st', 'atg')
           AND p.hvt_id IS NULL
           AND p.letrot_id IS NULL
        """
    )
    return cur.fetchone()[0]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--execute", action="store_true",
                    help="Actually delete (default is dry-run).")
    ap.add_argument("--keep-tracks", action="store_true",
                    help="Don't delete galopp tracks (only races/entries).")
    args = ap.parse_args()

    conn = get_connection()
    try:
        with conn.cursor() as cur:
            print("=" * 60)
            print(f"galopp purge — {'EXECUTE' if args.execute else 'DRY-RUN'}")
            print("=" * 60)

            galopp_tracks = _galopp_track_summary(cur)
            track_ids = [t[0] for t in galopp_tracks]
            print(f"\ngalopp tracks (sport='gallop'): {len(track_ids)}")
            for tid, name, country, atg_id, st_code, primary in galopp_tracks[:20]:
                print(f"  track_id={tid:<6} {country or '??':<2} "
                      f"{name or '?':<28} atg_id={atg_id} st_code={st_code or '-'}"
                      f" src={primary}")
            if len(galopp_tracks) > 20:
                print(f"  ... and {len(galopp_tracks) - 20} more")

            race_ids = _race_ids_for_tracks(cur, track_ids)
            print(f"\nraces on those tracks: {len(race_ids):,}")

            n_entries = 0
            if race_ids:
                n_entries = _count(
                    cur,
                    "SELECT COUNT(*) FROM entry WHERE race_id = ANY(%s)",
                    (race_ids,),
                )
            print(f"entries on those races: {n_entries:,}")

            if not args.execute:
                print("\n[dry-run] not deleting anything.")
                print("re-run with --execute to apply.")
                return 0

            print("\nexecuting deletes...")

            if race_ids:
                cur.execute(
                    "DELETE FROM entry WHERE race_id = ANY(%s)",
                    (race_ids,),
                )
                print(f"  deleted entries: {cur.rowcount:,}")

                cur.execute(
                    "DELETE FROM race WHERE race_id = ANY(%s)",
                    (race_ids,),
                )
                print(f"  deleted races: {cur.rowcount:,}")

            if track_ids and not args.keep_tracks:
                cur.execute(
                    "DELETE FROM track WHERE track_id = ANY(%s)",
                    (track_ids,),
                )
                print(f"  deleted tracks: {cur.rowcount:,}")
            elif args.keep_tracks:
                print(f"  kept {len(track_ids)} galopp tracks (--keep-tracks)")

            n_orphan_h = _orphan_horses_count(cur)
            n_orphan_p = _orphan_persons_count(cur)
            print(f"\norphans (NOT auto-deleted, surface in merge tool):")
            print(f"  horses with no entries + no pedigree role: {n_orphan_h:,}")
            print(f"  persons no longer referenced anywhere:     {n_orphan_p:,}")

        conn.commit()
        print("\ncommitted.")
    finally:
        conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
