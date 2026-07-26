"""Re-derive the true track for historical ATG races misfiled by rotating
slot ids.

ATG's numeric track ids are NOT stable venue identifiers for foreign/guest
tracks: the same id is a per-country slot whose venue changes between
racedays (id 54 was Odense on 2026-05-08 and Århus on 2026-06-20, verified
against ATG's own API). Before the name-guard fix in etl.matching.upsert_track,
every race scraped under such a slot was filed at whichever track first
claimed the id — e.g. ~9.5k Danish races all filed at "Ålborg".

This script repairs history:

  1. Collect ATG races at the suspect (former slot-holder) tracks, grouped
     by (race_date, slot) — every race in a group shares one physical venue.
  2. Determine the true venue name per group: from a locally stored
     atg_race_raw payload when available, else ONE live ATG API call per
     group (any race id in the group).
  3. If the true name disagrees with the current track row, resolve/create
     the correct track and repoint every race in the group. Each repoint is
     logged to track_change_log (change_type='race_repointed').

Repointing intentionally creates same-(track, date, number) duplicates
where an ST copy of the race already exists at the correct track — the
cleanup pipeline's duplicate-race phases fold those afterwards.

Usage
-----
    python3 -m scripts.backfill_atg_track_misfiles                 # dry-run
    python3 -m scripts.backfill_atg_track_misfiles --execute
    python3 -m scripts.backfill_atg_track_misfiles --execute --limit 200
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import httpx  # noqa: E402

from core.db import get_connection  # noqa: E402
from etl.matching import _track_names_agree, upsert_track  # noqa: E402

# Former slot-holder tracks (atg_track_id cleared in phase 1 of the DQ
# overhaul). All their ATG races are suspect. Italian slot 70 (Dei Sauri)
# and Australian slot 51 (Australien placeholder) were found in the
# phase-3 foreign-ETL trial and are included here.
DEFAULT_SUSPECT_TRACK_IDS = (33, 42, 67, 92, 136, 149, 166, 107)


def _all_foreign_slot_track_ids(cur) -> list[int]:
    """Every non-SE track that currently (or recently) holds an atg_track_id
    and has ATG races — all foreign ATG slots rotate."""
    cur.execute(
        """
        SELECT DISTINCT t.track_id
          FROM track t
          JOIN race r ON r.track_id = t.track_id AND r.atg_race_id IS NOT NULL
         WHERE t.atg_track_id IS NOT NULL
           AND COALESCE(t.country, '') <> 'SE'
         ORDER BY t.track_id
        """
    )
    return [r[0] for r in cur.fetchall()]

ATG_RACE_URL = "https://www.atg.se/services/racinginfo/v1/api/races/{race_id}"
UA = {"User-Agent": "Mozilla/5.0"}


def _collect_groups(cur, track_ids: tuple[int, ...]) -> list[dict]:
    cur.execute(
        """
        SELECT r.race_date,
               split_part(r.atg_race_id, '_', 2)  AS slot,
               r.track_id,
               t.name                             AS track_name,
               t.country                          AS track_country,
               array_agg(r.race_id ORDER BY r.race_id)      AS race_ids,
               array_agg(r.atg_race_id ORDER BY r.race_id)  AS atg_ids
          FROM race r
          JOIN track t ON t.track_id = r.track_id
         WHERE r.atg_race_id IS NOT NULL
           AND r.track_id = ANY(%s)
         GROUP BY 1, 2, 3, 4, 5
         ORDER BY 1 DESC
        """,
        (list(track_ids),),
    )
    cols = [d.name for d in cur.description]
    return [dict(zip(cols, r)) for r in cur.fetchall()]


def _truth_from_raw(cur, atg_ids: list[str]) -> tuple[str, str] | None:
    cur.execute(
        """
        SELECT raw_json->'track'->>'name', raw_json->'track'->>'countryCode'
          FROM atg_race_raw
         WHERE atg_race_id = ANY(%s)
           AND raw_json->'track'->>'name' IS NOT NULL
         LIMIT 1
        """,
        (atg_ids,),
    )
    row = cur.fetchone()
    return (row[0], row[1]) if row else None


def _truth_from_api(client: httpx.Client, atg_ids: list[str],
                    sleep: float) -> tuple[str, str] | None:
    for atg_id in atg_ids[:2]:  # one retry with a second race id
        try:
            resp = client.get(ATG_RACE_URL.format(race_id=atg_id), timeout=20)
            time.sleep(sleep)
            if resp.status_code != 200:
                continue
            track = (resp.json() or {}).get("track") or {}
            if track.get("name"):
                return track["name"], track.get("countryCode")
        except Exception:
            continue
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--execute", action="store_true",
                    help="Apply changes (default is dry-run).")
    ap.add_argument("--limit", type=int, default=None,
                    help="Max raceday groups to process.")
    ap.add_argument("--track-ids", type=int, nargs="*",
                    default=None,
                    help="Suspect track_ids to scan (default: known "
                         "slot-holders). Use --all-foreign for every "
                         "non-SE track that holds an atg_track_id.")
    ap.add_argument("--all-foreign", action="store_true",
                    help="Scan every non-SE track with an atg_track_id.")
    ap.add_argument("--sleep", type=float, default=0.25,
                    help="Delay between ATG API calls (seconds).")
    ap.add_argument("--clear-slot-ids", action="store_true",
                    help="NULL out atg_track_id on the scanned tracks "
                         "before repointing (recommended with --execute; "
                         "stops the slot from re-attracting new races).")
    args = ap.parse_args()

    conn = get_connection()
    cur = conn.cursor()

    if args.all_foreign:
        track_ids = _all_foreign_slot_track_ids(cur)
    elif args.track_ids is not None:
        track_ids = args.track_ids
    else:
        track_ids = list(DEFAULT_SUSPECT_TRACK_IDS)

    if args.clear_slot_ids and args.execute and track_ids:
        cur.execute(
            """
            SELECT track_id, name, atg_track_id FROM track
             WHERE track_id = ANY(%s) AND atg_track_id IS NOT NULL
            """,
            (track_ids,),
        )
        cleared = cur.fetchall()
        for tid, name, slot in cleared:
            cur.execute(
                """
                INSERT INTO track_change_log
                    (track_id, change_type, old_value, new_value, reason, changed_by)
                VALUES (%s, 'atg_track_id_cleared', %s, %s, %s, %s)
                """,
                (
                    tid,
                    json.dumps({"atg_track_id": slot, "name": name}),
                    json.dumps({"atg_track_id": None}),
                    "Foreign ATG slot ids rotate across venues; "
                    "name is authoritative",
                    "scripts.backfill_atg_track_misfiles",
                ),
            )
        cur.execute(
            "UPDATE track SET atg_track_id = NULL, last_updated_at = NOW() "
            " WHERE track_id = ANY(%s) AND atg_track_id IS NOT NULL",
            (track_ids,),
        )
        conn.commit()
        print(f"cleared atg_track_id on {len(cleared)} tracks: "
              f"{[(n, s) for _, n, s in cleared]}")

    # Fix known country mislabel: Milano was stamped GB.
    if args.execute:
        cur.execute(
            """
            UPDATE track
               SET country = 'IT', last_updated_at = NOW()
             WHERE name = 'Milano' AND COALESCE(country, '') <> 'IT'
            RETURNING track_id, country
            """
        )
        milano_fix = cur.fetchall()
        if milano_fix:
            for tid, _ in milano_fix:
                cur.execute(
                    """
                    INSERT INTO track_change_log
                        (track_id, change_type, old_value, new_value,
                         reason, changed_by)
                    VALUES (%s, 'country_corrected', %s, %s, %s, %s)
                    """,
                    (
                        tid,
                        json.dumps({"country": "GB"}),
                        json.dumps({"country": "IT"}),
                        "Milano is Italian; country was mis-stamped GB",
                        "scripts.backfill_atg_track_misfiles",
                    ),
                )
            conn.commit()
            print(f"corrected Milano country -> IT ({len(milano_fix)} row)")

    groups = _collect_groups(cur, tuple(track_ids))
    if args.limit:
        groups = groups[: args.limit]
    print(f"{'EXECUTE' if args.execute else 'DRY-RUN'} — "
          f"{len(groups)} raceday groups at suspect tracks {track_ids}")

    client = httpx.Client(headers=UA, follow_redirects=True)
    moves: Counter = Counter()          # (from_name, to_name) -> races
    unresolved: Counter = Counter()     # from_name -> races
    confirmed = 0
    races_repointed = 0
    target_cache: dict[tuple[str, str | None], int] = {}

    for i, g in enumerate(groups):
        truth = _truth_from_raw(cur, g["atg_ids"]) or _truth_from_api(
            client, g["atg_ids"], args.sleep
        )
        if truth is None:
            unresolved[g["track_name"]] += len(g["race_ids"])
            continue
        true_name, true_country = truth

        if _track_names_agree(g["track_name"], true_name):
            confirmed += len(g["race_ids"])
            continue

        moves[(g["track_name"], true_name)] += len(g["race_ids"])
        races_repointed += len(g["race_ids"])

        if not args.execute:
            continue

        cache_key = (true_name.strip().lower(), true_country)
        target_id = target_cache.get(cache_key)
        if target_id is None:
            target_id = upsert_track(
                cur, "atg", int(g["slot"]) if g["slot"].isdigit() else None,
                {"name": true_name, "country": true_country, "sport": "trot"},
            )
            target_cache[cache_key] = target_id
        if target_id == g["track_id"]:
            continue

        for race_id in g["race_ids"]:
            cur.execute(
                "UPDATE race SET track_id = %s, last_updated_at = NOW() "
                " WHERE race_id = %s",
                (target_id, race_id),
            )
            cur.execute(
                """
                INSERT INTO track_change_log
                    (track_id, race_id, change_type, old_value, new_value,
                     reason, changed_by)
                VALUES (%s, %s, 'race_repointed', %s, %s, %s, %s)
                """,
                (
                    target_id, race_id,
                    json.dumps({"track_id": g["track_id"],
                                "track_name": g["track_name"]}),
                    json.dumps({"track_id": target_id,
                                "track_name": true_name}),
                    f"ATG slot {g['slot']} on {g['race_date']} was "
                    f"{true_name!r} per ATG payload/API",
                    "scripts.backfill_atg_track_misfiles",
                ),
            )
        if (i + 1) % 50 == 0:
            conn.commit()
            print(f"  ... {i + 1}/{len(groups)} groups processed "
                  f"({races_repointed} races to repoint so far)")

    if args.execute:
        conn.commit()

    print(f"\nconfirmed correctly filed: {confirmed} races")
    print(f"repoint{'ed' if args.execute else ' needed'}: "
          f"{races_repointed} races across {sum(moves.values()) and len(moves)} track pairs")
    for (frm, to), n in moves.most_common():
        print(f"  {frm!r} -> {to!r}: {n} races")
    if unresolved:
        print("unresolved (no raw payload, API 404 — left in place):")
        for frm, n in unresolved.most_common():
            print(f"  {frm!r}: {n} races")
    conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
