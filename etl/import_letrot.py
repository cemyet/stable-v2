"""Import LeTROT (French trotting) data into stable_v2.

Given a Le Trot course-page (one HTTP call), we get a full race + every
horse + every person + every result. This module turns that payload into
upserts against the v2 master tables.

Strategy:
    - One race per (date, reunion_id, course_number); letrot_race_id =
      "<date>_<reunion>_<course>".
    - One canonical horse row per `letrot_id`. Cross-source matching by
      letrot_id only (no name-based auto-merge).
    - Driver + trainer upserted as person rows with letrot_id.
    - Track upserted by (name, country='FR'). Le Trot doesn't expose stable
      track ids — `reunion_id` is per-day, not per-track.
    - Prize money parsed in EUR and converted to SEK via core.exchange.

Public entry-points:

    import_course(conn, race_date, reunion_id, course_number)
        Scrape + upsert one race.

    import_day(conn, race_date)
        Walk every course advertised for `race_date` and import each.
"""

from __future__ import annotations

import logging
import time
from datetime import date as Date
from typing import Iterable

import httpx
from psycopg2.extras import Json

from core.db import buffer_prune
from core.exchange import to_sek
from etl.matching import upsert_horse, upsert_person, upsert_race, upsert_entry
from scrapers.letrot import (
    make_client,
    fetch_course,
    fetch_horse_identity,
    list_today,
)

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Track upsert (FR — match by name+country)
# ---------------------------------------------------------------------------

def _upsert_track_fr(cur, track_name: str | None) -> int | None:
    if not track_name:
        return None
    cur.execute(
        """
        SELECT track_id FROM track
         WHERE country = 'FR' AND lower(name) = lower(%s)
         LIMIT 1
        """,
        (track_name,),
    )
    row = cur.fetchone()
    if row:
        return row[0]
    cur.execute(
        """
        INSERT INTO track (name, country, sport, source_data, primary_source, last_updated_at)
        VALUES (%s, 'FR', 'trot',
                jsonb_build_object('letrot', jsonb_build_object('discovered_via', 'course_page')),
                'letrot', NOW())
        RETURNING track_id
        """,
        (track_name,),
    )
    return cur.fetchone()[0]


# ---------------------------------------------------------------------------
# Horse upsert (light — uses runner row only; does NOT fetch identity page)
# ---------------------------------------------------------------------------

def _upsert_horse_from_runner(cur, runner: dict) -> int | None:
    if not runner.get("horse_letrot_id"):
        return None
    canonical = {
        "name":          runner.get("horse_name"),
        "gender_code":   runner.get("sex"),
        # Birth year inferred from age at race date (refined later when
        # we fetch the identity page).
    }
    raw = {
        "musique":           runner.get("musique"),
        "record_text":       runner.get("record_text"),
        "gains_lifetime_eur": runner.get("gains_lifetime_eur"),
        "discovered_via":    "course_page",
        "fer":               runner.get("fer"),
    }
    return upsert_horse(
        cur, "letrot", runner["horse_letrot_id"],
        canonical, raw_payload=raw,
    )


def _upsert_person(cur, person: dict | None, role: str) -> int | None:
    if not person or not person.get("letrot_id"):
        return None
    flag = {
        "driver":  {"is_driver":  True},
        "jockey":  {"is_driver":  True},
        "entraineur": {"is_trainer": True},
        "trainer": {"is_trainer": True},
        "proprietaire": {"is_owner": True},
        "eleveur": {"is_breeder": True},
    }.get(role, {"is_driver": True})

    return upsert_person(
        cur, "letrot", person["letrot_id"],
        {"name": person.get("name"), "license_country": "FR"},
        raw_payload={"role": role, "letrot_slug": person.get("slug")},
        role_flags=flag,
    )


# ---------------------------------------------------------------------------
# Race + entries
# ---------------------------------------------------------------------------

def _upsert_race(cur, parsed: dict) -> int | None:
    if not parsed.get("letrot_race_id") or not parsed.get("track_name"):
        return None

    track_id = _upsert_track_fr(cur, parsed["track_name"])
    if not track_id:
        return None

    # Distance — most runners share the same distance; pick the modal.
    distances = [r.get("distance") for r in parsed.get("runners") or [] if r.get("distance")]
    distance = max(set(distances), key=distances.count) if distances else None

    canonical = {
        "race_date":   Date.fromisoformat(parsed["race_date"]),
        "track_id":    track_id,
        "race_number": parsed.get("race_number"),
        "distance":    distance,
        "heading":     parsed.get("race_name"),
        "status":      "results",
    }
    payload = {
        "reunion_id": parsed.get("reunion_id"),
        "race_name":  parsed.get("race_name"),
        "raw_h1":     parsed.get("raw_h1"),
        "raw_title":  parsed.get("raw_title"),
    }
    return upsert_race(
        cur, "letrot", parsed["letrot_race_id"], canonical,
        raw_payload=payload,
    )


def import_course(
    conn,
    race_date: Date | str,
    reunion_id: str | int,
    course_number: int | str,
    *,
    client: httpx.Client | None = None,
) -> dict:
    own_client = client is None
    if own_client:
        client = make_client()

    if isinstance(race_date, Date):
        race_date_str = race_date.isoformat()
    else:
        race_date_str = race_date
        race_date = Date.fromisoformat(race_date_str)

    summary = {
        "letrot_race_id":  f"{race_date_str}_{reunion_id}_{course_number}",
        "race_id":         None,
        "horses_upserted": 0,
        "entries_upserted": 0,
        "skipped":         0,
    }

    try:
        parsed = fetch_course(client, race_date_str, reunion_id, course_number)
        if not parsed or not parsed.get("runners"):
            summary["skipped"] = 1
            return summary

        with conn.cursor() as cur:
            race_id = _upsert_race(cur, parsed)
            if not race_id:
                summary["skipped"] = 1
                conn.rollback()
                return summary
            summary["race_id"] = race_id

            for runner in parsed["runners"]:
                horse_id = _upsert_horse_from_runner(cur, runner)
                if not horse_id:
                    summary["skipped"] += 1
                    continue
                summary["horses_upserted"] += 1

                driver_id  = _upsert_person(cur, runner.get("driver"),  "driver")
                trainer_id = _upsert_person(cur, runner.get("trainer"), "trainer")

                prize_kr, prize_orig, fx_rate, fx_date = to_sek(
                    runner.get("prize_eur"), "EUR", race_date
                )

                entry_canonical = {
                    "program_number": runner.get("program_number"),
                    "post":           runner.get("program_number"),  # FR programme = post
                    "distance":       runner.get("distance"),
                    "placement":      runner.get("placement"),
                    "placement_text": runner.get("placement_text"),
                    "time_text":      runner.get("time_text"),
                    "time_seconds":   runner.get("km_time_seconds"),
                    "prize_kr":       prize_kr or 0,
                    "prize_currency": "EUR",
                    "prize_original": prize_orig,
                    "prize_fx_rate":  fx_rate,
                    "prize_fx_date":  fx_date,
                    "driver_id":      driver_id,
                    "trainer_id":     trainer_id,
                    "age":            runner.get("age"),
                    "sex":            runner.get("sex"),
                    "shoe_code":      runner.get("fer"),
                    "withdrawn":      runner.get("placement") is None and runner.get("placement_text") == "0",
                }
                entry_payload = {
                    "musique":      runner.get("musique"),
                    "record_text":  runner.get("record_text"),
                    "gains_lifetime_eur": runner.get("gains_lifetime_eur"),
                    "avis_trainer": runner.get("avis_trainer"),
                    "odds_text":    runner.get("odds_text"),
                    "prize_eur":    runner.get("prize_eur"),
                    "km_time_text": runner.get("km_time_text"),
                }
                upsert_entry(cur, "letrot", race_id, horse_id,
                             entry_canonical, raw_payload=entry_payload)
                summary["entries_upserted"] += 1

        conn.commit()
    finally:
        if own_client:
            client.close()
    return summary


# ---------------------------------------------------------------------------
# Day import
# ---------------------------------------------------------------------------

_WHEN_BY_OFFSET = {-1: "hier", 0: "aujourd-hui", 1: "demain"}


def import_day(
    conn,
    when: str | Date = "hier",
    *,
    client: httpx.Client | None = None,
) -> dict:
    """Walk every course Le Trot advertises for `when` and import each.

    `when` accepts 'hier'/'aujourd-hui'/'demain' (relative) or an ISO date
    string (we'll filter the listing to that date).
    """
    own_client = client is None
    if own_client:
        client = make_client()

    summary = {
        "scraped":         0,
        "imported":        0,
        "skipped":         0,
        "horses_upserted": 0,
        "entries_upserted": 0,
    }

    try:
        if isinstance(when, Date):
            target_date = when.isoformat()
            rows = list_today(client, "hier")  # listing endpoint we use most
            rows = [r for r in rows if r["race_date"] == target_date]
        elif when in ("hier", "aujourd-hui", "demain"):
            rows = list_today(client, when)
        else:
            target_date = when
            rows = list_today(client, "hier")
            rows = [r for r in rows if r["race_date"] == target_date]

        for row in rows:
            summary["scraped"] += 1
            res = import_course(
                conn, row["race_date"], row["reunion_id"], row["course_number"],
                client=client,
            )
            if res.get("race_id"):
                summary["imported"] += 1
                summary["horses_upserted"]  += res["horses_upserted"]
                summary["entries_upserted"] += res["entries_upserted"]
            else:
                summary["skipped"] += 1
            time.sleep(0.1)

        buffer_prune(conn, "letrot")
    finally:
        if own_client:
            client.close()
    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    import sys
    from core.db import get_connection
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print("usage: python -m etl.import_letrot course <YYYY-MM-DD> <reunion> <course> | day [hier|aujourd-hui|demain|YYYY-MM-DD]")
        return

    conn = get_connection()
    try:
        cmd = sys.argv[1]
        if cmd == "course":
            print(import_course(conn, sys.argv[2], sys.argv[3], sys.argv[4]))
        elif cmd == "day":
            when = sys.argv[2] if len(sys.argv) >= 3 else "hier"
            print(import_day(conn, when))
        else:
            raise SystemExit(f"unknown command: {cmd!r}")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
