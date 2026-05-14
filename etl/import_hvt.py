"""Import HVT (Hauptverband für Traberzucht) data into stable_v2.

HVT gives us four primitives via scrapers.hvt:
    - search_horses(name)           -> list of search candidates
    - fetch_grunddaten(traberid)    -> canonical horse fields
    - fetch_formenspiegel(traberid) -> race-by-race history
    - fetch_pedigree(traberid)      -> sire/dam ids by generation

This module turns those primitives into idempotent UPSERTs against the v2
master tables. Cross-source matching is handled by etl.matching:
    - by hvt_id            (we always set this on a successful import)
    - by registration_number / ueln_number (when HVT publishes UELN)
    - we NEVER auto-merge by name only.

Key entry-points:

    import_horse(conn, hvt_id, *, with_history=True, with_pedigree=False)
        UPSERT a single horse + (optionally) all its German races.

    discover_for_unmatched_st_horses(conn, limit=100)
        Walk v2 horses with primary_source='st' and no hvt_id, search HVT
        by name+year, and import the best unique match. Soft, additive —
        skips ambiguous matches.
"""

from __future__ import annotations

import logging
import time
from datetime import date as Date
from typing import Iterable, Optional

import httpx
from psycopg2.extras import Json

from core.db import buffer_insert, buffer_prune
from core.exchange import to_sek
from etl.matching import upsert_horse, upsert_person, upsert_track, upsert_race, upsert_entry
from scrapers.hvt import (
    make_client,
    search_horses,
    fetch_grunddaten,
    fetch_formenspiegel,
    fetch_pedigree,
)

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Track upsert (German tracks — HVT doesn't expose stable IDs, so we key by
# (name, country='DE'))
# ---------------------------------------------------------------------------

def _upsert_track(cur, track_name: str) -> int | None:
    if not track_name:
        return None

    cur.execute(
        """
        SELECT track_id, source_data, primary_source FROM track
         WHERE country = 'DE'
           AND lower(name) = lower(%s)
         LIMIT 1
        """,
        (track_name,),
    )
    row = cur.fetchone()
    if row:
        return row[0]

    # Insert a fresh DE track. country='DE', sport='trot'.
    cur.execute(
        """
        INSERT INTO track (name, country, sport, source_data, primary_source, last_updated_at)
        VALUES (%s, 'DE', 'trot', jsonb_build_object('hvt', jsonb_build_object('discovered_via', 'formenspiegel')), 'hvt', NOW())
        RETURNING track_id
        """,
        (track_name,),
    )
    return cur.fetchone()[0]


# ---------------------------------------------------------------------------
# Person upsert (driver/trainer — HVT gives us a 6-digit fahrerid)
# ---------------------------------------------------------------------------

def _upsert_driver(cur, name: str | None, hvt_id: str | None) -> int | None:
    if not name and not hvt_id:
        return None
    return upsert_person(
        cur, "hvt", hvt_id or f"name:{name}",
        {"name": name, "license_country": "DE"},
        raw_payload={"name": name, "hvt_id": hvt_id, "role": "driver"},
        role_flags={"is_driver": True},
    )


# ---------------------------------------------------------------------------
# Horse + pedigree upsert
# ---------------------------------------------------------------------------

_GERMAN_TIME_RE = None
def _parse_de_time_to_seconds(s: str | None) -> float | None:
    """'1:13,7' -> 73.7 (seconds per km)."""
    if not s or s in ("dis.r.", "n.p.", "-", "n. p."):
        return None
    s = s.strip()
    m_full = None
    import re as _re
    if not _GERMAN_TIME_RE:
        globals()["_GERMAN_TIME_RE"] = _re.compile(r"^(\d+):(\d+)[,.](\d+)$")
    m = globals()["_GERMAN_TIME_RE"].match(s)
    if not m:
        return None
    return int(m[1]) * 60 + int(m[2]) + int(m[3]) / 10.0


def import_horse(
    conn,
    hvt_id: str | int,
    *,
    client: httpx.Client | None = None,
    with_history: bool = True,
    with_pedigree: bool = False,
) -> dict:
    """Upsert one HVT horse. Returns {hvt_id, horse_id, races_imported, ...}."""
    own_client = client is None
    if own_client:
        client = make_client()

    summary = {
        "hvt_id": str(hvt_id),
        "horse_id": None,
        "races_imported": 0,
        "races_skipped": 0,
        "ancestors_linked": 0,
    }

    try:
        gd = fetch_grunddaten(client, hvt_id)
        if not gd or not gd.get("name"):
            log.warning("hvt %s: no grunddaten", hvt_id)
            summary["error"] = "no grunddaten"
            return summary

        horse_id = _upsert_horse_from_grunddaten(conn, gd)
        summary["horse_id"] = horse_id

        if with_history:
            races = fetch_formenspiegel(client, hvt_id)
            for r in races:
                if _import_one_race(conn, horse_id, gd, r):
                    summary["races_imported"] += 1
                else:
                    summary["races_skipped"] += 1
            conn.commit()

        if with_pedigree:
            ped = fetch_pedigree(client, hvt_id, gens=5)
            for p in ped:
                _upsert_pedigree_node(conn, p)
                summary["ancestors_linked"] += 1
            conn.commit()
    finally:
        if own_client:
            client.close()

    return summary


def _upsert_horse_from_grunddaten(conn, gd: dict) -> int:
    canonical = {
        "name":                 gd.get("name"),
        "date_of_birth":        gd.get("date_of_birth"),
        "gender_code":          gd.get("gender_code"),
        "color":                gd.get("color"),
        "birth_country":        gd.get("birth_country"),
        "bred_country":         gd.get("bred_country"),
        "registration_country": gd.get("registration_country"),
        "sire_name":            gd.get("sire_name"),
        "dam_name":             gd.get("dam_name"),
        "scraped_starts":       gd.get("scraped_starts"),
        "scraped_wins":         gd.get("scraped_wins"),
        "scraped_record":       gd.get("scraped_record"),
    }
    raw = {
        "raw_grunddaten":  gd.get("raw_grunddaten"),
        "owner_name":      gd.get("owner_name"),
        "breeder_name":    gd.get("breeder_name"),
        "last_trainer":    gd.get("last_trainer_name"),
        "last_driver":     gd.get("last_driver_name"),
        "winnings_eur":    gd.get("scraped_prize_money_eur"),
        "chip_number":     gd.get("chip_number"),
        "sire_hvt_id":     gd.get("sire_hvt_id"),
        "dam_hvt_id":      gd.get("dam_hvt_id"),
    }
    with conn.cursor() as cur:
        return upsert_horse(
            cur, "hvt", str(gd["hvt_id"]),
            canonical,
            raw_payload=raw,
            ueln_number=gd.get("ueln_number"),
        )


def _upsert_pedigree_node(conn, ped_node: dict) -> int:
    """Insert/upsert a pedigree ancestor (limited canonical fields)."""
    canonical = {
        "name":                 ped_node.get("name"),
        "registration_country": ped_node.get("country") if ped_node.get("country") != "-" else None,
        "scraped_record":       ped_node.get("record"),
    }
    raw = {"discovered_via": "pedigree", "side": ped_node.get("side")}
    with conn.cursor() as cur:
        return upsert_horse(
            cur, "hvt", str(ped_node["traberid"]),
            canonical, raw_payload=raw,
        )


def _import_one_race(conn, horse_id: int, gd: dict, race: dict) -> bool:
    """Insert one race + entry from a Formenspiegel row.

    Returns True iff we created the entry (or it already existed and we
    refreshed it). Returns False on data-quality skips.
    """
    if not race.get("race_date") or not race.get("track_name"):
        return False

    with conn.cursor() as cur:
        track_id = _upsert_track(cur, race["track_name"])
        if not track_id:
            return False

        # Synthesise an HVT race id since HVT doesn't expose one. Use the
        # natural key (date + track + program_number-bucket). When the same
        # day has multiple races we collide on this key — but the entry
        # UNIQUE(race_id, horse_id) still saves us; we'd just attach
        # multiple entries to the same race row. Acceptable for now.
        synth_race_id = f"{race['race_date'].isoformat()}_{track_id}"

        race_canonical = {
            "race_date":    race["race_date"],
            "track_id":     track_id,
            "race_number":  None,  # HVT doesn't publish race-number
            "distance":     race.get("distance"),
            "start_method": race.get("start_method")[:1] if race.get("start_method") else None,
            "status":       "results",
        }
        race_payload = {
            "race_type":    race.get("race_type"),
            "top3_text":    race.get("top3_text"),
            "synth_id":     synth_race_id,
        }

        race_row_id = upsert_race(
            cur, "hvt", synth_race_id, race_canonical,
            raw_payload=race_payload,
        )

        driver_id = _upsert_driver(cur, race.get("driver_name"), race.get("driver_hvt_id"))

        time_secs = _parse_de_time_to_seconds(race.get("time_text"))
        prize_eur = race.get("prize_eur")
        prize_kr, prize_orig, fx_rate, fx_date = to_sek(
            prize_eur, "EUR", race["race_date"]
        )

        entry_canonical = {
            "program_number": race.get("program_number"),
            "distance":       race.get("distance"),
            "placement":      race.get("placement"),
            "placement_text": race.get("placement_text"),
            "time_text":      race.get("time_text"),
            "time_seconds":   time_secs,
            "auto":           (race.get("start_method") or "").startswith("A"),
            "prize_kr":       prize_kr or 0,
            "prize_currency": "EUR",
            "prize_original": prize_orig,
            "prize_fx_rate":  fx_rate,
            "prize_fx_date":  fx_date,
            "driver_id":      driver_id,
            # Foreign disqualification text — translate common German codes
            "disqualified":   "dis." in (race.get("time_text") or "").lower(),
        }
        entry_payload = {
            "evq":            race.get("evq"),
            "race_type":      race.get("race_type"),
            "top3":           race.get("top3_text"),
            "driver_name":    race.get("driver_name"),
            "driver_hvt_id":  race.get("driver_hvt_id"),
            "prize_eur":      prize_eur,
        }
        upsert_entry(
            cur, "hvt", race_row_id, horse_id,
            entry_canonical,
            raw_payload=entry_payload,
        )
    conn.commit()
    return True


# ---------------------------------------------------------------------------
# Discovery: find HVT matches for v2 horses we already know about
# ---------------------------------------------------------------------------

def discover_for_unmatched_horses(
    conn,
    limit: int = 100,
    *,
    only_foreign: bool = True,
    client: httpx.Client | None = None,
) -> dict:
    """Walk a batch of v2 horses without an hvt_id and try to find them on HVT.

    `only_foreign=True` restricts to horses whose registration_country isn't SE
    (those are the ones HVT is most likely to know about beyond what TravSport
    captured).
    """
    own_client = client is None
    if own_client:
        client = make_client()

    where = ["hvt_id IS NULL", "name IS NOT NULL", "date_of_birth IS NOT NULL"]
    if only_foreign:
        where.append("(registration_country IS NULL OR registration_country <> 'SE')")
    sql = (
        f"SELECT horse_id, name, date_of_birth, registration_country "
        f"  FROM horse WHERE {' AND '.join(where)} "
        f"  ORDER BY scraped_prize_money_kr DESC NULLS LAST "
        f"  LIMIT {int(limit)}"
    )

    summary = {"checked": 0, "matched": 0, "ambiguous": 0, "missing": 0, "errors": 0}

    try:
        with conn.cursor() as cur:
            cur.execute(sql)
            candidates = cur.fetchall()

        for horse_id, name, dob, country in candidates:
            summary["checked"] += 1
            try:
                hits = search_horses(client, name)
            except Exception as e:
                log.warning("hvt search failed for %r: %s", name, e)
                summary["errors"] += 1
                continue

            target_year = dob.year if isinstance(dob, Date) else None
            picks = [h for h in hits if h["year"] == target_year and h["name"].lower() == (name or "").lower()]
            if len(picks) == 1:
                imp = import_horse(conn, picks[0]["traberid"], client=client, with_history=False)
                if imp.get("horse_id"):
                    summary["matched"] += 1
                else:
                    summary["errors"] += 1
            elif len(picks) > 1:
                summary["ambiguous"] += 1
            else:
                summary["missing"] += 1
            time.sleep(0.05)
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
        print("usage: python -m etl.import_hvt horse <traberid> [--with-pedigree] | discover [--limit N]")
        return

    conn = get_connection()
    try:
        cmd = sys.argv[1]
        if cmd == "horse":
            tid = sys.argv[2]
            with_pedigree = "--with-pedigree" in sys.argv
            print(import_horse(conn, tid, with_pedigree=with_pedigree))
        elif cmd == "discover":
            limit = 50
            if "--limit" in sys.argv:
                limit = int(sys.argv[sys.argv.index("--limit") + 1])
            print(discover_for_unmatched_horses(conn, limit=limit))
        else:
            raise SystemExit(f"unknown command: {cmd!r}")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
