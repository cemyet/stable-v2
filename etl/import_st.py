"""
TravSport (`st`) source importer.

Two entry points:

  * `backfill_from_v1(v1_conn, v2_conn, *, batch_size=2000)`
        Streams v1's already-parsed clean tables (horse, person, track,
        race_day, race, race_result/entry, history tables) into the v2
        master tables, using etl.matching helpers.

  * `upsert_horse_from_st(v2_conn, payload)` / `upsert_raceday_from_st(...)`
        Live-mode: scrapers parse a TravSport HTML page or raceday API
        response in-memory and call these to write to v2.

Pedigree FK fixup is done in two passes:
  Pass 1 inserts every horse with sire_id/dam_id = NULL but stores the
         v1 father_horse_id / mother_horse_id under source_data.st.
  Pass 2 walks horses that have father_horse_id / mother_horse_id under
         source_data.st and resolves them to canonical v2 sire_id/dam_id.
"""

from __future__ import annotations

import re
import sys
from typing import Iterator

import psycopg2.extras

from core.common import (
    classify_placement,
    derive_start_method_from_time,
    normalize_country,
    parse_heading,
    parse_iso_date,
    parse_km_time_seconds,
    parse_money_kr,
    parse_position_distance,
)
from core.identity import resolve_horse, resolve_person

from .matching import (
    upsert_horse,
    upsert_person,
    upsert_track,
    upsert_race,
    upsert_entry,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _named_cursor(conn, name: str = "st_stream"):
    """Server-side cursor for streaming large result sets."""
    return conn.cursor(name=name)


def _print(msg: str) -> None:
    sys.stdout.write(msg + "\n")
    sys.stdout.flush()


# ---------------------------------------------------------------------------
# TRACK
# ---------------------------------------------------------------------------

def _galopp_v1_track_codes(v1_conn) -> set[str]:
    """Set of ST track_codes that v1 knows are galopp (via atg_track.sport).

    Used to skip galopp races on the ST side (v1.race uses race_day.track_code,
    which is an ST code; some galopp tracks share an st_code with their ATG
    counterpart).
    """
    with v1_conn.cursor() as cur:
        cur.execute(
            "SELECT DISTINCT track_code FROM atg_track "
            " WHERE sport = 'gallop' AND track_code IS NOT NULL"
        )
        return {r[0] for r in cur.fetchall()}


def _galopp_v2_track_ids(v2_conn) -> set[int]:
    """Set of v2 track_ids tagged as galopp."""
    with v2_conn.cursor() as cur:
        cur.execute("SELECT track_id FROM track WHERE sport = 'gallop'")
        return {r[0] for r in cur.fetchall()}


def backfill_tracks(v1_conn, v2_conn) -> int:
    """Copy v1.track + v1.atg_track → v2.track, skipping galopp tracks.

    ST (`v1.track`) has no sport column, so we use `atg_track.track_code`
    as the cross-reference: any ST track_code that v1 knows as galopp is
    skipped on the ST side too.
    """
    galopp_codes = _galopp_v1_track_codes(v1_conn)
    n = 0
    skipped_galopp = 0

    with v1_conn.cursor() as src, v2_conn.cursor() as dst:
        src.execute(
            "SELECT code, name, country, first_seen_at, last_seen_at FROM track"
        )
        for code, name, country, fs, ls in src.fetchall():
            if code in galopp_codes:
                skipped_galopp += 1
                continue
            upsert_track(
                dst, "st", code,
                {
                    "name": name,
                    "country": country,
                    "first_seen_at": fs,
                    "last_seen_at": ls,
                    "sport": "trot",
                },
            )
            n += 1
    v2_conn.commit()

    with v1_conn.cursor() as src, v2_conn.cursor() as dst:
        src.execute(
            "SELECT atg_track_id, name, country_code, sport, track_code, "
            "       first_seen, last_seen "
            "  FROM atg_track "
            " WHERE sport != 'gallop' OR sport IS NULL"
        )
        for atg_id, name, country, sport, st_code, fs, ls in src.fetchall():
            fields = {
                "name": name,
                "country": country,
                "sport": sport or "trot",
                "first_seen_at": fs,
                "last_seen_at": ls,
            }
            # upsert_track now handles cross-source dedup by (lower(name),
            # country) automatically — no need for the manual st_code lookup
            # we used to do here.
            upsert_track(dst, "atg", atg_id, fields)
            n += 1
    v2_conn.commit()
    if skipped_galopp:
        _print(f"  skipped {skipped_galopp:,} galopp st_code tracks")
    return n


# ---------------------------------------------------------------------------
# PERSON
# ---------------------------------------------------------------------------

PERSON_FIELDS = (
    "person_id, name, person_type, organisation, source_of_data, "
    "is_driver, is_trainer, is_owner, is_breeder, short_name"
)


def backfill_persons(v1_conn, v2_conn, *, batch_size: int = 5000) -> int:
    """Copy all v1.person rows → v2.person."""
    n = 0
    with _named_cursor(v1_conn) as src, v2_conn.cursor() as dst:
        src.execute(f"SELECT {PERSON_FIELDS} FROM person")
        while True:
            rows = src.fetchmany(batch_size)
            if not rows:
                break
            for row in rows:
                (pid, name, ptype, org, sod,
                 d, t, o, b, short) = row
                upsert_person(
                    dst, "st", pid,
                    {
                        "name": name,
                        "person_type": ptype,
                        "short_name": short,
                    },
                    role_flags={
                        "is_driver":  bool(d),
                        "is_trainer": bool(t),
                        "is_owner":   bool(o),
                        "is_breeder": bool(b),
                    },
                )
                n += 1
            v2_conn.commit()
            _print(f"  persons: {n:,}")
    return n


# ---------------------------------------------------------------------------
# HORSE  (two-pass: insert all rows first, then resolve pedigree FKs)
# ---------------------------------------------------------------------------

HORSE_FIELDS = (
    "horse_id, name, date_of_birth, gender_code, color, breed_code, "
    "registration_number, ueln_number, birth_country, bred_country, "
    "registration_country, is_dead, is_guest_horse, has_offspring, "
    "breed_index, inbreed_coefficient, "
    "father_horse_id, father_name, mother_horse_id, mother_name, "
    "scraped_starts, scraped_wins, scraped_prize_money_kr, scraped_record"
)


def backfill_horses_pass1(v1_conn, v2_conn, *, batch_size: int = 2000) -> int:
    """Pass 1 of horse import: every horse upserted, pedigree FKs left NULL.

    The original v1 father_horse_id / mother_horse_id are kept under
    source_data.st.father_horse_id so pass 2 can resolve them to v2 ids.
    """
    n = 0
    with _named_cursor(v1_conn) as src, v2_conn.cursor() as dst:
        src.execute(f"SELECT {HORSE_FIELDS} FROM horse")
        while True:
            rows = src.fetchmany(batch_size)
            if not rows:
                break
            for r in rows:
                (h_id, name, dob, gender, color, breed,
                 reg, ueln, bcountry, bred_c, reg_c,
                 is_dead, is_guest, has_off,
                 breed_idx, inbreed,
                 father_id, father_name, mother_id, mother_name,
                 sc_starts, sc_wins, sc_prize, sc_record) = r
                fields = {
                    "name": name,
                    "date_of_birth": dob,
                    "gender_code": gender,
                    "color": color,
                    "breed_code": breed,
                    "birth_country": bcountry,
                    "bred_country": bred_c,
                    "registration_country": reg_c,
                    "is_dead": is_dead,
                    "is_guest_horse": is_guest,
                    "has_offspring": has_off,
                    "breed_index": breed_idx,
                    "inbreed_coefficient": inbreed,
                    "sire_name": father_name,
                    "dam_name": mother_name,
                    "scraped_starts": sc_starts,
                    "scraped_wins": sc_wins,
                    "scraped_prize_money_kr": sc_prize,
                    "scraped_record": sc_record,
                }
                # Stash v1 pedigree pointers in source_data for pass 2.
                payload = {
                    "father_horse_id": father_id,
                    "mother_horse_id": mother_id,
                }
                upsert_horse(
                    dst, "st", h_id, fields,
                    raw_payload=payload,
                    registration_number=reg,
                    ueln_number=ueln,
                )
                n += 1
            v2_conn.commit()
            _print(f"  horses pass1: {n:,}")
    return n


def backfill_horses_pass2(v2_conn) -> int:
    """Resolve sire_id / dam_id from source_data.st.{father,mother}_horse_id."""
    with v2_conn.cursor() as cur:
        cur.execute(
            """
            UPDATE horse h
               SET sire_id = sire.horse_id
              FROM horse sire
             WHERE h.sire_id IS NULL
               AND sire.st_id = (h.source_data->'st'->>'father_horse_id')::int
               AND (h.source_data->'st'->>'father_horse_id') ~ '^\\d+$'
            """
        )
        n_sire = cur.rowcount
        cur.execute(
            """
            UPDATE horse h
               SET dam_id = dam.horse_id
              FROM horse dam
             WHERE h.dam_id IS NULL
               AND dam.st_id = (h.source_data->'st'->>'mother_horse_id')::int
               AND (h.source_data->'st'->>'mother_horse_id') ~ '^\\d+$'
            """
        )
        n_dam = cur.rowcount
    v2_conn.commit()
    _print(f"  pedigree resolved: {n_sire:,} sires, {n_dam:,} dams")
    return n_sire + n_dam


# ---------------------------------------------------------------------------
# RACE_DAY + RACE   (TravSport keeps race_day separate; we collapse it)
# ---------------------------------------------------------------------------

def backfill_races(v1_conn, v2_conn, *, batch_size: int = 2000) -> int:
    """Copy v1.race + v1.race_day → v2.race (one row per race).

    Galopp races are skipped — any race whose track is missing in v2 (because
    we filtered it in backfill_tracks) or tagged sport='gallop' is dropped.
    We materialise the join so the loader pulls track + date for each race.
    """
    galopp_track_ids = _galopp_v2_track_ids(v2_conn)
    skipped_galopp = 0
    n = 0
    with _named_cursor(v1_conn) as src, v2_conn.cursor() as dst:
        src.execute(
            """
            SELECT r.race_id, r.race_day_id, r.race_number, r.start_time,
                   r.distance, r.start_method, r.heading, r.proposition_text,
                   r.track_conditions, r.victory_margin, r.tempo_text,
                   r.total_prize_kr,
                   r.turnover_vinnare_kr, r.turnover_plats_kr,
                   r.turnover_trio_kr, r.turnover_tvilling_kr,
                   r.result_and_odds_trio,
                   rd.race_date, rd.track_code, rd.heading AS day_heading,
                   rd.info AS day_info, rd.attendance, rd.organisation
            FROM race r
            LEFT JOIN race_day rd ON rd.race_day_id = r.race_day_id
            """
        )
        while True:
            rows = src.fetchmany(batch_size)
            if not rows:
                break
            for r in rows:
                (race_id, race_day_id, race_no, start_time,
                 distance, start_method, heading, prop, track_cond,
                 victory_margin, tempo, total_prize,
                 turn_v, turn_p, turn_trio, turn_tvilling,
                 trio_result,
                 race_date, track_code, day_heading,
                 day_info, attendance, org) = r
                dst.execute(
                    "SELECT track_id FROM track WHERE st_code = %s",
                    (track_code,),
                )
                row = dst.fetchone()
                track_id = row[0] if row else None
                if track_id is None or track_id in galopp_track_ids:
                    skipped_galopp += 1
                    continue

                fields = {
                    "race_date": race_date,
                    "track_id": track_id,
                    "race_number": race_no,
                    "start_time": start_time,
                    "distance": distance,
                    "start_method": start_method,
                    "heading": heading,
                    "proposition_text": prop,
                    "track_conditions": track_cond,
                    "victory_margin": victory_margin,
                    "tempo_text": tempo,
                    "total_prize_kr": total_prize,
                    "st_race_day_id": race_day_id,
                }
                payload = {
                    "race_day_id": race_day_id,
                    "day_heading": day_heading,
                    "day_info": day_info,
                    "attendance": attendance,
                    "organisation": org,
                    "turnover_vinnare_kr":  turn_v,
                    "turnover_plats_kr":    turn_p,
                    "turnover_trio_kr":     turn_trio,
                    "turnover_tvilling_kr": turn_tvilling,
                    "result_and_odds_trio": trio_result,
                }
                upsert_race(
                    dst, "st", race_id, fields,
                    raw_payload=payload,
                )
                n += 1
            v2_conn.commit()
            _print(f"  races: {n:,}")
    if skipped_galopp:
        _print(f"  skipped {skipped_galopp:,} galopp races")
    return n


# ---------------------------------------------------------------------------
# ENTRY
# ---------------------------------------------------------------------------

ENTRY_FIELDS = (
    "race_id, atg_id, horse_id, number, post, distance, "
    "placement, finish_order, placement_text, "
    "time_val, time_text, auto, gal, dq, withdrawn, "
    "odds, prize, "
    "driver_id, driver_changed, trainer_id, "
    "age, sex, earnings_pre, "
    "shoe_code, shoe_front_changed, shoe_back_changed, "
    "sulky, sulky_changed, source, tillagg"
)


def backfill_entries(v1_conn, v2_conn, *, batch_size: int = 10000) -> int:
    """Copy v1.entry → v2.entry.

    Performance-critical (~4.3M rows). We avoid the generic upsert path
    here because v2.entry starts empty: in-memory lookup maps + bulk
    INSERT via execute_values. Falls back to UPSERT semantics on rerun
    via ON CONFLICT (race_id, horse_id) DO UPDATE.
    """
    from psycopg2.extras import execute_values

    _print("  building lookup maps...")
    with v2_conn.cursor() as cur:
        cur.execute("SELECT horse_id, st_id FROM horse WHERE st_id IS NOT NULL")
        horse_map: dict = {st_id: hid for hid, st_id in cur.fetchall()}
        cur.execute("SELECT race_id, st_race_id FROM race WHERE st_race_id IS NOT NULL")
        race_st_map: dict = {st_id: rid for rid, st_id in cur.fetchall()}
        cur.execute("SELECT race_id, atg_race_id FROM race WHERE atg_race_id IS NOT NULL")
        race_atg_map: dict = {atg_id: rid for rid, atg_id in cur.fetchall()}
        cur.execute("SELECT person_id, st_id FROM person WHERE st_id IS NOT NULL")
        person_map: dict = {st_id: pid for pid, st_id in cur.fetchall()}
    _print(
        f"    horses={len(horse_map):,} races={len(race_st_map):,}+{len(race_atg_map):,} "
        f"persons={len(person_map):,}"
    )

    INSERT_SQL = """
        INSERT INTO entry (
            race_id, horse_id, program_number, post, distance, tillagg,
            placement, finish_order, placement_text,
            withdrawn, galopp, disqualified,
            time_seconds, time_text, auto,
            odds, prize_kr,
            driver_id, driver_changed, trainer_id,
            age, sex, earnings_pre,
            shoe_code, shoe_front_changed, shoe_back_changed,
            sulky, sulky_changed,
            source_data, primary_source
        ) VALUES %s
        ON CONFLICT (race_id, horse_id) DO NOTHING
    """

    n = 0
    skipped_no_race = 0
    skipped_no_horse = 0
    with _named_cursor(v1_conn) as src, v2_conn.cursor() as dst:
        src.execute(f"SELECT {ENTRY_FIELDS} FROM entry")
        while True:
            rows = src.fetchmany(batch_size)
            if not rows:
                break
            batch = []
            for r in rows:
                (st_race_id, atg_id, st_horse_id, number, post, distance,
                 placement, finish_order, placement_text,
                 time_val, time_text, auto, gal, dq, withdrawn,
                 odds, prize,
                 driver_st_id, driver_changed, trainer_st_id,
                 age, sex, earnings_pre,
                 shoe_code, shoe_front_changed, shoe_back_changed,
                 sulky, sulky_changed, source, tillagg) = r

                # ST stores sex Swedish-lowercase (v/h/s); the canonical
                # entry.sex convention is uppercase H/V/S.
                if isinstance(sex, str) and sex.lower() in ("v", "h", "s"):
                    sex = sex.upper()

                race_id = race_st_map.get(st_race_id) if st_race_id else None
                if race_id is None and atg_id:
                    race_id = race_atg_map.get(atg_id)
                if race_id is None:
                    skipped_no_race += 1
                    continue
                horse_id = horse_map.get(st_horse_id)
                if horse_id is None:
                    skipped_no_horse += 1
                    continue

                driver_id  = person_map.get(driver_st_id)  if driver_st_id  else None
                trainer_id = person_map.get(trainer_st_id) if trainer_st_id else None

                primary_source = source if source in ("st", "atg") else "st"
                source_data = '{"' + primary_source + '": {}}'  # tiny placeholder

                batch.append((
                    race_id, horse_id, number, post, distance, tillagg,
                    placement, finish_order, placement_text,
                    bool(withdrawn), bool(gal), bool(dq),
                    time_val, time_text, auto,
                    odds, prize or 0,
                    driver_id, bool(driver_changed), trainer_id,
                    age, sex, earnings_pre,
                    shoe_code, shoe_front_changed, shoe_back_changed,
                    sulky, sulky_changed,
                    source_data, primary_source,
                ))
            if batch:
                execute_values(dst, INSERT_SQL, batch, page_size=1000)
                v2_conn.commit()
                n += len(batch)
                _print(f"  entries: {n:,}")
    if skipped_no_race or skipped_no_horse:
        _print(f"  skipped: no_race={skipped_no_race:,} no_horse={skipped_no_horse:,}")
    return n


def _lookup_canonical(cur, table: str, source_id_col: str, source_id) -> int | None:
    """Return canonical pk for a (source_id_col, source_id) match, or None."""
    if source_id is None:
        return None
    pk_col = {"horse": "horse_id", "race": "race_id", "person": "person_id", "track": "track_id"}[table]
    cur.execute(f"SELECT {pk_col} FROM {table} WHERE {source_id_col} = %s", (source_id,))
    row = cur.fetchone()
    return row[0] if row else None


# ---------------------------------------------------------------------------
# OWNER + TRAINER history
# ---------------------------------------------------------------------------

def backfill_history(v1_conn, v2_conn) -> int:
    """Copy owner/trainer history tables. Lookups translate v1 ids → v2."""
    n_owner = 0
    n_trainer = 0
    with v1_conn.cursor() as src, v2_conn.cursor() as dst:
        src.execute(
            "SELECT horse_id, owner_id, owner_name, ownership_form, from_date, to_date "
            "FROM horse_owner_history"
        )
        for st_horse, st_owner, name, form, fd, td in src.fetchall():
            h = _lookup_canonical(dst, "horse", "st_id", st_horse)
            o = _lookup_canonical(dst, "person", "st_id", st_owner) if st_owner else None
            if h is None:
                continue
            dst.execute(
                """
                INSERT INTO horse_owner_history
                  (horse_id, owner_id, owner_name, ownership_form, from_date, to_date, source)
                VALUES (%s,%s,%s,%s,%s,%s,'st')
                ON CONFLICT (horse_id, from_date) DO UPDATE
                  SET owner_id = EXCLUDED.owner_id,
                      owner_name = EXCLUDED.owner_name,
                      ownership_form = EXCLUDED.ownership_form,
                      to_date = EXCLUDED.to_date
                """,
                (h, o, name, form, fd, td),
            )
            n_owner += 1
        src.execute(
            "SELECT horse_id, trainer_id, trainer_name, license_text, from_date, to_date "
            "FROM horse_trainer_history"
        )
        for st_horse, st_trainer, name, lic, fd, td in src.fetchall():
            h = _lookup_canonical(dst, "horse", "st_id", st_horse)
            t = _lookup_canonical(dst, "person", "st_id", st_trainer) if st_trainer else None
            if h is None:
                continue
            dst.execute(
                """
                INSERT INTO horse_trainer_history
                  (horse_id, trainer_id, trainer_name, license_text, from_date, to_date, source)
                VALUES (%s,%s,%s,%s,%s,%s,'st')
                ON CONFLICT (horse_id, from_date) DO UPDATE
                  SET trainer_id = EXCLUDED.trainer_id,
                      trainer_name = EXCLUDED.trainer_name,
                      license_text = EXCLUDED.license_text,
                      to_date = EXCLUDED.to_date
                """,
                (h, t, name, lic, fd, td),
            )
            n_trainer += 1
    v2_conn.commit()
    _print(f"  owner history: {n_owner:,}; trainer history: {n_trainer:,}")
    return n_owner + n_trainer


# ---------------------------------------------------------------------------
# ORCHESTRATION
# ---------------------------------------------------------------------------

def backfill_from_v1(v1_conn, v2_conn) -> dict:
    """Run all backfill steps in dependency order. Returns counts."""
    counts = {}
    _print(">> tracks")
    counts["tracks"] = backfill_tracks(v1_conn, v2_conn)
    _print(">> persons")
    counts["persons"] = backfill_persons(v1_conn, v2_conn)
    _print(">> horses (pass 1)")
    counts["horses"] = backfill_horses_pass1(v1_conn, v2_conn)
    _print(">> horses (pass 2 — pedigree FKs)")
    counts["pedigree_links"] = backfill_horses_pass2(v2_conn)
    _print(">> races")
    counts["races"] = backfill_races(v1_conn, v2_conn)
    _print(">> entries")
    counts["entries"] = backfill_entries(v1_conn, v2_conn)
    _print(">> owner + trainer history")
    counts["history"] = backfill_history(v1_conn, v2_conn)
    return counts


# ===========================================================================
# NATIVE LIVE MODE — ETL from the durable st_horse_raw blobs (no v1 bridge).
#
# Mirrors v1's etl/horses.py field map but writes through resolve_horse /
# resolve_person so it dedups against existing ATG/cross-source rows (and
# honours manual-merge redirects) instead of keying on horse_id directly.
# ===========================================================================

# Data types we need from st_horse_raw to build a horse passport.
ST_HORSE_DATA_TYPES = (
    "horse-basic-information", "horse-history", "lineage-small", "horse-statistics",
)

_MONEY_RE = re.compile(r"[\d\s]+")


def _parse_livs_stats(stats_blob) -> dict:
    """Lifetime ('Livs') stats from the horse-statistics blob.

    Returns {starts, wins, prize_money_kr, record} (all nullable).
    Identical semantics to v1 etl/horses._parse_livs_stats.
    """
    out: dict = {"starts": None, "wins": None, "prize_money_kr": None, "record": None}
    if not isinstance(stats_blob, dict):
        return out
    for s in (stats_blob.get("statistics") or []):
        if not isinstance(s, dict) or s.get("year") != "Livs":
            continue
        try:
            out["starts"] = int(s["numberOfStarts"])
        except (KeyError, ValueError, TypeError):
            pass
        parts = (s.get("placements") or "").split("-")
        if parts and parts[0]:
            try:
                out["wins"] = int(parts[0])
            except (ValueError, TypeError):
                pass
        pm = _MONEY_RE.search((s.get("prizeMoney") or "").replace("\xa0", " "))
        if pm:
            try:
                out["prize_money_kr"] = int(pm.group().replace(" ", ""))
            except ValueError:
                pass
        out["record"] = s.get("mark") or None
        break
    return out


def _resolve_history_persons(cur, horse_id: int, history: dict) -> None:
    """Rewrite owner/trainer history for `horse_id` from the history blob.

    The upstream lists ARE the canonical state, so we delete+reinsert (matching
    v1). Each owner/trainer is resolved through resolve_person so the row's
    *_id points at the canonical person, not a raw ST id.
    """
    if not isinstance(history, dict):
        return

    owners = history.get("owners")
    if owners:
        cur.execute("DELETE FROM horse_owner_history WHERE horse_id = %s AND source = 'st'",
                    (horse_id,))
        for o in owners:
            if not isinstance(o, dict):
                continue
            o_sid = o.get("id")
            from_d = parse_iso_date(o.get("from"))
            if o_sid is None or from_d is None:
                continue
            pid = resolve_person(
                cur, source="st", source_id=o_sid,
                canonical_fields={"name": o.get("name")},
                raw_payload={"organisation": o.get("organisation"),
                             "source_of_data": o.get("sourceOfData")},
                role_flags={"is_owner": True},
            )
            cur.execute(
                """
                INSERT INTO horse_owner_history
                    (horse_id, owner_id, owner_name, ownership_form, from_date, to_date, source)
                VALUES (%s, %s, %s, %s, %s, %s, 'st')
                ON CONFLICT (horse_id, from_date) DO UPDATE SET
                    owner_id       = EXCLUDED.owner_id,
                    owner_name     = EXCLUDED.owner_name,
                    ownership_form = EXCLUDED.ownership_form,
                    to_date        = EXCLUDED.to_date,
                    source         = 'st'
                """,
                (horse_id, pid, o.get("name"), o.get("ownershipForm"),
                 from_d, parse_iso_date(o.get("to"))),
            )

    trainers = history.get("trainers")
    if trainers:
        cur.execute("DELETE FROM horse_trainer_history WHERE horse_id = %s AND source = 'st'",
                    (horse_id,))
        for t in trainers:
            if not isinstance(t, dict):
                continue
            t_sid = t.get("id")
            from_d = parse_iso_date(t.get("from"))
            if t_sid is None or from_d is None:
                continue
            pid = resolve_person(
                cur, source="st", source_id=t_sid,
                canonical_fields={"name": t.get("name")},
                raw_payload={"organisation": t.get("organisation"),
                             "source_of_data": t.get("sourceOfData")},
                role_flags={"is_trainer": True},
            )
            cur.execute(
                """
                INSERT INTO horse_trainer_history
                    (horse_id, trainer_id, trainer_name, license_text, from_date, to_date, source)
                VALUES (%s, %s, %s, %s, %s, %s, 'st')
                ON CONFLICT (horse_id, from_date) DO UPDATE SET
                    trainer_id   = EXCLUDED.trainer_id,
                    trainer_name = EXCLUDED.trainer_name,
                    license_text = EXCLUDED.license_text,
                    to_date      = EXCLUDED.to_date,
                    source       = 'st'
                """,
                (horse_id, pid, t.get("name"), t.get("licenseText"),
                 from_d, parse_iso_date(t.get("to"))),
            )


def _link_pedigree_fks(cur, horse_id: int, father_st_id, mother_st_id) -> None:
    """Set sire_id / dam_id when the parent horse already exists in v2 (by st_id)."""
    if father_st_id:
        cur.execute(
            "UPDATE horse h SET sire_id = p.horse_id "
            "FROM horse p WHERE p.st_id = %s AND h.horse_id = %s",
            (father_st_id, horse_id),
        )
    if mother_st_id:
        cur.execute(
            "UPDATE horse h SET dam_id = p.horse_id "
            "FROM horse p WHERE p.st_id = %s AND h.horse_id = %s",
            (mother_st_id, horse_id),
        )


def upsert_horse_from_st(cur, st_horse_id: int, blobs: dict) -> int | None:
    """Upsert one horse from its st_horse_raw blobs. Returns canonical horse_id.

    `blobs` maps data_type -> parsed JSON (as stored in st_horse_raw). Writes
    are made on `cur`; the caller owns the transaction/commit.
    """
    basic = blobs.get("horse-basic-information") or {}
    if not isinstance(basic, dict) or "id" not in basic:
        return None
    history = blobs.get("horse-history") or {}
    lineage = blobs.get("lineage-small") or {}
    livs = _parse_livs_stats(blobs.get("horse-statistics"))

    reg_status = basic.get("registrationStatus") or {}
    trot_extra = basic.get("trotAdditionalInformation") or {}

    father = lineage.get("father") if isinstance(lineage, dict) else None
    mother = lineage.get("mother") if isinstance(lineage, dict) else None
    father = father if isinstance(father, dict) else {}
    mother = mother if isinstance(mother, dict) else {}
    father_st_id = father.get("horseId")
    mother_st_id = mother.get("horseId")

    canonical_fields = {
        "name": basic.get("name"),
        "date_of_birth": parse_iso_date(basic.get("dateOfBirth")),
        "gender_code": (basic.get("horseGender") or {}).get("code"),
        "color": basic.get("color"),
        "breed_code": (basic.get("horseBreed") or {}).get("code"),
        "birth_country": normalize_country(basic.get("birthCountryCode")),
        "bred_country": normalize_country(basic.get("bredCountryCode")),
        "registration_country": normalize_country(basic.get("registrationCountryCode")),
        "is_dead": reg_status.get("dead"),
        "is_guest_horse": basic.get("guestHorse"),
        "has_offspring": basic.get("offspringExists"),
        "breed_index": trot_extra.get("breedIndex"),
        "inbreed_coefficient": trot_extra.get("inbreedCoefficient"),
        "sire_name": father.get("name"),
        "dam_name": mother.get("name"),
        "scraped_starts": livs["starts"],
        "scraped_wins": livs["wins"],
        "scraped_prize_money_kr": livs["prize_money_kr"],
        "scraped_record": livs["record"],
    }
    payload = {
        "father_horse_id": father_st_id,
        "mother_horse_id": mother_st_id,
        "year_of_death": reg_status.get("yearOfDeath"),
    }

    horse_id = resolve_horse(
        cur, source="st", source_id=st_horse_id,
        canonical_fields=canonical_fields,
        raw_payload=payload,
        registration_number=basic.get("registrationNumber"),
        ueln_number=basic.get("uelnNumber"),
        sire_name=father.get("name"),
        dam_name=mother.get("name"),
    )
    if horse_id is None:
        return None

    # Breeder: keep the person row fresh (v2 horse has no breeder FK).
    breeder = basic.get("breeder") or {}
    if breeder.get("id"):
        resolve_person(
            cur, source="st", source_id=breeder["id"],
            canonical_fields={"name": breeder.get("name")},
            role_flags={"is_breeder": True},
        )

    _resolve_history_persons(cur, horse_id, history)
    _link_pedigree_fks(cur, horse_id, father_st_id, mother_st_id)
    return horse_id


def _iter_st_raw_horse_ids(conn, since=None) -> Iterator[int]:
    """st_horse_raw horse_ids that have a basic-information object."""
    sql = (
        "SELECT DISTINCT horse_id FROM st_horse_raw "
        "WHERE data_type = 'horse-basic-information' "
        "  AND jsonb_typeof(raw_json) = 'object'"
    )
    params: list = []
    if since is not None:
        sql += " AND scraped_at >= %s"
        params.append(since)
    sql += " ORDER BY horse_id"
    with conn.cursor(name="st_raw_ids") as cur:
        cur.itersize = 5000
        cur.execute(sql, params)
        for (hid,) in cur:
            yield hid


def _fetch_st_blobs(conn, horse_ids: list[int]) -> dict[int, dict]:
    """{horse_id: {data_type: raw_json}} for the requested ids."""
    if not horse_ids:
        return {}
    out: dict[int, dict] = {}
    with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
        cur.execute(
            "SELECT horse_id, data_type, raw_json FROM st_horse_raw "
            "WHERE horse_id = ANY(%s) AND data_type = ANY(%s)",
            (horse_ids, list(ST_HORSE_DATA_TYPES)),
        )
        for row in cur:
            out.setdefault(row["horse_id"], {})[row["data_type"]] = row["raw_json"]
    return out


def load_st_horses_from_raw(conn, *, batch_size: int = 1000, since=None,
                            limit: int | None = None) -> int:
    """ETL all (or recently-scraped) st_horse_raw horses into master tables.

    Returns the number of horses processed. Idempotent: every write upserts.
    """
    processed = 0
    batch: list[int] = []

    def _flush(ids: list[int]) -> int:
        blobs = _fetch_st_blobs(conn, ids)
        done = 0
        with conn.cursor() as wcur:
            for hid in ids:
                bag = blobs.get(hid)
                if not bag:
                    continue
                if upsert_horse_from_st(wcur, hid, bag) is not None:
                    done += 1
        conn.commit()
        return done

    for hid in _iter_st_raw_horse_ids(conn, since=since):
        batch.append(hid)
        if len(batch) >= batch_size:
            processed += _flush(batch)
            batch = []
            if limit is not None and processed >= limit:
                return processed
    if batch and (limit is None or processed < limit):
        processed += _flush(batch)
    return processed


def load_st_horses_for_ids(conn, horse_ids: list[int]) -> int:
    """ETL a specific set of st_horse_raw horse ids (targeted, not full-scan)."""
    ids = list(dict.fromkeys(int(h) for h in horse_ids))
    if not ids:
        return 0
    done = 0
    skipped = 0
    with conn.cursor() as wcur:
        for chunk_start in range(0, len(ids), 1000):
            chunk = ids[chunk_start:chunk_start + 1000]
            blobs = _fetch_st_blobs(conn, chunk)
            for hid in chunk:
                bag = blobs.get(hid)
                if not bag:
                    continue
                # Per-horse savepoint so one bad row (e.g. a registration/UELN
                # collision with a pre-existing duplicate that the cleanup pass
                # will merge later) can't poison the whole batch.
                wcur.execute("SAVEPOINT st_horse")
                try:
                    if upsert_horse_from_st(wcur, hid, bag) is not None:
                        done += 1
                    wcur.execute("RELEASE SAVEPOINT st_horse")
                except Exception as exc:  # noqa: BLE001
                    wcur.execute("ROLLBACK TO SAVEPOINT st_horse")
                    skipped += 1
                    _print(f"  ! st horse {hid} skipped: {exc!r}")
            conn.commit()
    if skipped:
        _print(f"load_st_horses_for_ids: {done} ok, {skipped} skipped (collisions)")
    return done


# ---------------------------------------------------------------------------
# Gap discovery — the fix for v1's sequential-id-scan blind spot.
#
# v1 discovers horses by probing ids in order with a high-water mark; an id
# probed before its passport existed (200-but-empty) is never re-probed. Those
# horses surface in v2 as ATG starters (numeric atg_id) that raced on an SE
# track but never got an st_id. We re-fetch them natively here.
# ---------------------------------------------------------------------------

def discover_gap_horse_ids(conn) -> list[int]:
    """TravSport ids (== numeric atg_id) of SE-racing horses still missing st_id."""
    with conn.cursor() as cur:
        cur.execute(
            r"""
            SELECT DISTINCT h.atg_id::int
              FROM horse h
             WHERE h.st_id IS NULL
               AND h.atg_id ~ '^[0-9]+$'
               AND EXISTS (
                     SELECT 1 FROM entry e
                       JOIN race r  ON r.race_id  = e.race_id
                       JOIN track t ON t.track_id = r.track_id
                      WHERE e.horse_id = h.horse_id AND t.country = 'SE'
                   )
             ORDER BY 1
            """
        )
        return [r[0] for r in cur.fetchall()]


def gap_fill_st_horses(conn, *, dry_run: bool = False, limit: int | None = None,
                       log=_print) -> dict:
    """Scrape + ETL the gap horses natively so they gain st_id/pedigree/pill.

    Returns counts. `dry_run` only reports the discovered ids (no network, no
    writes). Idempotent and additive: ETL goes through resolve_horse so it
    attaches st_id to the existing ATG row instead of duplicating.
    """
    ids = discover_gap_horse_ids(conn)
    if limit is not None:
        ids = ids[:limit]
    out: dict = {"gap_horses": len(ids)}
    log(f"discovered {len(ids)} gap horses (numeric atg_id, st_id NULL, raced SE)")
    if dry_run:
        out["dry_run"] = True
        out["ids"] = ids
        return out
    if not ids:
        return out

    from scrapers import st_horse  # etl -> scrapers (no cycle: scrapers has no etl import)
    out["scrape"] = st_horse.scrape_horse_ids(conn, ids, skip_done=False)

    blobs = _fetch_st_blobs(conn, ids)
    etled = 0
    attached = 0
    with conn.cursor() as wcur:
        for hid in ids:
            bag = blobs.get(hid)
            if not bag:
                continue
            rid = upsert_horse_from_st(wcur, hid, bag)
            if rid is not None:
                etled += 1
        conn.commit()
        # How many of the targeted ids now actually carry st_id?
        wcur.execute(
            "SELECT count(*) FROM horse WHERE st_id = ANY(%s)", (ids,)
        )
        attached = wcur.fetchone()[0]
    out["etled"] = etled
    out["st_id_attached"] = attached
    log(f"native ST gap-fill: scraped={out['scrape']}, etled={etled}, "
        f"st_id_attached={attached}")
    return out


# ===========================================================================
# NATIVE LIVE MODE — racedays (results API) -> race + entry
#
# Mirrors v1 etl/racedays.py but writes through upsert_race / upsert_entry /
# resolve_horse / resolve_person. Track + date come from the raceday `heading`
# (parse_heading); per-row distance/start-method are derived like v1.
# ===========================================================================

def _build_st_track_code_map(conn) -> dict[int, str]:
    """{race_day_id: track_code} harvested from st_horse_raw race-results blobs.

    The raceday API has no inline track code, but the horse-side race-results
    blobs carry `trackCode` + `raceInformation.raceDayId`. Used as a secondary
    track resolver behind the heading track-name match.
    """
    out: dict[int, str] = {}
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT (elem->'raceInformation'->>'raceDayId')::int AS rdid,
                   elem->>'trackCode' AS tc, COUNT(*) c
              FROM st_horse_raw, jsonb_array_elements(raw_json) elem
             WHERE data_type = 'race-results'
               AND jsonb_typeof(raw_json) = 'array'
               AND elem->'raceInformation'->>'raceDayId' IS NOT NULL
               AND elem->>'trackCode' IS NOT NULL
             GROUP BY 1, 2
            """
        )
        best: dict[int, tuple[str, int]] = {}
        for rdid, tc, c in cur.fetchall():
            if rdid is None or not tc:
                continue
            if rdid not in best or c > best[rdid][1]:
                best[rdid] = (tc, c)
    out = {rd: code for rd, (code, _) in best.items()}
    return out


def _resolve_st_track_id(cur, track_label: str | None, track_code: str | None) -> int | None:
    """Resolve a v2 track_id: by st_code first, then by heading track name."""
    if track_code:
        cur.execute("SELECT track_id FROM track WHERE st_code = %s", (track_code,))
        row = cur.fetchone()
        if row:
            return row[0]
    if track_label:
        cur.execute(
            "SELECT track_id FROM track WHERE lower(name) = lower(%s) "
            "ORDER BY (country = 'SE') DESC NULLS LAST LIMIT 1",
            (track_label.strip(),),
        )
        row = cur.fetchone()
        if row:
            return row[0]
    return None


def _st_galopp(time_text: str | None) -> bool:
    """ST encodes a gait-break with a trailing 'g' on the km-time ('19,3g',
    '9g'). Auto-start times end in 'a' ('18,3a'); galopp+auto would be 'ag'."""
    t = (time_text or "").strip().lower()
    return t.endswith("g")


def _st_placement_text(placement: int | None, disqualified: bool, galopp: bool,
                       withdrawn: bool, placement_display: str | None) -> str | None:
    """Same convention as etl.import_atg._placement_text_from_result so the
    'g' marker / numeric retention behaves identically across sources."""
    if withdrawn:
        return "utg"
    if disqualified:
        return "d"
    if galopp:
        return str(placement) if (placement and placement >= 1) else "g"
    if placement and placement >= 1:
        return str(placement)
    pdv = (placement_display or "").strip()
    return pdv or None


def upsert_raceday_from_st(cur, race_day_id: int, raw: dict,
                           track_code_map: dict[int, str] | None = None) -> dict:
    """ETL one raceday JSON into v2 race + entry. Returns counts.

    `track_code_map` is the optional {race_day_id: track_code} from
    `_build_st_track_code_map`; pass it for batch runs to avoid rebuilding.
    """
    counts = {"races": 0, "entries": 0, "skipped_no_track": 0}
    if not isinstance(raw, dict):
        return counts
    track_code_map = track_code_map or {}

    track_label, race_date = parse_heading(raw.get("heading"))
    track_code = track_code_map.get(race_day_id)
    track_id = _resolve_st_track_id(cur, track_label, track_code)
    if track_id is None:
        counts["skipped_no_track"] = 1
        return counts

    races = raw.get("racesWithReadyResult") or []
    if not isinstance(races, list):
        return counts

    basic_by_id: dict[int, dict] = {}
    for b in (raw.get("racesWithBasicInfoAndResultStatus") or []):
        if isinstance(b, dict) and "id" in b:
            basic_by_id[b["id"]] = b

    for race in races:
        if not isinstance(race, dict) or "raceId" not in race:
            continue
        st_race_id = race["raceId"]
        gi = race.get("generalInfo") or {}
        result_rows = [r for r in (race.get("raceResultRows") or []) if isinstance(r, dict)]
        prohibited = [r for r in (race.get("prohibitedHorses") or []) if isinstance(r, dict)]

        # Derive majority start-method + distance from the result-row times.
        method_votes: dict[str, int] = {}
        dist_votes: dict[int, int] = {}
        for rr in result_rows:
            m = derive_start_method_from_time(rr.get("time"))
            if m:
                method_votes[m] = method_votes.get(m, 0) + 1
            _, d = parse_position_distance(rr.get("startPositionAndDistance"))
            if d:
                dist_votes[d] = dist_votes.get(d, 0) + 1
        start_method = max(method_votes, key=method_votes.get) if method_votes else None
        majority_distance = max(dist_votes, key=dist_votes.get) if dist_votes else None

        bbi = basic_by_id.get(st_race_id) or {}
        proposition = (bbi.get("description") or "").strip() or None

        race_fields = {
            "race_date": race_date,
            "track_id": track_id,
            "race_number": gi.get("raceNumber"),
            "distance": majority_distance,
            "start_method": start_method,
            "heading": gi.get("heading"),
            "proposition_text": proposition,
            "track_conditions": gi.get("trackConditions"),
            "victory_margin": gi.get("victoryMargin"),
            "tempo_text": gi.get("tempoText"),
            "total_prize_kr": parse_money_kr(gi.get("totalPriceSum")),
            "st_race_day_id": race_day_id,
        }
        race_id = upsert_race(cur, "st", st_race_id, race_fields,
                              raw_payload={"race_day_id": race_day_id})
        counts["races"] += 1

        ran_horse_ids: set[int] = set()
        auto = (start_method == "A") if start_method else None

        for rr in result_rows:
            horse = rr.get("horse") or {}
            h_sid = horse.get("id")
            if h_sid is None:
                continue
            prog = None
            try:
                prog = int(rr.get("programNumber")) if rr.get("programNumber") not in (None, "") else None
            except (TypeError, ValueError):
                prog = None

            horse_id = resolve_horse(
                cur, source="st", source_id=h_sid,
                canonical_fields={"name": horse.get("name")},
                race_id=race_id, program_number=prog,
            )
            if horse_id is None:
                continue
            ran_horse_ids.add(h_sid)

            driver = rr.get("driver") or {}
            trainer = rr.get("trainer") or {}
            driver_id = resolve_person(
                cur, source="st", source_id=driver.get("id"),
                canonical_fields={"name": driver.get("name")},
                role_flags={"is_driver": True},
            ) if driver.get("id") else None
            trainer_id = resolve_person(
                cur, source="st", source_id=trainer.get("id"),
                canonical_fields={"name": trainer.get("name")},
                role_flags={"is_trainer": True},
            ) if trainer.get("id") else None

            placement, disqualified = classify_placement(
                str(rr.get("placementNumber")) if rr.get("placementNumber") is not None else None,
                rr.get("placementDisplay"),
            )
            galopp = _st_galopp(rr.get("time"))
            start_pos, distance = parse_position_distance(rr.get("startPositionAndDistance"))
            time_text = rr.get("time")
            ptext = _st_placement_text(placement, disqualified, galopp, False,
                                       rr.get("placementDisplay"))

            # entry.sex: canonical uppercase H/V/S from the resolved horse.
            cur.execute("SELECT gender_code FROM horse WHERE horse_id = %s", (horse_id,))
            grow = cur.fetchone()
            sex = (grow[0].upper() if grow and grow[0] else None)

            shoe = ((rr.get("equipmentSelection") or {}).get("shoeOption") or {})

            entry_fields = {
                "program_number": prog,
                "post": start_pos,
                "distance": distance or majority_distance,
                "placement": placement,
                "placement_text": ptext,
                "withdrawn": False,
                "disqualified": disqualified,
                "galopp": galopp,
                "time_seconds": parse_km_time_seconds(time_text),
                "time_text": time_text,
                "auto": auto,
                "driver_id": driver_id,
                "driver_changed": bool(rr.get("driverChanged")),
                "trainer_id": trainer_id,
                "sex": sex,
                "shoe_code": shoe.get("code"),
            }
            upsert_entry(cur, "st", race_id, horse_id, entry_fields,
                         raw_payload={"st_race_id": st_race_id})
            counts["entries"] += 1

        # prohibitedHorses (scratched / not allowed to start).
        for wh in prohibited:
            wh_sid = wh.get("id")
            if wh_sid is None or wh_sid in ran_horse_ids:
                continue
            horse_id = resolve_horse(
                cur, source="st", source_id=wh_sid,
                canonical_fields={"name": wh.get("name")},
                race_id=race_id,
            )
            if horse_id is None:
                continue
            upsert_entry(cur, "st", race_id, horse_id, {
                "program_number": _safe_prog(wh.get("programNumber")),
                "withdrawn": True,
                "placement_text": "utg",
            }, raw_payload={"st_race_id": st_race_id, "cause": wh.get("cause")})
            counts["entries"] += 1

    return counts


def _safe_prog(v) -> int | None:
    try:
        return int(v) if v not in (None, "") else None
    except (TypeError, ValueError):
        return None


def _iter_st_raceday_ids(conn, since=None) -> Iterator[int]:
    sql = ("SELECT race_day_id FROM st_raceday_raw "
           "WHERE jsonb_typeof(raw_json) = 'object'")
    params: list = []
    if since is not None:
        sql += " AND scraped_at >= %s"
        params.append(since)
    sql += " ORDER BY race_day_id"
    with conn.cursor(name="st_rd_ids") as cur:
        cur.itersize = 2000
        cur.execute(sql, params)
        for (rd,) in cur:
            yield rd


def load_st_racedays_from_raw(conn, *, batch_size: int = 100, since=None,
                              limit: int | None = None, log=_print) -> dict:
    """ETL all (or recent) st_raceday_raw racedays into race + entry.

    Returns aggregate counts. Idempotent.
    """
    track_map = _build_st_track_code_map(conn)
    log(f"track-code map: {len(track_map):,} race_day_ids")
    totals = {"racedays": 0, "races": 0, "entries": 0, "skipped_no_track": 0}
    ids = list(_iter_st_raceday_ids(conn, since=since))
    if limit is not None:
        ids = ids[:limit]
    with conn.cursor() as wcur:
        for i, rdid in enumerate(ids, 1):
            wcur.execute("SELECT raw_json FROM st_raceday_raw WHERE race_day_id = %s", (rdid,))
            row = wcur.fetchone()
            if not row:
                continue
            c = upsert_raceday_from_st(wcur, rdid, row[0], track_map)
            totals["racedays"] += 1
            for k in ("races", "entries", "skipped_no_track"):
                totals[k] += c.get(k, 0)
            if i % batch_size == 0:
                conn.commit()
                log(f"  racedays {i}/{len(ids)} — {totals}")
        conn.commit()
    return totals


def load_st_racedays_for_ids(conn, race_day_ids: list[int],
                             track_map: dict[int, str] | None = None,
                             log=_print) -> dict:
    """ETL a specific set of st_raceday_raw racedays (targeted, not full-scan)."""
    ids = list(dict.fromkeys(int(r) for r in race_day_ids))
    totals = {"racedays": 0, "races": 0, "entries": 0, "skipped_no_track": 0}
    if not ids:
        return totals
    if track_map is None:
        track_map = _build_st_track_code_map(conn)
    with conn.cursor() as wcur:
        for i, rdid in enumerate(ids, 1):
            wcur.execute("SELECT raw_json FROM st_raceday_raw WHERE race_day_id = %s", (rdid,))
            row = wcur.fetchone()
            if not row:
                continue
            c = upsert_raceday_from_st(wcur, rdid, row[0], track_map)
            totals["racedays"] += 1
            for k in ("races", "entries", "skipped_no_track"):
                totals[k] += c.get(k, 0)
            if i % 50 == 0:
                conn.commit()
        conn.commit()
    return totals


# ---------------------------------------------------------------------------
# Raceday-driven discovery loop (durable replacement for v1's id-scan).
# ---------------------------------------------------------------------------

def discover_raceday_ids_from_horse_raw(conn, *, since=None) -> list[int]:
    """raceDayIds referenced by st_horse_raw race-results but not yet scraped."""
    with conn.cursor() as cur:
        sql = """
            SELECT DISTINCT (elem->'raceInformation'->>'raceDayId')::int AS rdid
              FROM st_horse_raw, jsonb_array_elements(raw_json) elem
             WHERE data_type = 'race-results'
               AND jsonb_typeof(raw_json) = 'array'
               AND elem->'raceInformation'->>'raceDayId' IS NOT NULL
        """
        params: list = []
        if since is not None:
            sql += " AND scraped_at >= %s"
            params.append(since)
        cur.execute(sql, params)
        candidate = {r[0] for r in cur.fetchall() if r[0] is not None}
        cur.execute("SELECT race_day_id FROM st_raceday_scrape_log WHERE http_status = 200")
        done = {r[0] for r in cur.fetchall()}
    return sorted(candidate - done)


def discover_new_starter_ids_from_racedays(conn, *, since=None) -> list[int]:
    """Horse ids appearing in scraped racedays that we have not fetched yet."""
    with conn.cursor() as cur:
        sql = """
            WITH rows AS (
                SELECT jsonb_array_elements(
                         jsonb_array_elements(raw_json->'racesWithReadyResult')->'raceResultRows'
                       ) AS rr
                  FROM st_raceday_raw
                 WHERE jsonb_typeof(raw_json) = 'object'
                   {since}
            )
            SELECT DISTINCT (rr->'horse'->>'id')::int AS hid FROM rows
             WHERE rr->'horse'->>'id' IS NOT NULL
        """.format(since="AND scraped_at >= %s" if since is not None else "")
        params = [since] if since is not None else []
        cur.execute(sql, params)
        candidate = {r[0] for r in cur.fetchall() if r[0] is not None}
        cur.execute("SELECT horse_id FROM st_horse_scrape_log")
        done = {r[0] for r in cur.fetchall()}
    return sorted(candidate - done)


# ===========================================================================
# NATIVE PIPELINE ORCHESTRATION (Phase 3)
#
# Bounded scrape->ETL->discover loop that owns ST ingestion end to end:
#   1. gap-fill the blind-spot horses (numeric atg_id, st_id NULL, raced SE)
#   2. scrape + ETL racedays referenced by freshly-scraped horse race-results
#   3. scrape + ETL the new starters those racedays reveal (bounded loops)
# Every step is idempotent and writes through resolve_*; the caps stop a
# first-run backlog from turning into an unbounded crawl.
# ===========================================================================

def run_native_st(conn, *, do_gap_fill: bool = True, max_racedays: int = 1500,
                  starter_loops: int = 1, max_new_horses: int = 4000,
                  log=_print) -> dict:
    """Run the native ST pipeline. Returns aggregate counts. Bounded + additive."""
    from scrapers import st_horse, st_raceday

    summary: dict = {}
    # Build the track-code map once; reuse across the whole run (the heading
    # name-match resolves any tracks the map misses, so a slightly stale map
    # is fine and avoids re-scanning st_horse_raw every loop).
    track_map = _build_st_track_code_map(conn)
    log(f"track-code map: {len(track_map):,} race_day_ids")

    if do_gap_fill:
        log("step 1 — gap-fill blind-spot horses")
        summary["gap_fill"] = gap_fill_st_horses(conn, log=log)

    # Step 2: scrape + ETL only the racedays referenced by scraped horses that
    # we haven't scraped yet (bounded). ETL is targeted to the ids we just
    # fetched, not a full re-scan.
    rd_ids = discover_raceday_ids_from_horse_raw(conn)
    if max_racedays is not None:
        rd_ids = rd_ids[:max_racedays]
    log(f"step 2 — {len(rd_ids)} racedays to scrape (capped at {max_racedays})")
    if rd_ids:
        summary["raceday_scrape"] = st_raceday.scrape_raceday_ids(conn, rd_ids, skip_done=True)
        summary["raceday_etl"] = load_st_racedays_for_ids(conn, rd_ids, track_map, log=log)

    # Step 3: bounded starter-discovery loops. Each loop scrapes the new
    # starters those racedays revealed, ETLs only their passports, then scrapes
    # any new racedays THOSE horses reference (also targeted ETL).
    loop_stats: list[dict] = []
    for i in range(starter_loops):
        new_horses = discover_new_starter_ids_from_racedays(conn)
        if max_new_horses is not None:
            new_horses = new_horses[:max_new_horses]
        if not new_horses:
            break
        log(f"step 3.{i+1} — {len(new_horses)} new starters to scrape")
        hs = st_horse.scrape_horse_ids(conn, new_horses, skip_done=True)
        he = load_st_horses_for_ids(conn, new_horses)
        more_rd = [r for r in discover_raceday_ids_from_horse_raw(conn)]
        if max_racedays is not None:
            more_rd = more_rd[:max_racedays]
        rd = {"scraped": None, "etl": None}
        if more_rd:
            rd["scraped"] = st_raceday.scrape_raceday_ids(conn, more_rd, skip_done=True)
            rd["etl"] = load_st_racedays_for_ids(conn, more_rd, track_map, log=log)
        loop_stats.append({"horses": hs, "horse_etl": he, "racedays": rd})
    summary["starter_loops"] = loop_stats
    return summary


def discover_new_starter_ids_for_racedays(conn, race_day_ids) -> list[int]:
    """Horse ids appearing in a SPECIFIC set of scraped racedays that we have
    not natively fetched yet. Used by the recent pipeline so starter discovery
    is scoped to the racedays the chain walk just found (not all of history)."""
    ids = [int(r) for r in race_day_ids]
    if not ids:
        return []
    with conn.cursor() as cur:
        cur.execute(
            """
            WITH rows AS (
                SELECT jsonb_array_elements(
                         jsonb_array_elements(raw_json->'racesWithReadyResult')->'raceResultRows'
                       ) AS rr
                  FROM st_raceday_raw
                 WHERE race_day_id = ANY(%s)
                   AND jsonb_typeof(raw_json) = 'object'
            )
            SELECT DISTINCT (rr->'horse'->>'id')::int AS hid FROM rows
             WHERE rr->'horse'->>'id' IS NOT NULL
            """,
            (ids,),
        )
        candidate = {r[0] for r in cur.fetchall() if r[0] is not None}
        cur.execute("SELECT horse_id FROM st_horse_scrape_log")
        done = {r[0] for r in cur.fetchall()}
    return sorted(candidate - done)


def discover_shallow_st_horse_ids(conn, *, lookback_days: int = 120,
                                  limit: int | None = None) -> list[int]:
    """ST ids of recently-active horses that carry a SHALLOW passport — a row
    exists (so they raced) but `sire_name` is NULL. These are normally brand-new
    horses the raceday ETL created on the fly; healing scrapes (if needed) +
    re-ETLs them so none stay stuck without pedigree. We do NOT exclude
    already-scraped ids: a horse can be scraped but not yet ETLed, and the heal
    step is cheap (ETL reads existing raw; scrape skips ids already fetched)."""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT DISTINCT h.st_id
              FROM horse h
             WHERE h.st_id IS NOT NULL
               AND h.sire_name IS NULL
               AND EXISTS (SELECT 1 FROM entry e
                             JOIN race r ON r.race_id = e.race_id
                            WHERE e.horse_id = h.horse_id
                              AND r.race_date >= CURRENT_DATE - %s)
             ORDER BY 1
            """,
            (lookback_days,),
        )
        ids = [r[0] for r in cur.fetchall()]
    return ids[:limit] if limit is not None else ids


def run_native_st_recent(conn, *, do_gap_fill: bool = True,
                         max_steps_per_chain: int = 80,
                         max_new_horses: int = 4000, log=_print) -> dict:
    """Recent-scoped native ST pipeline — the cutover-safe daily runner.

    Unlike `run_native_st` (which drains every raceday reachable from already-
    scraped horses, i.e. all of history), this walks the per-track
    `nextRaceDayId` chain forward from the latest known raceday, so it only
    touches racedays published since the last run. Steps:

      1. gap-fill blind-spot horses (numeric atg_id, st_id NULL, raced SE)
      2. chain-walk forward → scrape only the newly-published racedays
      3. ETL just those racedays
      4. scrape + ETL the new starters those racedays reveal (no st passport yet)

    We deliberately do NOT re-scrape existing horses' passports: race results
    arrive via the ATG bridge, and passports (pedigree/UELN/career) rarely
    change, so new horses (gap-fill + new starters) are the only passport work.
    Everything is idempotent and bounded.
    """
    from scrapers import st_horse, st_raceday

    summary: dict = {}
    track_map = _build_st_track_code_map(conn)
    log(f"track-code map: {len(track_map):,} race_day_ids")

    if do_gap_fill:
        log("step 1 — gap-fill blind-spot horses")
        summary["gap_fill"] = gap_fill_st_horses(conn, log=log)

    # Heal any recently-active horses left with a shallow passport (no sire)
    # by an earlier raceday-ETL that ran before their passport was scraped.
    # Normally empty; bounded so a backlog can't blow up the daily run.
    shallow = discover_shallow_st_horse_ids(conn, limit=max_new_horses)
    summary["shallow_heal_candidates"] = len(shallow)
    if shallow:
        log(f"step 1b — healing {len(shallow)} shallow-passport horses")
        st_horse.scrape_horse_ids(conn, shallow, skip_done=True)
        summary["shallow_heal_etled"] = load_st_horses_for_ids(conn, shallow)

    log("step 2 — chain-walk forward from latest known raceday per track")
    seeds = st_raceday.seed_raceday_ids(conn)
    log(f"  {len(seeds)} track seeds")
    walk = st_raceday.walk_forward_racedays(
        conn, seeds, max_steps_per_chain=max_steps_per_chain, log=log)
    rd_ids = walk.pop("scraped_ids", [])
    summary["raceday_walk"] = walk
    log(f"  walked {walk['steps']} steps, {len(rd_ids)} new racedays scraped")

    if rd_ids:
        starters = discover_new_starter_ids_for_racedays(conn, rd_ids)

        log(f"step 3 — ETL {len(rd_ids)} new racedays")
        summary["raceday_etl"] = load_st_racedays_for_ids(conn, rd_ids, track_map, log=log)

        # Scrape passports only for starters lacking a FULL passport: brand-new
        # horses (not yet in v2) and shallow rows the raceday ETL just created
        # (`sire_name IS NULL`). Existing starters (the vast majority) already
        # carry a rich passport from the historical backfill, and career stats
        # come from the entry table via the materialized view — so re-scraping
        # them is pure waste. The `sire_name` proxy self-heals shallow rows
        # regardless of scrape/ETL ordering and keeps the daily run cheap (tens
        # of horses, not tens of thousands).
        with conn.cursor() as cur:
            cur.execute(
                "SELECT st_id FROM horse "
                "WHERE st_id = ANY(%s) AND sire_name IS NOT NULL",
                (starters,),
            )
            rich = {r[0] for r in cur.fetchall()}
        new_horses = [h for h in starters if h not in rich]
        if max_new_horses is not None:
            new_horses = new_horses[:max_new_horses]
        log(f"step 4 — {len(new_horses)} starters need a passport "
            f"(of {len(starters)} total; rest already rich in v2)")
        if new_horses:
            summary["starter_scrape"] = st_horse.scrape_horse_ids(
                conn, new_horses, skip_done=True)
            summary["starter_etl"] = load_st_horses_for_ids(conn, new_horses)
        summary["starters_total"] = len(starters)
        summary["starters_new"] = len(new_horses)
    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    from core.db import get_connection, get_v1_connection

    ap = argparse.ArgumentParser(description="TravSport (st) importer")
    ap.add_argument(
        "command", nargs="?", default="backfill",
        choices=("backfill", "gap-fill", "load-horses", "load-racedays",
                 "native", "native-recent"),
        help="backfill = full v1->v2 mirror; gap-fill = native scrape+ETL of "
             "SE horses missing st_id; load-horses / load-racedays = ETL existing "
             "raw; native = full bounded native ST pipeline (drains history); "
             "native-recent = recent-scoped chain-walk pipeline (daily runner)",
    )
    ap.add_argument("--dry-run", action="store_true",
                    help="gap-fill: only report discovered ids")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--max-racedays", type=int, default=1500)
    args = ap.parse_args()

    if args.command == "backfill":
        v1 = get_v1_connection()
        v2 = get_connection()
        try:
            counts = backfill_from_v1(v1, v2)
            _print("DONE")
            for k, v in counts.items():
                _print(f"  {k}: {v:,}")
        finally:
            v1.close()
            v2.close()
    elif args.command == "gap-fill":
        v2 = get_connection()
        try:
            res = gap_fill_st_horses(v2, dry_run=args.dry_run, limit=args.limit)
            _print(f"DONE: {res}")
        finally:
            v2.close()
    elif args.command == "load-racedays":
        v2 = get_connection()
        try:
            res = load_st_racedays_from_raw(v2, limit=args.limit)
            _print(f"DONE: {res}")
        finally:
            v2.close()
    elif args.command == "native":
        v2 = get_connection()
        try:
            res = run_native_st(v2, max_racedays=args.max_racedays)
            _print(f"DONE: {res}")
        finally:
            v2.close()
    elif args.command == "native-recent":
        v2 = get_connection()
        try:
            res = run_native_st_recent(v2)
            _print(f"DONE: {res}")
        finally:
            v2.close()
    else:  # load-horses
        v2 = get_connection()
        try:
            n = load_st_horses_from_raw(v2, limit=args.limit)
            _print(f"DONE: {n} horses ETLed from st_horse_raw")
        finally:
            v2.close()
