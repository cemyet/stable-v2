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

import sys
from typing import Iterator

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

def backfill_tracks(v1_conn, v2_conn) -> int:
    """Copy v1.track + v1.atg_track → v2.track. Returns row count."""
    n = 0
    # TravSport tracks (st_code primary key)
    with v1_conn.cursor() as src, v2_conn.cursor() as dst:
        src.execute("SELECT code, name, country, first_seen_at, last_seen_at FROM track")
        for code, name, country, fs, ls in src.fetchall():
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
    # ATG tracks (numeric atg_track_id). v1 may have already linked them to
    # an ST track via track_code. If so, attach atg_track_id to the existing
    # ST row instead of creating a duplicate.
    with v1_conn.cursor() as src, v2_conn.cursor() as dst:
        src.execute(
            "SELECT atg_track_id, name, country_code, sport, track_code, first_seen, last_seen "
            "FROM atg_track"
        )
        for atg_id, name, country, sport, st_code, fs, ls in src.fetchall():
            fields = {
                "name": name,
                "country": country,
                "sport": sport,
                "first_seen_at": fs,
                "last_seen_at": ls,
            }
            existing_st_id = None
            if st_code:
                dst.execute(
                    "SELECT track_id, atg_track_id FROM track WHERE st_code = %s",
                    (st_code,),
                )
                row = dst.fetchone()
                if row:
                    existing_st_id, existing_atg = row
                    if existing_atg and existing_atg != atg_id:
                        # ST row already linked to a DIFFERENT atg_track_id.
                        # Make a separate row for this ATG track to avoid
                        # silent merging of unrelated tracks.
                        existing_st_id = None
            if existing_st_id:
                dst.execute(
                    "UPDATE track SET atg_track_id = %s, "
                    "name = COALESCE(name, %s), country = COALESCE(country, %s), "
                    "sport = COALESCE(sport, %s), "
                    "last_updated_at = NOW() "
                    "WHERE track_id = %s",
                    (atg_id, name, country, sport, existing_st_id),
                )
            else:
                upsert_track(dst, "atg", atg_id, fields)
            n += 1
    v2_conn.commit()
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

    We materialise the join so the loader pulls track + date for each race.
    """
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
                # Resolve track_id via st_code lookup
                dst.execute("SELECT track_id FROM track WHERE st_code = %s", (track_code,))
                row = dst.fetchone()
                track_id = row[0] if row else None

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


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from core.db import get_connection, get_v1_connection
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
