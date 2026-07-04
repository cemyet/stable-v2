"""
ATG (Aktiebolaget Trav och Galopp) source importer.

Reads v1's `v2_atg_race_raw` table (the per-race raw JSON cache) and
writes canonical rows into v2's master tables. This is what closes the
"foreign-race sparsity" gap — ST's pipeline only imports entries for
TravSport-registered horses, so non-SE races (Laval, Bjerke, Vincennes,
Momarken, ...) end up with 1 horse per race in v2 even though ATG knows
all 13 starters.

Public entry points:
  * `ingest_race(v2_conn, atg_race_id, raw_json)` — upsert one race.
  * `backfill_from_v1_raw(v1_conn, v2_conn, ...)` — sweep v1.v2_atg_race_raw.

Identity quirks handled:
  * Swedish horses/persons: ATG's numeric `horse.id` / `person.id` are the
    same TravSport ids stored in v2 as `st_id`. matching.upsert_horse /
    upsert_person fall back to st_id when atg passes an int source_id.
  * Foreign horses/persons: no ATG id. We mint a deterministic synthetic
    string id `f"x:{country_code}:{NORMALIZED_NAME}"` so they dedupe
    across races. A later admin merge tool can attach them to Le Trot /
    HVT / etc rows.

Skipped:
  * `sport='gallop'` races (kept entirely out of v2).
  * Entries where the horse name is empty (data noise).
"""

from __future__ import annotations

import re
import sys
from datetime import date, datetime
from typing import Any

from .matching import (
    upsert_horse,
    upsert_person,
    upsert_track,
    upsert_race,
    upsert_entry,
)
from core.identity import resolve_horse, resolve_person


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _print(msg: str) -> None:
    sys.stdout.write(msg + "\n")
    sys.stdout.flush()


_KM_PER_SECOND = 1.0  # kmTime is already seconds-per-km in ATG's payload


def _km_time_to_seconds(km: dict | None) -> float | None:
    """ATG's nested {minutes,seconds,tenths} → flat seconds-per-km."""
    if not km:
        return None
    m = km.get("minutes") or 0
    s = km.get("seconds") or 0
    t = km.get("tenths") or 0
    val = m * 60 + s + t / 10.0
    return float(val) if val > 0 else None


def _km_time_to_text(km: dict | None) -> str | None:
    """Render '14,4' / '1.13,2' from ATG's nested time object."""
    if not km:
        return None
    m = km.get("minutes") or 0
    s = km.get("seconds") or 0
    t = km.get("tenths") or 0
    if m == 0 and s == 0 and t == 0:
        return None
    sec_str = f"{s},{t}"
    return sec_str if m <= 1 else f"{m-1}.{sec_str}"


_START_METHOD = {
    "auto": ("A", True),
    "volte": ("V", False),
    "line": ("L", None),
}


def _safe_int(v: Any) -> int | None:
    try:
        if v is None:
            return None
        return int(v)
    except (TypeError, ValueError):
        return None


_TOTAL_PRIZE_RE = re.compile(r"max total:\s*([\d.\s]+)\s*kr", re.IGNORECASE)


def _extract_total_prize(prize_str: str | None) -> int | None:
    """Parse 'Prispengar max total: 84.700 kr.' style strings."""
    if not prize_str:
        return None
    m = _TOTAL_PRIZE_RE.search(prize_str)
    if not m:
        return None
    digits = m.group(1).replace(".", "").replace(" ", "").replace(",", "")
    try:
        return int(digits)
    except ValueError:
        return None


def _extract_proposition(terms: list | None) -> str | None:
    """Join ATG's `terms` array into a single proposition text."""
    if not terms:
        return None
    parts = [str(t).strip() for t in terms if t]
    return "\n".join(parts) if parts else None


# ATG reports `sex` as a full word that is English for SE/most horses
# (stallion/gelding/mare) but FRENCH for French starters
# (mâle/hongre/femelle). The canonical convention for both
# horse.gender_code and entry.sex is the Swedish H/V/S. Map on the full
# word so the French "mâle" (stallion) is never confused with the English
# "mare" — both truncate to 'M' but mean opposite sexes.
_SEX_WORD_TO_HVS = {
    "stallion": "H", "gelding": "V", "mare": "S",        # ATG English
    "hingst": "H", "valack": "V", "sto": "S",            # Swedish
    "hongre": "V", "femelle": "S", "male": "H",          # French (ascii)
    "mâle": "H", "entier": "H",                          # French (accented / intact)
}


def _sex_to_hvs(raw: str | None) -> str | None:
    """Normalize any ATG sex word (EN/FR/SV) to canonical H/V/S."""
    if not raw:
        return None
    t = str(raw).strip().lower()
    if t in _SEX_WORD_TO_HVS:
        return _SEX_WORD_TO_HVS[t]
    # Single-letter inputs: only the unambiguous ones. 'm'/'s'/'h' are
    # ambiguous across EN/FR/SV (mare vs mâle, stallion vs sto, hingst vs
    # hongre) so we never guess them here — the word map above is the only
    # reliable disambiguator and already runs first.
    return {"v": "V", "g": "V", "f": "S"}.get(t)


def _placement_text_from_result(result: dict, *, scratched: bool = False) -> str | None:
    """ATG returns numeric `place`; render as the same string convention
    we use elsewhere ('1'..'15', '0' for unplaced finishers, 'd' for DQ,
    'utg' for scratched). `scratched` may be derived from a race-level
    `scratchings` array (per-start `scratched` is rarely present).
    """
    if scratched:
        return "utg"
    if result is None:
        return None
    if result.get("disqualified"):
        return "d"
    p = result.get("place")
    if result.get("galloped"):
        # A gait-breaker that was still CLASSIFIED keeps its real numeric
        # finishing position. The gait-break itself is carried by the
        # `entry.galopp` boolean (and shown with a small 'g' marker in the
        # UI). Only fall back to bare 'g' when there is no real placement.
        if p is not None and p >= 1:
            return str(p)
        return "g"
    if p is None:
        return "0" if result.get("finishOrder") is not None else None
    return str(p)


# ---------------------------------------------------------------------------
# Per-concept extraction
# ---------------------------------------------------------------------------

def _upsert_track_from_raw(cur, raw: dict) -> int | None:
    track = raw.get("track") or {}
    atg_id = _safe_int(track.get("id"))
    name = track.get("name")
    country = track.get("countryCode")
    if not name and atg_id is None:
        return None
    sport = raw.get("sport") or "trot"
    if sport == "gallop":
        return None
    fields = {
        "name": name,
        "country": country,
        "sport": sport if sport in ("trot", "gallop") else "trot",
    }
    return upsert_track(
        cur, "atg", atg_id if atg_id is not None else f"x:{country}:{name}",
        fields,
        source_id_column="atg_track_id" if atg_id is not None else None,
    )


def _upsert_horse_from_start(cur, horse_block: dict, race_date: date | None,
                             *, fallback_country: str | None = None,
                             race_id: int | None = None,
                             program_number: int | None = None) -> int | None:
    """Resolve / create a horse for an ATG start row.

    Delegates all identity decisions to `core.identity.resolve_horse`,
    which applies the strict matching protocol (strong IDs → cross-IDs →
    race context → pedigree triangulation → synth key → INSERT). NEVER
    matches by name only.
    """
    name = (horse_block or {}).get("name")
    if not name:
        return None
    atg_horse_id = _safe_int(horse_block.get("id"))
    nationality = horse_block.get("nationality") or fallback_country
    age = _safe_int(horse_block.get("age"))
    sex = horse_block.get("sex")
    color = horse_block.get("color")
    pedigree = horse_block.get("pedigree") or {}
    sire = (pedigree.get("father") or {}).get("name")
    dam = (pedigree.get("mother") or {}).get("name")

    gender_code = _sex_to_hvs(sex)

    fields: dict[str, Any] = {"name": name}
    if gender_code:
        fields["gender_code"] = gender_code
    if color:
        fields["color"] = color
    if nationality:
        fields["birth_country"] = nationality
    if sire:
        fields["sire_name"] = sire
    if dam:
        fields["dam_name"] = dam
    if age and race_date:
        fields["date_of_birth"] = date(race_date.year - age, 1, 1)

    # Strong id when ATG provides a numeric horse.id; otherwise leave None
    # and let resolve_horse try race-context, pedigree, then synth key.
    source_id: Any = atg_horse_id if (atg_horse_id is not None and atg_horse_id > 0) else None
    raw_payload = {"name": name, "nationality": nationality}
    if source_id is None:
        raw_payload["synthetic"] = True

    return resolve_horse(
        cur,
        source="atg",
        source_id=source_id,
        canonical_fields=fields,
        raw_payload=raw_payload,
        race_id=race_id,
        program_number=program_number,
        sire_name=sire,
        dam_name=dam,
    )


def _upsert_person_from_block(cur, block: dict | None, *, role: str,
                              nationality: str | None) -> int | None:
    if not block:
        return None
    first = (block.get("firstName") or "").strip()
    last = (block.get("lastName") or "").strip()
    full = " ".join(p for p in (last, first) if p) or block.get("name")
    if not full:
        return None
    pid = _safe_int(block.get("id"))
    fields: dict[str, Any] = {
        "name": full,
        "short_name": block.get("shortName"),
    }
    role_flags = {
        "is_driver":  role == "driver",
        "is_trainer": role == "trainer",
    }
    source_id: Any = pid if (pid is not None and pid > 0) else None
    raw_payload = {"name": full, "role": role}
    if source_id is None:
        raw_payload["synthetic"] = True
    return resolve_person(
        cur,
        source="atg",
        source_id=source_id,
        canonical_fields=fields,
        raw_payload=raw_payload,
        role_flags=role_flags,
        country=nationality,
    )


def _parse_race_date(raw: dict) -> date | None:
    """ATG payload has both `date` (YYYY-MM-DD) and `startTime` ISO."""
    d = raw.get("date")
    if d:
        try:
            return datetime.strptime(d, "%Y-%m-%d").date()
        except ValueError:
            pass
    st = raw.get("startTime")
    if st:
        try:
            return datetime.fromisoformat(st.replace("Z", "+00:00")).date()
        except ValueError:
            pass
    return None


# Track names that v1's ST import used as country-level catch-alls when it
# didn't know the real foreign track. When ATG ingest later identifies the
# actual track, we migrate the race off the placeholder onto the real row.
_GENERIC_TRACK_NAMES = {
    "frankrike", "norge", "finland", "danmark", "tyskland", "italien",
    "belgien", "holland", "usa", "storbritannien", "hong kong",
    "sydafrika", "nya zeeland", "australien", "sverige",
}


def _migrate_generic_track_race(cur, race_date: date, race_number: int | None,
                                country: str | None, new_track_id: int) -> int | None:
    """If a race with the same (date, number, country) currently lives on a
    generic catch-all track, repoint it at the real ATG track. Returns the
    migrated race_id, or None if no migration happened.
    """
    if race_number is None or country is None:
        return None
    cur.execute(
        """
        SELECT r.race_id FROM race r
          JOIN track  t ON t.track_id = r.track_id
         WHERE r.race_date = %s AND r.race_number = %s
           AND r.track_id <> %s
           AND lower(t.name) = ANY(%s)
           AND COALESCE(t.country, %s) = %s
         LIMIT 1
        """,
        (race_date, race_number, new_track_id,
         list(_GENERIC_TRACK_NAMES), country, country),
    )
    row = cur.fetchone()
    if not row:
        return None
    race_id = row[0]
    cur.execute(
        "UPDATE race SET track_id = %s WHERE race_id = %s",
        (new_track_id, race_id),
    )
    return race_id


# ---------------------------------------------------------------------------
# Public: per-race ingest
# ---------------------------------------------------------------------------

def ingest_race(v2_cur, atg_race_id: str, raw: dict) -> dict:
    """Upsert one ATG race + its track + all starters + drivers + trainers.

    Returns counts: {races, entries, horses, persons, skipped_reason}
    """
    stats = {"races": 0, "entries": 0, "horses": 0, "persons": 0}
    if not raw or not isinstance(raw, dict):
        return {**stats, "skipped_reason": "empty_raw"}

    sport = raw.get("sport") or "trot"
    if sport == "gallop":
        return {**stats, "skipped_reason": "gallop"}

    race_date = _parse_race_date(raw)
    starts = raw.get("starts") or []
    if not race_date or not starts:
        return {**stats, "skipped_reason": "no_date_or_starts"}

    track_id = _upsert_track_from_raw(v2_cur, raw)
    if track_id is None:
        return {**stats, "skipped_reason": "no_track"}

    track_country = ((raw.get("track") or {}).get("countryCode"))
    race_number = _safe_int(raw.get("number"))

    # Before upserting the race: if a v1-era ST race already exists for
    # this (date, number, country) but is stuck on a generic catch-all
    # track (e.g. 'Frankrike'), migrate it onto the real ATG track first.
    # This prevents a duplicate race row when upsert_race falls back to
    # (track_id, date, number) matching.
    _migrate_generic_track_race(v2_cur, race_date, race_number,
                                track_country, track_id)

    sm_raw = raw.get("startMethod")
    sm_code, sm_auto = _START_METHOD.get(sm_raw or "", (None, None))

    # Race-level extras: total prize sum, victory margin, scratchings.
    race_result = raw.get("result") or {}
    victory_margin = race_result.get("victoryMargin")
    scratched_nums = set(race_result.get("scratchings") or [])
    total_prize = _extract_total_prize(raw.get("prize"))
    proposition = _extract_proposition(raw.get("terms"))

    race_fields = {
        "race_date": race_date,
        "track_id": track_id,
        "race_number": race_number,
        "distance": _safe_int(raw.get("distance")),
        "start_method": sm_code,
        "heading": raw.get("heading") or raw.get("name"),
        "atg_race_day_id": atg_race_id.rsplit("_", 1)[0] if "_" in atg_race_id else None,
        "victory_margin": victory_margin,
        "total_prize_kr": total_prize,
        "proposition_text": proposition,
    }
    race_id = upsert_race(v2_cur, "atg", atg_race_id, race_fields,
                          raw_payload={"sport": sport, "prize": raw.get("prize")})
    stats["races"] = 1

    for s in starts:
        h = s.get("horse") or {}
        if not h.get("name"):
            continue
        prog = _safe_int(s.get("number"))
        horse_id = _upsert_horse_from_start(v2_cur, h, race_date,
                                            fallback_country=track_country,
                                            race_id=race_id,
                                            program_number=prog)
        if horse_id is None:
            continue
        stats["horses"] += 1

        # Driver from start.driver; trainer is nested under horse.trainer.
        nationality = h.get("nationality") or track_country
        driver_id = _upsert_person_from_block(v2_cur, s.get("driver"),
                                              role="driver", nationality=nationality)
        trainer_id = _upsert_person_from_block(v2_cur, h.get("trainer"),
                                               role="trainer", nationality=nationality)
        if driver_id:  stats["persons"] += 1
        if trainer_id: stats["persons"] += 1

        result = s.get("result") or {}
        km = result.get("kmTime")
        place = _safe_int(result.get("place"))
        finish_order = _safe_int(result.get("finishOrder"))
        post = _safe_int(s.get("postPosition"))
        # Match v2's existing logic: 'placement' is the integer position
        # for placed finishers only. We trust place when it's >= 1.
        placement = place if (place is not None and place >= 1) else None

        # Race-level `scratchings: [<startNumber>]` is ATG's primary scratched
        # signal; per-start `result.scratched` is rare/absent.
        start_num = _safe_int(s.get("number"))
        scratched = (start_num in scratched_nums) or bool(result.get("scratched"))
        ptext = _placement_text_from_result(result, scratched=scratched)

        entry_fields: dict[str, Any] = {
            "program_number": start_num,
            "distance": _safe_int(s.get("distance")) or _safe_int(raw.get("distance")),
            "placement": placement,
            "finish_order": finish_order,
            "placement_text": ptext,
            "withdrawn": scratched,
            "disqualified": bool(result.get("disqualified")),
            "galopp": bool(result.get("galloped")),
            "time_seconds": _km_time_to_seconds(km),
            "time_text": _km_time_to_text(km),
            "auto": sm_auto,
            "odds": result.get("finalOdds"),
            "prize_kr": _safe_int(result.get("prizeMoney")) or 0,
            "driver_id": driver_id,
            "trainer_id": trainer_id,
            "age": _safe_int(h.get("age")),
            "sex": _sex_to_hvs(h.get("sex")),
        }

        # Shoe/sulky equipment when reported.
        shoes = h.get("shoes") or {}
        if shoes.get("reported"):
            front = (shoes.get("front") or {})
            back = (shoes.get("back") or {})
            # ATG flags: hasShoe true/false. shoe_code is v2's TravSport-style
            # 1=none / 2=back only / 3=front only / 4=all four shoes.
            fs = bool(front.get("hasShoe"))
            bs = bool(back.get("hasShoe"))
            if fs and bs:        code = "4"
            elif fs and not bs:  code = "3"
            elif not fs and bs:  code = "2"
            else:                code = "1"
            entry_fields["shoe_code"] = code
            entry_fields["shoe_front_changed"] = bool(front.get("changed"))
            entry_fields["shoe_back_changed"]  = bool(back.get("changed"))

        # Sulky: use `.type.code` (Am/Va/Bg/Hy) — NOT `.colour.code` which
        # is the sulky color (Gu=Gul, Bl=Blå). The type is what races classify.
        sulky = h.get("sulky") or {}
        if sulky.get("reported"):
            stype = sulky.get("type") or {}
            entry_fields["sulky"] = stype.get("code")
            entry_fields["sulky_changed"] = bool(stype.get("changed"))

        upsert_entry(v2_cur, "atg", race_id, horse_id, entry_fields,
                     raw_payload={"start_id": s.get("id")})
        # ATG is the source of truth for result columns. The matching
        # layer's "don't overwrite with NULL" rule otherwise leaves a
        # stale `placement=1` on entries we previously ingested from ST
        # with wrong defaults (the Jabalpur class of bug). Force-clear.
        v2_cur.execute(
            """
            UPDATE entry
               SET placement      = %s,
                   placement_text = %s,
                   finish_order   = %s,
                   disqualified   = %s,
                   withdrawn      = %s,
                   galopp         = %s,
                   time_seconds   = %s,
                   time_text      = %s,
                   odds           = %s,
                   prize_kr       = %s,
                   sulky          = %s,
                   sulky_changed  = %s,
                   shoe_code      = %s,
                   shoe_front_changed = %s,
                   shoe_back_changed  = %s,
                   -- Post position: ATG reports it in `postPosition` but we
                   -- only FILL NULLs here so we never clobber a good ST post.
                   post           = COALESCE(post, %s)
             WHERE race_id = %s AND horse_id = %s
            """,
            (placement, ptext, finish_order,
             bool(result.get("disqualified")),
             scratched,
             bool(result.get("galloped")),
             entry_fields.get("time_seconds"),
             entry_fields.get("time_text"),
             entry_fields.get("odds"),
             entry_fields.get("prize_kr", 0),
             entry_fields.get("sulky"),
             entry_fields.get("sulky_changed"),
             entry_fields.get("shoe_code"),
             entry_fields.get("shoe_front_changed"),
             entry_fields.get("shoe_back_changed"),
             post,
             race_id, horse_id),
        )
        stats["entries"] += 1

    return stats


# ---------------------------------------------------------------------------
# Public: bulk backfill
# ---------------------------------------------------------------------------

def load_atg_from_raw(v2_conn, *,
                      since: str | None = None,
                      until: str | None = None,
                      limit: int | None = None,
                      only_foreign: bool = False,
                      batch_size: int = 200,
                      progress_every: int = 500,
                      read_conn=None) -> dict:
    """Sweep v2-local `atg_race_raw` and ingest each race into v2 master tables.

    This is the v2-native counterpart to `backfill_from_v1_raw`: the raw JSON now
    lives in v2's own `atg_race_raw` (populated by scrapers.atg), so we no longer
    need a v1 connection. Reads stream from a dedicated connection (`read_conn`,
    default a fresh v2 connection) so the per-batch commits on `v2_conn` don't
    invalidate the server-side read cursor.

    Args mirror `backfill_from_v1_raw`. `since`/`until` filter on `race_date`.
    """
    from core.db import get_connection

    where = ["raw_json IS NOT NULL"]
    params: list = []
    if since:
        where.append("race_date >= %s")
        params.append(since)
    if until:
        where.append("race_date <= %s")
        params.append(until)
    where_sql = " AND ".join(where)
    limit_sql = f" LIMIT {int(limit)}" if limit else ""
    sql = (
        f"SELECT atg_race_id, raw_json FROM atg_race_raw "
        f" WHERE {where_sql} ORDER BY race_date, atg_race_id{limit_sql}"
    )

    totals = {"scanned": 0, "ingested": 0, "skipped": 0,
              "races": 0, "entries": 0, "horses": 0, "persons": 0,
              "by_skip": {}}

    own_read = read_conn is None
    read_conn = read_conn or get_connection()
    read_conn.set_session(readonly=True)
    try:
        with read_conn.cursor(name="atg_local_raw_stream") as src:
            src.itersize = batch_size
            src.execute(sql, params)
            batch_count = 0
            with v2_conn.cursor() as dst:
                for atg_race_id, raw in src:
                    totals["scanned"] += 1
                    if only_foreign:
                        country = ((raw or {}).get("track") or {}).get("countryCode")
                        if country == "SE":
                            totals["skipped"] += 1
                            totals["by_skip"]["se_filter"] = totals["by_skip"].get("se_filter", 0) + 1
                            continue
                    dst.execute("SAVEPOINT race_sp")
                    try:
                        s = ingest_race(dst, atg_race_id, raw)
                    except Exception as exc:
                        dst.execute("ROLLBACK TO SAVEPOINT race_sp")
                        totals["skipped"] += 1
                        totals["by_skip"]["error"] = totals["by_skip"].get("error", 0) + 1
                        if totals["by_skip"]["error"] <= 20:
                            _print(f"  ERR {atg_race_id}: {exc!r}")
                        continue
                    else:
                        dst.execute("RELEASE SAVEPOINT race_sp")
                    if s.get("skipped_reason"):
                        totals["skipped"] += 1
                        k = s["skipped_reason"]
                        totals["by_skip"][k] = totals["by_skip"].get(k, 0) + 1
                    else:
                        totals["ingested"] += 1
                        for k in ("races", "entries", "horses", "persons"):
                            totals[k] += s.get(k, 0)
                    batch_count += 1
                    if batch_count >= batch_size:
                        v2_conn.commit()
                        batch_count = 0
                    if totals["scanned"] % progress_every == 0:
                        _print(
                            f"  [atg] scanned={totals['scanned']:,} "
                            f"ingested={totals['ingested']:,} "
                            f"entries={totals['entries']:,} "
                            f"skipped={totals['skipped']:,}"
                        )
                v2_conn.commit()
    finally:
        if own_read:
            read_conn.close()
    return totals


def backfill_from_v1_raw(v1_conn, v2_conn, *,
                         since: str | None = None,
                         until: str | None = None,
                         limit: int | None = None,
                         only_foreign: bool = False,
                         batch_size: int = 200,
                         progress_every: int = 500) -> dict:
    """Sweep v1.v2_atg_race_raw and ingest each race into v2.

    Args:
        since / until: 'YYYY-MM-DD' filters on race_date prefix of
            atg_race_id (which starts with the date).
        limit: cap on rows processed (debug / dry-run).
        only_foreign: skip races whose track is in Sweden (SE) — useful
            for a fast first pass that only adds the foreign coverage
            we don't already have via ST.
        batch_size: rows fetched + committed per batch.
        progress_every: stdout heartbeat cadence.
    """
    where = ["raw_json IS NOT NULL"]
    params: list = []
    if since:
        where.append("atg_race_id >= %s")
        params.append(since)
    if until:
        where.append("atg_race_id < %s")
        params.append(until + "_zzz")
    where_sql = " AND ".join(where)
    limit_sql = f" LIMIT {int(limit)}" if limit else ""
    sql = (
        f"SELECT atg_race_id, raw_json FROM v2_atg_race_raw "
        f" WHERE {where_sql} ORDER BY atg_race_id{limit_sql}"
    )

    totals = {"scanned": 0, "ingested": 0, "skipped": 0,
              "races": 0, "entries": 0, "horses": 0, "persons": 0,
              "by_skip": {}}

    with v1_conn.cursor(name="atg_raw_stream") as src:
        src.itersize = batch_size
        src.execute(sql, params)
        batch_count = 0
        with v2_conn.cursor() as dst:
            for atg_race_id, raw in src:
                totals["scanned"] += 1
                # Optional foreign-only filter (peek at countryCode quickly).
                if only_foreign:
                    country = ((raw or {}).get("track") or {}).get("countryCode")
                    if country == "SE":
                        totals["skipped"] += 1
                        totals["by_skip"]["se_filter"] = totals["by_skip"].get("se_filter", 0) + 1
                        continue
                # Per-race savepoint so a single bad race doesn't lose the
                # whole batch's worth of successful upserts.
                dst.execute("SAVEPOINT race_sp")
                try:
                    s = ingest_race(dst, atg_race_id, raw)
                except Exception as exc:
                    dst.execute("ROLLBACK TO SAVEPOINT race_sp")
                    totals["skipped"] += 1
                    totals["by_skip"]["error"] = totals["by_skip"].get("error", 0) + 1
                    if totals["by_skip"]["error"] <= 20:
                        _print(f"  ERR {atg_race_id}: {exc!r}")
                    continue
                else:
                    dst.execute("RELEASE SAVEPOINT race_sp")
                if s.get("skipped_reason"):
                    totals["skipped"] += 1
                    k = s["skipped_reason"]
                    totals["by_skip"][k] = totals["by_skip"].get(k, 0) + 1
                else:
                    totals["ingested"] += 1
                    for k in ("races", "entries", "horses", "persons"):
                        totals[k] += s.get(k, 0)
                batch_count += 1
                if batch_count >= batch_size:
                    v2_conn.commit()
                    batch_count = 0
                if totals["scanned"] % progress_every == 0:
                    _print(
                        f"  [atg] scanned={totals['scanned']:,} "
                        f"ingested={totals['ingested']:,} "
                        f"entries={totals['entries']:,} "
                        f"skipped={totals['skipped']:,}"
                    )
            v2_conn.commit()
    return totals
