"""
Cross-source identity resolution for stable-v2.

This module is the *only* place where decisions are made about whether a
horse / person seen by a source matches an existing canonical row.

Matching protocol (strongest to weakest):

  1. Source ID — same source seeing a row it already wrote.
  2. Cross-source ID — registration_number, UELN, SE id-equivalence
     (atg.id == st_id for Swedish horses/persons).
  3. Race-context — when ingesting a foreign starter on a race we already
     have entries for, same `program_number` is the same starter.
  4. Pedigree triangulation — name + birth_year + sire_name + dam_name all
     match. Auto-merge eligible: 735 known groups in audit.
  5. Foreign synthetic key — `x:CC:NORMALIZED_NAME`, used for foreign
     horses/persons without any stable id.

We **never** auto-match by name alone. When ambiguity remains, we insert a
new row and surface the duplicate in the admin matching dashboard.

All merges go through `merge_horses` / `merge_persons` which write to
`horse_merge_log` / `person_merge_log` (with full row snapshot for
rollback).

Per-source contract (what each importer should pass)
----------------------------------------------------

Every importer in etl/import_<source>.py uses `resolve_horse` /
`resolve_person`. The right call shape:

  resolve_horse(
      cur,
      source="atg",                       # required, must match a known source
      source_id=horse_block.get("id"),    # numeric ID when available, else None
      canonical_fields={                  # canonical horse columns to fill
          "name": ..., "date_of_birth": ..., "gender_code": ...,
          "birth_country": ..., "sire_name": ..., "dam_name": ...,
      },
      raw_payload={...},                  # stored under source_data[source]
      registration_number=...,            # if known (HVT, ST, Breedly)
      ueln_number=...,                    # if known (HVT, ST)
      race_id=race_id,                    # MUST pass when we already have a
      program_number=prog,                # race row — enables race-context match
      sire_name=...,                      # pass when known — pedigree triangulation
      dam_name=...,
  )

  resolve_person(
      cur,
      source="atg",
      source_id=numeric_or_None,
      canonical_fields={"name": ..., "short_name": ...},
      raw_payload={...},
      role_flags={"is_driver": True, "is_trainer": False},
      country="FR",                       # used for synthetic-key fallback
  )

Source-specific guidance:

  * ATG — Swedish horses share st_id ≡ atg horse.id; the resolver handles
    that internally. Foreign starters with no numeric id MUST go through
    `resolve_horse(..., race_id=..., program_number=...)` so race-context
    can reuse an existing ST/LeTrot row.
  * LeTrot — always pass race_id + program_number. LeTrot races we ingest
    are typically already in our DB from ST/ATG. Race-context plus
    pedigree triangulation prevent the "Indy Rock" orphan case.
  * HVT — pass `ueln_number` (and `registration_number` when scraper
    extracts it). Reg/UELN unique indexes will dedupe across sources.
  * Breedly — Breedly publishes `stId` for SE horses; combine with the
    pre-attach UPDATE in import_breedly.py.
  * USTA / kmtid — when wiring these, follow the same pattern.
"""

from __future__ import annotations

import re
import unicodedata
from datetime import date
from typing import Any

from psycopg2.extras import Json

from etl.matching import upsert_horse, upsert_person


# ---------------------------------------------------------------------------
# Name normalization
# ---------------------------------------------------------------------------

_NAME_SUFFIX_RE = re.compile(r"[\*\u00b7]+|\s*\([A-Z]{1,3}\)\s*$|\s*\?\s*$")

# Atomic Nordic / German / Polish letters that NFKD does not decompose.
# Mirrors Postgres `unaccent` extension's defaults so Python and SQL agree.
_EXTRA_TRANSLIT = str.maketrans({
    "ø": "o", "Ø": "O",
    "æ": "ae", "Æ": "AE",
    "œ": "oe", "Œ": "OE",
    "ß": "ss",
    "ð": "d", "Ð": "D",
    "þ": "th", "Þ": "TH",
    "ł": "l", "Ł": "L",
    "đ": "d", "Đ": "D",
    "ħ": "h", "Ħ": "H",
    "ı": "i", "İ": "I",
    "ŧ": "t", "Ŧ": "T",
    "ŋ": "n", "Ŋ": "N",
})


def normalize_name(name: str | None) -> str:
    """Uppercase + strip diacritics + strip TravSport guest suffix.

    Removes:
      * Diacritics (Redén -> REDEN, Müller -> MULLER, Åby -> ABY,
        Bjørn -> BJORN, Kjær -> KJAER, Lønborg -> LONBORG)
      * TravSport guest suffix (`* (FR)`, `(IT)`)
      * Trailing `?`, decorative stars/middots

    NOTE: trailing `X.Y.` patterns like ` I.T.` / ` F.R.` look like
    country codes in some ATG rows (Italian Trotter, etc.) but are also
    used in ST as owner-initial disambiguators (`Anna K.J.`, `Kasper T.T.`)
    that distinguish actually-different horses. They are NOT stripped here —
    the `Demon I.T.` ↔ `Demon* (IT)` style cases must be merged manually.

    Used for pedigree-triangulation lookups and synthetic-key generation —
    NEVER for matching alone.
    """
    if not name:
        return ""
    nfkd = unicodedata.normalize("NFKD", name)
    ascii_name = "".join(ch for ch in nfkd if not unicodedata.combining(ch))
    ascii_name = ascii_name.translate(_EXTRA_TRANSLIT)
    n = _NAME_SUFFIX_RE.sub("", ascii_name).strip().upper()
    return re.sub(r"\s+", " ", n)


def synth_id(country: str | None, name: str | None) -> str | None:
    """Deterministic foreign-horse/person key: `x:CC:NORMALIZED NAME`."""
    n = normalize_name(name)
    if not n:
        return None
    return f"x:{(country or '??').upper()}:{n}"


# ---------------------------------------------------------------------------
# Identity redirects — make merges permanent across future imports.
#
# `merge_persons`/`merge_horses` register the losing row's source ids and
# synthetic name keys in the `identity_redirect` table. resolve_* / upsert_*
# consult it before INSERT so a re-import of a merged-away duplicate lands on
# the keeper instead of minting the duplicate all over again.
# ---------------------------------------------------------------------------

def _resolve_redirect(cur, entity: str, source: str, source_key: Any) -> int | None:
    """Return the canonical pk that a merged-away key now points to (or None).

    Matches the key under its own `source` OR the generic `synth` bucket,
    so a name-derived synthetic key minted by any source is caught.
    """
    if source_key is None or source_key == "":
        return None
    cur.execute(
        "SELECT to_id FROM identity_redirect "
        " WHERE entity = %s AND source_key = %s AND source IN (%s, 'synth') "
        " ORDER BY (source = 'synth') ASC LIMIT 1",
        (entity, str(source_key), source),
    )
    row = cur.fetchone()
    return row[0] if row else None


def _redirect_synth_country(entity: str, row: dict) -> str | None:
    if entity == "person":
        return row.get("license_country")
    return row.get("birth_country") or row.get("registration_country")


def _register_redirects(cur, entity: str, src_row: dict, to_id: int,
                        from_id: int) -> None:
    """Point every key of the losing row (`src_row`/`from_id`) at `to_id`.

    Also repoints any pre-existing redirects that targeted the losing row,
    so chains of successive merges stay flat and correct.
    """
    pk = "horse_id" if entity == "horse" else "person_id"
    id_cols = (_HORSE_SOURCE_ID_COLS if entity == "horse"
               else _PERSON_SOURCE_ID_COLS)

    # 1. Re-point redirects that aimed at the now-deleted row.
    cur.execute(
        "UPDATE identity_redirect SET to_id = %s "
        " WHERE entity = %s AND to_id = %s",
        (to_id, entity, from_id),
    )

    def _put(source: str, key: Any) -> None:
        if key is None or key == "":
            return
        cur.execute(
            "INSERT INTO identity_redirect (entity, source, source_key, to_id) "
            "VALUES (%s, %s, %s, %s) "
            "ON CONFLICT (entity, source, source_key) "
            "DO UPDATE SET to_id = EXCLUDED.to_id",
            (entity, source, str(key), to_id),
        )

    # 2. Per-source ids on the losing row. Synthetic ids (`x:CC:NAME`) are
    #    also mirrored into the generic 'synth' bucket so a no-id re-import
    #    from ANY source (not just the one that minted it) is caught.
    for col in id_cols:
        val = src_row.get(col)
        _put(col[:-3], val)  # 'st_id' -> source 'st'
        if isinstance(val, str) and val.startswith("x:"):
            _put("synth", val)

    # 3. Synthetic name keys (so a no-id re-import of the same name lands here).
    country = _redirect_synth_country(entity, src_row)
    names = [src_row.get("name")]
    if entity == "person":
        names.append(src_row.get("short_name"))
    for nm in names:
        for cc in (country, None):
            sid = synth_id(cc, nm)
            if sid:
                _put("synth", sid)


def _unregister_redirects(cur, entity: str, restored_row: dict,
                          to_id: int) -> None:
    """Remove redirects created by a merge that is being rolled back.

    Only deletes redirects that point at the keeper (`to_id`) AND match a
    key the restored row owns, so unrelated redirects to the same keeper
    survive.
    """
    id_cols = (_HORSE_SOURCE_ID_COLS if entity == "horse"
               else _PERSON_SOURCE_ID_COLS)
    keys: list[tuple[str, str]] = []
    for col in id_cols:
        val = restored_row.get(col)
        if val not in (None, ""):
            keys.append((col[:-3], str(val)))
    country = _redirect_synth_country(entity, restored_row)
    names = [restored_row.get("name")]
    if entity == "person":
        names.append(restored_row.get("short_name"))
    for nm in names:
        for cc in (country, None):
            sid = synth_id(cc, nm)
            if sid:
                keys.append(("synth", sid))
    for source, key in keys:
        cur.execute(
            "DELETE FROM identity_redirect "
            " WHERE entity = %s AND source = %s AND source_key = %s AND to_id = %s",
            (entity, source, key, to_id),
        )


# ---------------------------------------------------------------------------
# Identity helpers used by all sources at ingest time
# ---------------------------------------------------------------------------

def _birth_year_from(dob: date | None, age: int | None, race_date: date | None) -> int | None:
    if dob is not None:
        return dob.year
    if age and race_date:
        return race_date.year - age
    return None


def find_horse_by_race_context(
    cur, race_id: int | None, program_number: int | None
) -> int | None:
    """Reuse an existing horse on the same race + program number.

    This catches the case where an ATG re-ingest of a foreign-only race
    finds a starter ST already wrote (with a real st_id), so we attach
    the ATG data to that horse instead of minting a synthetic.
    """
    if race_id is None or program_number is None:
        return None
    cur.execute(
        "SELECT horse_id FROM entry "
        " WHERE race_id = %s AND program_number = %s "
        " LIMIT 1",
        (race_id, program_number),
    )
    row = cur.fetchone()
    return row[0] if row else None


def find_horse_by_pedigree(
    cur,
    name: str,
    birth_year: int | None,
    sire_name: str | None,
    dam_name: str | None,
    *,
    country: str | None = None,
) -> int | None:
    """Pedigree triangulation: same name + birth_year + sire_name + dam_name.

    All four fields must be non-empty and match (case-insensitive, after
    name suffix normalization). Returns the horse_id of the unique match
    or None if zero or multiple candidates.
    """
    norm = normalize_name(name)
    sn = normalize_name(sire_name)
    dn = normalize_name(dam_name)
    if not norm or not birth_year or not sn or not dn:
        return None
    cur.execute(
        """
        SELECT horse_id FROM horse
         WHERE v2_normalize_name(name) = %s
           AND EXTRACT(year FROM date_of_birth)::int = %s
           AND v2_normalize_name(coalesce(sire_name, '')) = %s
           AND v2_normalize_name(coalesce(dam_name,  '')) = %s
        LIMIT 2
        """,
        (norm, birth_year, sn, dn),
    )
    rows = cur.fetchall()
    if len(rows) == 1:
        return rows[0][0]
    return None  # zero or ambiguous


# ---------------------------------------------------------------------------
# resolve_horse — main entry point used by every source's importer
# ---------------------------------------------------------------------------

def resolve_horse(
    cur,
    *,
    source: str,
    source_id: Any,
    canonical_fields: dict,
    raw_payload: dict | None = None,
    registration_number: str | None = None,
    ueln_number: str | None = None,
    race_id: int | None = None,
    program_number: int | None = None,
    sire_name: str | None = None,
    dam_name: str | None = None,
) -> int:
    """Resolve / create a horse for an incoming `source` start.

    Matching order:
      1. Strong IDs (source_id, registration_number, ueln_number) — delegated
         to `etl.matching.upsert_horse`.
      2. Race-context — same race_id + program_number reuses existing horse.
      3. Pedigree triangulation — name + birth_year + sire_name + dam_name.
      4. Synthetic key reuse — `x:CC:NAME` for foreign horses.
      5. INSERT — never matched, mint new row.

    `source_id` may be a real numeric id (preferred) or None (we'll mint a
    synthetic). Pass `race_id` + `program_number` whenever a race row is
    already created, so race-context can short-circuit when other sources
    already entered the same starter.
    """
    name = canonical_fields.get("name")
    country = canonical_fields.get("birth_country") or canonical_fields.get("registration_country")
    dob = canonical_fields.get("date_of_birth")

    # 0. Redirect: this (source, source_id) may have been merged away into a
    #    canonical keeper. Honour the redirect so we never re-mint the dup.
    if source_id is not None and source_id != "" and not _is_synth_id(source_id):
        rid = _resolve_redirect(cur, "horse", source, source_id)
        if rid is not None:
            _attach_source_to_horse(cur, rid, source, source_id, canonical_fields,
                                    raw_payload, registration_number, ueln_number)
            return rid

    # 1. Strong-ID path: delegate to matching.upsert_horse which already
    #    handles source_id, reg_number, UELN, and the SE id-equivalence quirk.
    if source_id is not None and source_id != "" and not _is_synth_id(source_id):
        return upsert_horse(
            cur, source, source_id, canonical_fields,
            raw_payload=raw_payload,
            registration_number=registration_number,
            ueln_number=ueln_number,
        )

    # 2. Race-context: same race + same program number → reuse, attach source id.
    rc_id = find_horse_by_race_context(cur, race_id, program_number)
    if rc_id is not None:
        _attach_source_to_horse(cur, rc_id, source, source_id, canonical_fields,
                                raw_payload, registration_number, ueln_number)
        return rc_id

    # 3. Pedigree triangulation: name + year + sire + dam (all required).
    ped_id = find_horse_by_pedigree(
        cur, name,
        _birth_year_from(dob, canonical_fields.get("age"), None),
        sire_name, dam_name,
        country=country,
    )
    if ped_id is not None:
        _attach_source_to_horse(cur, ped_id, source, source_id, canonical_fields,
                                raw_payload, registration_number, ueln_number)
        return ped_id

    # 4. Synthetic key reuse for foreign horses without a stable id.
    if source_id is None or source_id == "":
        sid = synth_id(country, name)
        if sid is None:
            return None  # can't even mint a key, give up
        # 4a. Redirect on the synth key: a merged-away dup of this name.
        rid = _resolve_redirect(cur, "horse", source, sid)
        if rid is not None:
            _attach_source_to_horse(cur, rid, source, sid, canonical_fields,
                                    raw_payload, registration_number, ueln_number)
            return rid
        # Quick check: synth already attached?
        col = f"{source}_id"
        cur.execute(f"SELECT horse_id FROM horse WHERE {col} = %s LIMIT 1", (sid,))
        row = cur.fetchone()
        if row:
            return row[0]
        source_id = sid

    # 5. Final: INSERT (or update if the synth happens to match) via upsert_horse.
    return upsert_horse(
        cur, source, source_id, canonical_fields,
        raw_payload=raw_payload,
        registration_number=registration_number,
        ueln_number=ueln_number,
    )


# ---------------------------------------------------------------------------
# resolve_person — same pattern, no pedigree
# ---------------------------------------------------------------------------

def resolve_person(
    cur,
    *,
    source: str,
    source_id: Any,
    canonical_fields: dict,
    raw_payload: dict | None = None,
    role_flags: dict[str, bool] | None = None,
    country: str | None = None,
) -> int | None:
    """Resolve / create a person.

    Matching order:
      1. Strong source_id (or SE int-equivalence).
      2. Synthetic key `x:CC:NORMALIZED_NAME` reuse.
      3. INSERT.

    Persons don't have UELN/reg numbers, and pedigree isn't applicable.
    """
    name = canonical_fields.get("name") or canonical_fields.get("short_name")
    if source_id is not None and source_id != "" and not _is_synth_id(source_id):
        rid = _resolve_redirect(cur, "person", source, source_id)
        if rid is not None:
            return rid
        return upsert_person(
            cur, source, source_id, canonical_fields,
            raw_payload=raw_payload, role_flags=role_flags,
        )
    if not name:
        return None
    sid = synth_id(country, name)
    if sid is None:
        return None
    rid = _resolve_redirect(cur, "person", source, sid)
    if rid is not None:
        return rid
    col = f"{source}_id"
    cur.execute(f"SELECT person_id FROM person WHERE {col} = %s LIMIT 1", (sid,))
    row = cur.fetchone()
    if row:
        # Still UPDATE flags/source_data via upsert_person.
        return upsert_person(
            cur, source, sid, canonical_fields,
            raw_payload=raw_payload, role_flags=role_flags,
        )
    return upsert_person(
        cur, source, sid, canonical_fields,
        raw_payload=raw_payload, role_flags=role_flags,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _is_synth_id(source_id: Any) -> bool:
    return isinstance(source_id, str) and source_id.startswith("x:")


def _attach_source_to_horse(
    cur, horse_id: int, source: str, source_id: Any, canonical_fields: dict,
    raw_payload: dict | None, registration_number: str | None,
    ueln_number: str | None,
) -> None:
    """Attach a (source, source_id) link to an existing horse row.

    Used when race-context or pedigree triangulation matched a different
    horse than `upsert_horse` would have. Keeps source_data, primary_source,
    and cross-id fields in sync without inserting a new row.
    """
    col = f"{source}_id"
    set_parts: list[str] = []
    params: list[Any] = []

    # Stringify text-typed source id columns.
    if col in {"atg_id", "usta_id", "letrot_id", "kmtid_id", "hvt_id", "breedly_id"}:
        stored = str(source_id) if source_id is not None else None
    else:
        stored = source_id

    if stored is not None:
        # Only attach if currently null OR equal (avoid violating unique index).
        cur.execute(f"SELECT {col} FROM horse WHERE horse_id=%s", (horse_id,))
        cur_val = cur.fetchone()[0]
        if cur_val is None:
            set_parts.append(f"{col} = %s")
            params.append(stored)
        elif cur_val != stored:
            # Conflict: row already has a different id from this source.
            # Stash the alternate id in source_data.<source>.aliases.
            cur.execute("SELECT source_data FROM horse WHERE horse_id=%s", (horse_id,))
            sd = cur.fetchone()[0] or {}
            block = dict(sd.get(source) or {})
            aliases = list(block.get("aliases") or [])
            if stored not in aliases:
                aliases.append(stored)
            block["aliases"] = aliases
            sd[source] = block
            set_parts.append("source_data = %s")
            params.append(Json(sd))

    if registration_number:
        cur.execute("SELECT registration_number FROM horse WHERE horse_id=%s", (horse_id,))
        if cur.fetchone()[0] is None:
            set_parts.append("registration_number = %s")
            params.append(registration_number)

    if ueln_number:
        cur.execute("SELECT ueln_number FROM horse WHERE horse_id=%s", (horse_id,))
        if cur.fetchone()[0] is None:
            set_parts.append("ueln_number = %s")
            params.append(ueln_number)

    # Merge raw_payload into source_data.<source>.
    if raw_payload:
        cur.execute("SELECT source_data FROM horse WHERE horse_id=%s", (horse_id,))
        sd = cur.fetchone()[0] or {}
        existing_block = sd.get(source) or {}
        # Keep existing aliases/keys, overlay raw_payload (latest wins).
        merged = {**existing_block, **raw_payload}
        sd[source] = merged
        # Avoid double source_data update if we already added it above.
        if not any(p.startswith("source_data") for p in set_parts):
            set_parts.append("source_data = %s")
            params.append(Json(sd))
        else:
            params[-1] = Json(sd)

    if not set_parts:
        return
    set_parts.append("last_updated_at = NOW()")
    params.append(horse_id)
    cur.execute(
        f"UPDATE horse SET {', '.join(set_parts)} WHERE horse_id = %s",
        params,
    )


# ---------------------------------------------------------------------------
# Merge — move all references from one horse/person row into another and
# delete the source row. All merges are logged for audit + rollback.
# ---------------------------------------------------------------------------

def _row_to_jsonable(row: dict) -> dict:
    """Convert a horse/person row to a JSON-serializable dict.

    `date` and `datetime` and `Decimal` become strings.
    """
    out: dict = {}
    for k, v in row.items():
        if v is None or isinstance(v, (str, int, float, bool, dict, list)):
            out[k] = v
        else:
            out[k] = str(v)
    return out


def _deep_jsonable(obj):
    """Recursively coerce arbitrary nested structures into JSON-safe values.

    Decimal / date / datetime / UUID etc. become strings. Tuples become
    lists. Used for audit blocks where the leaves come from raw DB rows
    and may include numeric (Decimal) or temporal types that psycopg2's
    Json() adapter can't serialize.
    """
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, dict):
        return {str(k): _deep_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_deep_jsonable(v) for v in obj]
    return str(obj)


def _fetch_horse(cur, horse_id: int) -> dict | None:
    cur.execute("SELECT * FROM horse WHERE horse_id = %s", (horse_id,))
    row = cur.fetchone()
    if not row:
        return None
    cols = [d.name for d in cur.description]
    return dict(zip(cols, row))


def _fetch_person(cur, person_id: int) -> dict | None:
    cur.execute("SELECT * FROM person WHERE person_id = %s", (person_id,))
    row = cur.fetchone()
    if not row:
        return None
    cols = [d.name for d in cur.description]
    return dict(zip(cols, row))


# Entry columns we look at when resolving conflicts on shared races.
# Kept for legacy callers (scripts.merge_duplicate_races still imports it
# as _ENTRY_SOURCE_PRIORITY constant).
_ENTRY_PRIORITY_BY_SOURCE = {
    "atg": 0,  # ATG wins entry-level (richer sulky/odds/equipment)
    "st":  1,
    "letrot": 2,
    "hvt": 3,
    "kmtid": 4,
    "usta": 5,
    "breedly": 6,
}


# ---------------------------------------------------------------------------
# Column-level entry merge
# ---------------------------------------------------------------------------
#
# Background: ATG and LeTrot disagree on which fields they're authoritative
# for. LeTrot has placement/time/shoe/disqualification for French races; ATG
# has odds and (Swedish-only) sulky. Older row-level merge resolved this by
# DELETing the "losing" row, which permanently lost the per-source data on
# the loser. This module merges column-by-column instead so the surviving
# entry has the best non-NULL value for every field, plus a full audit
# block written to `horse_merge_log.from_snapshot.entry_merges[]` for
# rollback.

# These are the only columns merge logic touches. `entry_id`, `race_id`,
# `horse_id`, `is_winner`, `is_placed` are excluded (identity / generated).
_ENTRY_COALESCE_COLS = (
    "program_number", "post", "distance", "tillagg",
    "finish_order", "placement_text",
    "time_text", "auto",
    "odds_plats_text",
    "prize_currency", "prize_original", "prize_fx_rate", "prize_fx_date",
    "driver_id", "trainer_id",
    "earnings_pre",
    "shoe_front_changed", "shoe_back_changed",
    "sulky_changed",
    "kmtid_first_200ms", "kmtid_last_200ms", "kmtid_best_100ms",
    "kmtid_best_100_start_m", "kmtid_actual_distance_m",
    "kmtid_actual_km_time_ms", "kmtid_slipstream_distance_m",
    "kmtid_intervals",
)

# Booleans that should be OR-merged (any source saying true wins).
_ENTRY_OR_COLS = ("withdrawn", "galopp", "disqualified",
                  "driver_changed", "trainer_changed")

_REAL_SULKY_CODES = ("VA", "AM", "HY", "BG")


def _decide_placement(keeper, loser):
    """Result-core fields: placement, time_seconds.

    Prefer non-null. On conflict, prefer the value paired with a finish
    time. If both have time, prefer the keeper but log the disagreement.
    """
    kv = keeper.get("placement")
    lv = loser.get("placement")
    ks = keeper.get("time_seconds")
    ls = loser.get("time_seconds")
    if lv is not None and kv is None:
        return "loser", lv
    if kv is not None and lv is None:
        return "keeper", kv
    if kv is None and lv is None:
        return "keeper", None
    if kv == lv:
        return "equal", kv
    # both set and different
    if ls is not None and ks is None:
        return "loser_has_time", lv
    if ks is not None and ls is None:
        return "keeper_has_time", kv
    return "keeper_disagree", kv


def _decide_time_seconds(keeper, loser):
    kv = keeper.get("time_seconds")
    lv = loser.get("time_seconds")
    if kv is None and lv is not None:
        return "loser", lv, None
    if kv is not None and lv is None:
        return "keeper", kv, None
    if kv is None and lv is None:
        return "both_null", None, None
    if kv == lv:
        return "equal", kv, None
    diff = abs(float(kv) - float(lv))
    if diff <= 1.0:
        return "keeper_close", kv, None
    return "keeper_disagree", kv, {"keeper": kv, "loser": lv, "delta_s": diff}


def _decide_odds(keeper, loser, keeper_src, loser_src):
    """odds: prefer ATG always. Coalesce otherwise."""
    kv = keeper.get("odds")
    lv = loser.get("odds")
    if lv is not None and kv is None:
        return "loser", lv
    if kv is not None and lv is None:
        return "keeper", kv
    if kv is None and lv is None:
        return "keeper", None
    if kv == lv:
        return "equal", kv
    # conflict
    if loser_src == "atg" and keeper_src != "atg":
        return "loser_atg", lv
    return "keeper", kv


def _decide_prize(keeper, loser, keeper_src, loser_src):
    """prize_kr: prefer non-zero, larger value, LeTrot wins ties."""
    kv = keeper.get("prize_kr") or 0
    lv = loser.get("prize_kr") or 0
    if kv == 0 and lv == 0:
        return "both_zero", 0
    if kv == 0:
        return "loser_nonzero", lv
    if lv == 0:
        return "keeper_nonzero", kv
    if kv == lv:
        return "equal", kv
    if lv > kv:
        return "loser_larger", lv
    if kv > lv and loser_src == "letrot" and keeper_src != "letrot":
        # Keep keeper's larger value, but if LeTrot is supposed to be the
        # truthier source for French races and its value is non-zero,
        # only swap if LeTrot's value is larger (already handled above).
        return "keeper_larger", kv
    return "keeper_larger", kv


def _decide_french_pref(keeper, loser, field, *, is_french_race,
                        keeper_src, loser_src):
    """Coalesce; on conflict, prefer LeTrot for French races, ATG otherwise."""
    kv = keeper.get(field)
    lv = loser.get(field)
    if lv is not None and kv is None:
        return "loser", lv
    if kv is not None and lv is None:
        return "keeper", kv
    if kv is None and lv is None:
        return "keeper", None
    if kv == lv:
        return "equal", kv
    if is_french_race and loser_src == "letrot" and keeper_src != "letrot":
        return "loser_french_pref", lv
    if (not is_french_race) and loser_src == "atg" and keeper_src != "atg":
        return "loser_atg_pref", lv
    return "keeper", kv


def _decide_sulky(keeper, loser):
    """sulky: prefer the "real" (equipment) code over colour codes."""
    kv = keeper.get("sulky")
    lv = loser.get("sulky")
    if lv and not kv:
        return "loser", lv
    if kv and not lv:
        return "keeper", kv
    if not kv and not lv:
        return "keeper", kv
    if kv == lv:
        return "equal", kv
    k_real = (kv or "").upper() in _REAL_SULKY_CODES
    l_real = (lv or "").upper() in _REAL_SULKY_CODES
    if l_real and not k_real:
        return "loser_real", lv
    return "keeper", kv


def _merge_source_data(keeper_sd, loser_sd, *, keeper_src, loser_src,
                       shoe_conflict_value, disagreements):
    """Shallow-merge source_data jsonb.

    Rules:
      - keeper-side per-source blocks win on key conflicts (loser overlays
        only missing keys).
      - Add loser's source_data.<loser_src> entirely if keeper has none.
      - Stash loser's shoe_code raw value (when shoe conflicted) under
        source_data.<loser_src>.shoe_code_raw.
      - `_contributors` list contains every distinct primary_source that
        has contributed to the keeper row (including the keeper's own).
      - `_disagreements` list grows with each new merge.
    """
    out = dict(keeper_sd or {})

    for k, v in (loser_sd or {}).items():
        if k.startswith("_"):
            continue  # meta keys handled below
        existing = out.get(k)
        if existing is None:
            out[k] = v
        elif isinstance(existing, dict) and isinstance(v, dict):
            merged = {**v, **existing}  # keeper wins per-key
            out[k] = merged

    if shoe_conflict_value is not None and loser_src:
        block = dict(out.get(loser_src) or {})
        block["shoe_code_raw"] = shoe_conflict_value
        out[loser_src] = block

    contribs = list(out.get("_contributors") or [])
    for s in (keeper_src, loser_src):
        if s and s not in contribs:
            contribs.append(s)
    out["_contributors"] = contribs

    if disagreements:
        prev = list(out.get("_disagreements") or [])
        prev.extend(disagreements)
        out["_disagreements"] = prev

    return out


def _merge_entries_columnwise(
    keeper: dict, loser: dict, *, is_french_race: bool = False
) -> tuple[dict, dict]:
    """Compute merged column values for the keeper entry plus an audit block.

    Returns:
      (set_dict, audit_block)

    `set_dict` contains only columns that *change* on the keeper, ready to
    feed into an `UPDATE entry SET ... WHERE entry_id = keeper_id`.

    The audit block is structured so `rollback_horse_merge` (or a future
    rollback_entry_merge) can perfectly reconstruct both rows.
    """
    keeper_src = (keeper.get("primary_source") or "").lower()
    loser_src  = (loser.get("primary_source")  or "").lower()

    set_dict: dict = {}
    decisions: dict = {}
    disagreements: list = []

    # Simple coalesce columns: only fill keeper if NULL.
    for col in _ENTRY_COALESCE_COLS:
        kv = keeper.get(col)
        lv = loser.get(col)
        if kv is None and lv is not None:
            set_dict[col] = lv
            decisions[col] = "loser_fill"
        elif kv is not None and lv is None:
            decisions[col] = "keeper_only"
        elif kv is None and lv is None:
            decisions[col] = "both_null"
        elif kv == lv:
            decisions[col] = "equal"
        else:
            decisions[col] = "keeper_conflict"

    # OR-merge booleans (true wins).
    for col in _ENTRY_OR_COLS:
        kv = bool(keeper.get(col))
        lv = bool(loser.get(col))
        merged = kv or lv
        if merged != kv:
            set_dict[col] = merged
        decisions[col] = "or_true" if merged else "or_false"

    # placement (decision logic depends on time_seconds availability).
    plc_outcome, plc_val = _decide_placement(keeper, loser)
    if plc_val != keeper.get("placement"):
        set_dict["placement"] = plc_val
    decisions["placement"] = plc_outcome
    if plc_outcome == "keeper_disagree":
        disagreements.append({
            "field": "placement", "keeper": keeper.get("placement"),
            "loser": loser.get("placement"),
            "loser_source": loser_src,
        })

    # time_seconds.
    ts_outcome, ts_val, ts_dis = _decide_time_seconds(keeper, loser)
    if ts_outcome.startswith("loser") and ts_val != keeper.get("time_seconds"):
        set_dict["time_seconds"] = ts_val
    decisions["time_seconds"] = ts_outcome
    if ts_dis:
        ts_dis["loser_source"] = loser_src
        disagreements.append({"field": "time_seconds", **ts_dis})

    # odds (always prefer ATG).
    odds_outcome, odds_val = _decide_odds(keeper, loser, keeper_src, loser_src)
    if odds_val != keeper.get("odds"):
        set_dict["odds"] = odds_val
    decisions["odds"] = odds_outcome

    # prize_kr.
    prz_outcome, prz_val = _decide_prize(keeper, loser, keeper_src, loser_src)
    if prz_val != (keeper.get("prize_kr") or 0):
        set_dict["prize_kr"] = prz_val
    decisions["prize_kr"] = prz_outcome

    # Demographics (age, sex): French-pref.
    for col in ("age", "sex"):
        out, val = _decide_french_pref(keeper, loser, col,
                                       is_french_race=is_french_race,
                                       keeper_src=keeper_src,
                                       loser_src=loser_src)
        if val != keeper.get(col):
            set_dict[col] = val
        decisions[col] = out

    # shoe_code: if conflict, keep keeper, stash loser raw in source_data.
    shoe_conflict_value = None
    kv_shoe = keeper.get("shoe_code")
    lv_shoe = loser.get("shoe_code")
    if kv_shoe is None and lv_shoe is not None:
        set_dict["shoe_code"] = lv_shoe
        decisions["shoe_code"] = "loser_fill"
    elif kv_shoe is not None and lv_shoe is not None and kv_shoe != lv_shoe:
        decisions["shoe_code"] = "keeper_conflict_stash_loser"
        shoe_conflict_value = lv_shoe
    elif kv_shoe == lv_shoe:
        decisions["shoe_code"] = "equal"
    else:
        decisions["shoe_code"] = "both_null"

    # sulky.
    sulky_out, sulky_val = _decide_sulky(keeper, loser)
    if sulky_val != keeper.get("sulky"):
        set_dict["sulky"] = sulky_val
    decisions["sulky"] = sulky_out

    # source_data shallow merge.
    merged_sd = _merge_source_data(
        keeper.get("source_data") or {},
        loser.get("source_data") or {},
        keeper_src=keeper_src,
        loser_src=loser_src,
        shoe_conflict_value=shoe_conflict_value,
        disagreements=disagreements,
    )
    # Always overwrite source_data to refresh _contributors / _disagreements.
    set_dict["source_data"] = merged_sd

    audit_block = {
        "from_entry_id": loser.get("entry_id"),
        "from_source":   loser_src,
        "to_entry_id":   keeper.get("entry_id"),
        "to_source":     keeper_src,
        "race_id":       keeper.get("race_id"),
        "horse_id":      keeper.get("horse_id"),
        "is_french_race": bool(is_french_race),
        "before_keeper": _row_to_jsonable(keeper),
        "before_loser":  _row_to_jsonable(loser),
        "field_decisions": _deep_jsonable(decisions),
        "applied_changes": _deep_jsonable({
            k: (v if not isinstance(v, dict) else "<jsonb>")
            for k, v in set_dict.items()
        }),
    }
    return set_dict, audit_block


def merge_horses(
    cur,
    from_horse_id: int,
    to_horse_id: int,
    *,
    reason: str,
    method: str,
    merged_by: str = "system",
    dry_run: bool = True,
) -> dict:
    """Merge `from_horse_id` into `to_horse_id`.

    Steps:
      1. Move entries (with same-race conflict resolution).
      2. Move horse_owner_history / horse_trainer_history.
      3. Re-point pedigree FKs (any horse where sire_id/dam_id = from_id).
      4. Merge source_data + per-source ID columns into destination.
      5. Snapshot the source row to horse_merge_log.
      6. DELETE the source horse row.

    Returns a dict summary: { entries_moved, conflicts_resolved,
    histories_moved, pedigree_refs_repointed, deleted }.

    With `dry_run=True`, no writes happen — returns the same summary as if
    it would, plus a `preview: True` flag.
    """
    if from_horse_id == to_horse_id:
        return {"error": "from == to", "preview": dry_run}

    src = _fetch_horse(cur, from_horse_id)
    dst = _fetch_horse(cur, to_horse_id)
    if not src:
        return {"error": f"from_horse_id {from_horse_id} not found", "preview": dry_run}
    if not dst:
        return {"error": f"to_horse_id {to_horse_id} not found", "preview": dry_run}

    summary: dict = {
        "from_horse_id": from_horse_id,
        "to_horse_id":   to_horse_id,
        "from_name":     src.get("name"),
        "to_name":       dst.get("name"),
        "method":        method,
        "preview":       dry_run,
    }

    # 1. Plan entry move (resolve same-race conflicts).
    cur.execute(
        "SELECT * FROM entry WHERE horse_id = %s",
        (from_horse_id,),
    )
    cols = [d.name for d in cur.description]
    src_entries = [dict(zip(cols, r)) for r in cur.fetchall()]

    cur.execute(
        "SELECT race_id, entry_id FROM entry WHERE horse_id = %s",
        (to_horse_id,),
    )
    dst_by_race = {race_id: eid for race_id, eid in cur.fetchall()}

    conflicts: list[tuple[int, int]] = []  # (src_entry_id, dst_entry_id)
    movable: list[int] = []
    for e in src_entries:
        if e["race_id"] in dst_by_race:
            conflicts.append((e["entry_id"], dst_by_race[e["race_id"]]))
        else:
            movable.append(e["entry_id"])

    summary["entries_moved"]      = len(movable)
    summary["conflicts_resolved"] = len(conflicts)

    # Pre-fetch French-ness flag for every conflict race; used by the
    # column-level merger to pick French-pref or ATG-pref on ambiguous
    # demographic columns.
    is_french_by_race: dict[int, bool] = {}
    if conflicts:
        conflict_race_ids = list({
            e["race_id"] for e in src_entries
            if e["race_id"] in dst_by_race
        })
        cur.execute(
            "SELECT r.race_id, t.country "
            "  FROM race r "
            "  LEFT JOIN track t ON r.track_id = t.track_id "
            " WHERE r.race_id = ANY(%s)",
            (conflict_race_ids,),
        )
        for rid, country in cur.fetchall():
            is_french_by_race[rid] = (country == "FR")

    # 2. Plan history move.
    cur.execute("SELECT COUNT(*) FROM horse_owner_history   WHERE horse_id=%s", (from_horse_id,))
    n_owner = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) FROM horse_trainer_history WHERE horse_id=%s", (from_horse_id,))
    n_trainer = cur.fetchone()[0]
    summary["histories_moved"] = n_owner + n_trainer

    # 3. Plan pedigree FK repoint.
    cur.execute("SELECT COUNT(*) FROM horse WHERE sire_id=%s OR dam_id=%s",
                (from_horse_id, from_horse_id))
    summary["pedigree_refs_repointed"] = cur.fetchone()[0]

    if dry_run:
        return summary

    # ------------------------------------------------------------------
    # EXECUTE
    # ------------------------------------------------------------------

    # Resolve conflicts: column-level merge into the keeper (dst entry),
    # then DELETE the loser (src entry). Every per-entry merge produces an
    # audit block stored later in horse_merge_log.from_snapshot.entry_merges.
    entry_audits: list[dict] = []
    for src_eid, dst_eid in conflicts:
        cur.execute(
            "SELECT * FROM entry WHERE entry_id IN (%s, %s)",
            (src_eid, dst_eid),
        )
        ecols = [d.name for d in cur.description]
        rows = [dict(zip(ecols, r)) for r in cur.fetchall()]
        by_id = {r["entry_id"]: r for r in rows}
        keeper = by_id[dst_eid]
        loser  = by_id[src_eid]
        is_french = is_french_by_race.get(keeper.get("race_id"), False)

        set_dict, audit_block = _merge_entries_columnwise(
            keeper, loser, is_french_race=is_french,
        )
        entry_audits.append(audit_block)

        if set_dict:
            cols_sql = ", ".join(f"{c} = %s" for c in set_dict)
            vals = [Json(v) if isinstance(v, dict) else v
                    for v in set_dict.values()]
            cur.execute(
                f"UPDATE entry SET {cols_sql}, last_updated_at = NOW() "
                f"WHERE entry_id = %s",
                [*vals, dst_eid],
            )
        cur.execute("DELETE FROM entry WHERE entry_id = %s", (src_eid,))

    # Move remaining src-only entries.
    if movable:
        cur.execute(
            "UPDATE entry SET horse_id = %s WHERE entry_id = ANY(%s)",
            (to_horse_id, movable),
        )

    # Move history (delete-and-insert to avoid PK collisions on shared dates).
    cur.execute(
        """
        DELETE FROM horse_owner_history o
         WHERE o.horse_id = %s
           AND EXISTS (SELECT 1 FROM horse_owner_history o2
                        WHERE o2.horse_id = %s AND o2.from_date = o.from_date)
        """,
        (from_horse_id, to_horse_id),
    )
    cur.execute(
        "UPDATE horse_owner_history SET horse_id = %s WHERE horse_id = %s",
        (to_horse_id, from_horse_id),
    )
    cur.execute(
        """
        DELETE FROM horse_trainer_history o
         WHERE o.horse_id = %s
           AND EXISTS (SELECT 1 FROM horse_trainer_history o2
                        WHERE o2.horse_id = %s AND o2.from_date = o.from_date)
        """,
        (from_horse_id, to_horse_id),
    )
    cur.execute(
        "UPDATE horse_trainer_history SET horse_id = %s WHERE horse_id = %s",
        (to_horse_id, from_horse_id),
    )

    # Repoint pedigree FKs to destination.
    cur.execute("UPDATE horse SET sire_id = %s WHERE sire_id = %s",
                (to_horse_id, from_horse_id))
    cur.execute("UPDATE horse SET dam_id  = %s WHERE dam_id  = %s",
                (to_horse_id, from_horse_id))

    # Merge per-source IDs and source_data from src into dst for automated
    # duplicate merges. Manual merges are often used for "move relationships
    # to this canonical horse" cleanups; in that workflow the losing row may
    # be a real horse whose own identity should not leak into the keeper.
    if method != "manual":
        _merge_horse_identity(cur, src, dst)

    # Snapshot + log. The new from_snapshot shape is
    #   {
    #     "horse_row": <full source row>,
    #     "to_horse_row": <full keeper row before merge>,
    #     "moved_entry_ids": [...],
    #     "entry_merges": [audit_block, ...],
    #   }
    # so rollback can restore the keeper row and move exact source entries
    # back instead of relying on primary_source heuristics.
    full_snapshot = {
        "horse_row":       _row_to_jsonable(src),
        "to_horse_row":    _row_to_jsonable(dst),
        "moved_entry_ids": list(movable),
        "entry_merges":    entry_audits,
    }
    cur.execute(
        """
        INSERT INTO horse_merge_log
            (from_horse_id, to_horse_id, reason, method,
             entries_moved, conflicts_resolved, from_snapshot, merged_by)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        """,
        (from_horse_id, to_horse_id, reason, method,
         summary["entries_moved"], summary["conflicts_resolved"],
         Json(full_snapshot), merged_by),
    )

    # Make the merge permanent: re-imports of the losing keys route to dst.
    _register_redirects(cur, "horse", src, to_horse_id, from_horse_id)

    # Delete source horse.
    cur.execute("DELETE FROM horse WHERE horse_id = %s", (from_horse_id,))
    summary["deleted"] = True
    return summary


def merge_persons(
    cur,
    from_person_id: int,
    to_person_id: int,
    *,
    reason: str,
    method: str,
    merged_by: str = "system",
    dry_run: bool = True,
) -> dict:
    """Merge `from_person_id` into `to_person_id` across driver/trainer roles."""
    if from_person_id == to_person_id:
        return {"error": "from == to", "preview": dry_run}

    src = _fetch_person(cur, from_person_id)
    dst = _fetch_person(cur, to_person_id)
    if not src:
        return {"error": "from not found", "preview": dry_run}
    if not dst:
        return {"error": "to not found",   "preview": dry_run}

    cur.execute(
        "SELECT COUNT(*) FROM entry WHERE driver_id = %s OR trainer_id = %s",
        (from_person_id, from_person_id),
    )
    n_entries = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) FROM horse_owner_history   WHERE owner_id   = %s", (from_person_id,))
    n_owner = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) FROM horse_trainer_history WHERE trainer_id = %s", (from_person_id,))
    n_trainer = cur.fetchone()[0]

    summary = {
        "from_person_id": from_person_id,
        "to_person_id":   to_person_id,
        "from_name":      src.get("name"),
        "to_name":        dst.get("name"),
        "method":         method,
        "entries_moved":  n_entries,
        "histories_moved": n_owner + n_trainer,
        "preview":        dry_run,
    }
    if dry_run:
        return summary

    cur.execute("UPDATE entry SET driver_id  = %s WHERE driver_id  = %s", (to_person_id, from_person_id))
    cur.execute("UPDATE entry SET trainer_id = %s WHERE trainer_id = %s", (to_person_id, from_person_id))
    cur.execute("UPDATE horse_owner_history   SET owner_id   = %s WHERE owner_id   = %s", (to_person_id, from_person_id))
    cur.execute("UPDATE horse_trainer_history SET trainer_id = %s WHERE trainer_id = %s", (to_person_id, from_person_id))

    _merge_person_identity(cur, src, dst)

    cur.execute(
        """
        INSERT INTO person_merge_log
            (from_person_id, to_person_id, reason, method, entries_moved, from_snapshot, merged_by)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        """,
        (from_person_id, to_person_id, reason, method, n_entries,
         Json(_row_to_jsonable(src)), merged_by),
    )
    # Make the merge permanent: re-imports of the losing keys route to dst.
    _register_redirects(cur, "person", src, to_person_id, from_person_id)
    cur.execute("DELETE FROM person WHERE person_id = %s", (from_person_id,))
    summary["deleted"] = True
    return summary


_HORSE_SOURCE_ID_COLS = ("st_id", "atg_id", "usta_id", "letrot_id",
                         "kmtid_id", "hvt_id", "breedly_id")
_PERSON_SOURCE_ID_COLS = ("st_id", "atg_id", "usta_id", "letrot_id", "hvt_id")

# Canonical fields that have a UNIQUE index. When moving these from src→dst
# during a merge we must also NULL them on src first to free the constraint.
_HORSE_UNIQUE_CANONICAL_COLS = ("registration_number", "ueln_number")

# Canonical fields that should fill from src when dst is missing them.
_HORSE_CANONICAL_FILL_COLS = (
    "name", "date_of_birth", "gender_code", "color", "breed_code",
    "birth_country", "bred_country", "registration_country",
    "registration_number", "ueln_number",
    "is_dead", "is_guest_horse", "has_offspring",
    "breed_index", "inbreed_coefficient", "sire_id", "dam_id",
    "sire_name", "dam_name",
    "scraped_starts", "scraped_wins", "scraped_prize_money_kr", "scraped_record",
)
_PERSON_CANONICAL_FILL_COLS = (
    "name", "short_name", "license_country",
    "is_driver", "is_trainer", "is_owner", "is_breeder",
)


_NAME_COUNTRY_SUFFIX_RX = re.compile(r"\((?!SE\b|SWE\b)[A-Z]{2,3}\)\s*$")
_NAME_NOISE_SUFFIX_RX = re.compile(r"\s*\((?:SE|SWE)\)\s*$", re.IGNORECASE)


def _pick_better_horse_name(dst_name: str | None, src_name: str | None) -> str | None:
    """Return whichever name carries more information.

    Priority:
      1. Non-empty wins over empty.
      2. Name ending with a country suffix like '(FR)' / '(USA)' wins
         (the frontend uses this suffix to render a flag).
      3. Mixed-case wins over ALL-UPPERCASE (LeTrot stores names in
         uppercase; ATG/ST keep the proper-case version).
      4. Otherwise keep dst.
    """
    if not dst_name:
        return src_name or None
    if not src_name:
        return dst_name

    # Strip noise suffixes (SE/SWE — SE is default, suffix is clutter)
    # BEFORE evaluating. We never want them to outrank a clean name.
    dst_name = _NAME_NOISE_SUFFIX_RX.sub("", dst_name).rstrip()
    src_name = _NAME_NOISE_SUFFIX_RX.sub("", src_name).rstrip()

    dst_has_suffix = bool(_NAME_COUNTRY_SUFFIX_RX.search(dst_name))
    src_has_suffix = bool(_NAME_COUNTRY_SUFFIX_RX.search(src_name))
    dst_is_caps = dst_name.upper() == dst_name and any(
        ch.isalpha() for ch in dst_name)
    src_is_caps = src_name.upper() == src_name and any(
        ch.isalpha() for ch in src_name)

    if src_has_suffix and not dst_has_suffix and not dst_is_caps and src_is_caps:
        code = _NAME_COUNTRY_SUFFIX_RX.search(src_name).group(0).strip()
        return f"{dst_name} {code}"
    if src_has_suffix and not dst_has_suffix:
        return src_name
    if dst_has_suffix and not src_has_suffix:
        return dst_name

    # Same suffix-status. Prefer mixed-case over ALL-CAPS.
    if dst_is_caps and not src_is_caps:
        return src_name
    return dst_name


def _merge_horse_identity(cur, src: dict, dst: dict) -> None:
    """Move per-source IDs and fill missing canonical fields from src into dst."""
    set_parts: list[str] = []
    params: list[Any] = []

    # Per-source IDs: take from src if dst is null. If both have one, stash
    # src's into source_data.<source>.aliases.
    new_sd = dict(dst.get("source_data") or {})
    src_sd = dict(src.get("source_data") or {})

    cols_to_clear_on_src: list[str] = []
    for col in _HORSE_SOURCE_ID_COLS:
        src_val = src.get(col)
        dst_val = dst.get(col)
        if src_val is None:
            continue
        if dst_val is None:
            set_parts.append(f"{col} = %s")
            params.append(src_val)
            cols_to_clear_on_src.append(col)
        elif dst_val != src_val:
            source_name = col.removesuffix("_id")
            block = dict(new_sd.get(source_name) or {})
            aliases = list(block.get("aliases") or [])
            if src_val not in aliases:
                aliases.append(src_val)
            block["aliases"] = aliases
            new_sd[source_name] = block

    # Fill missing canonical fields.
    for col in _HORSE_CANONICAL_FILL_COLS:
        if col in ("sire_id", "dam_id"):
            continue  # handled by pedigree repoint
        if col == "name":
            # Special-case: even if dst.name is set, prefer src.name when it
            # carries strictly more information than dst.name (e.g. ST's
            # "Aubrion du Gers (FR)" should win over LeTrot's all-caps
            # "AUBRION DU GERS" so the frontend's country-suffix flag logic
            # keeps working).
            better = _pick_better_horse_name(dst.get("name"), src.get("name"))
            if better and better != dst.get("name"):
                set_parts.append("name = %s")
                params.append(better)
            continue
        if dst.get(col) is None and src.get(col) is not None:
            set_parts.append(f"{col} = %s")
            params.append(src.get(col))
            if col in _HORSE_UNIQUE_CANONICAL_COLS:
                cols_to_clear_on_src.append(col)

    # Free unique constraints on src BEFORE we copy those IDs/values onto dst.
    # The src row will be DELETEd shortly anyway. Without this step,
    # `UPDATE dst SET st_id = src.st_id` (and same for registration_number /
    # ueln_number) fails with the unique-index violation because src still
    # holds the same value.
    if cols_to_clear_on_src:
        cur.execute(
            f"UPDATE horse SET {', '.join(c + '=NULL' for c in cols_to_clear_on_src)} "
            f"WHERE horse_id = %s",
            (src["horse_id"],),
        )

    # Merge source_data shallowly: keep dst values, only fill missing sub-keys.
    for src_key, src_val in src_sd.items():
        existing = new_sd.get(src_key)
        if existing is None:
            new_sd[src_key] = src_val
        elif isinstance(existing, dict) and isinstance(src_val, dict):
            merged = {**src_val, **existing}  # dst wins per-key
            new_sd[src_key] = merged
    set_parts.append("source_data = %s")
    params.append(Json(new_sd))

    set_parts.append("last_updated_at = NOW()")
    params.append(dst["horse_id"])
    cur.execute(
        f"UPDATE horse SET {', '.join(set_parts)} WHERE horse_id = %s",
        params,
    )


def _merge_person_identity(cur, src: dict, dst: dict) -> None:
    set_parts: list[str] = []
    params: list[Any] = []
    new_sd = dict(dst.get("source_data") or {})
    src_sd = dict(src.get("source_data") or {})

    cols_to_clear_on_src: list[str] = []
    for col in _PERSON_SOURCE_ID_COLS:
        src_val = src.get(col)
        dst_val = dst.get(col)
        if src_val is None:
            continue
        if dst_val is None:
            set_parts.append(f"{col} = %s")
            params.append(src_val)
            cols_to_clear_on_src.append(col)
        elif dst_val != src_val:
            source_name = col.removesuffix("_id")
            block = dict(new_sd.get(source_name) or {})
            aliases = list(block.get("aliases") or [])
            if src_val not in aliases:
                aliases.append(src_val)
            block["aliases"] = aliases
            new_sd[source_name] = block

    # Free unique constraints on src before copying values onto dst (src will
    # be DELETEd shortly). Mirrors the horse merge fix.
    if cols_to_clear_on_src:
        cur.execute(
            f"UPDATE person SET {', '.join(c + '=NULL' for c in cols_to_clear_on_src)} "
            f"WHERE person_id = %s",
            (src["person_id"],),
        )

    for col in _PERSON_CANONICAL_FILL_COLS:
        if dst.get(col) is None and src.get(col) is not None:
            set_parts.append(f"{col} = %s")
            params.append(src.get(col))
        elif col in ("is_driver", "is_trainer", "is_owner", "is_breeder"):
            if src.get(col) and not dst.get(col):
                set_parts.append(f"{col} = %s")
                params.append(True)

    for src_key, src_val in src_sd.items():
        existing = new_sd.get(src_key)
        if existing is None:
            new_sd[src_key] = src_val
        elif isinstance(existing, dict) and isinstance(src_val, dict):
            new_sd[src_key] = {**src_val, **existing}
    set_parts.append("source_data = %s")
    params.append(Json(new_sd))

    set_parts.append("last_updated_at = NOW()")
    params.append(dst["person_id"])
    cur.execute(
        f"UPDATE person SET {', '.join(set_parts)} WHERE person_id = %s",
        params,
    )


# ---------------------------------------------------------------------------
# Rollback — re-insert the snapshotted row and move entries back.
# ---------------------------------------------------------------------------

def _split_snapshot(snap: dict) -> tuple[dict, list[dict]]:
    """Extract (horse_row, entry_merges) from a snapshot in either the new
    `{horse_row, entry_merges}` shape or the legacy flat-row shape used
    before column-level merging was introduced."""
    if isinstance(snap, dict) and "horse_row" in snap:
        return (snap.get("horse_row") or {}, list(snap.get("entry_merges") or []))
    return (snap or {}, [])


def _restore_horse_row(cur, horse_id: int, snap: dict) -> None:
    """Restore all non-PK horse columns from a merge snapshot."""
    restore_cols = [c for c in snap.keys() if c != "horse_id"]
    if not restore_cols:
        return
    sets = ", ".join(f"{c} = %s" for c in restore_cols)
    vals = [
        Json(snap[c]) if c == "source_data" and snap[c] is not None else snap[c]
        for c in restore_cols
    ]
    cur.execute(
        f"UPDATE horse SET {sets} WHERE horse_id = %s",
        [*vals, horse_id],
    )


def rollback_horse_merge(cur, merge_id: int) -> dict:
    """Undo a horse merge by re-inserting the source horse row, moving
    movable entries back, and reverting any column-level coalesces on
    conflict entries (then re-inserting their original loser rows).

    Snapshot shape (new):
        {
          "horse_row":    {...full horse row...},
          "entry_merges": [{
              "from_entry_id": int, "to_entry_id": int,
              "before_keeper": {...keeper row pre-merge...},
              "before_loser":  {...loser row pre-merge...},
              ...
          }]
        }

    Legacy snapshots (just a flat horse row) are also supported — those
    rolled merges had no column-level entry data so the rollback can only
    move the surviving entries.
    """
    cur.execute(
        "SELECT * FROM horse_merge_log WHERE merge_id = %s AND NOT rolled_back",
        (merge_id,),
    )
    row = cur.fetchone()
    if not row:
        return {"error": "merge_id not found or already rolled back"}
    cols = [d.name for d in cur.description]
    rec = dict(zip(cols, row))

    raw_snapshot = rec["from_snapshot"]
    horse_snap, entry_audits = _split_snapshot(raw_snapshot)
    to_snap = raw_snapshot.get("to_horse_row") if isinstance(raw_snapshot, dict) else None
    moved_entry_ids = (
        list(raw_snapshot.get("moved_entry_ids") or [])
        if isinstance(raw_snapshot, dict) else []
    )
    from_id = rec["from_horse_id"]
    to_id = rec["to_horse_id"]

    # 1. If available, first restore the keeper row. This frees any unique
    # values copied from the source before we re-insert the source row.
    if to_snap:
        _restore_horse_row(cur, to_id, to_snap)

    # 2. Re-insert original horse row with its old horse_id (override SERIAL).
    insert_cols: list[str] = []
    insert_vals: list[Any] = []
    for k, v in horse_snap.items():
        if k == "horse_id":
            continue
        if k == "source_data" and v is not None:
            insert_vals.append(Json(v))
        else:
            insert_vals.append(v)
        insert_cols.append(k)

    cur.execute(
        f"INSERT INTO horse (horse_id, {', '.join(insert_cols)}) "
        f"VALUES (%s, {', '.join(['%s'] * len(insert_cols))})",
        [from_id] + insert_vals,
    )

    # 3. Revert column-level entry merges first: restore keeper to its
    # pre-merge state, then re-insert the loser entry with its original id.
    entries_restored = 0
    for audit in entry_audits:
        keeper_id = audit.get("to_entry_id")
        loser_snap = audit.get("before_loser") or {}
        keeper_snap = audit.get("before_keeper") or {}
        if not keeper_id or not keeper_snap or not loser_snap:
            continue

        # Restore keeper row to its pre-merge values.
        restore_cols = [c for c in keeper_snap.keys()
                        if c not in ("entry_id", "is_winner", "is_placed")]
        if restore_cols:
            sets = ", ".join(f"{c} = %s" for c in restore_cols)
            vals = [Json(keeper_snap[c]) if c == "source_data"
                                          and keeper_snap[c] is not None
                    else keeper_snap[c]
                    for c in restore_cols]
            cur.execute(
                f"UPDATE entry SET {sets} WHERE entry_id = %s",
                [*vals, keeper_id],
            )

        # Re-insert loser entry, preserving entry_id so any external
        # references survive. `is_winner`/`is_placed` are generated.
        loser_id = loser_snap.get("entry_id")
        loser_cols = [c for c in loser_snap.keys()
                      if c not in ("is_winner", "is_placed")]
        loser_vals = [Json(loser_snap[c]) if c == "source_data"
                                            and loser_snap[c] is not None
                      else loser_snap[c]
                      for c in loser_cols]
        # Point the restored loser at the (now also restored) from_id horse.
        if "horse_id" in loser_cols:
            loser_vals[loser_cols.index("horse_id")] = from_id
        cur.execute(
            f"INSERT INTO entry ({', '.join(loser_cols)}) "
            f"VALUES ({', '.join(['%s'] * len(loser_cols))}) "
            f"ON CONFLICT (entry_id) DO NOTHING",
            loser_vals,
        )
        if loser_id:
            entries_restored += 1

    # 4. Move any remaining movable entries (those that weren't conflicts)
    # back to the restored horse. New snapshots carry exact entry IDs; old
    # snapshots fall back to the previous primary_source heuristic.
    if moved_entry_ids:
        cur.execute(
            """
            UPDATE entry SET horse_id = %s
             WHERE horse_id = %s
               AND entry_id = ANY(%s::bigint[])
            """,
            (from_id, to_id, moved_entry_ids),
        )
    else:
        audited_keeper_ids = {a.get("to_entry_id") for a in entry_audits
                              if a.get("to_entry_id")}
        snap_primary = horse_snap.get("primary_source")
        if snap_primary:
            cur.execute(
                """
                UPDATE entry SET horse_id = %s
                 WHERE horse_id = %s
                   AND primary_source = %s
                   AND NOT (entry_id = ANY(%s::bigint[]))
                """,
                (from_id, to_id, snap_primary, list(audited_keeper_ids) or [0]),
            )

    # Drop the redirects this merge created so re-imports resolve to the
    # freshly-restored row again instead of the old keeper.
    _unregister_redirects(cur, "horse", horse_snap, to_id)

    cur.execute(
        "UPDATE horse_merge_log SET rolled_back = TRUE, rolled_back_at = NOW() "
        "WHERE merge_id = %s",
        (merge_id,),
    )

    return {
        "rolled_back": True,
        "horse_restored": from_id,
        "merge_id": merge_id,
        "entries_restored": entries_restored,
    }


def rollback_person_merge(cur, merge_id: int) -> dict:
    """Undo a person merge by re-inserting the snapshot and moving its
    entries / histories back to the original person row. Returns a summary.

    Like `rollback_horse_merge`, this can't perfectly restore destructively
    resolved conflicts — only rows that still belong to the merge target
    with the original `primary_source` are moved back.
    """
    cur.execute(
        "SELECT * FROM person_merge_log WHERE merge_id = %s AND NOT rolled_back",
        (merge_id,),
    )
    row = cur.fetchone()
    if not row:
        return {"error": "merge_id not found or already rolled back"}
    cols = [d.name for d in cur.description]
    rec = dict(zip(cols, row))

    snap = rec["from_snapshot"]
    from_id = rec["from_person_id"]
    to_id = rec["to_person_id"]

    insert_cols: list[str] = []
    insert_vals: list[Any] = []
    for k, v in snap.items():
        if k == "person_id":
            continue
        if k == "source_data" and v is not None:
            insert_vals.append(Json(v))
        else:
            insert_vals.append(v)
        insert_cols.append(k)

    cur.execute(
        f"INSERT INTO person (person_id, {', '.join(insert_cols)}) "
        f"VALUES (%s, {', '.join(['%s'] * len(insert_cols))})",
        [from_id] + insert_vals,
    )

    # Move driver/trainer entries back where primary_source matches the
    # original row's primary_source — best-effort.
    snap_primary = snap.get("primary_source")
    if snap_primary:
        cur.execute(
            """
            UPDATE entry SET driver_id = %s
             WHERE driver_id = %s
               AND primary_source = %s
            """,
            (from_id, to_id, snap_primary),
        )
        cur.execute(
            """
            UPDATE entry SET trainer_id = %s
             WHERE trainer_id = %s
               AND primary_source = %s
            """,
            (from_id, to_id, snap_primary),
        )

    # Histories: we don't know which rows were originally on `from`, so we
    # leave them on `to`. (Histories are rebuilt from raw on next sync anyway.)

    _unregister_redirects(cur, "person", snap, to_id)

    cur.execute(
        "UPDATE person_merge_log SET rolled_back = TRUE, rolled_back_at = NOW() "
        "WHERE merge_id = %s",
        (merge_id,),
    )

    return {"rolled_back": True, "person_restored": from_id, "merge_id": merge_id}
