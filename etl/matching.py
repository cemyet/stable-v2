"""
Cross-source identity matching for the master tables.

Every source that wants to write to the master tables goes through these
helpers. The matching protocol is the same for every concept:

  1. Lookup by (source, source_id). If found -> UPDATE.
  2. Else lookup by cross-source identifier (registration_number / ueln /
     name+date+track for races, etc.) when caller passes it. If found ->
     attach source_id, UPDATE.
  3. Else INSERT a new canonical row.

We NEVER auto-merge by name only. Worst case = duplicates that we can
manually merge later via a small admin tool.

Conflict resolution:
  When updating an existing row, canonical fields are overwritten only if
  the incoming source has higher priority than the row's current
  `primary_source` (or the existing value is NULL). The original raw
  values for ALL sources accumulate in `source_data` JSONB.
"""

from __future__ import annotations

from typing import Any, Iterable
from psycopg2.extras import Json
import re


# ---------------------------------------------------------------------------
# Source priority — lower index = higher priority for that field family.
# ---------------------------------------------------------------------------

PRIORITY = {
    "horse":  ("usta", "st", "letrot", "hvt", "atg", "breedly", "kmtid"),
    "race":   ("atg", "st", "usta", "letrot", "hvt", "kmtid"),
    "entry":  ("atg", "st", "usta", "letrot", "hvt", "kmtid"),
    "person": ("st", "atg", "usta", "letrot", "hvt", "kmtid"),
    "track":  ("st", "atg", "usta", "letrot", "hvt", "kmtid"),
}


# Sources whose reported name IS the registry name, and may therefore replace
# a name outright. The aggregator feeds are excluded: ATG reports a rename
# within hours, but it also has the weakest identity links in the system (see
# scripts/split_polluted_atg_ids.py — 21 Italian rows currently carry an
# unrelated horse's name under source_data.atg). They may still re-dress a
# name (case, country suffix, ST's rename star); they just can't replace it.
RENAME_SOURCES = frozenset({"st", "usta", "letrot", "hvt"})


def _rank(concept: str, source: str) -> int:
    order = PRIORITY[concept]
    try:
        return order.index(source)
    except ValueError:
        return 999  # unknown source — lowest priority


def _winning(concept: str, incoming_source: str, current_source: str | None) -> bool:
    """True iff `incoming_source` should overwrite the row's canonical fields.

    Same source always wins (re-fetch refreshes the values); otherwise it
    must outrank the current primary_source.
    """
    if not current_source:
        return True
    if incoming_source == current_source:
        return True
    return _rank(concept, incoming_source) < _rank(concept, current_source)


# ---------------------------------------------------------------------------
# Merge canonical fields with priority + JSONB accumulation
# ---------------------------------------------------------------------------

def _merge_source_data(
    existing: dict | None, source: str, payload: dict | None
) -> dict:
    """Insert/replace `payload` under key `source` in `existing` JSONB."""
    out = dict(existing or {})
    if payload:
        out[source] = payload
    return out


def _record_name_history(
    source_data: dict | None, old_name: str | None, new_name: str, source: str
) -> dict:
    """Append a rename to source_data._name_history (same shape as the
    backfill scripts use, so the existing inspection tooling still reads it)."""
    out = dict(source_data or {})
    hist = list(out.get("_name_history") or [])
    hist.append({"old": old_name, "new": new_name, "by": source})
    out["_name_history"] = hist
    return out


def _build_canonical_update(
    concept: str,
    incoming_source: str,
    current_source: str | None,
    incoming_fields: dict,
    current_row: dict,
) -> dict:
    """Decide which canonical fields to overwrite, given priority rules.

    Returns a dict of {column: value} suitable for an UPDATE.
    """
    overwrite_all = _winning(concept, incoming_source, current_source)
    out: dict = {}
    for col, val in incoming_fields.items():
        if val is None:
            continue
        if concept == "horse" and col == "name" and current_row.get("name"):
            val = _pick_better_horse_name(
                current_row.get("name"), val,
                allow_rename=incoming_source in RENAME_SOURCES,
            )
            if val == current_row.get("name"):
                continue
        if overwrite_all or current_row.get(col) is None:
            out[col] = val
    return out


_NAME_COUNTRY_SUFFIX_RX = re.compile(r"\((?!SE\b|SWE\b)[A-Z]{2,3}\)\s*$")
_NAME_NOISE_SUFFIX_RX = re.compile(r"\s*\((?:SE|SWE)\)\s*$", re.IGNORECASE)


def _name_identity_key(name: str | None) -> str:
    """Decoration-free key used to tell a rename from a re-dressed name."""
    # Local import: core.identity imports this module at module level, so a
    # top-level import here would be circular.
    from core.identity import normalize_name

    key = normalize_name(name)
    # Sources disagree on how to fold Nordic vowels ("Gjärla" vs "Gjaerla").
    # Folding both ways round keeps that out of the rename decision.
    for digraph, plain in (("AE", "A"), ("OE", "O"), ("UE", "U")):
        key = key.replace(digraph, plain)
    return key


def _is_name_variant(current: str, incoming: str) -> bool:
    """True when one name is just the other plus extra trailing tokens.

    ST appends owner initials that other feeds drop ("Keep Going F.R." ->
    "Keep Going", "Crazy Muscles S." -> "Crazy Muscles"). Those initials
    distinguish actually-different horses, so shedding them is a downgrade
    rather than a rename.
    """
    current_tokens = _name_identity_key(current).split()
    incoming_tokens = _name_identity_key(incoming).split()
    if not current_tokens or not incoming_tokens:
        return False
    shorter, longer = sorted((current_tokens, incoming_tokens), key=len)
    return longer[:len(shorter)] == shorter


def _pick_better_horse_name(current_name: str | None, incoming_name: str | None,
                            *, allow_rename: bool = True) -> str | None:
    """Keep the richer display name during source refreshes.

    Two different questions hide behind this one field:

      * Same name, different dress — LeTrot/ATG often refresh with all-caps
        names without a country suffix. If a row already has a suffix such as
        "(FR)" or a mixed-case version, don't let a refresh degrade it.
      * A genuine rename — horses are routinely renamed before their first
        start, and the old logic could not tell that from a re-dressed name,
        so it kept the stale name even when the registry itself reported the
        new one. A rename now wins, subject to `allow_rename` (registry
        sources only, see RENAME_SOURCES) and the caller's priority gate.
    """
    if not current_name:
        return incoming_name or None
    if not incoming_name:
        return current_name

    # Strip noise suffixes (SE/SWE — SE is default, suffix is clutter)
    # so they can be replaced by a clean version naturally.
    current_name = _NAME_NOISE_SUFFIX_RX.sub("", current_name).rstrip()
    incoming_name = _NAME_NOISE_SUFFIX_RX.sub("", incoming_name).rstrip()

    current_has_suffix = bool(_NAME_COUNTRY_SUFFIX_RX.search(current_name))
    incoming_has_suffix = bool(_NAME_COUNTRY_SUFFIX_RX.search(incoming_name))
    current_is_caps = current_name.upper() == current_name and any(ch.isalpha() for ch in current_name)
    incoming_is_caps = incoming_name.upper() == incoming_name and any(ch.isalpha() for ch in incoming_name)

    if _name_identity_key(current_name) != _name_identity_key(incoming_name):
        if not allow_rename:
            return current_name
        if _is_name_variant(current_name, incoming_name):
            return current_name
        if incoming_is_caps and not current_is_caps:
            return current_name
        if current_has_suffix and not incoming_has_suffix:
            # A rename doesn't change where the horse was born, and the
            # frontend reads the suffix to pick a flag.
            code = _NAME_COUNTRY_SUFFIX_RX.search(current_name).group(0).strip()
            return f"{incoming_name} {code}"
        return incoming_name

    if incoming_has_suffix and not current_has_suffix and not current_is_caps and incoming_is_caps:
        code = _NAME_COUNTRY_SUFFIX_RX.search(incoming_name).group(0).strip()
        return f"{current_name} {code}"
    if current_has_suffix and not incoming_has_suffix:
        return current_name
    if incoming_has_suffix and not current_has_suffix:
        return incoming_name

    if current_is_caps and not incoming_is_caps:
        return incoming_name
    # Same name and same case quality: let ST's rename star back in if an
    # earlier source recorded the new name without it.
    if "*" in incoming_name and "*" not in current_name:
        return incoming_name
    return current_name


# ---------------------------------------------------------------------------
# Generic upsert by (source_id_column, source_id) lookup
# ---------------------------------------------------------------------------

def _fetch_row(cur, table: str, where_col: str, where_val: Any) -> dict | None:
    cur.execute(f"SELECT * FROM {table} WHERE {where_col} = %s", (where_val,))
    row = cur.fetchone()
    if not row:
        return None
    cols = [d.name for d in cur.description]
    return dict(zip(cols, row))


def _redirect_to(cur, entity: str, source: str, source_key: Any) -> int | None:
    """Look up a merge redirect (identity_redirect) for a direct upsert.

    Mirrors core.identity._resolve_redirect but kept dependency-free here so
    importers that call upsert_* directly (ST backfill, HVT) also honour
    merges. Matches the key under its own source or the generic 'synth'.
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


def _fetch_row_first_match(
    cur, table: str, candidates: Iterable[tuple[str, Any]]
) -> dict | None:
    """Try each (column, value) candidate in order until one finds a row."""
    for col, val in candidates:
        if val is None:
            continue
        row = _fetch_row(cur, table, col, val)
        if row:
            return row
    return None


def _do_update(
    cur,
    table: str,
    pk_col: str,
    pk_val: Any,
    cols_values: dict,
    source: str,
    new_primary_source: str | None,
    source_data: dict,
) -> None:
    """Apply an UPDATE that bumps last_updated_at + source_data."""
    cols_values = dict(cols_values)
    cols_values["source_data"] = Json(source_data)
    cols_values["last_updated_at"] = "NOW()"  # sentinel — substituted below
    if new_primary_source is not None:
        cols_values["primary_source"] = new_primary_source

    set_parts = []
    params: list = []
    for col, val in cols_values.items():
        if val == "NOW()" and col == "last_updated_at":
            set_parts.append(f"{col} = NOW()")
        else:
            set_parts.append(f"{col} = %s")
            params.append(val)
    params.append(pk_val)
    sql = f"UPDATE {table} SET {', '.join(set_parts)} WHERE {pk_col} = %s"
    cur.execute(sql, params)


def _do_insert(
    cur,
    table: str,
    pk_col: str,
    cols_values: dict,
    source: str,
    source_data: dict,
) -> int:
    """INSERT a new row. Returns the new pk."""
    cols_values = dict(cols_values)
    cols_values["primary_source"] = source
    cols_values["source_data"] = Json(source_data)
    cols = list(cols_values.keys())
    placeholders = ", ".join(["%s"] * len(cols))
    sql = (
        f"INSERT INTO {table} ({', '.join(cols)}) "
        f"VALUES ({placeholders}) RETURNING {pk_col}"
    )
    cur.execute(sql, [cols_values[c] for c in cols])
    return cur.fetchone()[0]


# ---------------------------------------------------------------------------
# Public: upsert helpers per concept
# ---------------------------------------------------------------------------

def find_horse_by_strong_ids(
    cur,
    source: str,
    source_id: Any,
    *,
    registration_number: str | None = None,
    ueln_number: str | None = None,
) -> dict | None:
    """Strong-ID horse lookup shared by upsert_horse and resolve_horse.

    Tries, in order: the source's own id column, registration_number,
    ueln_number, and the SE id-equivalence fallbacks (ATG horse.id ==
    TravSport st_id for SE-registered horses, in both directions).
    Returns the full horse row dict or None.
    """
    source_id_col = f"{source}_id"
    lookup_source_id = str(source_id) if source_id_col in {
        "atg_id", "usta_id", "letrot_id", "kmtid_id", "hvt_id", "breedly_id"
    } else source_id
    candidates: list[tuple[str, Any]] = [(source_id_col, lookup_source_id)]
    if registration_number:
        candidates.append(("registration_number", registration_number))
    if ueln_number:
        candidates.append(("ueln_number", ueln_number))
    if source == "atg" and isinstance(source_id, int) and source_id > 0:
        candidates.append(("st_id", source_id))
    # Symmetric to the above: the NATIVE ST scraper upserts with the TravSport
    # id as `st_id`, but SE horses were usually created first from ATG and only
    # carry `atg_id` (== the same integer, as a string). Without this fallback,
    # resolve_horse(source="st", source_id=815348) on a horse that has
    # atg_id='815348' but st_id IS NULL would INSERT a duplicate instead of
    # attaching st_id to the existing row (this is exactly the Giovaz case).
    if source == "st" and isinstance(source_id, int) and source_id > 0:
        candidates.append(("atg_id", str(source_id)))
    return _fetch_row_first_match(cur, "horse", candidates)


def upsert_horse(
    cur,
    source: str,
    source_id: Any,
    canonical_fields: dict,
    *,
    raw_payload: dict | None = None,
    registration_number: str | None = None,
    ueln_number: str | None = None,
) -> int:
    """Upsert a horse from `source`. Returns canonical horse_id.

    `canonical_fields` keys must match horse table columns.
    `raw_payload` (optional) is stored under source_data[source].

    Cross-source id quirk: for Swedish horses, ATG's `horse.id` and
    TravSport's `st_id` are the same integer (ATG inherits TravSport
    ids for SE-registered horses). When upserting from ATG with an int
    source_id, also try matching by st_id as a fallback.
    """
    source_id_col = f"{source}_id"

    lookup_source_id = str(source_id) if source_id_col in {
        "atg_id", "usta_id", "letrot_id", "kmtid_id", "hvt_id", "breedly_id"
    } else source_id

    existing = find_horse_by_strong_ids(
        cur, source, source_id,
        registration_number=registration_number,
        ueln_number=ueln_number,
    )

    # Merge redirect: route a merged-away source id to its canonical keeper.
    via_redirect = False
    if not existing:
        red_id = _redirect_to(cur, "horse", source, lookup_source_id)
        if red_id is not None:
            existing = _fetch_row(cur, "horse", "horse_id", red_id)
            via_redirect = existing is not None

    fields = dict(canonical_fields)
    if registration_number is not None:
        fields.setdefault("registration_number", registration_number)
    if ueln_number is not None:
        fields.setdefault("ueln_number", ueln_number)

    if existing:
        new_source_data = _merge_source_data(existing.get("source_data"), source, raw_payload)
        upd = _build_canonical_update(
            "horse",
            source,
            existing.get("primary_source"),
            fields,
            existing,
        )
        if "name" in upd:
            new_source_data = _record_name_history(
                new_source_data, existing.get("name"), upd["name"], source,
            )
        # Always set the source_id column we just learned, even if the row
        # was matched via registration_number / ueln (i.e. we now have a
        # cross-link we didn't have before).
        stored_source_id = str(source_id) if source_id_col in {
            "atg_id", "usta_id", "letrot_id", "kmtid_id", "hvt_id", "breedly_id"
        } else source_id
        # When matched via a redirect, only fill the source-id column if the
        # keeper has none — never clobber the keeper's own id with the
        # merged-away duplicate's id.
        if existing.get(source_id_col) != stored_source_id and (
            not via_redirect or existing.get(source_id_col) in (None, "")
        ):
            upd[source_id_col] = stored_source_id

        new_primary = source if _winning("horse", source, existing.get("primary_source")) else None
        _do_update(
            cur, "horse", "horse_id", existing["horse_id"],
            upd, source, new_primary, new_source_data,
        )
        return existing["horse_id"]

    # INSERT
    fields[source_id_col] = str(source_id) if source_id_col in {
        "atg_id", "usta_id", "letrot_id", "kmtid_id", "hvt_id", "breedly_id"
    } else source_id
    return _do_insert(
        cur, "horse", "horse_id",
        fields, source,
        _merge_source_data(None, source, raw_payload),
    )


def upsert_person(
    cur,
    source: str,
    source_id: Any,
    canonical_fields: dict,
    *,
    raw_payload: dict | None = None,
    role_flags: dict[str, bool] | None = None,
) -> int:
    """Upsert a person from `source`. Returns canonical person_id.

    `role_flags` like {'is_driver': True, 'is_trainer': True} OR them in
    rather than overwriting (a trainer that we later see as a driver should
    end up with both flags True).
    """
    source_id_col = f"{source}_id"
    lookup_source_id = str(source_id) if source_id_col in {
        "atg_id", "usta_id", "letrot_id", "hvt_id"
    } else source_id
    candidates: list[tuple[str, Any]] = [(source_id_col, lookup_source_id)]
    # Same SE id-equivalence quirk as horses: ATG's person.id == TravSport st_id.
    if source == "atg" and isinstance(source_id, int) and source_id > 0:
        candidates.append(("st_id", source_id))
    # Symmetric fallback for the native ST scraper (see upsert_horse).
    if source == "st" and isinstance(source_id, int) and source_id > 0:
        candidates.append(("atg_id", str(source_id)))
    existing = _fetch_row_first_match(cur, "person", candidates)

    via_redirect = False
    if not existing:
        red_id = _redirect_to(cur, "person", source, lookup_source_id)
        if red_id is not None:
            existing = _fetch_row(cur, "person", "person_id", red_id)
            via_redirect = existing is not None

    fields = dict(canonical_fields)

    if existing:
        new_source_data = _merge_source_data(existing.get("source_data"), source, raw_payload)
        upd = _build_canonical_update(
            "person", source, existing.get("primary_source"), fields, existing
        )
        stored_source_id = str(source_id) if source_id_col in {
            "atg_id", "usta_id", "letrot_id", "hvt_id"
        } else source_id
        if existing.get(source_id_col) != stored_source_id and (
            not via_redirect or existing.get(source_id_col) in (None, "")
        ):
            upd[source_id_col] = stored_source_id
        # OR-merge role flags
        if role_flags:
            for flag, val in role_flags.items():
                if val and not existing.get(flag):
                    upd[flag] = True
        new_primary = source if _winning("person", source, existing.get("primary_source")) else None
        _do_update(
            cur, "person", "person_id", existing["person_id"],
            upd, source, new_primary, new_source_data,
        )
        return existing["person_id"]

    fields[source_id_col] = str(source_id) if source_id_col in {
        "atg_id", "usta_id", "letrot_id", "hvt_id"
    } else source_id
    if role_flags:
        for flag, val in role_flags.items():
            fields[flag] = bool(val)
    return _do_insert(
        cur, "person", "person_id", fields, source,
        _merge_source_data(None, source, raw_payload),
    )


def _normalize_track_name(name: str | None) -> str | None:
    """Title-case track names that arrive ALL CAPS or all lowercase.

    Sources like ATG and Le Trot send track names as 'ENGHIEN' / 'GELSENKIRCHEN'.
    Mixed-case names ('Bjerke Travbane') are kept as-is — we don't want to
    mangle proper-noun casing the source already got right.
    """
    if not name:
        return name
    s = name.strip()
    if not s:
        return s
    if s == s.upper() or s == s.lower():
        return s.title()
    return s


def _squash_track_name(name: str | None) -> str:
    """Lowercase and strip everything but letters/digits so punctuation and
    spacing variants compare equal ('Cagnes-Sur-Mer' == 'Cagnes Sur Mer')."""
    import re as _re

    return _re.sub(r"[^0-9a-zåäöæøüéèêàç]", "", (name or "").lower())


def _track_names_agree(a: str | None, b: str | None) -> bool:
    """Loose same-venue check: squashed-equal or one contains the other
    ('Orkla' vs 'Orkla Arena'). Missing names count as agreement (nothing
    to contradict)."""
    sa, sb = _squash_track_name(a), _squash_track_name(b)
    if not sa or not sb:
        return True
    return sa == sb or sa in sb or sb in sa


# Sources whose numeric track ids are NOT stable venue identifiers. ATG
# reuses a per-country pool of ids for guest/foreign tracks: id 54 was
# Odense on 2026-05-08 and Århus on 2026-06-20 (verified against ATG's own
# API). An id match therefore only counts if the payload's track NAME also
# agrees with the matched row.
UNSTABLE_TRACK_ID_SOURCES = {"atg"}


def upsert_track(
    cur,
    source: str,
    source_id: Any,
    canonical_fields: dict,
    *,
    raw_payload: dict | None = None,
    source_id_column: str | None = None,
) -> int:
    """Upsert a track with cross-source dedup.

    Match order:
      1. (source, source_id) — same source seeing a track it already wrote.
         For UNSTABLE_TRACK_ID_SOURCES (ATG's rotating foreign slots) the
         match is only accepted when the track name also agrees.
      2. (lower(name), country) — different source / different source_id but
         the same physical track. Attaches the new source_id to the canonical
         row so future imports converge on it. Falls back to a squashed-name
         comparison so punctuation variants don't fork rows.
      3. New row.

    When the existing row already has a *different* source_id from this same
    source (e.g. ATG re-registered Enghien under a second atg_track_id), the
    new id is appended to source_data.<source>.aliases instead of being
    inserted as a duplicate row.
    """
    if source_id_column is None:
        if source == "st":
            source_id_column = "st_code"
        elif source == "atg":
            source_id_column = "atg_track_id"
        else:
            source_id_column = f"{source}_id"

    fields = dict(canonical_fields)
    if fields.get("name"):
        fields["name"] = _normalize_track_name(fields["name"])

    name = fields.get("name")
    country = fields.get("country")

    # 1. Lookup by source id.
    existing = _fetch_row(cur, "track", source_id_column, source_id)
    id_match_rejected = False
    if (
        existing
        and source in UNSTABLE_TRACK_ID_SOURCES
        and not _track_names_agree(existing.get("name"), name)
    ):
        # Rotating slot id currently parked on a different venue — the name
        # is the truth. Resolve by name+country below instead.
        existing = None
        id_match_rejected = True

    # 2. Cross-source fallback: same name + same country = same physical track.
    if not existing and name:
        if country:
            # Strict: exact country match.
            cur.execute(
                "SELECT * FROM track "
                " WHERE country = %s AND lower(name) = lower(%s) "
                " LIMIT 1",
                (country, name),
            )
            row = cur.fetchone()
            # Loose: name match against rows where country is unknown.
            # ST imports often arrive country=NULL, so HVT's 'Gelsenkirchen DE'
            # should still find the existing 'GELSENKIRCHEN ?' and propagate
            # the country onto it.
            if not row:
                cur.execute(
                    "SELECT * FROM track "
                    " WHERE country IS NULL AND lower(name) = lower(%s) "
                    " LIMIT 1",
                    (name,),
                )
                row = cur.fetchone()
        else:
            # Incoming has no country — match against same-name rows
            # regardless (best we can do).
            cur.execute(
                "SELECT * FROM track "
                " WHERE lower(name) = lower(%s) "
                " ORDER BY country NULLS LAST LIMIT 1",
                (name,),
            )
            row = cur.fetchone()
        # Squashed fallback: punctuation/spacing variants of the same venue
        # ('Cagnes-Sur-Mer' vs 'Cagnes Sur Mer') must not fork a new row.
        if not row:
            squash_sql = (
                "lower(regexp_replace(name, '[^0-9A-Za-z"
                "\u00e5\u00e4\u00f6\u00c5\u00c4\u00d6\u00e6\u00c6\u00f8\u00d8"
                "\u00fc\u00dc\u00e9\u00c9\u00e8\u00c8\u00ea\u00ca\u00e7\u00c7]', '', 'g'))"
            )
            squashed = _squash_track_name(name)
            if squashed:
                if country:
                    cur.execute(
                        f"SELECT * FROM track "
                        f" WHERE (country = %s OR country IS NULL) "
                        f"   AND {squash_sql} = %s "
                        f" ORDER BY (country IS NULL), track_id LIMIT 1",
                        (country, squashed),
                    )
                else:
                    cur.execute(
                        f"SELECT * FROM track "
                        f" WHERE {squash_sql} = %s "
                        f" ORDER BY country NULLS LAST, track_id LIMIT 1",
                        (squashed,),
                    )
                row = cur.fetchone()
        if row:
            cols = [d.name for d in cur.description]
            existing = dict(zip(cols, row))

    if existing:
        new_source_data = _merge_source_data(
            existing.get("source_data"), source, raw_payload
        )
        upd = _build_canonical_update(
            "track", source, existing.get("primary_source"), fields, existing
        )

        # If the existing name is ALL CAPS but we just got a nicer-cased
        # version, force-update regardless of priority — uppercase is
        # never the right canonical form.
        ex_name = existing.get("name") or ""
        in_name = fields.get("name") or ""
        if (
            ex_name and in_name
            and ex_name == ex_name.upper()
            and in_name != in_name.upper()
        ):
            upd["name"] = in_name

        # Attach this source_id to the canonical row. Never stamp an id that
        # is still parked on another row (unique index) or that we just
        # rejected as a rotating slot — stash those as aliases instead.
        cur_val = existing.get(source_id_column)
        id_taken_elsewhere = False
        if cur_val is None and source_id is not None and not id_match_rejected:
            cur.execute(
                f"SELECT 1 FROM track WHERE {source_id_column} = %s "
                f"  AND track_id != %s LIMIT 1",
                (source_id, existing["track_id"]),
            )
            id_taken_elsewhere = cur.fetchone() is not None
        if cur_val is None and source_id is not None and not id_match_rejected \
                and not id_taken_elsewhere:
            upd[source_id_column] = source_id
        elif cur_val is None and source_id is not None:
            src_block = dict(new_source_data.get(source) or {})
            aliases = list(src_block.get("aliases") or [])
            if source_id not in aliases:
                aliases.append(source_id)
            src_block["aliases"] = aliases
            new_source_data[source] = src_block
        elif cur_val is not None and cur_val != source_id:
            # Same source, multiple ids for the same physical track.
            # Stash the extra id under source_data.<source>.aliases so we
            # don't lose the cross-link for future imports.
            src_block = dict(new_source_data.get(source) or {})
            aliases = list(src_block.get("aliases") or [])
            if source_id not in aliases:
                aliases.append(source_id)
            src_block["aliases"] = aliases
            new_source_data[source] = src_block

        new_primary = (
            source
            if _winning("track", source, existing.get("primary_source"))
            else None
        )
        _do_update(
            cur, "track", "track_id", existing["track_id"],
            upd, source, new_primary, new_source_data,
        )
        return existing["track_id"]

    # 3. New row. A rejected rotating-slot id is still held by another track
    # row, so it must go into source_data aliases rather than the unique
    # id column.
    source_data = _merge_source_data(None, source, raw_payload)
    if id_match_rejected:
        src_block = dict(source_data.get(source) or {})
        aliases = list(src_block.get("aliases") or [])
        if source_id not in aliases:
            aliases.append(source_id)
        src_block["aliases"] = aliases
        source_data[source] = src_block
    else:
        fields[source_id_column] = source_id
    return _do_insert(
        cur, "track", "track_id", fields, source,
        source_data,
    )


def upsert_race(
    cur,
    source: str,
    source_id: Any,
    canonical_fields: dict,
    *,
    raw_payload: dict | None = None,
    source_id_column: str | None = None,
) -> int:
    """Upsert a race. `source_id_column` defaults to '<source>_race_id'.

    Cross-source fallback: when no row matches the source id, try
    matching by (track_id, race_date, race_number) if those are all
    present in `canonical_fields`. This lets a second source attach
    its id to a race already imported by another source.
    """
    if source_id_column is None:
        source_id_column = f"{source}_race_id"

    existing = _fetch_row(cur, "race", source_id_column, source_id)
    fields = dict(canonical_fields)

    if not existing:
        track_id = fields.get("track_id")
        race_date = fields.get("race_date")
        race_number = fields.get("race_number")
        if track_id is not None and race_date is not None and race_number is not None:
            cur.execute(
                "SELECT * FROM race "
                " WHERE track_id = %s AND race_date = %s AND race_number = %s "
                " LIMIT 1",
                (track_id, race_date, race_number),
            )
            row = cur.fetchone()
            if row:
                cols = [d.name for d in cur.description]
                candidate = dict(zip(cols, row))
                # Distance sanity check: foreign aggregator races (esp. st's
                # "FRANKRIKE" bucket) sometimes share (track_id, date, number)
                # with a different physical race. If both rows declare a
                # distance and they disagree by more than 50 m, reject the
                # match and let this row be inserted separately.
                new_dist = fields.get("distance")
                old_dist = candidate.get("distance")
                if (
                    new_dist is not None
                    and old_dist is not None
                    and abs(int(new_dist) - int(old_dist)) > 50
                ):
                    pass  # skip — distances don't match
                else:
                    existing = candidate

    if existing:
        new_source_data = _merge_source_data(existing.get("source_data"), source, raw_payload)
        upd = _build_canonical_update(
            "race", source, existing.get("primary_source"), fields, existing
        )
        # Attach this source's id to the existing canonical row (we may
        # have matched via track+date+number instead of source id).
        if existing.get(source_id_column) is None:
            upd[source_id_column] = source_id
        new_primary = source if _winning("race", source, existing.get("primary_source")) else None
        _do_update(
            cur, "race", "race_id", existing["race_id"],
            upd, source, new_primary, new_source_data,
        )
        return existing["race_id"]

    fields[source_id_column] = source_id
    return _do_insert(
        cur, "race", "race_id", fields, source,
        _merge_source_data(None, source, raw_payload),
    )


def upsert_entry(
    cur,
    source: str,
    race_id: int,
    horse_id: int,
    canonical_fields: dict,
    *,
    raw_payload: dict | None = None,
) -> int:
    """Upsert one (race_id, horse_id) entry. Returns entry_id."""
    cur.execute(
        "SELECT * FROM entry WHERE race_id = %s AND horse_id = %s",
        (race_id, horse_id),
    )
    row = cur.fetchone()
    if row:
        cols = [d.name for d in cur.description]
        existing = dict(zip(cols, row))
        new_source_data = _merge_source_data(existing.get("source_data"), source, raw_payload)
        upd = _build_canonical_update(
            "entry", source, existing.get("primary_source"), canonical_fields, existing
        )
        new_primary = source if _winning("entry", source, existing.get("primary_source")) else None
        _do_update(
            cur, "entry", "entry_id", existing["entry_id"],
            upd, source, new_primary, new_source_data,
        )
        return existing["entry_id"]

    fields = dict(canonical_fields)
    fields["race_id"] = race_id
    fields["horse_id"] = horse_id
    return _do_insert(
        cur, "entry", "entry_id", fields, source,
        _merge_source_data(None, source, raw_payload),
    )
