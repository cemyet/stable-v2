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
        if overwrite_all or current_row.get(col) is None:
            out[col] = val
    return out


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
    """
    source_id_col = f"{source}_id"

    candidates: list[tuple[str, Any]] = [(source_id_col, source_id)]
    if registration_number:
        candidates.append(("registration_number", registration_number))
    if ueln_number:
        candidates.append(("ueln_number", ueln_number))

    existing = _fetch_row_first_match(cur, "horse", candidates)

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
        # Always set the source_id column we just learned, even if the row
        # was matched via registration_number / ueln (i.e. we now have a
        # cross-link we didn't have before).
        if existing.get(source_id_col) != source_id:
            upd[source_id_col] = source_id

        new_primary = source if _winning("horse", source, existing.get("primary_source")) else None
        _do_update(
            cur, "horse", "horse_id", existing["horse_id"],
            upd, source, new_primary, new_source_data,
        )
        return existing["horse_id"]

    # INSERT
    fields[source_id_col] = source_id
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
    existing = _fetch_row(cur, "person", source_id_col, source_id)

    fields = dict(canonical_fields)

    if existing:
        new_source_data = _merge_source_data(existing.get("source_data"), source, raw_payload)
        upd = _build_canonical_update(
            "person", source, existing.get("primary_source"), fields, existing
        )
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

    fields[source_id_col] = source_id
    if role_flags:
        for flag, val in role_flags.items():
            fields[flag] = bool(val)
    return _do_insert(
        cur, "person", "person_id", fields, source,
        _merge_source_data(None, source, raw_payload),
    )


def upsert_track(
    cur,
    source: str,
    source_id: Any,
    canonical_fields: dict,
    *,
    raw_payload: dict | None = None,
    source_id_column: str | None = None,
) -> int:
    """Upsert a track. `source_id_column` lets you override e.g. 'st_code' or 'atg_track_id'."""
    if source_id_column is None:
        # Default mapping: st->st_code, atg->atg_track_id, others -> <source>_id
        if source == "st":
            source_id_column = "st_code"
        elif source == "atg":
            source_id_column = "atg_track_id"
        else:
            source_id_column = f"{source}_id"

    existing = _fetch_row(cur, "track", source_id_column, source_id)
    fields = dict(canonical_fields)

    if existing:
        new_source_data = _merge_source_data(existing.get("source_data"), source, raw_payload)
        upd = _build_canonical_update(
            "track", source, existing.get("primary_source"), fields, existing
        )
        new_primary = source if _winning("track", source, existing.get("primary_source")) else None
        _do_update(
            cur, "track", "track_id", existing["track_id"],
            upd, source, new_primary, new_source_data,
        )
        return existing["track_id"]

    fields[source_id_column] = source_id
    return _do_insert(
        cur, "track", "track_id", fields, source,
        _merge_source_data(None, source, raw_payload),
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
    """Upsert a race. `source_id_column` defaults to '<source>_race_id'."""
    if source_id_column is None:
        source_id_column = f"{source}_race_id"

    existing = _fetch_row(cur, "race", source_id_column, source_id)
    fields = dict(canonical_fields)

    if existing:
        new_source_data = _merge_source_data(existing.get("source_data"), source, raw_payload)
        upd = _build_canonical_update(
            "race", source, existing.get("primary_source"), fields, existing
        )
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
