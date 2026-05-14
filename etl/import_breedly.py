"""Import Breedly horse + pedigree data into stable_v2.

Breedly is *pedigree enrichment*. Each horse page returns the focal horse
plus 5-7 generations of ancestors — every one with a breedly_id and (when
known) a TravSport st_id. That's a free cross-source link for the whole
pedigree tree.

Strategy:
    Pass 1: For every Horse in the Apollo cache, upsert into v2.horse by
            breedly_id. If the row also carries an `stId`, pass it as a
            hint so cross-source matching attaches it to the right
            existing v2 row (same st_id we built v2 from).
    Pass 2: For every Horse, resolve its `fatherId`/`motherId` to v2
            horse_id and write `sire_id`/`dam_id`.

Public entry-points:

    import_horse(conn, slug)     -- scrape + upsert + pedigree-resolve
    import_by_breedly_id(conn, breedly_id)
                                 -- look up the slug from db, then import
    discover_for_unmatched_horses(conn, limit=100)
                                 -- scan v2.horse rows without a breedly_id,
                                    search Breedly by name+year, import the
                                    unique match.
"""

from __future__ import annotations

import logging
import time
from typing import Iterable

import httpx
from psycopg2.extras import Json

from core.db import buffer_prune
from etl.matching import upsert_horse
from scrapers.breedly import (
    make_client,
    fetch_horse_by_slug,
    search_horses,
)

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Breedly Horse → v2 horse upsert
# ---------------------------------------------------------------------------

_GENDER_BREEDLY_TO_CODE = {
    "s": "H",   # stallion -> Hingst
    "m": "S",   # mare      -> Sto
    "g": "V",   # gelding   -> Valack
}


def _records_text_to_record(s: str | None) -> str | None:
    """horseDisplayRecordAuto / horseBestRecord come as e.g. '1.11,2ak' — store as-is."""
    if not s:
        return None
    return s.strip() or None


def _parse_first_int_in(s: str | None) -> int | None:
    """'35 (8-3-4)' -> 35   (used for starts/wins/placed parsing)."""
    if not s:
        return None
    import re
    m = re.match(r"\s*(\d+)", s)
    return int(m[1]) if m else None


def _placements_breakdown(s: str | None) -> tuple[int | None, int | None, int | None, int | None]:
    """'35 (8-3-4)' -> (35, 8, 3, 4) = (starts, wins, seconds, thirds)."""
    if not s:
        return None, None, None, None
    import re
    m = re.match(r"\s*(\d+)\s*\((\d+)-(\d+)-(\d+)\)", s)
    if not m:
        return _parse_first_int_in(s), None, None, None
    return int(m[1]), int(m[2]), int(m[3]), int(m[4])


def _upsert_breedly_horse(cur, h: dict) -> int:
    """UPSERT one breedly Horse dict and return the v2 horse_id.

    Cross-source matching priority:
        1. Existing row with breedly_id == this horseId  -> update.
        2. Existing row with st_id   == this stId        -> attach breedly_id, update.
        3. Otherwise INSERT a new canonical row.
    """
    breedly_id = str(h["horseId"])
    st_id = h.get("stId")
    try:
        st_id_int = int(st_id) if st_id is not None else None
    except (ValueError, TypeError):
        st_id_int = None

    # Decide which row (if any) to merge into BEFORE calling upsert_horse,
    # so we never insert a duplicate. If both ids point at different rows,
    # prefer the st-rooted one (it's older / has more attached history).
    existing_id: int | None = None

    cur.execute("SELECT horse_id FROM horse WHERE breedly_id = %s", (breedly_id,))
    row = cur.fetchone()
    if row:
        existing_id = row[0]

    if st_id_int is not None:
        cur.execute("SELECT horse_id FROM horse WHERE st_id = %s", (st_id_int,))
        st_row = cur.fetchone()
        if st_row:
            if existing_id and existing_id != st_row[0]:
                # Two existing rows for the same horse — attach breedly to
                # st-rooted row and soft-mark the duplicate for later merge.
                cur.execute(
                    """
                    UPDATE horse
                       SET breedly_id = NULL,
                           source_data = COALESCE(source_data, '{}'::jsonb)
                                      || jsonb_build_object('_merge_target', %s)
                     WHERE horse_id = %s
                    """,
                    (st_row[0], existing_id),
                )
            existing_id = st_row[0]

    starts, wins, seconds, thirds = _placements_breakdown(h.get("horseDisplayPlacements"))

    canonical = {
        "name":          h.get("horseDisplayName") or h.get("horseName"),
        "gender_code":   _GENDER_BREEDLY_TO_CODE.get(h.get("gender")),
        "bred_country":  h.get("bornCountry"),
        "birth_country": h.get("bornCountry"),
        "registration_country": h.get("bornCountry"),
        "scraped_record": _records_text_to_record(h.get("horseBestRecord")
                                                  or h.get("horseDisplayRecordAuto")),
        "scraped_starts": starts,
        "scraped_wins":   wins,
    }

    raw = {
        "horseId":      h.get("horseId"),
        "slug":         h.get("slug"),
        "stId":         st_id,
        "source":       h.get("source"),
        "type":         h.get("type"),
        "gender_raw":   h.get("gender"),
        "bornYear":     h.get("bornYear"),
        "inbreeding":   h.get("inbreeding"),
        "blup":         h.get("blup"),
        "blupAccuracy": h.get("blupAccuracy"),
        "breederName":  h.get("breederName"),
        "horseDisplayPlacements":   h.get("horseDisplayPlacements"),
        "horseDisplayTotalEarnings": h.get("horseDisplayTotalEarnings"),
        "fatherId":     h.get("fatherId"),
        "fatherName":   h.get("fatherName"),
        "motherId":     h.get("motherId"),
        "motherName":   h.get("motherName"),
        "isAvailableStallion": h.get("isAvailableStallion"),
        "isEliteMare":  h.get("isEliteMare"),
    }

    if existing_id is not None:
        # Update path: stamp breedly_id (and st_id if missing) onto the row
        # we found, then run the canonical merge via upsert_horse.
        cur.execute(
            """
            UPDATE horse
               SET breedly_id = COALESCE(breedly_id, %s),
                   st_id      = COALESCE(st_id,      %s),
                   last_updated_at = NOW()
             WHERE horse_id = %s
            """,
            (breedly_id, st_id_int, existing_id),
        )

    horse_id = upsert_horse(
        cur, "breedly", breedly_id,
        canonical, raw_payload=raw,
    )

    if st_id_int is not None:
        cur.execute(
            "UPDATE horse SET st_id = COALESCE(st_id, %s) WHERE horse_id = %s",
            (st_id_int, horse_id),
        )

    return horse_id


def _resolve_pedigree_links(
    cur, breedly_id_to_v2: dict[str, int], focal_and_ancestors: dict[str, dict]
) -> int:
    """Pass 2: set sire_id / dam_id on every horse we just upserted."""
    n_set = 0
    for bid, h in focal_and_ancestors.items():
        v2_id = breedly_id_to_v2.get(bid)
        if not v2_id:
            continue
        father_v2 = breedly_id_to_v2.get(str(h.get("fatherId"))) if h.get("fatherId") else None
        mother_v2 = breedly_id_to_v2.get(str(h.get("motherId"))) if h.get("motherId") else None
        if father_v2 is None and mother_v2 is None:
            continue
        cur.execute(
            """
            UPDATE horse
               SET sire_id = COALESCE(sire_id, %s),
                   dam_id  = COALESCE(dam_id,  %s),
                   last_updated_at = NOW()
             WHERE horse_id = %s
            """,
            (father_v2, mother_v2, v2_id),
        )
        n_set += 1
    return n_set


# ---------------------------------------------------------------------------
# Public entry-points
# ---------------------------------------------------------------------------

def import_horse(conn, slug: str, *, client: httpx.Client | None = None) -> dict:
    """Scrape + import one Breedly horse (focal + all pedigree ancestors)."""
    own_client = client is None
    if own_client:
        client = make_client()

    summary = {
        "slug":           slug,
        "focal_v2_id":    None,
        "horses_upserted": 0,
        "ancestors":      0,
        "pedigree_links": 0,
    }

    try:
        res = fetch_horse_by_slug(client, slug)
        if not res or not res.get("focal"):
            log.warning("breedly %s: no focal horse", slug)
            return summary

        focal = res["focal"]
        ancestors = res["ancestors"]
        all_horses = {str(focal["horseId"]): focal}
        all_horses.update(ancestors)

        breedly_id_to_v2: dict[str, int] = {}
        with conn.cursor() as cur:
            for bid, h in all_horses.items():
                v2_id = _upsert_breedly_horse(cur, h)
                breedly_id_to_v2[bid] = v2_id
                summary["horses_upserted"] += 1

            # Pass 2: pedigree links
            summary["pedigree_links"] = _resolve_pedigree_links(
                cur, breedly_id_to_v2, all_horses,
            )

        conn.commit()
        summary["focal_v2_id"] = breedly_id_to_v2.get(str(focal["horseId"]))
        summary["ancestors"]   = len(ancestors)
    finally:
        if own_client:
            client.close()
    return summary


def discover_for_unmatched_horses(
    conn, limit: int = 100, *, client: httpx.Client | None = None,
) -> dict:
    """Search Breedly for v2 horses that don't yet have a breedly_id."""
    own_client = client is None
    if own_client:
        client = make_client()

    summary = {"checked": 0, "matched": 0, "ambiguous": 0, "missing": 0, "errors": 0}

    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT horse_id, name, EXTRACT(YEAR FROM date_of_birth)::int AS y,
                       registration_country
                  FROM horse
                 WHERE breedly_id IS NULL
                   AND name IS NOT NULL
                   AND date_of_birth IS NOT NULL
                 ORDER BY scraped_prize_money_kr DESC NULLS LAST
                 LIMIT %s
                """,
                (limit,),
            )
            cands = cur.fetchall()

        for horse_id, name, year, country in cands:
            summary["checked"] += 1
            try:
                hits = search_horses(client, name)
            except Exception as e:
                log.warning("breedly search %r failed: %s", name, e)
                summary["errors"] += 1
                continue

            # Score candidates: name + year (and country if we have it)
            picks = []
            for h in hits:
                slug = h["slug"]
                # The slug usually contains the year; treat as match when it does.
                if str(year) in slug.split("-"):
                    if country and country.lower() in slug.split("-"):
                        picks.append(h)
                    elif not country:
                        picks.append(h)
            if len(picks) == 1:
                imp = import_horse(conn, picks[0]["slug"], client=client)
                if imp.get("focal_v2_id"):
                    summary["matched"] += 1
                else:
                    summary["errors"] += 1
            elif len(picks) > 1:
                summary["ambiguous"] += 1
            else:
                summary["missing"] += 1
            time.sleep(0.05)

        buffer_prune(conn, "breedly")
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
        print("usage: python -m etl.import_breedly horse <slug> | discover [--limit N]")
        return

    conn = get_connection()
    try:
        cmd = sys.argv[1]
        if cmd == "horse":
            print(import_horse(conn, sys.argv[2]))
        elif cmd == "discover":
            limit = 50
            if "--limit" in sys.argv:
                limit = int(sys.argv[sys.argv.index("--limit") + 1])
            print(discover_for_unmatched_horses(conn, limit=limit))
        else:
            raise SystemExit(f"unknown cmd: {cmd!r}")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
