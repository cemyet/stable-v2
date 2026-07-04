"""
Scrape LeTrot horse-identity pages to backfill pedigree (sire / dam),
color, lifetime gains, record, trainer / owner / breeder for every horse
already in our DB with a `letrot_id`.

Approach
--------

LeTrot's course pages (the per-race scrape we already do daily) only
expose the per-row partants info: musique, record, gains. They do NOT
expose sire/dam. The horse-identity page
(`/stats/chevaux/<slug>/<letrot_id>/courses`) does — one HTTP request
per horse.

To stay efficient and avoid millions of redundant requests we:
  1. Deduplicate by `horse.letrot_id`. The same horse races dozens of
     times but we only need ONE identity fetch.
  2. Skip horses already processed. The marker is
     `source_data['letrot']['identity_fetched_at']` (ISO timestamp), so
     re-runs only touch new horses.
  3. Parallelize with a thread pool of ~5 workers (`--workers`). The
     LeTrot site handles this comfortably; we throttle gently so we
     don't get rate-limited.
  4. Batch DB writes (`--commit-every`, default 100).

Linking sire / dam to existing horse rows
-----------------------------------------

When the identity page exposes a `sire_letrot_id`, we resolve the sire
via the canonical horse-identity protocol (`core.identity.resolve_horse`)
— same path the course-page scrape uses. That handles strong-id matches
and creates a minimal placeholder row when nothing exists.

When LeTrot *does not* expose a `letrot_id` for the parent (older
sires/dams that pre-date LeTrot's IDed catalogue), we still write the
text name to `horse.sire_name` / `horse.dam_name` and leave the FK NULL.
The companion `scripts.link_pedigree_by_name` pass then attaches
`sire_id` / `dam_id` to existing canonical rows via normalised-name
lookup (handles the JOSH POWER → OFFSHORE DREAM case where ST already
owns Offshore Dream as horse #43073).

Even *with* a `sire_letrot_id`, we first try a normalised-name lookup
against existing horses so that ST's "Offshore Dream (FR)" gets the
LeTrot id attached rather than orphaning a new row. The match is
constrained to gender_code in {'S', NULL} (sires) / {'M', NULL} (dams)
to avoid false attachments.

Usage
-----

    # Smoke test: one horse
    python -m scripts.scrape_letrot_pedigree --letrot-id ZWt8ZAQBBgEZ \\
        --execute

    # Small sample first
    python -m scripts.scrape_letrot_pedigree --limit 10 --execute

    # Full run (resumable; safe to re-run)
    python -m scripts.scrape_letrot_pedigree --execute

    # Tune throughput
    python -m scripts.scrape_letrot_pedigree --execute --workers 8 \\
        --commit-every 200

    # Re-scrape horses fetched more than N days ago
    python -m scripts.scrape_letrot_pedigree --execute --refresh-days 30

`--dry-run` (default when `--execute` is absent) walks the candidate
list but performs no HTTP and no DB writes, useful for verifying scope.
"""

from __future__ import annotations

import argparse
import logging
import os
import queue
import sys
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from psycopg2.extras import Json  # noqa: E402

from core.db import get_connection  # noqa: E402
from core.identity import normalize_name, resolve_horse  # noqa: E402
from scrapers.letrot import (  # noqa: E402
    make_client,
    fetch_horse_identity,
)

log = logging.getLogger("scrape_letrot_pedigree")


# ---------------------------------------------------------------------------
# Candidate selection
# ---------------------------------------------------------------------------

def _fetch_candidates(conn, *, limit: int | None,
                      refresh_days: int | None,
                      single_letrot_id: str | None) -> list[tuple]:
    """Return [(horse_id, letrot_id, slug)] of horses to scrape."""
    if single_letrot_id:
        sql = """
            SELECT horse_id,
                   letrot_id,
                   COALESCE(source_data->'letrot'->>'slug', 'x') AS slug
              FROM horse
             WHERE letrot_id = %s
        """
        params: tuple = (single_letrot_id,)
    elif refresh_days is not None:
        # Re-fetch horses whose marker is older than the cutoff (or absent).
        cutoff = datetime.utcnow() - timedelta(days=refresh_days)
        sql = """
            SELECT horse_id,
                   letrot_id,
                   COALESCE(source_data->'letrot'->>'slug', 'x') AS slug
              FROM horse
             WHERE letrot_id IS NOT NULL
               AND ( source_data->'letrot'->>'identity_fetched_at' IS NULL
                  OR (source_data->'letrot'->>'identity_fetched_at')::timestamp < %s )
             ORDER BY horse_id
        """
        params = (cutoff,)
    else:
        sql = """
            SELECT horse_id,
                   letrot_id,
                   COALESCE(source_data->'letrot'->>'slug', 'x') AS slug
              FROM horse
             WHERE letrot_id IS NOT NULL
               AND NOT (source_data->'letrot' ? 'identity_fetched_at')
             ORDER BY horse_id
        """
        params = ()

    if limit and not single_letrot_id:
        sql += f" LIMIT {int(limit)}"

    with conn.cursor() as cur:
        cur.execute(sql, params)
        rows = cur.fetchall()
    return [(r[0], r[1], r[2] or "x") for r in rows]


# ---------------------------------------------------------------------------
# Per-horse identity processing
# ---------------------------------------------------------------------------

_FRENCH_SEX_WORD_TO_HVS = {
    "male": "H", "mâle": "H", "entier": "H",   # stallion
    "hongre": "V",                              # gelding
    "femelle": "S",                             # mare
}


def _french_sex_word_to_hvs(text: str | None) -> str | None:
    """LeTrot identity-page 'Sexe' (free text / single letter) → H/V/S."""
    if not text:
        return None
    t = str(text).strip().lower()
    if t in _FRENCH_SEX_WORD_TO_HVS:
        return _FRENCH_SEX_WORD_TO_HVS[t]
    # Single French letters H(ongre)/F(emelle)/M(âle).
    return {"h": "V", "f": "S", "m": "H"}.get(t[:1])


def _resolve_parent(cur, *, name: str | None, letrot_id: str | None,
                    expect_gender: str | None) -> int | None:
    """Return canonical horse_id for a sire (expect_gender='H') or dam
    ('S'), attaching the letrot_id to an existing row when possible.

    Lookup order:
      1. If `letrot_id` already known → return that horse_id.
      2. Unique normalised-name match constrained by gender_code
         (or NULL) → attach the letrot_id to that row and return it.
      3. Fall back to `resolve_horse` which will either match by some
         other strong id or INSERT a fresh minimal letrot-only row.
    """
    if not name:
        return None
    name = name.strip()
    if not name:
        return None

    # 1. letrot_id already in DB.
    if letrot_id:
        cur.execute("SELECT horse_id FROM horse WHERE letrot_id = %s",
                    (letrot_id,))
        row = cur.fetchone()
        if row:
            return row[0]

    # 2a. Try strict normalised-name match with consistent gender.
    cur.execute(
        """
        SELECT horse_id, letrot_id
          FROM horse
         WHERE v2_normalize_name(name) = v2_normalize_name(%s)
           AND v2_normalize_name(name) <> ''
           AND (gender_code = %s OR gender_code IS NULL OR gender_code = '')
         LIMIT 3
        """,
        (name, expect_gender),
    )
    rows = cur.fetchall()

    # 2b. Fallback: drop the gender filter entirely. Legacy rows in our DB
    #     carry the French codes H / F for many old foreign sires/dams
    #     (e.g. Offshore Dream → gender_code='H' = French hongre/gelding,
    #     but he's actually a stallion). Without this fallback the JOSH
    #     POWER → OFFSHORE DREAM link would stay NULL forever. Risk of a
    #     false attachment is low when the normalised name matches
    #     uniquely.
    if not rows:
        cur.execute(
            """
            SELECT horse_id, letrot_id
              FROM horse
             WHERE v2_normalize_name(name) = v2_normalize_name(%s)
               AND v2_normalize_name(name) <> ''
             LIMIT 3
            """,
            (name,),
        )
        rows = cur.fetchall()

    if len(rows) == 1:
        existing_id, existing_letrot = rows[0]
        if letrot_id and not existing_letrot:
            # Attach the LeTrot id to the existing canonical row.
            # Race condition: another worker may have just claimed this
            # letrot_id elsewhere; the UNIQUE constraint will throw.
            # We catch that and skip the attach — resolve_horse will
            # then route to the existing row.
            try:
                cur.execute(
                    "UPDATE horse SET letrot_id = %s, last_updated_at = NOW() "
                    " WHERE horse_id = %s AND letrot_id IS NULL",
                    (letrot_id, existing_id),
                )
            except Exception as exc:
                log.debug("skip parent letrot_id attach: %r", exc)
        return existing_id

    # 3. Fall back to canonical resolver. It'll either find by some
    #    other id we didn't try, or INSERT a fresh minimal row.
    if letrot_id:
        return resolve_horse(
            cur,
            source="letrot",
            source_id=letrot_id,
            canonical_fields={
                "name": name,
                "gender_code": expect_gender if expect_gender in ("H", "S") else None,
            },
            raw_payload={"discovered_via": "pedigree_scrape"},
        )
    return None


def _apply_identity(cur, horse_id: int, ident: dict, *, now_iso: str) -> dict:
    """Apply a parsed identity dict to a horse row. Returns a small audit
    summary so the worker thread can report back to the orchestrator.
    """
    audit: dict = {"horse_id": horse_id}

    sire_id = _resolve_parent(
        cur,
        name=ident.get("sire_name"),
        letrot_id=ident.get("sire_letrot_id"),
        expect_gender="H",
    )
    dam_id = _resolve_parent(
        cur,
        name=ident.get("dam_name"),
        letrot_id=ident.get("dam_letrot_id"),
        expect_gender="S",
    )

    # Build the SET clause defensively — every column is opt-in so a
    # missing identity field never clobbers an existing canonical value.
    sets: list[str] = []
    params: list = []

    def _setif(col: str, val):
        # Only overwrite when we have a non-empty new value.
        if val is None or val == "":
            return
        sets.append(f"{col} = COALESCE(%s, {col})")
        params.append(val)

    _setif("sire_name", ident.get("sire_name"))
    _setif("dam_name",  ident.get("dam_name"))
    _setif("color",     ident.get("color"))

    # Promote the identity-page "Sexe" to the canonical gender_code when we
    # don't already have one. This rescues French horses whose race rows
    # never carried a usable sex (e.g. NEWTON DE MONGOCHY).
    gender_code = _french_sex_word_to_hvs(ident.get("sex"))
    if gender_code:
        sets.append("gender_code = COALESCE(gender_code, %s)")
        params.append(gender_code)
        audit["gender_code"] = gender_code
    if sire_id is not None and sire_id != horse_id:
        sets.append("sire_id = COALESCE(sire_id, %s)")
        params.append(sire_id)
        audit["sire_id"] = sire_id
    if dam_id is not None and dam_id != horse_id:
        sets.append("dam_id = COALESCE(dam_id, %s)")
        params.append(dam_id)
        audit["dam_id"] = dam_id

    if ident.get("birth_year"):
        # Only set DOB if we don't have it; LeTrot only gives the year.
        sets.append(
            "date_of_birth = COALESCE(date_of_birth, "
            "MAKE_DATE(%s, 1, 1))"
        )
        params.append(int(ident["birth_year"]))

    # Always patch source_data['letrot'] with everything LeTrot gave us
    # plus the fetched-at marker so the next run skips this horse.
    letrot_blob = {
        k: v for k, v in ident.items()
        if k in (
            "sex", "birth_year", "color", "gains_total_eur",
            "record_text", "sire_name", "sire_letrot_id",
            "dam_name", "dam_letrot_id",
            "trainer_name", "trainer_letrot_id",
            "owner_name", "owner_letrot_id",
            "breeder_name", "breeder_letrot_id",
        )
        and v not in (None, "")
    }
    letrot_blob["identity_fetched_at"] = now_iso

    sets.append(
        "source_data = jsonb_set("
        "  COALESCE(source_data, '{}'::jsonb),"
        "  '{letrot}',"
        "  COALESCE(source_data->'letrot', '{}'::jsonb) || %s::jsonb,"
        "  true)"
    )
    params.append(Json(letrot_blob))

    sets.append("last_updated_at = NOW()")
    params.append(horse_id)

    sql = f"UPDATE horse SET {', '.join(sets)} WHERE horse_id = %s"
    cur.execute(sql, params)
    return audit


# ---------------------------------------------------------------------------
# Worker thread — fetches identity pages, returns to writer thread
# ---------------------------------------------------------------------------

_FetchResult = tuple  # (horse_id, letrot_id, slug, ident_dict_or_None, error_str_or_None)


def _worker(in_q: "queue.Queue[tuple[int, str, str] | None]",
            out_q: "queue.Queue[_FetchResult]",
            client_lock: threading.Lock) -> None:
    # Each worker gets its own httpx.Client — connection pool, kept alive
    # across requests.
    client = make_client()
    try:
        while True:
            item = in_q.get()
            if item is None:
                in_q.task_done()
                break
            horse_id, letrot_id, slug = item
            try:
                ident = fetch_horse_identity(client, letrot_id, slug)
                err = None
                if ident is None:
                    err = "identity page returned None"
            except Exception as exc:
                ident = None
                err = repr(exc)[:200]
            out_q.put((horse_id, letrot_id, slug, ident, err))
            in_q.task_done()
    finally:
        client.close()


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def run(*, execute: bool, limit: int | None, refresh_days: int | None,
        single_letrot_id: str | None, workers: int, commit_every: int) -> int:
    conn = get_connection()
    try:
        candidates = _fetch_candidates(
            conn, limit=limit, refresh_days=refresh_days,
            single_letrot_id=single_letrot_id,
        )
    finally:
        conn.close()

    log.info("found %d horses to scrape "
             "(limit=%s, refresh_days=%s, single=%s)",
             len(candidates), limit, refresh_days, single_letrot_id)
    if not candidates:
        return 0

    if not execute:
        log.info("DRY-RUN — no HTTP or DB writes. First 5 candidates:")
        for hid, lid, slug in candidates[:5]:
            log.info("  horse_id=%d letrot_id=%s slug=%s", hid, lid, slug)
        return 0

    in_q: "queue.Queue" = queue.Queue(maxsize=workers * 4)
    out_q: "queue.Queue" = queue.Queue(maxsize=workers * 8)
    client_lock = threading.Lock()

    workers_list = [
        threading.Thread(target=_worker, args=(in_q, out_q, client_lock),
                         daemon=True)
        for _ in range(workers)
    ]
    for w in workers_list:
        w.start()

    # Feed inputs from main thread so we can throttle to the worker pace.
    def _feeder():
        for c in candidates:
            in_q.put(c)
        # Sentinel per worker to shut down cleanly.
        for _ in range(workers):
            in_q.put(None)

    feeder = threading.Thread(target=_feeder, daemon=True)
    feeder.start()

    # Single writer thread (the main thread) consumes out_q and does
    # all DB work — no thread contention on the connection.
    write_conn = get_connection()
    t0 = time.time()
    processed = 0
    ok = 0
    errors = 0
    err_samples: list[str] = []
    in_batch = 0

    try:
        write_ms_total = 0.0
        get_ms_total = 0.0
        while processed < len(candidates):
            t_get0 = time.monotonic()
            horse_id, letrot_id, slug, ident, err = out_q.get()
            get_ms_total += (time.monotonic() - t_get0) * 1000
            processed += 1

            if err:
                errors += 1
                if len(err_samples) < 5:
                    err_samples.append(f"horse_id={horse_id}: {err}")
            elif ident:
                t_w0 = time.monotonic()
                try:
                    with write_conn.cursor() as cur:
                        now_iso = datetime.utcnow().isoformat()
                        _apply_identity(cur, horse_id, ident, now_iso=now_iso)
                    ok += 1
                    in_batch += 1
                except Exception as exc:
                    errors += 1
                    if len(err_samples) < 5:
                        err_samples.append(
                            f"horse_id={horse_id} DB write: {exc!r}"[:300])
                    write_conn.rollback()
                write_ms_total += (time.monotonic() - t_w0) * 1000

            if in_batch >= commit_every:
                write_conn.commit()
                in_batch = 0
                rate = processed / max(1e-6, time.time() - t0)
                remaining = len(candidates) - processed
                eta_min = remaining / max(1e-6, rate) / 60.0
                log.info(
                    "  …%d/%d scraped  ok=%d errors=%d  %.1f horses/s  "
                    "ETA %.1f min",
                    processed, len(candidates), ok, errors, rate, eta_min,
                )

        # Final flush.
        if in_batch:
            write_conn.commit()
    finally:
        write_conn.close()
        for w in workers_list:
            w.join(timeout=5)

    log.info("done — %d processed in %.1fs  ok=%d  errors=%d  "
             "(wait-on-fetch=%.1fs, db-writes=%.1fs)",
             processed, time.time() - t0, ok, errors,
             get_ms_total / 1000.0, write_ms_total / 1000.0)
    if err_samples:
        log.info("first error samples:")
        for s in err_samples:
            log.info("  %s", s)
    return 0 if errors < max(10, len(candidates) // 100) else 1


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--execute", action="store_true",
                   help="actually fetch + write (default is dry-run)")
    p.add_argument("--limit", type=int, default=None,
                   help="cap on horses to process (for testing)")
    p.add_argument("--letrot-id", default=None,
                   help="single horse by letrot_id (overrides --limit, "
                        "ignores --refresh-days filter)")
    p.add_argument("--refresh-days", type=int, default=None,
                   help="re-fetch horses whose identity_fetched_at marker "
                        "is older than N days (or missing). Default: only "
                        "horses with no marker at all.")
    p.add_argument("--workers", type=int, default=5,
                   help="parallel HTTP workers (default 5; LeTrot tolerates "
                        "this comfortably)")
    p.add_argument("--commit-every", type=int, default=100,
                   help="DB commit batch size (default 100)")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    return run(
        execute=args.execute,
        limit=args.limit,
        refresh_days=args.refresh_days,
        single_letrot_id=args.letrot_id,
        workers=args.workers,
        commit_every=args.commit_every,
    )


if __name__ == "__main__":
    sys.exit(main())
