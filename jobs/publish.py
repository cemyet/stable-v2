"""
Publish the local v2 "serving set" to Supabase (local → cloud sync).

Architecture: LOCAL is the writer/ML lab (scrapes, derives, trains models);
CLOUD (Supabase) is the read replica the Railway web app serves. This job runs
after the nightly update and incrementally upserts everything the cloud app
needs — master tables, ML feature/derived tables, and ML outputs — but NOT the
raw archives (atg_race_raw, st_*_raw, *_buffer) which stay local-only.

Sync strategy per table:
  * watermark tables: pull rows whose watermark column advanced since the last
    publish and upsert them (idempotent). Watermarks are stored in `publish_state`.
  * history tables (no per-row timestamp): pull rows for horses whose
    `last_updated_at` advanced.
  * materialized views: REFRESH on the cloud after base tables are synced.

Usage:
    # One-time, right after the initial full restore (Supabase already has the
    # data): set watermarks to the current local max WITHOUT copying anything.
    python3 -m jobs.publish --init

    # Nightly incremental (default):
    python3 -m jobs.publish

    # Force a full re-push of the serving set (ignores watermarks):
    python3 -m jobs.publish --full

    # Skip the (slower) materialized-view refresh on the cloud:
    python3 -m jobs.publish --no-matviews

Requires SUPABASE_DATABASE_URL in the environment (or .env).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import psycopg2
from psycopg2.extras import execute_values, Json

from core import config


# FK-safe order. (table, watermark_column). watermark_column drives the
# incremental window; None means "sync via horse.last_updated_at" (history) or
# handled specially below.
SERVING_TABLES: list[tuple[str, str | None]] = [
    ("track",                 "last_updated_at"),
    ("person",                "last_updated_at"),
    ("horse",                 "last_updated_at"),
    ("race",                  "last_updated_at"),
    ("entry",                 "last_updated_at"),
    ("horse_owner_history",   None),   # via horse watermark
    ("horse_trainer_history", None),   # via horse watermark
    # computed_at (not race_date): relabel_recent() back-fills result labels onto
    # rows whose race already ran (often days after their race_date), bumping
    # computed_at. Watermarking on computed_at guarantees those corrections — and
    # any future in-place column back-fill — actually reach the cloud, which a
    # race_date watermark (only ~3 days of overlap) would silently drop.
    ("entry_features",        "computed_at"),
    ("entry_outperf",         "race_date"),
    ("entry_perf",            "race_date"),
    ("trainer_form_exp",      "race_date"),
    ("identity_redirect",     "created_at"),
    ("watchlist",             "added_at"),
    ("ml_slice",              "created_at"),
    ("ml_model",              "created_at"),
    ("ml_prediction",         "updated_at"),
]

# Materialized views refreshed on the cloud after base tables land (order: the
# horse_*/person_* career views feed the browse-stat views).
MATVIEWS = [
    "horse_career_stats", "horse_year_stats", "person_career_stats",
    "track_stats", "horse_stats", "person_stats", "track_post_stats",
]

# Overlap re-pulled each run so boundary rows (equal timestamps / late same-day
# derived rows) are never missed. Upserts make the re-pull idempotent.
_TS_OVERLAP = "INTERVAL '1 hour'"
_DATE_OVERLAP = "INTERVAL '3 days'"

_BATCH = 5000


# TCP keepalives so a connection whose peer vanished mid-operation (e.g. the
# laptop slept on battery and the socket to Supabase went dead) is detected and
# raised within ~1 min instead of blocking forever on a recv() that never
# returns. Without this, an interrupted publish leaves the process hung — and,
# worse, half-reconciled (pre-pass ran, post-pass didn't). A clean error lets
# the run fail and the next idempotent publish self-heal.
_KEEPALIVE_KW = dict(
    connect_timeout=15,
    keepalives=1,
    keepalives_idle=30,
    keepalives_interval=10,
    keepalives_count=3,
)


def _connect(url: str, *, readonly: bool = False):
    conn = psycopg2.connect(url, **_KEEPALIVE_KW)
    if readonly:
        conn.set_session(readonly=True)
    return conn


class _ConnectionLost(Exception):
    """The cloud connection died mid-publish (classic cause: the laptop slept
    on battery, so the socket to Supabase went dead). Raised so run_publish's
    retry wrapper reconnects and re-runs — every step is idempotent."""


def _conn_dead(conn) -> bool:
    # psycopg2 flips .closed to nonzero once the server connection is gone.
    return getattr(conn, "closed", 0) != 0


def _safe_rollback(conn) -> None:
    # Rolling back a connection whose socket already died itself raises
    # InterfaceError ("connection already closed"), burying the real cause
    # under a noisy secondary traceback. Swallow it — the caller inspects
    # _conn_dead() to decide whether to retry.
    try:
        conn.rollback()
    except Exception:
        pass


def _ensure_state(local) -> None:
    with local.cursor() as cur:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS publish_state (
                table_name  TEXT PRIMARY KEY,
                watermark   TEXT,
                updated_at  TIMESTAMP NOT NULL DEFAULT NOW()
            )
        """)
    local.commit()


def _get_watermark(local, table: str) -> str | None:
    with local.cursor() as cur:
        cur.execute("SELECT watermark FROM publish_state WHERE table_name=%s", (table,))
        row = cur.fetchone()
    return row[0] if row else None


def _set_watermark(local, table: str, value) -> None:
    with local.cursor() as cur:
        cur.execute("""
            INSERT INTO publish_state (table_name, watermark, updated_at)
            VALUES (%s, %s, NOW())
            ON CONFLICT (table_name)
            DO UPDATE SET watermark = EXCLUDED.watermark, updated_at = NOW()
        """, (table, str(value) if value is not None else None))
    local.commit()


def _columns(conn, table: str) -> list[str]:
    """Insertable columns, in ordinal order. Generated columns (attgenerated
    <> '') are excluded — they can't be written and are recomputed on the
    cloud from their base columns."""
    with conn.cursor() as cur:
        cur.execute("""
            SELECT a.attname
            FROM pg_attribute a
            JOIN pg_class c ON c.oid = a.attrelid
            JOIN pg_namespace n ON n.oid = c.relnamespace
            WHERE n.nspname='public' AND c.relname=%s
              AND a.attnum > 0 AND NOT a.attisdropped
              AND a.attgenerated = ''
            ORDER BY a.attnum
        """, (table,))
        return [r[0] for r in cur.fetchall()]


def _json_columns(conn, table: str) -> set[str]:
    """Column names of json/jsonb type — their values come back as Python
    dict/list and must be re-wrapped in Json() before re-inserting."""
    with conn.cursor() as cur:
        cur.execute("""
            SELECT column_name FROM information_schema.columns
            WHERE table_schema='public' AND table_name=%s
              AND data_type IN ('json', 'jsonb')
        """, (table,))
        return {r[0] for r in cur.fetchall()}


def _pk_columns(conn, table: str) -> list[str]:
    with conn.cursor() as cur:
        cur.execute("""
            SELECT a.attname
            FROM pg_index i
            JOIN pg_class c ON c.oid = i.indrelid
            JOIN pg_namespace n ON n.oid = c.relnamespace
            JOIN pg_attribute a ON a.attrelid = c.oid AND a.attnum = ANY(i.indkey)
            WHERE i.indisprimary AND n.nspname='public' AND c.relname=%s
            ORDER BY array_position(i.indkey, a.attnum)
        """, (table,))
        return [r[0] for r in cur.fetchall()]


def _upsert_sql(table: str, cols: list[str], pk: list[str]) -> str:
    col_list = ", ".join(cols)
    updates = [c for c in cols if c not in pk]
    set_clause = ", ".join(f"{c}=EXCLUDED.{c}" for c in updates)
    conflict = (f"ON CONFLICT ({', '.join(pk)}) DO UPDATE SET {set_clause}"
                if updates else f"ON CONFLICT ({', '.join(pk)}) DO NOTHING")
    return f"INSERT INTO {table} ({col_list}) VALUES %s {conflict}"


def _stream_upsert(local, cloud, table: str, where_sql: str, params: list,
                   cols: list[str], pk: list[str], *, log=print) -> int:
    """Stream rows matching where_sql from local and upsert into cloud."""
    sql = _upsert_sql(table, cols, pk)
    col_list = ", ".join(cols)
    json_cols = _json_columns(local, table)
    json_idx = [i for i, c in enumerate(cols) if c in json_cols]
    select = f"SELECT {col_list} FROM {table}"
    if where_sql:
        select += f" WHERE {where_sql}"

    def _prep(row):
        if not json_idx:
            return row
        r = list(row)
        for i in json_idx:
            if r[i] is not None:
                r[i] = Json(r[i])
        return r

    pushed = 0
    with local.cursor(name=f"pub_{table}") as src:
        src.itersize = _BATCH
        src.execute(select, params)
        with cloud.cursor() as dst:
            batch = []
            for row in src:
                batch.append(_prep(row))
                if len(batch) >= _BATCH:
                    execute_values(dst, sql, batch, page_size=1000)
                    cloud.commit()
                    pushed += len(batch)
                    batch = []
            if batch:
                execute_values(dst, sql, batch, page_size=1000)
                cloud.commit()
                pushed += len(batch)
    if pushed:
        log(f"  {table}: +{pushed:,} rows")
    return pushed


def _new_max(local, table: str, wm_col: str, where_sql: str, params: list):
    with local.cursor() as cur:
        q = f"SELECT MAX({wm_col}) FROM {table}"
        if where_sql:
            q += f" WHERE {where_sql}"
        cur.execute(q, params)
        return cur.fetchone()[0]


def sync_table(local_read, state, cloud, table: str, wm_col: str, *,
               full: bool, log=print) -> int:
    cols = _columns(local_read, table)
    pk = _pk_columns(local_read, table)
    if not pk:
        log(f"  {table}: no PK, skipping")
        return 0

    is_date = wm_col == "race_date"
    overlap = _DATE_OVERLAP if is_date else _TS_OVERLAP
    wm = None if full else _get_watermark(state, table)

    if wm is None:
        where_sql, params = "", []
    else:
        where_sql = f"{wm_col} >= %s::timestamptz - {overlap}" if not is_date \
            else f"{wm_col} >= %s::date - {overlap}"
        params = [wm]

    pushed = _stream_upsert(local_read, cloud, table, where_sql, params, cols, pk, log=log)
    new_max = _new_max(local_read, table, wm_col, where_sql, params)
    if new_max is not None:
        _set_watermark(state, table, new_max)
    return pushed


def sync_history(local_read, cloud, table: str, *, full: bool,
                 horse_wm: str | None, log=print) -> int:
    """History tables have no per-row timestamp; sync rows for horses whose
    last_updated_at advanced since the horse watermark (or all rows if full)."""
    cols = _columns(local_read, table)
    pk = _pk_columns(local_read, table)
    if full or horse_wm is None:
        where_sql, params = "", []
    else:
        where_sql = (f"horse_id IN (SELECT horse_id FROM horse "
                     f"WHERE last_updated_at >= %s::timestamptz - {_TS_OVERLAP})")
        params = [horse_wm]
    return _stream_upsert(local_read, cloud, table, where_sql, params, cols, pk, log=log)


# ---------------------------------------------------------------------------
# Merge reconciliation — mirror local horse/person merges onto the cloud.
#
# merge_horses()/merge_persons() delete the losing row locally, re-home its
# entries/histories/pedigree onto the keeper, and register an identity_redirect.
# The publish job only UPSERTS, so without this step the cloud accumulates
# orphaned loser rows forever: they hog unique ids (e.g. letrot_id), keep old
# entries pointed at a ghost horse, and double-count in every stat matview.
# Eventually a restored id (e.g. a manual re-merge) collides with a lingering
# loser's unique column and the whole horse/entry/entry_features push aborts.
#
# We replay each merge on the cloud, driven by the local *_merge_log (source of
# truth). It is idempotent and self-limiting: once a loser is deleted on the
# cloud it never appears again. A single pre-pass frees the doomed losers'
# unique columns so the keeper upsert in the main loop can't collide; the
# post-pass (after all upserts) re-homes references and deletes the losers.
# ---------------------------------------------------------------------------

# ref tuple: (table, fk_column, dedup_other_keys)
#   dedup_other_keys is None  -> plain UPDATE (fk_column is not part of a PK)
#   dedup_other_keys is a list-> the OTHER PK columns; loser rows that would
#                                PK-collide with the keeper are deleted first
#                                (mirrors the delete-and-move in core.identity).
_MERGE_SPECS: list[dict] = [
    dict(
        entity="horse", log="horse_merge_log",
        from_col="from_horse_id", to_col="to_horse_id",
        master="horse", pk="horse_id",
        refs=[
            # dedup on race_id: entry has UNIQUE(race_id, horse_id), so if the
            # loser and keeper both ran the same race (locally already resolved
            # to a single keeper entry), re-homing the loser's stale entry would
            # collide — delete it (its entry_features cascade) and keep the
            # keeper's, mirroring merge_horses' same-race conflict resolution.
            ("entry",                 "horse_id", ["race_id"]),
            ("watchlist",             "horse_id", []),
            ("horse_owner_history",   "horse_id", ["from_date"]),
            ("horse_trainer_history", "horse_id", ["from_date"]),
            ("horse",                 "sire_id",  None),
            ("horse",                 "dam_id",   None),
        ],
    ),
    dict(
        entity="person", log="person_merge_log",
        from_col="from_person_id", to_col="to_person_id",
        master="person", pk="person_id",
        refs=[
            ("entry",                 "driver_id",  None),
            ("entry",                 "trainer_id", None),
            ("horse_owner_history",   "owner_id",   None),
            ("horse_trainer_history", "trainer_id", None),
        ],
    ),
    dict(
        entity="race", log="race_merge_log",
        from_col="from_race_id", to_col="to_race_id",
        master="race", pk="race_id",
        refs=[
            # dedup on horse_id: entry has UNIQUE(race_id, horse_id), so when a
            # duplicate race copy is folded into the real one, both may hold an
            # entry for the same horse. Locally that is already resolved to a
            # single entry — re-homing the loser's copy would collide, so drop
            # it (entry_features cascades), mirroring the horse spec above.
            ("entry", "race_id", ["horse_id"]),
        ],
    ),
]

# Tables where local surgery can move a single-column unique value between two
# surviving rows without any merge: re-pointing ATG's rotating track slots, or
# detaching a wrongly-attached synthetic id from one horse and giving it to the
# horse that really owns it. Neither leaves a merge-log entry, so the reconcile
# above cannot see them. Compared wholesale — keep this to tables where that is
# cheap (horse is ~78k non-null atg_ids and takes a few seconds); anything
# larger should be scoped to the push window instead.
_MOVED_UNIQUE_TABLES = ["track", "horse"]


def _single_col_unique_cols(conn, table: str, pk: list[str]) -> list[str]:
    """Columns that are the sole key of a UNIQUE index (excluding the PK) — the
    ones that can trip a single-row unique violation when a keeper adopts a
    value a lingering loser still holds."""
    with conn.cursor() as cur:
        cur.execute("""
            SELECT a.attname
            FROM pg_index i
            JOIN pg_class c ON c.oid = i.indrelid
            JOIN pg_attribute a ON a.attrelid = c.oid AND a.attnum = i.indkey[0]
            WHERE c.relname = %s AND i.indisunique AND i.indnkeyatts = 1
        """, (table,))
        cols = [r[0] for r in cur.fetchall()]
    return [c for c in cols if c not in pk]


def _build_merge_plan(local_read, cloud, spec: dict) -> list[tuple[int, int]]:
    """(loser_id, terminal_keeper_id) pairs for losers still lingering on the
    cloud but already deleted locally. Chains of successive merges are flattened
    to the terminal keeper."""
    with local_read.cursor() as cur:
        cur.execute(
            f"SELECT {spec['from_col']}, {spec['to_col']} "
            f"FROM {spec['log']} WHERE rolled_back = false"
        )
        nxt = {int(f): int(t) for f, t in cur.fetchall()}
    if not nxt:
        return []

    def resolve(start: int) -> tuple[int, list[int]]:
        """Terminal keeper plus every id visited on the way there.

        The chain can close back on itself: a horse split off into a new row
        and later merged back into the original produces from->to edges in both
        directions. Returning the visited set lets the caller settle those
        against local existence instead of silently dropping the pair.
        """
        seen: list[int] = []
        cur_id = start
        while cur_id in nxt and cur_id not in seen:
            seen.append(cur_id)
            cur_id = nxt[cur_id]
        return cur_id, seen

    losers = list(nxt.keys())
    with cloud.cursor() as cc:
        cc.execute(
            f"SELECT {spec['pk']} FROM {spec['master']} WHERE {spec['pk']} = ANY(%s)",
            (losers,),
        )
        cloud_losers = [int(r[0]) for r in cc.fetchall()]
    if not cloud_losers:
        return []
    with local_read.cursor() as lc:
        lc.execute(
            f"SELECT {spec['pk']} FROM {spec['master']} WHERE {spec['pk']} = ANY(%s)",
            (cloud_losers,),
        )
        still_local = {int(r[0]) for r in lc.fetchall()}

    plan, keepers = [], set()
    cyclic: list[tuple[int, list[int]]] = []
    for loser in cloud_losers:
        if loser in still_local:      # not actually merged away; leave it alone
            continue
        keeper, chain = resolve(loser)
        if keeper == loser:
            cyclic.append((loser, chain))
            continue
        plan.append((loser, keeper))
        keepers.add(keeper)

    # Settle cyclic chains: exactly one member of the cycle survives locally,
    # and that is the real keeper. Without this the pair is dropped and the
    # loser lingers on the cloud forever, blocking every later push of whatever
    # unique value it still holds.
    if cyclic:
        members = sorted({m for _, chain in cyclic for m in chain})
        with local_read.cursor() as lc:
            lc.execute(
                f"SELECT {spec['pk']} FROM {spec['master']} "
                f"WHERE {spec['pk']} = ANY(%s)",
                (members,),
            )
            alive = {int(r[0]) for r in lc.fetchall()}
        for loser, chain in cyclic:
            survivors = [m for m in chain if m in alive]
            if len(survivors) == 1:
                plan.append((loser, survivors[0]))
                keepers.add(survivors[0])

    if not plan:
        return []

    # The terminal keeper must exist locally (it is the surviving row); drop any
    # pair whose keeper is missing locally rather than risk orphaning entries.
    with local_read.cursor() as lc:
        lc.execute(
            f"SELECT {spec['pk']} FROM {spec['master']} WHERE {spec['pk']} = ANY(%s)",
            (sorted(keepers),),
        )
        keep_ok = {int(r[0]) for r in lc.fetchall()}
    return [(l, k) for (l, k) in plan if k in keep_ok]


def reconcile_free_unique(local_read, cloud, spec: dict,
                          plan: list[tuple[int, int]], *, log=print) -> None:
    """Pre-pass: null the doomed losers' single-column unique values so the
    keeper upsert in the main loop cannot collide with them."""
    if not plan:
        return
    ucols = _single_col_unique_cols(local_read, spec["master"], [spec["pk"]])
    if not ucols:
        return
    losers = [l for l, _ in plan]
    set_sql = ", ".join(f"{c} = NULL" for c in ucols)
    with cloud.cursor() as cc:
        cc.execute(
            f"UPDATE {spec['master']} SET {set_sql} WHERE {spec['pk']} = ANY(%s)",
            (losers,),
        )
    cloud.commit()
    log(f"  reconcile[{spec['entity']}]: freed unique cols on "
        f"{len(losers):,} lingering cloud rows")


#: Candidate values sent to the cloud per round trip. Bounded because the
#: comparison used to pull the whole column down instead, and `horse.atg_id`
#: alone is ~78k rows — a single fetch big enough to blow the SSL read timeout
#: and take the entire publish with it.
_MOVED_UNIQUE_CHUNK = 5_000


def reconcile_moved_unique(local_read, state, cloud, table: str, *,
                           full: bool, log=print) -> int:
    """Free cloud unique values that local has since moved to a different row.

    The incremental push is keyed on the primary key, so when a unique value
    migrates between two rows that both still exist — an ATG track slot
    re-pointed from the venue that used to hold it to the one that really owns
    it — the cloud still has the old holder and the new owner's upsert trips
    the unique constraint. Nulling the stale copy lets the push through; the
    old holder receives its own corrected value in the same run.

    Only rows this run will actually push can collide, so candidates come from
    the watermark window and the match is evaluated server-side: local sends
    (value, owner) pairs up and the cloud returns just the rows holding a value
    under the wrong key. The window is not a weaker guarantee than scanning
    everything — a move that never bumped `last_updated_at` would not be pushed
    either, so there would be no upsert to collide with.
    """
    pks = _pk_columns(local_read, table)
    if len(pks) != 1:
        return 0
    pk = pks[0]
    wm = None if full else _get_watermark(state, table)
    freed = 0
    for col in _single_col_unique_cols(local_read, table, [pk]):
        with local_read.cursor() as lc:
            if wm is None:
                lc.execute(
                    f"SELECT {col}::text, {pk} FROM {table} "
                    f" WHERE {col} IS NOT NULL")
            else:
                lc.execute(
                    f"SELECT {col}::text, {pk} FROM {table} "
                    f" WHERE {col} IS NOT NULL "
                    f"   AND last_updated_at > %s::timestamp - {_TS_OVERLAP}",
                    (wm,),
                )
            pairs = lc.fetchall()
        if not pairs:
            continue
        stale: list[int] = []
        for i in range(0, len(pairs), _MOVED_UNIQUE_CHUNK):
            batch = pairs[i: i + _MOVED_UNIQUE_CHUNK]
            with cloud.cursor() as cc:
                cc.execute(
                    f"SELECT t.{pk} FROM {table} t "
                    f"  JOIN unnest(%s::text[], %s::bigint[]) AS p(val, owner) "
                    f"    ON t.{col}::text = p.val "
                    f" WHERE t.{pk} <> p.owner",
                    ([v for v, _ in batch], [k for _, k in batch]),
                )
                stale.extend(r[0] for r in cc.fetchall())
        if not stale:
            continue
        with cloud.cursor() as cc:
            cc.execute(f"UPDATE {table} SET {col} = NULL WHERE {pk} = ANY(%s)",
                       (stale,))
        cloud.commit()
        freed += len(stale)
    if freed:
        log(f"  reconcile[{table}]: freed {freed:,} moved unique values")
    return freed


def reconcile_moved_entry_keys(local_read, state, cloud, *, full: bool,
                               log=print) -> int:
    """Drop cloud entries whose (race_id, horse_id) local now holds elsewhere.

    `entry` is keyed on entry_id but carries UNIQUE(race_id, horse_id). When a
    merge rebuilds an entry against the keeper race it gets a fresh entry_id,
    so the cloud still holds that pair under the old id and the incoming upsert
    trips the unique constraint. Only rows this run is about to push can
    collide, so the comparison is scoped to the watermark window rather than
    all 7.6M entries. A full push rewrites everything and needs no pre-pass.
    """
    wm = None if full else _get_watermark(state, "entry")
    if wm is None:
        return 0
    with local_read.cursor() as lc:
        lc.execute(
            "SELECT entry_id, race_id, horse_id FROM entry "
            f"WHERE last_updated_at > %s::timestamp - {_TS_OVERLAP}",
            (wm,),
        )
        rows = lc.fetchall()
    if not rows:
        return 0
    want = {(int(r), int(h)): int(e) for e, r, h in rows}
    races = [r for _, r, _ in rows]
    horses = [h for _, _, h in rows]
    with cloud.cursor() as cc:
        cc.execute(
            "SELECT e.entry_id, e.race_id, e.horse_id FROM entry e "
            "JOIN unnest(%s::bigint[], %s::bigint[]) AS p(race_id, horse_id) "
            "  ON p.race_id = e.race_id AND p.horse_id = e.horse_id",
            (races, horses),
        )
        stale = [int(e) for e, r, h in cc.fetchall()
                 if want.get((int(r), int(h))) != int(e)]
        if stale:
            cc.execute("DELETE FROM entry WHERE entry_id = ANY(%s)", (stale,))
    cloud.commit()
    if stale:
        log(f"  reconcile[entry]: dropped {len(stale):,} cloud rows whose "
            f"(race_id, horse_id) moved to a new entry_id")
    return len(stale)


# Children before parents so an orphan sweep never trips a foreign key.
_ORPHAN_SWEEP_TABLES = ["entry", "race", "horse", "person", "track"]

# A sweep should only ever remove the tail of a cleanup. If a table's cloud
# copy diverges by more than this, something is wrong (wrong database, a
# half-restored local, an interrupted merge) and mass-deleting the difference
# would be far worse than leaving it — so warn and skip instead.
_ORPHAN_SWEEP_MAX_FRACTION = 0.02


def _fk_referrers(conn, table: str, pk: str) -> list[tuple[str, str]]:
    """(child_table, child_column) pairs with a foreign key onto table.pk."""
    with conn.cursor() as cur:
        cur.execute("""
            SELECT tc.table_name, kcu.column_name
              FROM information_schema.table_constraints tc
              JOIN information_schema.key_column_usage kcu
                ON kcu.constraint_name = tc.constraint_name
              JOIN information_schema.constraint_column_usage ccu
                ON ccu.constraint_name = tc.constraint_name
             WHERE tc.constraint_type = 'FOREIGN KEY'
               AND ccu.table_name = %s AND ccu.column_name = %s
        """, (table, pk))
        return [(r[0], r[1]) for r in cur.fetchall()]


def _rehome_refs_from_local(local_read, cloud, table: str, pk: str,
                            orphans: list[int], *, log=print) -> None:
    """Repoint cloud rows still referencing soon-to-be-deleted orphan parents.

    A track folded into another locally leaves its races pointing at the keeper
    here, but the cloud copies of those races may pre-date the merge and still
    reference the loser. Local is authoritative, so each stale reference is
    refreshed from whatever local now holds for that same child row.
    """
    for child, col in _fk_referrers(local_read, table, pk):
        child_pks = _pk_columns(local_read, child)
        if len(child_pks) != 1:
            continue
        cpk = child_pks[0]
        with cloud.cursor() as cc:
            cc.execute(f"SELECT {cpk} FROM {child} WHERE {col} = ANY(%s)",
                       (orphans,))
            ids = [int(r[0]) for r in cc.fetchall()]
        if not ids:
            continue
        with local_read.cursor() as lc:
            lc.execute(f"SELECT {cpk}, {col} FROM {child} WHERE {cpk} = ANY(%s)",
                       (ids,))
            pairs = [(int(a), b) for a, b in lc.fetchall() if b is not None]
        if not pairs:
            continue
        with cloud.cursor() as cc:
            execute_values(
                cc,
                f"UPDATE {child} SET {col} = v.newval FROM (VALUES %s) "
                f"AS v(id, newval) WHERE {child}.{cpk} = v.id",
                pairs, page_size=1000,
            )
        cloud.commit()
        log(f"  orphan-sweep[{table}]: re-homed {len(pairs):,} "
            f"{child}.{col} refs off orphan rows")


def reconcile_orphans(local_read, cloud, *, log=print) -> dict:
    """Delete cloud rows whose primary key no longer exists locally.

    The merge-log reconcile only knows about rows folded into a keeper. Plain
    deletions — a duplicate race dropped by hand, entries removed when a
    wrongly-attached synthetic id was detached — leave no audit trail, so the
    cloud keeps serving rows the local database has long since removed.

    Counts are compared first because they are cheap server-side; the primary
    key diff (which streams every id of a 7.6M-row table) only runs for tables
    that actually diverge, so the usual no-drift case costs five counts.
    """
    out: dict[str, int] = {}
    for table in _ORPHAN_SWEEP_TABLES:
        pks = _pk_columns(local_read, table)
        if len(pks) != 1:
            continue
        pk = pks[0]
        with local_read.cursor() as lc:
            lc.execute(f"SELECT COUNT(*) FROM {table}")
            n_local = lc.fetchone()[0]
        with cloud.cursor() as cc:
            cc.execute(f"SELECT COUNT(*) FROM {table}")
            n_cloud = cc.fetchone()[0]
        if n_cloud <= n_local:
            continue

        with cloud.cursor(name=f"orph_{table}") as cc:
            cc.itersize = _BATCH
            cc.execute(f"SELECT {pk} FROM {table}")
            cloud_ids = {int(r[0]) for r in cc}
        with local_read.cursor(name=f"orphl_{table}") as lc:
            lc.itersize = _BATCH
            lc.execute(f"SELECT {pk} FROM {table}")
            local_ids = {int(r[0]) for r in lc}
        orphans = sorted(cloud_ids - local_ids)
        if not orphans:
            continue
        if n_local and len(orphans) / n_local > _ORPHAN_SWEEP_MAX_FRACTION:
            log(f"  orphan-sweep[{table}]: SKIPPED — {len(orphans):,} orphans "
                f"is over {_ORPHAN_SWEEP_MAX_FRACTION:.0%} of {n_local:,} local "
                f"rows; refusing to mass-delete, investigate first")
            continue
        _rehome_refs_from_local(local_read, cloud, table, pk, orphans, log=log)
        with cloud.cursor() as cc:
            for i in range(0, len(orphans), _BATCH):
                cc.execute(f"DELETE FROM {table} WHERE {pk} = ANY(%s)",
                           (orphans[i:i + _BATCH],))
                cloud.commit()
        out[table] = len(orphans)
        log(f"  orphan-sweep[{table}]: deleted {len(orphans):,} cloud rows "
            f"absent locally")
    return out


def reconcile_apply(local_read, cloud, spec: dict, plan: list[tuple[int, int]],
                    *, log=print) -> None:
    """Post-pass: ensure keepers exist on the cloud, re-home every reference
    from loser to keeper, then delete the loser rows."""
    if not plan:
        return
    keepers = sorted({k for _, k in plan})
    # Guarantee FK targets exist on the cloud (keepers of old merges may pre-date
    # the incremental window and thus not be re-pushed by the main loop).
    cols = _columns(local_read, spec["master"])
    pk = _pk_columns(local_read, spec["master"])
    _stream_upsert(local_read, cloud, spec["master"],
                   f"{spec['pk']} = ANY(%s)", [keepers], cols, pk, log=log)

    with cloud.cursor() as cc:
        cc.execute("CREATE TEMP TABLE _merge_map (loser bigint PRIMARY KEY, "
                   "keeper bigint) ON COMMIT DROP")
        execute_values(cc, "INSERT INTO _merge_map (loser, keeper) VALUES %s",
                       plan, page_size=1000)
        for table, col, dedup in spec["refs"]:
            if dedup is not None:
                cond = " AND ".join(
                    [f"t2.{col} = m.keeper"] + [f"t2.{k} = t.{k}" for k in dedup]
                )
                cc.execute(
                    f"DELETE FROM {table} t USING _merge_map m "
                    f"WHERE t.{col} = m.loser "
                    f"  AND EXISTS (SELECT 1 FROM {table} t2 WHERE {cond})"
                )
            cc.execute(
                f"UPDATE {table} t SET {col} = m.keeper "
                f"FROM _merge_map m WHERE t.{col} = m.loser"
            )
        cc.execute(
            f"DELETE FROM {spec['master']} "
            f"WHERE {spec['pk']} IN (SELECT loser FROM _merge_map)"
        )
    cloud.commit()
    log(f"  reconcile[{spec['entity']}]: re-homed refs and deleted "
        f"{len(plan):,} merged-away rows")


def refresh_matviews(cloud, *, log=print) -> None:
    for mv in MATVIEWS:
        t0 = time.time()
        with cloud.cursor() as cur:
            try:
                cur.execute(f"REFRESH MATERIALIZED VIEW CONCURRENTLY {mv}")
                cloud.commit()
            except Exception as exc:
                _safe_rollback(cloud)
                if _conn_dead(cloud):
                    raise _ConnectionLost(f"matview {mv}: {exc!r}") from exc
                try:
                    cur.execute(f"REFRESH MATERIALIZED VIEW {mv}")
                    cloud.commit()
                except Exception as exc2:
                    _safe_rollback(cloud)
                    if _conn_dead(cloud):
                        raise _ConnectionLost(f"matview {mv}: {exc2!r}") from exc2
                    log(f"  matview {mv}: FAILED {exc2!r}")
                    continue
        log(f"  matview {mv}: refreshed in {time.time()-t0:.1f}s")


def init_watermarks(local, *, log=print) -> None:
    """Set each watermark table's watermark to its current local MAX without
    copying — used once after the initial full restore so the first incremental
    run only pushes genuinely new rows."""
    _ensure_state(local)
    for table, wm_col in SERVING_TABLES:
        if wm_col is None:
            continue
        with local.cursor() as cur:
            cur.execute(f"SELECT MAX({wm_col}) FROM {table}")
            mx = cur.fetchone()[0]
        _set_watermark(local, table, mx)
        log(f"  {table}: watermark = {mx}")
    # history tables ride the horse watermark, already set above.
    log("watermarks initialized.")


def _run_publish_once(*, full: bool = False, do_matviews: bool = True, log=print) -> dict:
    if not config.SUPABASE_DATABASE_URL:
        raise SystemExit("SUPABASE_DATABASE_URL is not set (env or .env).")
    local_read = _connect(config.DATABASE_URL, readonly=True)   # streaming reads
    state = _connect(config.DATABASE_URL)                       # publish_state rw
    cloud = _connect(config.SUPABASE_DATABASE_URL)              # upsert target
    totals = {"tables": {}, "pushed": 0}
    t0 = time.time()
    try:
        _ensure_state(state)

        # Merge reconciliation (pre-pass): free unique columns on cloud rows
        # that local merges have deleted, so keeper upserts below can't collide.
        merge_plans: list[tuple[dict, list[tuple[int, int]]]] = []
        try:
            for spec in _MERGE_SPECS:
                plan = _build_merge_plan(local_read, cloud, spec)
                merge_plans.append((spec, plan))
                if plan:
                    reconcile_free_unique(local_read, cloud, spec, plan, log=log)
        except Exception as exc:
            _safe_rollback(cloud)
            if _conn_dead(cloud):
                raise _ConnectionLost(f"reconcile pre-pass: {exc!r}") from exc
            log(f"  reconcile pre-pass: ERROR {exc!r}")

        for _tbl in _MOVED_UNIQUE_TABLES:
            try:
                reconcile_moved_unique(local_read, state, cloud, _tbl,
                                       full=full, log=log)
            except Exception as exc:
                _safe_rollback(cloud)
                if _conn_dead(cloud):
                    raise _ConnectionLost(
                        f"reconcile_moved_unique[{_tbl}]: {exc!r}") from exc
                log(f"  reconcile_moved_unique[{_tbl}]: ERROR {exc!r}")

        try:
            reconcile_moved_entry_keys(local_read, state, cloud,
                                       full=full, log=log)
        except Exception as exc:
            _safe_rollback(cloud)
            if _conn_dead(cloud):
                raise _ConnectionLost(f"reconcile_moved_entry_keys: {exc!r}") from exc
            log(f"  reconcile_moved_entry_keys: ERROR {exc!r}")

        horse_wm = None if full else _get_watermark(state, "horse")
        for table, wm_col in SERVING_TABLES:
            try:
                if wm_col is None:
                    n = sync_history(local_read, cloud, table, full=full,
                                     horse_wm=horse_wm, log=log)
                else:
                    n = sync_table(local_read, state, cloud, table, wm_col,
                                   full=full, log=log)
                totals["tables"][table] = n
                totals["pushed"] += n
            except Exception as exc:
                _safe_rollback(cloud)
                if _conn_dead(cloud):
                    raise _ConnectionLost(f"{table}: {exc!r}") from exc
                log(f"  {table}: ERROR {exc!r}")
                totals["tables"][table] = f"error: {exc!r}"

        # Merge reconciliation (post-pass): re-home references and delete the
        # merged-away loser rows now that all keepers/entries are on the cloud.
        for spec, plan in merge_plans:
            if not plan:
                continue
            try:
                reconcile_apply(local_read, cloud, spec, plan, log=log)
                totals[f"reconciled_{spec['entity']}"] = len(plan)
            except Exception as exc:
                _safe_rollback(cloud)
                if _conn_dead(cloud):
                    raise _ConnectionLost(f"reconcile[{spec['entity']}] apply: {exc!r}") from exc
                log(f"  reconcile[{spec['entity']}] apply: ERROR {exc!r}")

        # Last, once every keeper and freshly-pushed row is on the cloud: drop
        # rows deleted locally with no merge-log trail. Runs after the merge
        # post-pass so re-homed references are already pointing at keepers.
        try:
            totals["orphans_swept"] = reconcile_orphans(local_read, cloud, log=log)
        except Exception as exc:
            _safe_rollback(cloud)
            if _conn_dead(cloud):
                raise _ConnectionLost(f"orphan sweep: {exc!r}") from exc
            log(f"  orphan sweep: ERROR {exc!r}")

        if do_matviews:
            log("refreshing materialized views on cloud...")
            refresh_matviews(cloud, log=log)
    finally:
        local_read.close()
        state.close()
        cloud.close()
    totals["seconds"] = round(time.time() - t0, 1)
    log(f"publish done in {totals['seconds']}s — pushed {totals['pushed']:,} rows")
    return totals


def run_publish(*, full: bool = False, do_matviews: bool = True, log=print,
                max_attempts: int = 4, retry_delay: float = 15.0) -> dict:
    """Idempotent Supabase publish with automatic reconnect-and-retry.

    A publish that dies because the cloud socket vanished mid-run — classic
    cause: the laptop slept on battery, the process froze, then woke to a dead
    connection — used to need a manual re-run. Every step here is idempotent
    (PK-keyed upserts, watermark-driven table sync, and a two-pass merge
    reconcile whose plan is re-derived each run), so we just reconnect and
    start over up to `max_attempts` times with escalating backoff.

    Only connection death is retried; a genuine logic error is caught
    per-table inside _run_publish_once and never reaches here, so it won't
    trigger a pointless retry. After the final attempt the last error is
    re-raised so the caller/log still sees a hard failure."""
    last_exc: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            return _run_publish_once(full=full, do_matviews=do_matviews, log=log)
        except (_ConnectionLost, psycopg2.OperationalError, psycopg2.InterfaceError) as exc:
            last_exc = exc
            if attempt >= max_attempts:
                break
            delay = retry_delay * attempt
            log(f"publish: cloud connection lost ({exc!r}) — attempt "
                f"{attempt}/{max_attempts} aborted, reconnecting in {delay:.0f}s")
            time.sleep(delay)
    log(f"publish: FAILED after {max_attempts} attempts — {last_exc!r}")
    raise last_exc


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--init", action="store_true",
                    help="set watermarks to current local max, copy nothing")
    ap.add_argument("--full", action="store_true",
                    help="ignore watermarks; push the entire serving set")
    ap.add_argument("--no-matviews", action="store_true",
                    help="skip refreshing materialized views on the cloud")
    ap.add_argument("--max-attempts", type=int, default=4,
                    help="reconnect-and-retry attempts on cloud connection death "
                         "(default 4; set 1 to disable retry)")
    ap.add_argument("--retry-delay", type=float, default=15.0,
                    help="base backoff seconds between retries; grows per attempt "
                         "(default 15 → 15s, 30s, 45s)")
    args = ap.parse_args()

    if args.init:
        conn = _connect(config.DATABASE_URL)
        try:
            init_watermarks(conn)
        finally:
            conn.close()
        return 0

    run_publish(full=args.full, do_matviews=not args.no_matviews,
                max_attempts=args.max_attempts, retry_delay=args.retry_delay)
    return 0


if __name__ == "__main__":
    sys.exit(main())
