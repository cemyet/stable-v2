"""
Split horse rows whose entries have an age that contradicts the horse's
date_of_birth (the "Indy Boy / Last Queen" class of pollution).

Why this exists:
    `split_polluted_atg_ids.py` looks for entries dated >18y after the
    horse's DOB. That misses every modern ST horse (DOB 2008+) where the
    gap to the polluted modern entries is only 14-18y, so it currently
    reports 0 candidates even though hundreds of polluted horses exist.

    The far more direct signal is per-entry: each entry carries `e.age`
    (horse's age at race time). If `|e.age - (race_year - dob_year)| > 5`
    the entry CANNOT belong to this horse — it's either a name-collision
    foreign horse or a different generation namesake.

Detection (per entry, conservative):
    * horse.date_of_birth IS NOT NULL
    * entry.age IS NOT NULL
    * race.race_date > horse.date_of_birth + 6 months   ← skip Type B
    * |entry.age - (race_year - dob_year)| > 5          ← high confidence

We deliberately use a wide (>5y) threshold so we don't trip on stray
ATG data-entry errors. Lower thresholds catch more cases but raise the
false-positive rate.

Type B (entries dated before the horse's DOB) is a DIFFERENT failure
mode — the DOB itself was incorrectly attached (often from a breedly
foal record being merged into an active racer). We REPORT those but
never auto-fix them here. Moving entries off would be wrong; the right
fix is to nullify the bad DOB. That's a separate workstream.

Action per Type-A polluted horse:
    1. Detach the horse's atg_id (only if it's synthetic `x:%`). This
       prevents future ATG ingests from re-polluting the row AND frees
       the synth key for the new clean target.
    2. Pick a target row for the polluted entries:
       - same v2_normalize_name(name)
       - DOB-year within plausible range for the polluted entries'
         median race year (race_year - 12 ... race_year - 1)
       - CANDIDATE MUST BE CLEAN (no age-vs-DOB pollution itself)
       - CANDIDATE MUST NOT BE THE SOURCE HORSE
    3. If exactly one clean candidate → move polluted entries there
       (same-race conflicts resolved by deleting the source entry, the
       destination one is canonical).
    4. If zero/multiple candidates → mint a NEW horse row with the
       detached atg_id and route polluted entries to it.
    5. Audit every action in horse_merge_log with method=
       'age_mismatch_split' and a full snapshot of the source row.

The source horse row is NEVER deleted — it retains its st_id and any
clean entries that DO match the DOB.

Compounding-problem guards:
    * `--execute` required to write; default is dry-run.
    * `--limit` caps the batch.
    * `--commit-every N` reduces blast radius if something goes wrong.
    * `--mismatch-threshold` defaults to 5 (years) — lower at your own
      risk, with a fresh dry-run review.
    * Target candidate is rejected if it ALSO has age-mismatch entries.
    * `polluted_entries / total_entries` ratio is recorded but does NOT
      gate behaviour (we trust the per-entry rule).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from statistics import median

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from psycopg2.extras import Json  # noqa: E402

from scripts._merge_helpers import build_argparser, script_runner  # noqa: E402


_DEFAULT_THRESHOLD = 5  # years; |entry.age - expected_age| > this = polluted


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------

def _fetch_candidates(cur, threshold: int, limit: int | None) -> list[dict]:
    """Return horses with at least one Type-A age-mismatched entry.

    Type-A only: any horse that ALSO has entries BEFORE its DOB (Type B)
    is excluded — those need DOB correction, not entry migration.
    """
    sql = """
    WITH pol_per_entry AS (
        SELECT h.horse_id, h.name, h.atg_id, h.st_id, h.date_of_birth,
               h.birth_country,
               e.entry_id, e.age AS entry_age, r.race_date,
               (EXTRACT(year FROM r.race_date)::int
                - EXTRACT(year FROM h.date_of_birth)::int) AS gap_years
          FROM horse h
          JOIN entry e ON e.horse_id = h.horse_id
          JOIN race  r ON r.race_id  = e.race_id
         WHERE h.date_of_birth IS NOT NULL
           AND e.age IS NOT NULL
    ),
    per_horse AS (
        SELECT horse_id, name, atg_id, st_id, date_of_birth, birth_country,
               COUNT(*) AS total_entries,
               COUNT(*) FILTER (
                   WHERE ABS(entry_age - gap_years) > %(threshold)s
                     AND race_date > date_of_birth + INTERVAL '6 months'
               ) AS polluted_entries,
               COUNT(*) FILTER (
                   WHERE race_date < date_of_birth - INTERVAL '6 months'
               ) AS pre_dob_entries
          FROM pol_per_entry
         GROUP BY horse_id, name, atg_id, st_id, date_of_birth, birth_country
    )
    SELECT *
      FROM per_horse
     WHERE polluted_entries > 0
       AND pre_dob_entries  = 0   -- skip Type B
     ORDER BY polluted_entries DESC, horse_id
    """
    if limit:
        sql += f"\n     LIMIT {int(limit)}"
    cur.execute(sql, {"threshold": threshold})
    cols = [d.name for d in cur.description]
    return [dict(zip(cols, r)) for r in cur.fetchall()]


def _fetch_polluted_entries(cur, horse_id: int, threshold: int) -> list[dict]:
    """Return the individual entries on `horse_id` that fail the age rule."""
    cur.execute(
        """
        SELECT e.entry_id, e.age, r.race_id, r.race_date,
               EXTRACT(year FROM r.race_date)::int AS race_year
          FROM entry e
          JOIN race  r ON r.race_id = e.race_id
          JOIN horse h ON h.horse_id = e.horse_id
         WHERE h.horse_id = %s
           AND h.date_of_birth IS NOT NULL
           AND e.age IS NOT NULL
           AND ABS(e.age - (EXTRACT(year FROM r.race_date)::int
                            - EXTRACT(year FROM h.date_of_birth)::int)) > %s
           AND r.race_date > h.date_of_birth + INTERVAL '6 months'
         ORDER BY r.race_date
        """,
        (horse_id, threshold),
    )
    cols = [d.name for d in cur.description]
    return [dict(zip(cols, r)) for r in cur.fetchall()]


def _has_age_pollution(cur, horse_id: int, threshold: int) -> bool:
    """True if `horse_id` has ANY age-mismatched entry.

    Used to disqualify a candidate row from being a merge target so we
    never compound pollution by moving entries into an already-broken row.
    """
    cur.execute(
        """
        SELECT 1
          FROM entry e JOIN race r ON r.race_id = e.race_id
          JOIN horse h ON h.horse_id = e.horse_id
         WHERE h.horse_id = %s
           AND h.date_of_birth IS NOT NULL
           AND e.age IS NOT NULL
           AND ABS(e.age - (EXTRACT(year FROM r.race_date)::int
                            - EXTRACT(year FROM h.date_of_birth)::int)) > %s
           AND r.race_date > h.date_of_birth + INTERVAL '6 months'
         LIMIT 1
        """,
        (horse_id, threshold),
    )
    return cur.fetchone() is not None


# ---------------------------------------------------------------------------
# Target selection
# ---------------------------------------------------------------------------

def _find_clean_target(
    cur,
    src_horse_id: int,
    name: str,
    country: str | None,
    polluted_entry_years: list[int],
    threshold: int,
) -> tuple[int | None, str]:
    """Find a single clean target row for the polluted entries.

    Returns (horse_id, reason_string). horse_id is None if no unique
    clean candidate found.

    Plausibility window: dob_year ∈ [median_race_year - 12, median_race_year - 1].
    """
    if not polluted_entry_years:
        return None, "no polluted entries"
    median_year = int(median(polluted_entry_years))
    lo = median_year - 12
    hi = median_year - 1

    cur.execute(
        """
        SELECT h.horse_id, h.date_of_birth, h.st_id, h.atg_id, h.birth_country
          FROM horse h
         WHERE v2_normalize_name(h.name) = v2_normalize_name(%s)
           AND v2_normalize_name(h.name) <> ''
           AND h.horse_id <> %s
           AND h.date_of_birth IS NOT NULL
           AND EXTRACT(year FROM h.date_of_birth)::int BETWEEN %s AND %s
           AND (h.birth_country = %s OR h.birth_country IS NULL OR %s IS NULL)
        """,
        (name, src_horse_id, lo, hi, country, country),
    )
    candidates = cur.fetchall()
    if not candidates:
        return None, f"no candidate (name+dob window {lo}-{hi})"
    if len(candidates) > 1:
        return None, f"ambiguous ({len(candidates)} candidates)"

    cand_id = candidates[0][0]
    # Critical safety: candidate itself must be clean.
    if _has_age_pollution(cur, cand_id, threshold):
        return None, f"candidate {cand_id} is also polluted"
    return cand_id, f"single clean candidate {cand_id}"


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------

def _detach_atg_id(cur, horse_id: int) -> str | None:
    """Detach a synthetic atg_id from a horse. Returns the old value."""
    cur.execute(
        "SELECT atg_id FROM horse WHERE horse_id = %s AND atg_id LIKE 'x:%%'",
        (horse_id,),
    )
    row = cur.fetchone()
    if not row:
        return None
    old = row[0]
    cur.execute(
        "UPDATE horse SET atg_id = NULL, last_updated_at = NOW() "
        "WHERE horse_id = %s",
        (horse_id,),
    )
    return old


def _mint_split_row(cur, polluted: dict, detached_atg_id: str | None) -> int:
    parts = (detached_atg_id or "").split(":", 2)
    country = parts[1] if len(parts) == 3 and parts[1] else polluted["birth_country"]
    cur.execute(
        """
        INSERT INTO horse (name, birth_country, atg_id, primary_source,
                           source_data, last_updated_at)
        VALUES (%s, %s, %s, 'atg', %s, NOW())
        RETURNING horse_id
        """,
        (
            polluted["name"], country, detached_atg_id,
            Json({
                "atg": {
                    "split_from_horse_id": polluted["horse_id"],
                    "split_reason": "age_mismatch",
                    "detached_atg_id": detached_atg_id,
                }
            }),
        ),
    )
    return cur.fetchone()[0]


def _move_entries(cur, entry_ids: list[int], dst_horse_id: int) -> tuple[int, int]:
    """Move the given entries to dst_horse_id, deleting same-race conflicts.

    Returns (moved, conflicts_deleted).
    """
    if not entry_ids:
        return 0, 0

    # Same-race conflicts: if dst already has an entry on that race, the
    # source entry is a duplicate. Delete it; keep destination.
    cur.execute(
        """
        WITH targets AS (
            SELECT entry_id, race_id FROM entry WHERE entry_id = ANY(%s)
        ),
        conflicts AS (
            SELECT t.entry_id
              FROM targets t
             WHERE EXISTS (
                SELECT 1 FROM entry e2
                 WHERE e2.race_id = t.race_id AND e2.horse_id = %s
             )
        )
        DELETE FROM entry WHERE entry_id IN (SELECT entry_id FROM conflicts)
        RETURNING entry_id
        """,
        (entry_ids, dst_horse_id),
    )
    conflicts = cur.rowcount

    cur.execute(
        "UPDATE entry SET horse_id = %s WHERE entry_id = ANY(%s)",
        (dst_horse_id, entry_ids),
    )
    moved = cur.rowcount
    return moved, conflicts


def _log_split(
    cur,
    polluted: dict,
    dst_id: int,
    detached: str | None,
    moved: int,
    conflicts: int,
    method: str,
    reason_detail: str,
) -> None:
    snap = {k: (v.isoformat() if hasattr(v, "isoformat") else v)
            for k, v in polluted.items()}
    snap["detached_atg_id"] = detached
    cur.execute(
        """
        INSERT INTO horse_merge_log
            (from_horse_id, to_horse_id, reason, method,
             entries_moved, conflicts_resolved, from_snapshot, merged_by)
        VALUES (%s, %s, %s, %s, %s, %s, %s, 'split_age_mismatch')
        """,
        (
            polluted["horse_id"], dst_id,
            f"age_mismatch_split: {reason_detail}; detached={detached!r}; "
            f"DOB={polluted['date_of_birth']}; "
            f"{moved}/{polluted['polluted_entries']} polluted entries moved",
            method, moved, conflicts, json.dumps(snap, default=str),
        ),
    )


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def _report_type_b(cur, log) -> int:
    """Count and report Type B horses for manual follow-up (no fix)."""
    cur.execute(
        """
        SELECT COUNT(*) FROM (
            SELECT h.horse_id
              FROM horse h
              JOIN entry e ON e.horse_id = h.horse_id
              JOIN race  r ON r.race_id  = e.race_id
             WHERE h.date_of_birth IS NOT NULL
             GROUP BY h.horse_id, h.date_of_birth
            HAVING MIN(r.race_date) < h.date_of_birth - INTERVAL '6 months'
        ) sub
        """
    )
    n = cur.fetchone()[0]
    if n > 0:
        log(f"[type-b] {n} horses have entries BEFORE their DOB — not auto-fixed.")
        log(f"[type-b] These are foal-data-merged-into-racer cases needing a separate")
        log(f"[type-b] DOB-correction script (drop/nullify the wrong DOB).")
    return int(n)


def main() -> int:
    parser = build_argparser("split_age_mismatch")
    parser.add_argument(
        "--mismatch-threshold", type=int, default=_DEFAULT_THRESHOLD,
        help=f"|entry.age - (race_year - dob_year)| > N counts as polluted "
             f"(default {_DEFAULT_THRESHOLD}; lower = more aggressive)",
    )
    args = parser.parse_args()
    threshold = max(1, int(args.mismatch_threshold))

    with script_runner("split_age_mismatch", args) as (conn, log, summary):
        summary["mismatch_threshold"] = threshold
        summary["splits_into_existing"] = 0
        summary["splits_into_new_row"]  = 0
        summary["skipped_ambiguous"]    = 0
        summary["skipped_target_polluted"] = 0
        summary["entries_moved"]        = 0
        summary["entry_conflicts_dropped"] = 0
        summary["type_b_skipped"] = 0

        log(f"[split_age_mismatch] execute={args.execute} "
            f"limit={args.limit} threshold={threshold}y")

        with conn.cursor() as cur:
            summary["type_b_skipped"] = _report_type_b(cur, log)
            candidates = _fetch_candidates(cur, threshold, args.limit)

        summary["candidates"] = len(candidates)
        log(f"\n[type-a] {len(candidates)} polluted horses (Type A).\n")

        for p in candidates:
            with conn.cursor() as cur:
                polluted_entries = _fetch_polluted_entries(
                    cur, p["horse_id"], threshold
                )
                pol_years = [e["race_year"] for e in polluted_entries]
                target_id, reason = _find_clean_target(
                    cur, p["horse_id"], p["name"], p["birth_country"],
                    pol_years, threshold,
                )

            kept = p["total_entries"] - p["polluted_entries"]
            log(
                f"  #{p['horse_id']:<7} {p['name'][:25]!r:<27} "
                f"atg={p['atg_id']!r:<28} DOB={p['date_of_birth']}  "
                f"polluted={p['polluted_entries']:<3} kept={kept:<3}  "
                f"→ {reason}"
            )

            if not args.execute:
                summary["merged"] += 1
                if target_id is not None:
                    summary["splits_into_existing"] += 1
                else:
                    summary["splits_into_new_row"] += 1
                summary["entries_moved"] += len(polluted_entries)
                continue

            try:
                with conn.cursor() as cur:
                    detached = _detach_atg_id(cur, p["horse_id"])
                    method = ("age_mismatch_existing"
                              if target_id is not None
                              else "age_mismatch_new_row")
                    if target_id is None:
                        target_id = _mint_split_row(cur, p, detached)
                        summary["splits_into_new_row"] += 1
                    else:
                        summary["splits_into_existing"] += 1
                    moved, conflicts = _move_entries(
                        cur, [e["entry_id"] for e in polluted_entries], target_id
                    )
                    _log_split(
                        cur, p, target_id, detached, moved, conflicts,
                        method, reason,
                    )
                summary["merged"] += 1
                summary["entries_moved"]          += moved
                summary["entry_conflicts_dropped"] += conflicts
                if summary["merged"] % args.commit_every == 0:
                    conn.commit()
                    log(f"    [commit] batch checkpoint at merged={summary['merged']}")
            except Exception as exc:
                conn.rollback()
                summary["errors"] += 1
                log(f"  ! error on horse {p['horse_id']}: {exc!r}")

        if args.execute:
            conn.commit()

    return 0


if __name__ == "__main__":
    sys.exit(main())
