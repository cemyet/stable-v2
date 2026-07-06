"""
Phase 2 — find races stored as separate rows under different tracks.

Background: when ATG and LeTrot both record a French race but disagree on
the track ("Le Mans" vs "Vincennes"), we end up with two race rows that
have the same date + race_number but different track_id. This script
finds those pairs by fingerprinting their entry lists (program_number +
normalized horse name) and emits merge candidates.

Resolution preference:
  - If the keeper has a non-FR track but the loser is LeTrot on an FR
    track, switch keeper.track_id to loser's (LeTrot is authoritative
    for French tracks). The actual switch happens in
    `scripts.merge_duplicate_races` after this script flags the pair.
  - Otherwise keep keeper's track_id.

Usage
-----

    python -m scripts.match_french_races              # dry-run, top 200
    python -m scripts.match_french_races --execute   # apply merges
    python -m scripts.match_french_races --min-score 0.85 --limit 1000
    python -m scripts.match_french_races --rollback-job <id>
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from psycopg2.extras import Json  # noqa: E402

from core.identity import normalize_name  # noqa: E402
from scripts._merge_helpers import (  # noqa: E402
    build_argparser,
    script_runner,
)


# ---------------------------------------------------------------------------
# Candidate discovery
# ---------------------------------------------------------------------------

_CANDIDATE_SQL = """
WITH cand AS (
  SELECT r_a.race_id  AS atg_race_id,
         r_l.race_id  AS letrot_race_id,
         r_a.race_date,
         r_a.race_number,
         r_a.track_id AS atg_track_id,
         r_l.track_id AS letrot_track_id,
         r_a.distance AS atg_distance,
         r_l.distance AS letrot_distance,
         t_a.country  AS atg_country,
         t_l.country  AS letrot_country
    FROM race r_a
    JOIN race r_l
      ON r_l.race_date    = r_a.race_date
     AND r_l.race_number  = r_a.race_number
     AND r_l.race_id <> r_a.race_id
     AND r_l.track_id  IS NOT NULL
     AND r_a.track_id  IS NOT NULL
     AND r_l.track_id <> r_a.track_id
    LEFT JOIN track t_a ON r_a.track_id = t_a.track_id
    LEFT JOIN track t_l ON r_l.track_id = t_l.track_id
   WHERE r_a.primary_source = 'atg'
     AND r_l.primary_source = 'letrot'
     AND r_a.atg_race_id    IS NOT NULL
     AND r_l.letrot_race_id IS NOT NULL
     -- Only races touching France: either side must be on a French track.
     AND (t_a.country = 'FR' OR t_l.country = 'FR')
)
SELECT * FROM cand
 {where_recent}
 ORDER BY race_date DESC
"""


def _fetch_candidates(cur, limit: int | None,
                      since_days: int | None = None) -> list[dict]:
    # Nightly runs pass a small `since_days` so we only fingerprint the races
    # just ingested. The unscoped join spans ALL history (LeTrot's full French
    # backfill × ATG internationals => ~260k date+race_number pairs since 2020),
    # each needing two per-race fingerprint queries — hours of pointless work
    # re-checking pairs that were already resolved (or never matched) long ago.
    params: list = []
    if since_days is not None:
        where_recent = "WHERE race_date >= CURRENT_DATE - %s::int"
        params.append(int(since_days))
    else:
        where_recent = ""
    sql = _CANDIDATE_SQL.format(where_recent=where_recent)
    if limit:
        sql += f" LIMIT {int(limit)}"
    cur.execute(sql, params)
    cols = [d.name for d in cur.description]
    return [dict(zip(cols, r)) for r in cur.fetchall()]


def _fingerprint(cur, race_id: int) -> list[str]:
    """List of 'program:NORMALIZED_NAME' tokens, sorted by program number."""
    cur.execute(
        """
        SELECT e.program_number, h.name
          FROM entry e
          JOIN horse h ON e.horse_id = h.horse_id
         WHERE e.race_id = %s
         ORDER BY e.program_number NULLS LAST, h.name
        """,
        (race_id,),
    )
    out: list[str] = []
    for prog, name in cur.fetchall():
        prog_s = str(prog) if prog is not None else "?"
        out.append(f"{prog_s}:{normalize_name(name)}")
    return out


def _jaccard(a: list[str], b: list[str]) -> float:
    """Jaccard similarity on the program:name token sets."""
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 0.0
    inter = len(sa & sb)
    union = len(sa | sb)
    return inter / union if union else 0.0


def _prog_only_jaccard(a: list[str], b: list[str]) -> float:
    """Jaccard on just program numbers — robustness check when names diverge
    across sources (e.g. spelling differences). Used as a secondary signal."""
    def progs(tokens):
        return {t.split(":", 1)[0] for t in tokens if ":" in t and t.split(":", 1)[0] != "?"}
    sa, sb = progs(a), progs(b)
    if not sa and not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def _pick_canonical_track(c: dict) -> tuple[int, str]:
    """Return (preferred_track_id, justification).

    For French races, LeTrot is authoritative. For others, prefer the
    keeper (ATG) as-is.
    """
    if c["letrot_country"] == "FR":
        return c["letrot_track_id"], "french_race_letrot_track"
    return c["atg_track_id"], "default_keeper_track"


def main() -> int:
    parser = build_argparser("match_french_races")
    parser.add_argument("--min-score", type=float, default=0.5,
                        help="minimum full-token Jaccard score (program:name) "
                             "to accept a pair as the same race (default 0.5)")
    parser.add_argument("--min-name-overlap", type=int, default=2,
                        help="minimum number of shared program:name tokens "
                             "required no matter what (default 2). Prevents "
                             "false positives from races that share only "
                             "their program-number set (1..12 etc.)")
    parser.add_argument("--since-days", type=int, default=None,
                        help="only consider races run within the last N days "
                             "(default: all history). The nightly passes a "
                             "small window so it never rescans years of French "
                             "history every run.")
    args = parser.parse_args()
    min_score = args.min_score
    min_overlap = args.min_name_overlap

    if args.rollback_job:
        return _rollback(args)

    with script_runner("match_french_races", args) as (conn, log, summary):
        log(f"[match_french_races] execute={args.execute} limit={args.limit} "
            f"min_score={min_score} since_days={args.since_days}")
        with conn.cursor() as cur:
            cands = _fetch_candidates(cur, args.limit, since_days=args.since_days)
        summary["candidates"] = len(cands)
        log(f"Found {len(cands)} cross-track candidates to fingerprint.")

        keep_pairs: list[dict] = []
        for c in cands:
            with conn.cursor() as cur:
                fp_a = _fingerprint(cur, c["atg_race_id"])
                fp_l = _fingerprint(cur, c["letrot_race_id"])
            score = _jaccard(fp_a, fp_l)
            prog_score = _prog_only_jaccard(fp_a, fp_l)
            shared = len(set(fp_a) & set(fp_l))
            # Skip empty fields (no entries in one side).
            if not fp_a or not fp_l:
                summary["skipped"] += 1
                continue
            # Must share at least `min_overlap` actual program:name tokens.
            # Without this guard, races with disjoint horse rosters that
            # happen to share program-number set (1..12) score prog_score
            # = 1.0 with score = 0.0 — pure false positives.
            if shared < min_overlap:
                continue
            if score < min_score:
                continue
            keep_pairs.append({
                **c,
                "score":      round(score, 3),
                "prog_score": round(prog_score, 3),
                "shared":     shared,
                "n_a": len(fp_a),
                "n_l": len(fp_l),
            })

        log(f"\n{len(keep_pairs)} pairs scored above threshold "
            f"(score >= {min_score}, shared_horses >= {min_overlap}):")
        for p in keep_pairs[:50]:
            log(f"  {p['race_date']} #{p['race_number']}  "
                f"atg={p['atg_race_id']} (track {p['atg_track_id']}/{p['atg_country']}) "
                f"<-> letrot={p['letrot_race_id']} (track {p['letrot_track_id']}/{p['letrot_country']})  "
                f"score={p['score']} shared={p['shared']}/{p['n_a']}/{p['n_l']}")
        if len(keep_pairs) > 50:
            log(f"  ... and {len(keep_pairs) - 50} more")

        if not args.execute:
            summary["merged"] = len(keep_pairs)  # dry-run preview count
            log("\nDRY-RUN — no DB changes. Use --execute to apply.")
            return 0

        # Execute: align track_id (when needed), then call the upgraded
        # merge_duplicate_races helper to do the column-level row merge.
        from scripts.merge_duplicate_races import (  # noqa: E402
            merge_two_races_columnwise,
        )

        for p in keep_pairs:
            preferred_track, justification = _pick_canonical_track(p)
            keeper_id = p["atg_race_id"]
            loser_id  = p["letrot_race_id"]

            with conn.cursor() as cur:
                # If keeper's current track doesn't match preferred,
                # switch it BEFORE the merge (preserve old in source_data
                # for audit). Switching the loser's track to keeper's would
                # also work, but switching the keeper is the explicit
                # "French is canonical" signal.
                cur.execute("SELECT track_id, source_data FROM race "
                            "WHERE race_id = %s", (keeper_id,))
                cur_track, cur_sd = cur.fetchone()
                if cur_track != preferred_track:
                    sd = cur_sd or {}
                    hist = list(sd.get("_track_history") or [])
                    hist.append({"prev_track_id": cur_track,
                                 "new_track_id": preferred_track,
                                 "reason": justification,
                                 "method": "match_french_races"})
                    sd["_track_history"] = hist
                    cur.execute(
                        "UPDATE race SET track_id = %s, source_data = %s, "
                        "last_updated_at = NOW() WHERE race_id = %s",
                        (preferred_track, Json(sd), keeper_id),
                    )
                # Loser also needs to point at the preferred track so the
                # subsequent column-level race merge sees them as consistent.
                cur.execute(
                    "UPDATE race SET track_id = %s WHERE race_id = %s "
                    "AND track_id <> %s",
                    (preferred_track, loser_id, preferred_track),
                )

                res = merge_two_races_columnwise(
                    cur,
                    keeper_id=keeper_id,
                    loser_id=loser_id,
                    method="match_french_races",
                )
            if "error" in res:
                summary["errors"] += 1
                log(f"  ! error merging {loser_id} -> {keeper_id}: {res['error']}")
                continue
            summary["merged"] += 1
            log(f"  merged race {loser_id} -> {keeper_id} "
                f"({res.get('moved')} moved, {res.get('conflicts')} conflicts)")

            if summary["merged"] % args.commit_every == 0:
                conn.commit()

        conn.commit()
    return 0


def _rollback(args) -> int:
    """Rollback all race merges performed under method='match_french_races'.

    Race-row merges are recorded in `race.source_data._merges[]` (not in
    horse_merge_log). Each entry carries a `from_snapshot` of the loser
    race row plus the moved entry count. Rolling back means re-inserting
    the loser race row + restoring its entries.

    NOTE: same-race entry conflicts that were column-merged into the keeper
    cannot be perfectly reversed here without per-entry snapshots in the
    race-merge audit; those are kept as-is (a future enhancement could
    extend `merge_two_races_columnwise` to stash entry audit blocks too).
    For now this rollback returns the loser race row and any entries that
    still exist on it; column-merged entry changes on the keeper persist.
    """
    from scripts._merge_helpers import script_runner as _runner
    with _runner("match_french_races_rollback", args) as (conn, log, summary):
        log(f"[rollback] job_run_id={args.rollback_job}")
        if args.rollback_job is None:
            log("Provide --rollback-job <id>. Aborting.")
            return 2
        # Find merges in any race.source_data._merges that match method
        # and (if job_run available) the time window.
        with conn.cursor() as cur:
            cur.execute("SELECT started_at, finished_at FROM job_run "
                        "WHERE job_run_id=%s", (args.rollback_job,))
            row = cur.fetchone()
            if not row:
                log(f"  no job_run row for id={args.rollback_job}")
                return 2
            started, finished = row

            cur.execute(
                """
                SELECT r.race_id, m
                  FROM race r,
                       jsonb_array_elements(coalesce(r.source_data->'_merges','[]'::jsonb)) AS m
                 WHERE (m->>'method') = 'match_french_races'
                   AND (m->>'merged_at')::timestamp BETWEEN %s AND COALESCE(%s, NOW())
                """,
                (started, finished),
            )
            entries = cur.fetchall()

        log(f"Found {len(entries)} race-merge audit rows in window")
        log("Race-merge rollback requires manual review; see MERGE_RUN_CHECKLIST.md "
            "section 3 for pg_dump restore steps.")
        summary["candidates"] = len(entries)
    return 0


if __name__ == "__main__":
    sys.exit(main())
