"""
Phase 2c — merge ST 'foreign-country stub races' into the real race rows.

Background
----------
When a Swedish horse races abroad, ST scrapes only that horse's individual
result and stores it under a synthetic track called the country in Swedish
("Frankrike", "Italien", "Tyskland", "Norge", "Belgien", "Danmark",
"Finland", "Australien"). The race row therefore typically has a single
entry — the visiting horse — while the real race (e.g. on LeTrot or ATG)
has the full field.

`merge_duplicate_races` and `match_french_races` can't bridge these because
their entry-fingerprint Jaccard requires several shared (program_number,
horse_name) tokens. With only one entry on the ST side, the fingerprint is
too weak.

This script bridges the gap using **horse_id co-occurrence** instead of
program-number fingerprinting:

  1. Find every stub race S = a race on a fake-country track with <= N
     entries (default 1).
  2. For each entry e on S (horse_id h, race_date d), find another race R
     on the same date that ALSO has horse_id h as a participant.
  3. If exactly one such R exists for that (h, d) pair AND R has more
     entries than S, treat (S, R) as a candidate pair.
  4. Confidence checks:
       - placement on the stub matches the placement of h in R (when both
         have a value)
       - time_seconds within ±0.5s (when both have a value)
       - prize_kr ratio within 0.5x..2x (FX differences allowed)
       - distance within 100 m (small rounding diffs between sources)
  5. Pass → call `merge_two_races_columnwise(keeper=R, loser=S)`.
     The column-level merger preserves ST's prize_kr/odds on h's entry
     while keeping all of R's other participants.

Rollback
--------
    python -m scripts.match_stub_races --rollback-job <id>

Usage
-----
    python -m scripts.match_stub_races               # dry-run preview
    python -m scripts.match_stub_races --execute --commit-every 100
    python -m scripts.match_stub_races --countries Frankrike,Italien
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts._merge_helpers import (  # noqa: E402
    build_argparser,
    script_runner,
    rollback_horse_merges_by_method,
)


# Tracks that are really country placeholders, not real tracks. ST creates
# these when a horse competes abroad and only the visitor's result row is
# imported. Keep this list in sync with import_st.py.
_FAKE_COUNTRY_TRACKS = (
    "Frankrike", "Italien", "Tyskland", "Norge",
    "Belgien", "Danmark", "Finland", "Australien",
    "Holland", "USA",
)


_CANDIDATES_SQL = """
WITH stub AS (
  SELECT r.race_id            AS stub_race_id,
         r.race_date,
         r.race_number        AS stub_num,
         r.distance           AS stub_dist,
         r.start_method       AS stub_sm,
         r.track_id           AS stub_track_id,
         t.name               AS stub_track,
         t.country            AS stub_country,
         (SELECT COUNT(*) FROM entry e WHERE e.race_id = r.race_id) AS n_entries,
         (SELECT array_agg(e.horse_id) FROM entry e WHERE e.race_id = r.race_id) AS horse_ids
    FROM race r JOIN track t ON r.track_id = t.track_id
   WHERE t.name = ANY(%(tracks)s)
     AND (SELECT COUNT(*) FROM entry e2 WHERE e2.race_id = r.race_id) >= 1
     AND (SELECT COUNT(*) FROM entry e2 WHERE e2.race_id = r.race_id)
         <= %(max_entries)s
),
linked AS (
  SELECT s.stub_race_id, s.race_date, s.stub_num, s.stub_dist, s.stub_sm,
         s.stub_track, s.stub_country, s.n_entries, h_id,
         (SELECT COUNT(*) FROM entry ee WHERE ee.race_id = r2.race_id) AS r2_entries,
         r2.race_id    AS real_race_id,
         r2.race_number AS real_num,
         r2.distance    AS real_dist,
         r2.start_method AS real_sm,
         t2.name        AS real_track,
         t2.country     AS real_country,
         r2.primary_source AS real_src
    FROM stub s
    CROSS JOIN LATERAL unnest(s.horse_ids) AS h_id
    JOIN entry e2 ON e2.horse_id = h_id
    JOIN race  r2 ON r2.race_id  = e2.race_id
    JOIN track t2 ON r2.track_id = t2.track_id
   WHERE r2.race_date = s.race_date
     AND r2.race_id <> s.stub_race_id
     AND t2.name <> ALL(%(tracks)s)
     -- Real race must contain strictly MORE entries than the stub
     -- (otherwise both are equal-fingerprint candidates and
     -- match_french_races should handle them).
     AND (SELECT COUNT(*) FROM entry ee WHERE ee.race_id = r2.race_id)
         > s.n_entries
)
SELECT * FROM linked
 ORDER BY race_date DESC, stub_race_id
"""


def _fetch_candidates(cur, tracks: tuple[str, ...], max_entries: int,
                      limit: int | None) -> list[dict]:
    sql = _CANDIDATES_SQL
    if limit:
        sql += f" LIMIT {int(limit)}"
    cur.execute(sql, {"tracks": list(tracks), "max_entries": max_entries})
    cols = [d.name for d in cur.description]
    return [dict(zip(cols, r)) for r in cur.fetchall()]


def _entry_summary(cur, race_id: int, horse_id: int) -> dict | None:
    """Fetch the participating entry for a single horse on a race."""
    cur.execute(
        "SELECT entry_id, placement, time_seconds, prize_kr, program_number "
        "  FROM entry WHERE race_id = %s AND horse_id = %s",
        (race_id, horse_id),
    )
    row = cur.fetchone()
    if not row:
        return None
    return {
        "entry_id":      row[0],
        "placement":     row[1],
        "time_seconds":  row[2],
        "prize_kr":      row[3],
        "program_number": row[4],
    }


def _looks_like_same_race(stub_entry: dict, real_entry: dict,
                          stub_dist: int | None, real_dist: int | None,
                          ) -> tuple[bool, str]:
    """Return (ok, reason) — confidence checks before merging."""
    # Placement match (when both known)
    sp = stub_entry.get("placement")
    rp = real_entry.get("placement")
    if sp is not None and rp is not None and sp != rp:
        return False, f"placement mismatch {sp}!={rp}"

    # Time within ±0.5s (km-time)
    st_t = stub_entry.get("time_seconds")
    rt_t = real_entry.get("time_seconds")
    if st_t is not None and rt_t is not None:
        if abs(float(st_t) - float(rt_t)) > 0.5:
            return False, f"time diff {st_t}!={rt_t}"

    # Distance within 100m (different sources round differently)
    if stub_dist is not None and real_dist is not None:
        if abs(int(stub_dist) - int(real_dist)) > 100:
            return False, f"distance diff {stub_dist}!={real_dist}"

    # Prize ratio sanity (FX/rounding allowed): if both present, must be
    # within 0.4x..2.5x of each other.
    sp_kr = stub_entry.get("prize_kr")
    rp_kr = real_entry.get("prize_kr")
    if sp_kr and rp_kr:
        ratio = float(sp_kr) / float(rp_kr)
        if not (0.4 <= ratio <= 2.5):
            return False, f"prize ratio {ratio:.2f} out of range"

    return True, "ok"


def main() -> int:
    parser = build_argparser("match_stub_races")
    parser.add_argument(
        "--countries",
        default=",".join(_FAKE_COUNTRY_TRACKS),
        help="comma-separated list of fake-country track names "
             "(default: all known foreign placeholders)",
    )
    parser.add_argument(
        "--max-stub-entries", type=int, default=12,
        help="treat races with up to N entries as stubs (default 12). "
             "Set higher to also fold near-complete stubs into the canonical "
             "race row.",
    )
    args = parser.parse_args()

    tracks = tuple(t.strip() for t in args.countries.split(",") if t.strip())

    if args.rollback_job:
        with script_runner("match_stub_races", args) as (conn, log, summary):
            log(f"[match_stub_races] rolling back job_run_id={args.rollback_job}")
            n = rollback_horse_merges_by_method(
                conn, log, method="match_stub_races",
                job_run_id=args.rollback_job,
            )
            summary["rolled_back"] = n
        return 0

    with script_runner("match_stub_races", args) as (conn, log, summary):
        log(f"[match_stub_races] execute={args.execute} limit={args.limit} "
            f"max_stub_entries={args.max_stub_entries} tracks={list(tracks)}")

        with conn.cursor() as cur:
            cands = _fetch_candidates(cur, tracks, args.max_stub_entries,
                                      args.limit)
        log(f"Found {len(cands)} (stub, real) pre-pairs to evaluate.")

        # Group by stub_race_id — accept only if a unique real race remains
        # after confidence-filtering.
        by_stub: dict[int, list[dict]] = {}
        for c in cands:
            by_stub.setdefault(c["stub_race_id"], []).append(c)

        accepted: list[tuple[dict, dict]] = []  # (stub_pair, real_pair) tuples
        rejected_multi = 0
        rejected_fail  = 0

        for stub_race_id, group in by_stub.items():
            survivors: list[dict] = []
            for c in group:
                with conn.cursor() as cur:
                    stub_e = _entry_summary(cur, c["stub_race_id"], c["h_id"])
                    real_e = _entry_summary(cur, c["real_race_id"], c["h_id"])
                if not stub_e or not real_e:
                    continue
                ok, why = _looks_like_same_race(
                    stub_e, real_e, c["stub_dist"], c["real_dist"]
                )
                if ok:
                    survivors.append({**c, "stub_e": stub_e, "real_e": real_e,
                                      "why": why})
            if not survivors:
                rejected_fail += 1
                continue
            if len({s["real_race_id"] for s in survivors}) > 1:
                rejected_multi += 1
                continue
            accepted.append(survivors[0])

        summary["candidates"] = len(by_stub)
        summary["accepted"]   = len(accepted)
        summary["rejected_multi_real"] = rejected_multi
        summary["rejected_failed_checks"] = rejected_fail
        log(f"{len(by_stub)} stub races, {len(accepted)} accepted, "
            f"{rejected_multi} rejected (multiple real candidates), "
            f"{rejected_fail} rejected (confidence checks failed).")

        # Show preview
        for p in accepted[:25]:
            log(f"  PREVIEW {p['race_date']} stub#{p['stub_race_id']} "
                f"({p['stub_track']}, {p['n_entries']} entry) "
                f"-> real#{p['real_race_id']} ({p['real_track']}/"
                f"{p['real_country']}, {p['r2_entries']} entries, "
                f"src={p['real_src']})")
        if len(accepted) > 25:
            log(f"  ... and {len(accepted) - 25} more")

        if not args.execute:
            summary["merged"] = len(accepted)
            log("\nDRY-RUN — no DB changes. Use --execute to apply.")
            return 0

        from scripts.merge_duplicate_races import (  # noqa: E402
            merge_two_races_columnwise,
        )

        merged = 0
        errs = 0
        for i, p in enumerate(accepted, 1):
            with conn.cursor() as cur:
                res = merge_two_races_columnwise(
                    cur,
                    keeper_id=p["real_race_id"],
                    loser_id=p["stub_race_id"],
                    method="match_stub_races",
                )
            if "error" in res:
                errs += 1
                log(f"  ERROR stub#{p['stub_race_id']}->real#{p['real_race_id']}"
                    f": {res['error']}")
            else:
                merged += 1
                log(f"  merged stub#{p['stub_race_id']} "
                    f"({p['stub_track']}) -> real#{p['real_race_id']} "
                    f"({p['real_track']})  moved={res.get('entries_moved')} "
                    f"conflicts={res.get('conflicts_resolved')}")
            if i % args.commit_every == 0:
                conn.commit()
                log(f"  ... commit ({i}/{len(accepted)} processed)")
        conn.commit()
        summary["merged"] = merged
        summary["errors"] = errs
        log(f"\nDone. merged={merged}  errors={errs}")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
