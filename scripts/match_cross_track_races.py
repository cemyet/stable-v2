"""Merge the same real-world race recorded at two different tracks.

Two sources can file one physical race under different track rows:

  * Moved racedays — a meeting relocated (weather, track conditions); ATG
    keeps the original calendar venue while ST records the actual one
    (e.g. the whole 2026-03-03 card exists as ATG@Gävle AND ST@Solvalla).
  * ATG's rotating foreign slot ids — historical ATG races misfiled at the
    slot-holder track (e.g. ATG@"Ålborg") while the ST copy sits at the
    real venue (Århus/Odense/...).
  * Track name variants that used to fork rows before the squashed-name
    matching fix.

Identity evidence is the STARTERS: a horse cannot race at two tracks on the
same day, so two races on the same date with the same race number sharing
most of their field are the same race. Guards:

  * >= --min-shared shared horse_ids (default 3)
  * Jaccard(entries_a, entries_b) >= --min-jaccard (default 0.5), OR the
    smaller race's starters are fully contained in the larger one — ST's
    copies of foreign racedays often list only the ST-registered subset of
    the field, which tanks Jaccard while still proving identity
  * distances (when both known) within 100 m — else skip for review
  * races at country-placeholder tracks ('Norge', 'Finland', ...) are
    excluded entirely — folding those stubs into real races needs the
    stricter placement/time evidence in scripts.match_stub_races
  * a race appearing in more than one candidate pair is skipped (ambiguous
    evidence, e.g. pre-existing double entries linking three copies)

Keeper choice: the copy whose primary_source is 'st' wins (ST is the
authority on the physical venue); otherwise more entries, then lower
race_id. The loser is column-merged into the keeper via
merge_two_races_columnwise (audited in race_merge_log, method='cross_track').

CLI
---
    python -m scripts.match_cross_track_races                    # dry-run all
    python -m scripts.match_cross_track_races --since-days 14    # window
    python -m scripts.match_cross_track_races --execute
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from etl.dq_sentinel import PLACEHOLDER_TRACK_NAMES  # noqa: E402
from scripts._merge_helpers import build_argparser, script_runner  # noqa: E402
from scripts.merge_duplicate_races import merge_two_races_columnwise  # noqa: E402


def _fetch_candidates(cur, *, since_days: int | None, min_shared: int,
                      limit: int | None) -> list[dict]:
    date_filter = ""
    params: list = [min_shared]
    if since_days:
        date_filter = "AND r1.race_date >= CURRENT_DATE - %s"
        params = [since_days, min_shared]
    sql = f"""
    WITH pairs AS (
        SELECT e1.race_id AS ra, e2.race_id AS rb, COUNT(*) AS shared
          FROM entry e1
          JOIN race r1  ON r1.race_id = e1.race_id
          JOIN entry e2 ON e2.horse_id = e1.horse_id AND e2.race_id > e1.race_id
          JOIN race r2  ON r2.race_id = e2.race_id
         WHERE r1.race_date   = r2.race_date
           AND r1.race_number = r2.race_number
           AND r1.race_number IS NOT NULL
           AND r1.track_id   != r2.track_id
           {date_filter}
         GROUP BY 1, 2
        HAVING COUNT(*) >= %s
    )
    SELECT p.ra, p.rb, p.shared,
           ra.race_date, ra.race_number,
           ra.track_id  AS track_a, rb.track_id  AS track_b,
           ta.name      AS track_a_name, tb.name  AS track_b_name,
           ra.primary_source AS src_a, rb.primary_source AS src_b,
           ra.distance  AS dist_a, rb.distance   AS dist_b,
           (SELECT COUNT(*) FROM entry WHERE race_id = p.ra) AS n_a,
           (SELECT COUNT(*) FROM entry WHERE race_id = p.rb) AS n_b
      FROM pairs p
      JOIN race ra ON ra.race_id = p.ra
      JOIN race rb ON rb.race_id = p.rb
      LEFT JOIN track ta ON ta.track_id = ra.track_id
      LEFT JOIN track tb ON tb.track_id = rb.track_id
     ORDER BY ra.race_date DESC, p.ra
    """
    if limit:
        sql += f" LIMIT {int(limit)}"
    cur.execute(sql, params)
    cols = [d.name for d in cur.description]
    return [dict(zip(cols, r)) for r in cur.fetchall()]


def _pick_keeper(c: dict) -> tuple[int, int]:
    """Return (keeper_race_id, loser_race_id). ST venue wins; then entry
    count; then lower race_id."""
    def rank(race_id, src, n):
        return (0 if src == "st" else 1, -int(n or 0), race_id)

    a = rank(c["ra"], c["src_a"], c["n_a"])
    b = rank(c["rb"], c["src_b"], c["n_b"])
    return (c["ra"], c["rb"]) if a <= b else (c["rb"], c["ra"])


def main() -> int:
    ap = build_argparser("match_cross_track_races")
    ap.add_argument("--since-days", type=int, default=None,
                    help="only consider races within this many days back")
    ap.add_argument("--min-shared", type=int, default=3,
                    help="minimum shared starters (default 3)")
    ap.add_argument("--min-jaccard", type=float, default=0.5,
                    help="minimum starter-set Jaccard overlap (default 0.5)")
    args = ap.parse_args()

    with script_runner("match_cross_track_races", args) as (conn, log, summary):
        log(f"[match_cross_track_races] execute={args.execute} "
            f"since_days={args.since_days} min_shared={args.min_shared}")
        with conn.cursor() as cur:
            cands = _fetch_candidates(
                cur, since_days=args.since_days,
                min_shared=args.min_shared, limit=args.limit,
            )
        summary["candidates"] = len(cands)
        log(f"candidate cross-track pairs: {len(cands)}")

        import re

        def _squash(s):
            return re.sub(r"[^0-9a-zåäö]", "", (s or "").lower())

        placeholders = {_squash(p) for p in PLACEHOLDER_TRACK_NAMES}
        cands = [
            c for c in cands
            if _squash(c["track_a_name"]) not in placeholders
            and _squash(c["track_b_name"]) not in placeholders
        ]
        log(f"after excluding placeholder tracks: {len(cands)}")

        # A race in >1 pair means ambiguous evidence (e.g. three copies
        # chained by pre-existing double entries) — leave for review.
        from collections import Counter
        seen = Counter()
        for c in cands:
            seen[c["ra"]] += 1
            seen[c["rb"]] += 1
        ambiguous = {rid for rid, n in seen.items() if n > 1}
        if ambiguous:
            n_amb = sum(1 for c in cands
                        if c["ra"] in ambiguous or c["rb"] in ambiguous)
            log(f"skipping {n_amb} pairs involving {len(ambiguous)} "
                f"ambiguous races (appear in multiple pairs)")
            summary["ambiguous_pairs"] = n_amb
            cands = [c for c in cands
                     if c["ra"] not in ambiguous and c["rb"] not in ambiguous]

        merged_races = set()
        for c in cands:
            if c["ra"] in merged_races or c["rb"] in merged_races:
                summary["skipped"] += 1
                continue

            union = int(c["n_a"]) + int(c["n_b"]) - int(c["shared"])
            jac = c["shared"] / union if union else 0.0
            label = (f"{c['race_date']} #{c['race_number']} "
                     f"{c['track_a_name']}({c['src_a']},n={c['n_a']}) vs "
                     f"{c['track_b_name']}({c['src_b']},n={c['n_b']}) "
                     f"shared={c['shared']} jac={jac:.2f}")

            contained = c["shared"] == min(int(c["n_a"]), int(c["n_b"]))
            if jac < args.min_jaccard and not contained:
                summary["skipped"] += 1
                log(f"  skip (low jaccard): {label}")
                continue
            if (c["dist_a"] and c["dist_b"]
                    and abs(int(c["dist_a"]) - int(c["dist_b"])) > 100):
                summary["skipped"] += 1
                log(f"  skip (distance mismatch {c['dist_a']} vs "
                    f"{c['dist_b']}): {label}")
                continue

            keeper, loser = _pick_keeper(c)
            log(f"  merge {loser} -> {keeper}: {label}")
            if not args.execute:
                summary["merged"] += 1
                merged_races.add(loser)
                continue

            with conn.cursor() as cur:
                res = merge_two_races_columnwise(
                    cur, keeper_id=keeper, loser_id=loser,
                    method="cross_track",
                )
            if "error" in res:
                summary["errors"] += 1
                log(f"  ! error: {res['error']}")
                continue
            merged_races.add(loser)
            summary["merged"] += 1
            if summary["merged"] % args.commit_every == 0:
                conn.commit()

        if args.execute:
            conn.commit()
    return 0


if __name__ == "__main__":
    sys.exit(main())
