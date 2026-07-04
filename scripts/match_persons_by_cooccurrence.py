"""
Phase 5 — auto-merge driver/trainer person rows that the data clearly
proves are the same person via horse co-occurrence.

If person A (primary_source='atg') and person B (primary_source='letrot')
appear behind the SAME merged horses across many distinct races, they're
almost certainly the same person — independent of what their names look
like.

Algorithm (horse-only join — robust to prior race+entry merges):
    WITH ad AS (
      SELECT DISTINCT horse_id, driver_id AS pid
        FROM entry
       WHERE primary_source='atg' AND driver_id IS NOT NULL
    ), ld AS (
      SELECT DISTINCT horse_id, driver_id AS pid
        FROM entry
       WHERE primary_source='letrot' AND driver_id IS NOT NULL
    )
    SELECT ad.pid, ld.pid, count(*) AS shared_horses
      FROM ad JOIN ld USING (horse_id)
     WHERE ad.pid <> ld.pid
     GROUP BY 1, 2
    HAVING count(*) >= <min_shared>

Why horse-only (not horse + race)? Earlier phases column-merge the
LeTrot-side entry into the ATG-side entry when both sources covered the
same race, so post-merge there are no (horse_id, race_id) pairs with
entries from both sources. The horse-only signal is preserved and is
still extremely diagnostic: 30+ shared horses between two French
drivers is essentially impossible by chance.

Sanity guard:
  - Never auto-merge two persons that both have a non-NULL `st_id`
    pointing at different ST persons (those are confirmed-distinct
    Swedish licensed people, not aliases).

Tiers (defaults reflect the horse-only signal strength):
  - shared_horses ≥ --min-shared (default 30) → auto-merge
  - --review-min (default 10) ≤ shared_horses < min-shared → review CSV

Roles handled: driver and trainer (independently).

Usage
-----

    python -m scripts.match_persons_by_cooccurrence              # dry-run
    python -m scripts.match_persons_by_cooccurrence --execute --min-shared 30
    python -m scripts.match_persons_by_cooccurrence --rollback-job <id>
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts._merge_helpers import (  # noqa: E402
    build_argparser,
    script_runner,
    perform_merge,
    rollback_person_merges_by_method,
)

_ROLES = ("driver", "trainer")


# ---------------------------------------------------------------------------
# Pair discovery
# ---------------------------------------------------------------------------

def _fetch_pairs(cur, role: str, min_shared: int,
                 limit: int | None) -> list[dict]:
    """Find (atg_person_id, letrot_person_id, shared_horses) tuples.

    Uses the horse-only join (not horse+race) because earlier phases
    column-merge the LeTrot entry into the ATG entry for any race both
    sources cover, leaving no (horse_id, race_id) pairs with both
    primary_sources represented. The horse-only signal is preserved
    across merges and is highly diagnostic at threshold ≥ ~30.
    """
    fk = f"{role}_id"
    sql = f"""
    WITH ad AS (
      SELECT DISTINCT horse_id, {fk} AS pid
        FROM entry
       WHERE primary_source = 'atg' AND {fk} IS NOT NULL
    ), ld AS (
      SELECT DISTINCT horse_id, {fk} AS pid
        FROM entry
       WHERE primary_source = 'letrot' AND {fk} IS NOT NULL
    )
    SELECT ad.pid AS atg_pid,
           ld.pid AS letrot_pid,
           count(*) AS shared
      FROM ad JOIN ld USING (horse_id)
     WHERE ad.pid <> ld.pid
     GROUP BY 1, 2
    HAVING count(*) >= %s
     ORDER BY shared DESC, atg_pid
    """
    if limit:
        sql += f" LIMIT {int(limit)}"
    cur.execute(sql, (min_shared,))
    cols = [d.name for d in cur.description]
    return [dict(zip(cols, r)) for r in cur.fetchall()]


# ---------------------------------------------------------------------------
# Safety guard: don't merge two confirmed-distinct ST-licensed persons.
# ---------------------------------------------------------------------------

def _safe_to_merge(cur, atg_pid: int, letrot_pid: int) -> tuple[bool, str]:
    cur.execute(
        "SELECT person_id, st_id, atg_id, letrot_id, name FROM person "
        " WHERE person_id IN (%s, %s)",
        (atg_pid, letrot_pid),
    )
    cols = [d.name for d in cur.description]
    rows = [dict(zip(cols, r)) for r in cur.fetchall()]
    if len(rows) != 2:
        return False, f"missing person row(s) (got {len(rows)})"
    by_id = {r["person_id"]: r for r in rows}
    a = by_id.get(atg_pid)
    l = by_id.get(letrot_pid)
    if not a or not l:
        return False, "missing one side"
    if a["st_id"] and l["st_id"] and a["st_id"] != l["st_id"]:
        return False, f"both have ST id ({a['st_id']} vs {l['st_id']}) — distinct"
    if a["letrot_id"] and l["letrot_id"] and a["letrot_id"] != l["letrot_id"]:
        return False, "both have LeTrot id and they differ"
    return True, ""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = build_argparser("match_persons_by_cooccurrence")
    parser.add_argument("--min-shared", type=int, default=30,
                        help="minimum shared horses for auto-merge "
                             "(default 30 — at this threshold the chance of "
                             "two distinct French trotting persons sharing "
                             "this many horses is negligible)")
    parser.add_argument("--review-min", type=int, default=10,
                        help="minimum shared horses for review-CSV tier "
                             "(default 10)")
    args = parser.parse_args()

    if args.rollback_job:
        return _rollback(args)

    out_path = _ROOT / "out" / "match_persons_review.csv"

    with script_runner("match_persons_by_cooccurrence", args) as (conn, log, summary):
        log(f"[match_persons_by_cooccurrence] execute={args.execute} "
            f"min_shared={args.min_shared} review_min={args.review_min}")
        review_rows: list[dict] = []
        definite_by_role: dict[str, list[dict]] = {}

        for role in _ROLES:
            with conn.cursor() as cur:
                # Auto-merge tier.
                auto_pairs = _fetch_pairs(cur, role, args.min_shared, args.limit)
                # Review tier.
                rev_pairs = _fetch_pairs(cur, role, args.review_min, args.limit)
            # Filter review_pairs to only those below auto-threshold.
            rev_pairs = [p for p in rev_pairs if p["shared"] < args.min_shared]

            log(f"  role={role}: {len(auto_pairs)} auto-candidates, "
                f"{len(rev_pairs)} review-candidates")

            # Dominance check: a pair is only safe to auto-merge if the
            # runner-up alternative for BOTH sides is substantially
            # weaker. Otherwise the pairing is ambiguous (e.g. one
            # popular horse driven by many people on each side creates
            # a combinatorial cloud of medium-strength pairs).
            DOMINANCE_RATIO = 0.5  # runner-up must be < 50 % of best
            best_for_atg: dict[int, int] = {}  # atg_pid -> best shared
            second_for_atg: dict[int, int] = {}
            best_for_letrot: dict[int, int] = {}
            second_for_letrot: dict[int, int] = {}
            for p in auto_pairs:
                a, l, s = p["atg_pid"], p["letrot_pid"], p["shared"]
                if s > best_for_atg.get(a, 0):
                    second_for_atg[a] = best_for_atg.get(a, 0)
                    best_for_atg[a] = s
                elif s > second_for_atg.get(a, 0):
                    second_for_atg[a] = s
                if s > best_for_letrot.get(l, 0):
                    second_for_letrot[l] = best_for_letrot.get(l, 0)
                    best_for_letrot[l] = s
                elif s > second_for_letrot.get(l, 0):
                    second_for_letrot[l] = s

            # Greedy 1-1 matching on the dominance-passing subset.
            seen_atg: set[int] = set()
            seen_letrot: set[int] = set()
            unique_pairs: list[dict] = []
            for p in sorted(auto_pairs, key=lambda x: (-x["shared"], x["atg_pid"])):
                a, l, s = p["atg_pid"], p["letrot_pid"], p["shared"]
                # Must be both sides' best AND the runner-up on either
                # side must be << best for it to be unambiguous.
                if s != best_for_atg[a] or s != best_for_letrot[l]:
                    review_rows.append({
                        "role": role, "atg_pid": a, "letrot_pid": l,
                        "shared": s,
                        "skipped_reason": "not_mutual_best",
                    })
                    continue
                if (second_for_atg.get(a, 0) >= DOMINANCE_RATIO * s
                        or second_for_letrot.get(l, 0) >= DOMINANCE_RATIO * s):
                    review_rows.append({
                        "role": role, "atg_pid": a, "letrot_pid": l,
                        "shared": s,
                        "skipped_reason": (
                            f"ambiguous (runner-up ≥ {DOMINANCE_RATIO:.0%} "
                            f"of best: atg2nd={second_for_atg.get(a, 0)} "
                            f"letrot2nd={second_for_letrot.get(l, 0)})"
                        ),
                    })
                    continue
                if a in seen_atg or l in seen_letrot:
                    review_rows.append({
                        "role": role, "atg_pid": a, "letrot_pid": l,
                        "shared": s,
                        "skipped_reason": "side_already_claimed",
                    })
                    continue
                seen_atg.add(a)
                seen_letrot.add(l)
                unique_pairs.append(p)

            # Safety-filter auto pairs (ST-id conflict, etc.).
            safe = []
            for p in unique_pairs:
                with conn.cursor() as cur:
                    ok, reason = _safe_to_merge(cur, p["atg_pid"], p["letrot_pid"])
                if ok:
                    safe.append(p)
                else:
                    review_rows.append({"role": role,
                                        "atg_pid": p["atg_pid"],
                                        "letrot_pid": p["letrot_pid"],
                                        "shared": p["shared"],
                                        "skipped_reason": reason})
            definite_by_role[role] = safe
            for p in rev_pairs:
                review_rows.append({"role": role,
                                    "atg_pid": p["atg_pid"],
                                    "letrot_pid": p["letrot_pid"],
                                    "shared": p["shared"],
                                    "skipped_reason": "below_auto_threshold"})

        # Write review CSV.
        if review_rows:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(out_path, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["role", "atg_pid", "letrot_pid", "shared",
                            "skipped_reason"])
                for r in review_rows:
                    w.writerow([r["role"], r["atg_pid"], r["letrot_pid"],
                                r["shared"], r["skipped_reason"]])
            log(f"Wrote review CSV: {out_path} ({len(review_rows)} rows)")

        total_auto = sum(len(v) for v in definite_by_role.values())
        summary["candidates"] = total_auto
        log(f"\n{total_auto} total auto-merges across {len(_ROLES)} roles")

        if not args.execute:
            summary["merged"] = total_auto
            log("DRY-RUN — no DB changes. Use --execute to apply.")
            return 0

        for role, pairs in definite_by_role.items():
            log(f"\n  Executing {len(pairs)} {role} merges...")
            for p in pairs:
                perform_merge(
                    conn, log, summary,
                    from_id=p["letrot_pid"],
                    to_id=p["atg_pid"],
                    reason=f"{role} co-occurrence shared={p['shared']}",
                    method="match_persons_by_cooccurrence",
                    dry_run=False,
                    commit_every=args.commit_every,
                    kind="person",
                )
        conn.commit()
    return 0


def _rollback(args) -> int:
    with script_runner("match_persons_rollback", args) as (conn, log, summary):
        res = rollback_person_merges_by_method(
            conn, log, "match_persons_by_cooccurrence",
            job_run_id=args.rollback_job,
        )
        summary.update(res)
    return 0


if __name__ == "__main__":
    sys.exit(main())
