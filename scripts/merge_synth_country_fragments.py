"""
Collapse ATG *synthetic* person rows that fragment a single human across
countries.

ATG mints a synthetic person row keyed `atg_id = 'x:<CC>:<NAME>'` whenever it
sees a name in a country context it can't map to a real (ST-licensed) person.
A touring trainer therefore ends up scattered across many rows — e.g. Anders
Lundström Wolden appears as x:NO / x:FR / x:DE / x:US … and Rolf Håfvenström as
one real ST row plus eight `x:CC:HAFVENSTRÖM ROLF` synth rows.

This script reunites them, but ONLY when it is safe:

  * Candidates are grouped by `v2_normalize_name(name)` (accent/case/country-suffix
    insensitive).
  * A group is processed only if it has AT MOST ONE non-synthetic "real" row
    (a row with an st_id, a numeric atg_id, or a letrot_id). Groups with >1
    distinct ST license are DISTINCT humans that merely share a name
    (there are ~4,200 such groups — nine different "Lars Andersson" trainers,
    etc.) and are skipped outright.
  * The anchor (keeper) is that single real row if present, else the synth row
    with the most starts.
  * Every other synth row is merged into the anchor ONLY IF it shares at least
    `--min-shared` horses with the anchor. The shared-horse test is what
    distinguishes a genuine touring stable (same horses across borders) from
    two unrelated foreigners who happen to share a name — the latter share 0
    horses and are routed to the review CSV instead.

All merges go through `core.identity.merge_persons` via the standard harness,
so they are logged to `person_merge_log` and fully reversible with
`--rollback-job <id>` (method = 'synth_country_fragments').

Usage
-----
    python -m scripts.merge_synth_country_fragments                 # dry-run
    python -m scripts.merge_synth_country_fragments --execute
    python -m scripts.merge_synth_country_fragments --execute --min-shared 1
    python -m scripts.merge_synth_country_fragments --rollback-job <id>
"""

from __future__ import annotations

import csv
import sys
from collections import defaultdict
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

_METHOD = "synth_country_fragments"


def _fetch_groups(cur, limit: int | None) -> list[dict]:
    """Return per-name-group member rows for groups that contain >=1 synth row
    and at most one distinct real ST license."""
    sql = r"""
    WITH t AS (
      SELECT p.person_id,
             v2_normalize_name(p.name) AS nn,
             p.name, p.st_id, p.atg_id, p.letrot_id,
             (p.atg_id LIKE 'x:%%')                              AS is_synth,
             (SELECT count(*) FROM entry e
                WHERE e.trainer_id = p.person_id
                   OR e.driver_id  = p.person_id)                AS starts
      FROM person p
    ),
    g AS (
      SELECT nn
        FROM t
       WHERE nn <> ''
       GROUP BY nn
      HAVING count(*) FILTER (WHERE is_synth) >= 1          -- has synth fragments
         AND count(DISTINCT st_id) <= 1                     -- <=1 real ST license
         AND count(*) FILTER (WHERE NOT is_synth) <= 1      -- <=1 non-synth row
    )
    SELECT t.nn, t.person_id, t.name, t.st_id, t.atg_id, t.letrot_id,
           t.is_synth, t.starts
      FROM t JOIN g USING (nn)
     ORDER BY t.nn, t.starts DESC
    """
    cur.execute(sql)
    cols = [d.name for d in cur.description]
    rows = [dict(zip(cols, r)) for r in cur.fetchall()]
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        groups[r["nn"]].append(r)
    out = list(groups.values())
    if limit:
        out = out[:limit]
    return out


def _shared_horse_counts(cur, anchor_id: int, other_ids: list[int]) -> dict[int, int]:
    """For each id in other_ids, count distinct horses it shares with anchor
    (as trainer or driver on either side)."""
    if not other_ids:
        return {}
    cur.execute(
        """
        WITH anchor AS (
          SELECT DISTINCT horse_id FROM entry
           WHERE trainer_id = %s OR driver_id = %s
        )
        SELECT e.pid, count(DISTINCT e.horse_id)
        FROM (
          SELECT horse_id, trainer_id AS pid FROM entry WHERE trainer_id = ANY(%s)
          UNION ALL
          SELECT horse_id, driver_id  AS pid FROM entry WHERE driver_id  = ANY(%s)
        ) e
        JOIN anchor a USING (horse_id)
        GROUP BY e.pid
        """,
        (anchor_id, anchor_id, other_ids, other_ids),
    )
    return {pid: n for pid, n in cur.fetchall()}


def main() -> int:
    parser = build_argparser(_METHOD)
    parser.add_argument("--min-shared", type=int, default=2,
                        help="min shared horses for synth-ONLY groups (no real "
                             "anchor) — guards against foreign namesakes "
                             "(default 2)")
    parser.add_argument("--real-anchor-min-shared", type=int, default=1,
                        help="min shared horses when the group has a single "
                             "confirmed real (ST/letrot/numeric-atg) anchor; "
                             "the unique real identity makes 1 conclusive "
                             "(default 1)")
    args = parser.parse_args()

    if args.rollback_job:
        with script_runner(f"{_METHOD}_rollback", args) as (conn, log, summary):
            res = rollback_person_merges_by_method(
                conn, log, _METHOD, job_run_id=args.rollback_job)
            summary.update(res)
        return 0

    out_path = _ROOT / "out" / "synth_country_fragments_review.csv"
    review: list[dict] = []

    with script_runner(_METHOD, args) as (conn, log, summary):
        with conn.cursor() as cur:
            groups = _fetch_groups(cur, args.limit)
        log(f"[{_METHOD}] execute={args.execute} min_shared={args.min_shared} "
            f"groups={len(groups)}")

        planned: list[dict] = []  # {from_id, to_id, shared, nn}
        for members in groups:
            nn = members[0]["nn"]
            reals = [m for m in members if not m["is_synth"]]
            synths = [m for m in members if m["is_synth"]]
            has_real = bool(reals)
            anchor = reals[0] if reals else max(synths, key=lambda m: m["starts"])
            min_shared = args.real_anchor_min_shared if has_real else args.min_shared
            others = [m for m in members if m["person_id"] != anchor["person_id"]]
            # Only synth rows are ever absorbed; never merge a non-synth real away.
            others = [m for m in others if m["is_synth"]]
            if not others:
                continue
            with conn.cursor() as cur:
                shared = _shared_horse_counts(
                    cur, anchor["person_id"], [m["person_id"] for m in others])
            for m in others:
                sh = shared.get(m["person_id"], 0)
                if min_shared and sh < min_shared:
                    review.append({
                        "nn": nn, "anchor_id": anchor["person_id"],
                        "anchor_name": anchor["name"], "from_id": m["person_id"],
                        "from_name": m["name"], "shared": sh,
                        "reason": ("below_min_shared_real_anchor" if has_real
                                   else "below_min_shared_synth_only"),
                    })
                    continue
                planned.append({
                    "from_id": m["person_id"], "to_id": anchor["person_id"],
                    "shared": sh, "nn": nn,
                    "anchor_name": anchor["name"], "from_name": m["name"],
                })

        summary["candidates"] = len(planned)
        log(f"  planned merges: {len(planned)}   review (skipped): {len(review)}")

        if review:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(out_path, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(review[0].keys()))
                w.writeheader()
                w.writerows(review)
            log(f"  wrote review CSV: {out_path} ({len(review)} rows)")

        if not args.execute:
            log("DRY-RUN — no DB changes. Use --execute to apply.")
            # Preview a handful so the operator can eyeball them.
            for p in planned[:25]:
                log(f"    PREVIEW {p['from_id']} -> {p['to_id']}  shared={p['shared']}  "
                    f"{p['from_name']!r} -> {p['anchor_name']!r}")
            return 0

        for p in planned:
            perform_merge(
                conn, log, summary,
                from_id=p["from_id"], to_id=p["to_id"],
                reason=f"synth country fragment shared={p['shared']} nn={p['nn']}",
                method=_METHOD, dry_run=False,
                commit_every=args.commit_every, kind="person",
            )
        conn.commit()
    return 0


if __name__ == "__main__":
    sys.exit(main())
