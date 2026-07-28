"""
Detach entries that a horse cannot possibly have run.

A horse with two entries on the same date at tracks in different countries is
holding at least one result that is not its own. The usual cause is a foreign
race imported with name-only matching: a French "Lucky Love" runs at Le Mans,
no strong id resolves, and the result lands on the Swedish "Lucky Love" that
happened to be racing at Mantorp the same afternoon.

Adjudication is not heuristic. Travsport's own per-horse race-results list the
raceDayId the horse actually started on, and every race row carries
`st_race_day_id`. Exactly one of the two races will match; that one is real and
the other entry is the intruder. Pairs where ST covers both racedays, neither,
or has nothing for the date are left alone — a wrong split is worse than a
known-bad pair we can revisit.

What happens to the intruder depends on whether it is a second horse's result
or a second copy of the same one. When the two entries carry the same finishing
position, program number and race number, they are one event recorded twice —
a Norwegian meeting that also landed on a Swedish card — and the loser is
deleted, because the surviving entry already holds every field. Re-homing those
would mint a horse that never existed, an `x:SE:SOLLI VIKING* (NO)`.

Genuine collisions get the opposite treatment: the entry moves to the
`x:CC:NAME` synthetic row for the track's country, the convention the ATG
importer already uses for unidentified foreign horses. The start stays
queryable and a later strong-id match can still fold it into the real animal,
where deleting would discard a result we cannot re-derive without re-scraping.

    python -m scripts.split_cross_country_entries              # dry-run
    python -m scripts.split_cross_country_entries --scrape     # fetch evidence
    python -m scripts.split_cross_country_entries --execute
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from psycopg2.extras import Json  # noqa: E402

from scripts._merge_helpers import build_argparser, script_runner  # noqa: E402

# Country-named "tracks" ST invents for foreign starts it will not name a venue
# for. They are excluded on both sides: a stub pairing with a real track is the
# stub matcher's job, and stub-vs-stub says nothing about identity.
PLACEHOLDER_TRACKS = (
    "Frankrike", "Italien", "Tyskland", "Norge", "Belgien", "Danmark",
    "Finland", "Australien", "Holland", "USA", "Osterrike", "Österrike",
)


def find_pairs(cur) -> list[dict]:
    """Same horse, same day, two tracks in different countries."""
    cur.execute(
        """
        WITH ent AS (
            SELECT e.entry_id, e.horse_id, r.race_date, r.race_id,
                   r.st_race_day_id, t.name AS track, t.country,
                   e.placement, e.program_number, r.race_number,
                   e.time_text, e.driver_id
              FROM entry e
              JOIN race  r ON r.race_id  = e.race_id
              JOIN track t ON t.track_id = r.track_id
             WHERE t.country IS NOT NULL
               AND t.name <> ALL(%s)
        )
        SELECT a.horse_id, h.name, h.st_id, a.race_date,
               a.entry_id, a.race_id, a.st_race_day_id, a.track, a.country,
               b.entry_id, b.race_id, b.st_race_day_id, b.track, b.country,
               a.placement, a.program_number, a.race_number, a.time_text, a.driver_id,
               b.placement, b.program_number, b.race_number, b.time_text, b.driver_id
          FROM ent a
          JOIN ent b ON b.horse_id = a.horse_id
                    AND b.race_date = a.race_date
                    AND b.race_id > a.race_id
          JOIN horse h ON h.horse_id = a.horse_id
         WHERE a.country <> b.country
         ORDER BY a.horse_id, a.race_date
        """,
        (list(PLACEHOLDER_TRACKS),),
    )
    cols = ("horse_id", "name", "st_id", "race_date",
            "a_entry", "a_race", "a_rd", "a_track", "a_country",
            "b_entry", "b_race", "b_rd", "b_track", "b_country",
            "a_plc", "a_prog", "a_rn", "a_time", "a_driver",
            "b_plc", "b_prog", "b_rn", "b_time", "b_driver")
    out = []
    for r in cur.fetchall():
        p = dict(zip(cols, r))
        p["same_result"], p["agree"] = _is_same_event(p)
        out.append(p)
    return out


def _norm_time(t) -> str | None:
    """Strip the trailing start-method/gallop marker from a km time.

    ATG writes the same run as '17,2' from one card and '17,2a' from another,
    so a raw comparison reports a difference that isn't one.
    """
    if not t:
        return None
    s = str(t).strip().lower().rstrip("agku ")
    return s or None


def _is_same_event(p: dict) -> tuple[bool, list[str]]:
    """Is this one race recorded twice, rather than two horses' results?

    Scored rather than all-or-nothing. Duplicated cards routinely disagree on a
    single field — one source records the Bergen placement, the other the
    Bollnäs one — so demanding exact agreement misclassifies them as separate
    horses and would mint a synthetic row for an animal that never existed.
    Three independent agreements is the threshold; a genuine collision between
    two same-named horses reaches at most one by chance.

    Only fields present on both sides vote. Two nulls are not evidence.
    """
    agree: list[str] = []
    if p["a_rn"] is not None and p["a_rn"] == p["b_rn"]:
        agree.append("race#")
    if p["a_prog"] is not None and p["a_prog"] == p["b_prog"]:
        agree.append("program#")
    if p["a_plc"] is not None and p["a_plc"] == p["b_plc"]:
        agree.append("placement")
    ta, tb = _norm_time(p["a_time"]), _norm_time(p["b_time"])
    if ta is not None and ta == tb:
        agree.append("time")
    if p["a_driver"] is not None and p["a_driver"] == p["b_driver"]:
        agree.append("driver")
    return len(agree) >= 3, agree


def st_racedays_for(cur, st_id: int | None, date) -> set[int] | None:
    """raceDayIds ST lists for this horse on this date, or None if no evidence.

    An empty set is meaningful and distinct from None: ST has the horse's
    results and does not place it anywhere that day, which by itself condemns
    both entries. We still return it and let the caller refuse to act, because
    "ST has never heard of this start" is also what a stale scrape looks like.
    """
    if not st_id:
        return None
    cur.execute(
        "SELECT raw_json FROM st_horse_raw "
        " WHERE horse_id = %s AND data_type = 'race-results' LIMIT 1",
        (st_id,),
    )
    row = cur.fetchone()
    if not row:
        return None
    raw = row[0]
    if isinstance(raw, str):
        raw = json.loads(raw)
    if not isinstance(raw, list):
        return None
    out: set[int] = set()
    for el in raw:
        info = (el or {}).get("raceInformation") or {}
        if str(info.get("date")) == str(date):
            rd = info.get("raceDayId")
            if rd is not None:
                out.add(int(rd))
    return out


def adjudicate(pair: dict, st_days: set[int] | None) -> tuple[str, str]:
    """Return (verdict, why). Verdict is 'a', 'b' or 'skip'.

    'a' means the A-side entry is real and B is the intruder.
    """
    if st_days is None:
        return "skip", "no ST race-results for this horse"
    if not st_days:
        return "skip", "ST lists no start on this date"
    a_ok = pair["a_rd"] in st_days
    b_ok = pair["b_rd"] in st_days
    if a_ok and not b_ok:
        return "a", f"ST raceday {pair['a_rd']} matches {pair['a_track']}"
    if b_ok and not a_ok:
        return "b", f"ST raceday {pair['b_rd']} matches {pair['b_track']}"
    if a_ok and b_ok:
        return "skip", "both racedays appear in ST — needs manual review"
    return "skip", f"neither raceday in ST {sorted(st_days)}"


def synth_key(country: str, name: str) -> str:
    return f"x:{country.upper()}:{name.upper()}"


def _name_country(name: str) -> str | None:
    """The '(NO)' in 'Philip Lyn (NO)' — ST's own note of a horse's origin."""
    n = (name or "").strip()
    if n.endswith(")") and "(" in n:
        tag = n[n.rfind("(") + 1: -1].strip().upper()
        if len(tag) == 2 and tag.isalpha():
            return tag
    return None


def _strip_country(name: str) -> str:
    """Drop the trailing origin marker; the synthetic key already names a country.

    It is not merely redundant. The result belongs to a same-named horse from
    the track's own country, so a marker copied off the polluted row describes
    the wrong animal — 'Hard Times (NO)' becoming x:SE:HARD TIMES (NO).
    """
    return name[: name.rfind("(")].strip() if _name_country(name) else name


def find_or_create_synth(cur, country: str, name: str,
                         *, execute: bool) -> tuple[int | None, bool]:
    """Locate the `x:CC:NAME` row for a country, creating it when absent."""
    key = synth_key(country, name)
    cur.execute("SELECT horse_id FROM horse WHERE atg_id = %s", (key,))
    row = cur.fetchone()
    if row:
        return row[0], False
    if not execute:
        return None, True
    cur.execute(
        """
        INSERT INTO horse (name, atg_id, birth_country, primary_source,
                           last_updated_at)
        VALUES (%s, %s, %s, 'atg', NOW())
        RETURNING horse_id
        """,
        (name, key, country.upper()),
    )
    return cur.fetchone()[0], True


def _entry_row(cur, entry_id: int) -> dict:
    """Full entry as JSON-safe scalars, so a delete can be undone by hand."""
    cur.execute("SELECT * FROM entry WHERE entry_id = %s", (entry_id,))
    row = cur.fetchone()
    if not row:
        return {}
    cols = [d.name for d in cur.description]
    out: dict = {}
    for k, v in zip(cols, row):
        out[k] = v if v is None or isinstance(
            v, (str, int, float, bool, dict, list)) else str(v)
    return out


def _log_action(cur, p: dict, *, dest: int | None, row: dict | None = None) -> None:
    """Audit row. `dest=None` records a delete, keyed to the surviving entry.

    horse_merge_log is reused rather than given its own table because these are
    entry-level corrections to a horse's history and belong on the same trail
    the merge tooling already reads. Nothing here is reversible by
    `rollback_horse_merge` — no horse row is removed — so the snapshot carries
    enough to reconstruct the change by hand.
    """
    action = "deleted as a duplicate copy" if dest is None else (
        f"re-homed to the {p['bad_country']} synthetic row")
    cur.execute(
        """
        INSERT INTO horse_merge_log
            (from_horse_id, to_horse_id, reason, method,
             entries_moved, conflicts_resolved, from_snapshot, merged_by)
        VALUES (%s, %s, %s, 'split_cross_country', %s, %s, %s,
                'split_cross_country')
        """,
        (
            p["horse_id"], dest if dest is not None else p["horse_id"],
            f"horse {p['horse_id']} ({p['name']!r}) held entries at "
            f"{p['a_track']} ({p['a_country']}) and {p['b_track']} "
            f"({p['b_country']}) on {p['race_date']}; {p['why']}, so entry "
            f"{p['bad_entry']} at {p['bad_track']} is not its own and was "
            f"{action}",
            0 if dest is None else 1,
            1 if dest is None else 0,
            Json({
                "entry_id": p["bad_entry"],
                "race_id": p["bad_race"],
                "race_date": str(p["race_date"]),
                "from_horse_id": p["horse_id"],
                "to_horse_id": dest,
                "action": "delete" if dest is None else "rehome",
                "kept_entry_id": p["keep_entry"],
                "kept_track": p["keep_track"],
                "moved_track": p["bad_track"],
                "evidence": p["why"],
                "agrees_on": p.get("agree"),
                "deleted_entry_row": row,
            }),
        ),
    )


def main() -> int:
    parser = build_argparser("split_cross_country_entries")
    parser.add_argument("--scrape", action="store_true",
                        help="fetch missing ST race-results before adjudicating")
    parser.add_argument("--scrape-delay", type=float, default=None,
                        help="override per-request delay when scraping")
    args = parser.parse_args()

    with script_runner("split_cross_country_entries", args) as (conn, log, summary):
        with conn.cursor() as cur:
            pairs = find_pairs(cur)
        log(f"[split_cross_country_entries] {len(pairs)} cross-country pairs "
            f"across {len({p['horse_id'] for p in pairs})} horses")

        if args.scrape:
            with conn.cursor() as cur:
                need = sorted({
                    p["st_id"] for p in pairs
                    if p["st_id"] and st_racedays_for(cur, p["st_id"],
                                                      p["race_date"]) is None
                })
            if need:
                from scrapers.st_horse import scrape_horse_ids
                log(f"  scraping ST race-results for {len(need)} horses…")
                counts = scrape_horse_ids(conn, need, skip_done=False,
                                          delay=args.scrape_delay)
                log(f"  scrape: {counts}")

        # Adjudicate every pair first so the dry-run report is complete.
        planned: list[dict] = []
        skipped: dict[str, int] = defaultdict(int)
        with conn.cursor() as cur:
            for p in pairs:
                days = st_racedays_for(cur, p["st_id"], p["race_date"])
                verdict, why = adjudicate(p, days)
                if verdict == "skip":
                    skipped[why.split("—")[0].strip()[:44]] += 1
                    continue
                keep, drop = ("a", "b") if verdict == "a" else ("b", "a")
                planned.append({
                    **p,
                    "keep_track": p[f"{keep}_track"],
                    "keep_entry": p[f"{keep}_entry"],
                    "bad_entry": p[f"{drop}_entry"],
                    "bad_race": p[f"{drop}_race"],
                    "bad_track": p[f"{drop}_track"],
                    "bad_country": p[f"{drop}_country"],
                    "why": why,
                })

        dup = sum(1 for p in planned if p["same_result"])
        summary["candidates"] = len(pairs)
        summary["adjudicated"] = len(planned)
        log(f"  adjudicated: {len(planned)}   skipped: {sum(skipped.values())}")
        log(f"    {dup} duplicate copies (delete), "
            f"{len(planned) - dup} foreign collisions (re-home)")
        for why, n in sorted(skipped.items(), key=lambda kv: -kv[1]):
            log(f"    skip [{n:3}] {why}")

        if args.limit:
            planned = planned[: args.limit]

        moved = deleted = created = blocked = 0
        for p in planned:
            tag = "" if args.execute else "PREVIEW "
            with conn.cursor() as cur:
                if p["same_result"]:
                    # Same event twice. The kept entry carries identical
                    # placement, program number and race number, so nothing is
                    # lost — and no plausible horse exists to re-home to.
                    if args.execute:
                        row = _entry_row(cur, p["bad_entry"])
                        cur.execute("DELETE FROM entry WHERE entry_id = %s",
                                    (p["bad_entry"],))
                        _log_action(cur, p, dest=None, row=row)
                    deleted += 1
                    log(f"  {tag}delete entry {p['bad_entry']} ({p['bad_track']}) "
                        f"h{p['horse_id']} {p['name']!r} {p['race_date']} — "
                        f"duplicate of entry {p['keep_entry']} at "
                        f"{p['keep_track']}, agrees on {'+'.join(p['agree'])}")
                    continue

                # The result belongs to a same-named horse from the track's own
                # country, so ST's origin marker on this horse ("Hard Times
                # (NO)") describes the wrong animal and must not ride along
                # into the key.
                dest_name = _strip_country(p["name"])
                dest, is_new = find_or_create_synth(
                    cur, p["bad_country"], dest_name, execute=args.execute)
                if dest is None:
                    created += 1
                    moved += 1
                    log(f"  {tag}move entry {p['bad_entry']} ({p['bad_track']}) "
                        f"h{p['horse_id']} {p['name']!r} {p['race_date']} -> NEW "
                        f"{synth_key(p['bad_country'], dest_name)}   [{p['why']}]")
                    continue
                if dest == p["horse_id"]:
                    blocked += 1
                    log(f"  - skip h{p['horse_id']}: synthetic destination is the "
                        f"same row")
                    continue
                cur.execute(
                    "SELECT 1 FROM entry WHERE race_id = %s AND horse_id = %s",
                    (p["bad_race"], dest),
                )
                if cur.fetchone():
                    blocked += 1
                    log(f"  - skip entry {p['bad_entry']}: destination h{dest} "
                        f"already has an entry in race {p['bad_race']}")
                    continue

                if is_new:
                    created += 1
                if args.execute:
                    cur.execute(
                        "UPDATE entry SET horse_id = %s, last_updated_at = NOW() "
                        " WHERE entry_id = %s",
                        (dest, p["bad_entry"]),
                    )
                    _log_action(cur, p, dest=dest)
                moved += 1
                log(f"  {tag}move entry {p['bad_entry']} ({p['bad_track']}) "
                    f"h{p['horse_id']} -> h{dest}   [{p['why']}]")
            if args.execute and (moved + deleted) % args.commit_every == 0:
                conn.commit()

        if args.execute:
            conn.commit()
        summary["merged"] = moved + deleted
        summary["entries_moved"] = moved
        summary["entries_deleted"] = deleted
        summary["synth_rows_created"] = created
        summary["blocked"] = blocked
        if not args.execute:
            log("\nDRY-RUN — no DB changes. Use --execute to apply.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
