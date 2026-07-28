"""
Give Le Trot-identified placeholder races their real venue back.

ST records a Swedish horse's foreign start without naming the track, so the race
is parked on a synthetic one called "Frankrike". For most of those rows that is
the whole story — but 5,793 of them were later matched to Le Trot, filled with
their full field of 14-odd runners, placements and times, and left sitting on
the placeholder anyway. The venue was never written back.

Nothing needs to be fetched to fix them. A `letrot_race_id` is
`<date>_<reunion>_<course>`, and a Le Trot reunion is one meeting at one track,
so any other race sharing that (date, reunion) already names the venue. Every
one of the 5,793 has such a sibling, and in every case the sibling set points at
exactly one track — the mapping is derivable, not guessed.

Reunion numbers alone are not enough: they get reused across dates, and 2,578 of
these resolve to two or three different tracks when the date is dropped. The
pairing is what makes it unambiguous.

Races whose target already holds a race with the same date and race number are
left alone and reported. Those are duplicate imports rather than misfiles, and
folding them is `scripts.merge_duplicate_races`' job.

    python -m scripts.repoint_letrot_placeholder_races            # dry-run
    python -m scripts.repoint_letrot_placeholder_races --execute
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from psycopg2.extras import Json  # noqa: E402

from scripts._merge_helpers import build_argparser, script_runner  # noqa: E402

PLACEHOLDER_TRACKS = (
    "Frankrike", "Italien", "Tyskland", "Norge", "Belgien", "Danmark",
    "Finland", "Australien", "Holland", "USA", "Osterrike", "Österrike",
)


def find_candidates(cur) -> list[dict]:
    """Placeholder races whose (date, reunion) siblings name exactly one track."""
    cur.execute(
        """
        WITH ph AS (
            SELECT r.race_id, r.race_date, r.race_number, r.letrot_race_id,
                   r.track_id AS old_track_id, t.name AS old_track,
                   split_part(r.letrot_race_id, '_', 2) AS reunion
              FROM race r
              JOIN track t ON t.track_id = r.track_id
             WHERE t.name = ANY(%s)
               AND r.letrot_race_id IS NOT NULL
        ),
        sib AS (
            SELECT r.race_date,
                   split_part(r.letrot_race_id, '_', 2) AS reunion,
                   t.track_id, t.name
              FROM race r
              JOIN track t ON t.track_id = r.track_id
             WHERE r.letrot_race_id IS NOT NULL
               AND t.country = 'FR'
               AND t.name <> ALL(%s)
             GROUP BY 1, 2, 3, 4
        )
        SELECT ph.race_id, ph.race_date, ph.race_number, ph.letrot_race_id,
               ph.old_track_id, ph.old_track, ph.reunion,
               MIN(sib.track_id) AS new_track_id,
               MIN(sib.name)     AS new_track,
               COUNT(DISTINCT sib.track_id) AS n_tracks,
               EXISTS (
                   SELECT 1 FROM race x
                    WHERE x.track_id = MIN(sib.track_id)
                      AND x.race_date = ph.race_date
                      AND x.race_number IS NOT DISTINCT FROM ph.race_number
                      AND x.race_id <> ph.race_id
               ) AS target_occupied
          FROM ph
          JOIN sib ON sib.race_date = ph.race_date
                  AND sib.reunion   = ph.reunion
         GROUP BY ph.race_id, ph.race_date, ph.race_number, ph.letrot_race_id,
                  ph.old_track_id, ph.old_track, ph.reunion
         ORDER BY ph.race_date
        """,
        (list(PLACEHOLDER_TRACKS), list(PLACEHOLDER_TRACKS)),
    )
    cols = ("race_id", "race_date", "race_number", "letrot_race_id",
            "old_track_id", "old_track", "reunion", "new_track_id",
            "new_track", "n_tracks", "target_occupied")
    return [dict(zip(cols, r)) for r in cur.fetchall()]


def main() -> int:
    parser = build_argparser("repoint_letrot_placeholder_races")
    args = parser.parse_args()

    with script_runner("repoint_letrot_placeholder_races", args) as (conn, log, summary):
        with conn.cursor() as cur:
            cands = find_candidates(cur)

        ambiguous = [c for c in cands if c["n_tracks"] != 1]
        occupied = [c for c in cands if c["n_tracks"] == 1 and c["target_occupied"]]
        actionable = [c for c in cands if c["n_tracks"] == 1
                      and not c["target_occupied"]]

        log(f"[repoint_letrot_placeholder_races] {len(cands):,} placeholder races "
            f"carry a letrot_race_id")
        log(f"  actionable (single unambiguous venue): {len(actionable):,}")
        log(f"  skipped, target already has that race number: {len(occupied):,}"
            f"  -> merge_duplicate_races")
        log(f"  skipped, ambiguous venue: {len(ambiguous):,}")

        by_track = Counter(c["new_track"] for c in actionable)
        log("\n  destination venues (top 15):")
        for name, n in by_track.most_common(15):
            log(f"    {name:24} {n:>5,}")
        log(f"    ...{len(by_track)} venues in total")

        if args.limit:
            actionable = actionable[: args.limit]

        summary["candidates"] = len(cands)
        summary["actionable"] = len(actionable)
        summary["skipped_occupied"] = len(occupied)
        summary["skipped_ambiguous"] = len(ambiguous)

        if not args.execute:
            log("\n  sample:")
            for c in actionable[:10]:
                log(f"    race {c['race_id']} {c['race_date']} "
                    f"{c['letrot_race_id']}  {c['old_track']} -> {c['new_track']}")
            log("\nDRY-RUN — no DB changes. Use --execute to apply.")
            return 0

        done = 0
        for c in actionable:
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE race SET track_id = %s, last_updated_at = NOW() "
                    " WHERE race_id = %s",
                    (c["new_track_id"], c["race_id"]),
                )
                cur.execute(
                    """
                    INSERT INTO track_change_log
                        (track_id, race_id, change_type, old_value, new_value,
                         reason, changed_at, changed_by)
                    VALUES (%s, %s, 'repoint_placeholder', %s, %s, %s, NOW(),
                            'repoint_letrot_placeholder_races')
                    """,
                    (
                        c["new_track_id"], c["race_id"],
                        Json({"track_id": c["old_track_id"],
                              "track": c["old_track"]}),
                        Json({"track_id": c["new_track_id"],
                              "track": c["new_track"]}),
                        f"letrot_race_id {c['letrot_race_id']} shares "
                        f"(date {c['race_date']}, reunion {c['reunion']}) with "
                        f"races already at {c['new_track']}; a Le Trot reunion "
                        f"is one meeting at one track, so the placeholder "
                        f"{c['old_track']!r} was the misfile",
                    ),
                )
            done += 1
            if done % args.commit_every == 0:
                conn.commit()
                log(f"  … {done:,}/{len(actionable):,}")
        conn.commit()
        summary["merged"] = done
        summary["repointed"] = done
        log(f"  repointed {done:,} races")

    return 0


if __name__ == "__main__":
    sys.exit(main())
