"""
Read-only identity-matching health audit.

Two uses:

  1. Library: imported by web/app.py and other scripts to fetch live counts
     for the admin /admin/matching dashboard.

  2. CLI: `python -m scripts.audit_matching [--json] [--job-run-id N]` —
     prints a category-by-category snapshot, optionally with full JSON.
     When --job-run-id is set it writes progress + summary into job_run
     for the admin runner.

Categories (counts cited in the plan):

  - A — polluted old ST horses (first-race > ~18y after DOB)
  - B — ST-guest ↔ ATG-synth pairs (numeric atg_id + same name x:CC:NAME synth)
  - C-pedigree — same-name pedigree groups (name + year + sire + dam)
  - C-loose   — remaining same-name foreign-synth duplicates without pedigree
  - D — orphan ST registry horses (0 entries)
  - E — UELN / registration_number unique-index violations
  - F — duplicate races sharing (track_id, race_date, race_number)
  - G — ATG-synth persons with no ST link / G-pairs (st + synth same name)
  - H — duplicate tracks (lower(name), country)
  - I — ATG-only foreign horses (Josh Power class — leave alone)
  - same_row — rows with BOTH st_id AND x:CC:NAME atg_id (Face Time Bourbon)
  - merges_recent — last 20 merges (horse + person)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from core.db import get_connection  # noqa: E402


# ---------------------------------------------------------------------------
# Individual category queries — each returns (count, entries_affected)
# ---------------------------------------------------------------------------


def _category_a(cur) -> dict:
    """Polluted old ST horses: x:CC:NAME atg_id + first race > 18y after DOB."""
    cur.execute(
        """
        WITH polluted AS (
          SELECT h.horse_id,
                 h.name,
                 h.date_of_birth,
                 MIN(r.race_date) AS first_race
            FROM horse h
            JOIN entry e ON e.horse_id = h.horse_id
            JOIN race  r ON r.race_id  = e.race_id
           WHERE h.atg_id LIKE 'x:%'
             AND h.date_of_birth IS NOT NULL
             AND h.st_id IS NOT NULL
           GROUP BY h.horse_id, h.name, h.date_of_birth
          HAVING (MIN(r.race_date) - h.date_of_birth) > 6570  -- 18y in days
        )
        SELECT COUNT(*),
               COALESCE(SUM(
                 (SELECT COUNT(*) FROM entry e2
                                  JOIN race r2 ON r2.race_id = e2.race_id
                   WHERE e2.horse_id = polluted.horse_id
                     AND (r2.race_date - polluted.date_of_birth) > 6570
                 )
               ), 0) AS entries_affected
          FROM polluted
        """,
    )
    count, ents = cur.fetchone()
    return {"count": int(count), "entries": int(ents or 0)}


def _category_b(cur) -> dict:
    """ST-guest with numeric atg_id ↔ separate ATG-synth row, same name+country."""
    cur.execute(
        """
        WITH pairs AS (
          SELECT real.horse_id AS real_id,
                 synth.horse_id AS synth_id
            FROM horse synth
            JOIN horse real
              ON real.atg_id ~ '^[0-9]+$'
             AND real.atg_id::int <> 0
             AND synth.atg_id LIKE 'x:%'
             AND v2_normalize_name(real.name) = v2_normalize_name(synth.name)
             AND v2_normalize_name(real.name) <> ''
             AND (
                  -- match by country if both have it, else any
                  real.birth_country = synth.birth_country
                  OR real.birth_country IS NULL
                  OR synth.birth_country IS NULL
             )
           WHERE synth.horse_id <> real.horse_id
        )
        SELECT COUNT(*) AS pairs,
               COALESCE(SUM(
                 (SELECT COUNT(*) FROM entry e WHERE e.horse_id IN (pairs.real_id, pairs.synth_id))
               ), 0) AS entries
          FROM pairs
        """,
    )
    pairs, ents = cur.fetchone()
    return {"count": int(pairs), "entries": int(ents or 0)}


def _category_c_pedigree(cur) -> dict:
    """Same name + birth year + sire + dam — high-confidence merge groups."""
    cur.execute(
        """
        SELECT COUNT(*) FROM (
          SELECT 1
            FROM horse
           WHERE name IS NOT NULL
             AND date_of_birth IS NOT NULL
             AND sire_name IS NOT NULL
             AND dam_name  IS NOT NULL
           GROUP BY v2_normalize_name(name),
                    EXTRACT(year FROM date_of_birth),
                    v2_normalize_name(sire_name),
                    v2_normalize_name(dam_name)
          HAVING COUNT(*) > 1
        ) g
        """,
    )
    n = cur.fetchone()[0]
    return {"count": int(n), "entries": None}


def _category_c_loose(cur) -> dict:
    """Same-name foreign-synth dupes WITHOUT full pedigree (manual review)."""
    cur.execute(
        """
        SELECT COUNT(*) FROM (
          SELECT 1
            FROM horse
           WHERE atg_id LIKE 'x:%'
           GROUP BY v2_normalize_name(name)
          HAVING COUNT(*) > 1
        ) g
        """,
    )
    return {"count": int(cur.fetchone()[0]), "entries": None}


def _category_d(cur) -> dict:
    """Orphan ST registry horses (0 entries) — informational, leave alone."""
    cur.execute(
        """
        SELECT COUNT(*) FROM horse h
         WHERE NOT EXISTS (SELECT 1 FROM entry e WHERE e.horse_id = h.horse_id)
           AND h.st_id IS NOT NULL
           AND h.atg_id IS NULL
        """
    )
    return {"count": int(cur.fetchone()[0]), "entries": 0}


def _category_e(cur) -> dict:
    """UELN / registration_number duplicates (unique indexes mean usually 0)."""
    cur.execute(
        """
        SELECT
          COALESCE((SELECT SUM(c) FROM (
              SELECT COUNT(*) c FROM horse
               WHERE registration_number IS NOT NULL
               GROUP BY registration_number
              HAVING COUNT(*) > 1
          ) g), 0) AS reg_dupes,
          COALESCE((SELECT SUM(c) FROM (
              SELECT COUNT(*) c FROM horse
               WHERE ueln_number IS NOT NULL
               GROUP BY ueln_number
              HAVING COUNT(*) > 1
          ) g), 0) AS ueln_dupes
        """
    )
    reg, ueln = cur.fetchone()
    return {"count": int((reg or 0) + (ueln or 0)), "entries": None,
            "reg_dupes": int(reg or 0), "ueln_dupes": int(ueln or 0)}


def _category_f(cur) -> dict:
    """Races sharing (track_id, race_date, race_number)."""
    cur.execute(
        """
        WITH dupes AS (
          SELECT track_id, race_date, race_number, COUNT(*) AS n
            FROM race
           WHERE track_id IS NOT NULL AND race_number IS NOT NULL
           GROUP BY track_id, race_date, race_number
          HAVING COUNT(*) > 1
        )
        SELECT COUNT(*),
               COALESCE(SUM(
                 (SELECT COUNT(*) FROM entry e
                    JOIN race r ON r.race_id = e.race_id
                   WHERE r.track_id = d.track_id
                     AND r.race_date = d.race_date
                     AND r.race_number = d.race_number)
               ), 0) AS entries
          FROM dupes d
        """,
    )
    c, ents = cur.fetchone()
    return {"count": int(c), "entries": int(ents or 0)}


def _category_g(cur) -> dict:
    """Persons: ATG-synth with no ST link / synth+real pairs."""
    cur.execute("SELECT COUNT(*) FROM person WHERE atg_id LIKE 'x:%' AND st_id IS NULL")
    n_synth = cur.fetchone()[0]
    cur.execute(
        """
        WITH pairs AS (
          SELECT real.person_id, synth.person_id AS synth_id
            FROM person synth
            JOIN person real
              ON real.atg_id ~ '^[0-9]+$'
             AND real.atg_id::int <> 0
             AND synth.atg_id LIKE 'x:%'
             AND upper(real.name) = upper(synth.name)
           WHERE synth.person_id <> real.person_id
        )
        SELECT COUNT(*) FROM pairs
        """,
    )
    n_pairs = cur.fetchone()[0]
    return {"count": int(n_pairs), "entries": None,
            "orphan_synth": int(n_synth)}


def _category_h(cur) -> dict:
    """Track duplicates (lower(name), country)."""
    cur.execute(
        """
        SELECT COUNT(*) FROM (
          SELECT 1 FROM track
           GROUP BY lower(name), country
          HAVING COUNT(*) > 1
        ) g
        """
    )
    return {"count": int(cur.fetchone()[0]), "entries": None}


def _category_i(cur) -> dict:
    """ATG-only foreign horses (Josh Power class — leave alone, identity-correct)."""
    cur.execute(
        """
        SELECT COUNT(*) FROM horse h
         WHERE h.atg_id IS NOT NULL
           AND h.atg_id !~ '^[0-9]+$'
           AND h.st_id IS NULL
           AND h.atg_id LIKE 'x:%'
           AND EXISTS (SELECT 1 FROM entry e WHERE e.horse_id = h.horse_id)
        """
    )
    return {"count": int(cur.fetchone()[0]), "entries": None}


def _category_same_row(cur) -> dict:
    """Horses with BOTH st_id AND x:CC:NAME atg_id (Face Time Bourbon class)."""
    cur.execute(
        """
        SELECT COUNT(*),
               COALESCE(SUM(
                 (SELECT COUNT(*) FROM entry e WHERE e.horse_id = h.horse_id)
               ), 0)
          FROM horse h
         WHERE h.st_id IS NOT NULL
           AND h.atg_id LIKE 'x:%'
        """,
    )
    c, ents = cur.fetchone()
    return {"count": int(c), "entries": int(ents or 0)}


def _recent_merges(cur, limit: int = 20) -> list[dict]:
    cur.execute(
        """
        SELECT merge_id, from_horse_id, to_horse_id, reason, method,
               entries_moved, conflicts_resolved, merged_at, merged_by, rolled_back
          FROM horse_merge_log
         ORDER BY merged_at DESC
         LIMIT %s
        """,
        (limit,),
    )
    rows = []
    for r in cur.fetchall():
        rows.append({
            "merge_id":          r[0],
            "from_horse_id":     r[1],
            "to_horse_id":       r[2],
            "reason":            r[3],
            "method":            r[4],
            "entries_moved":     r[5],
            "conflicts_resolved": r[6],
            "merged_at":         r[7].isoformat() if r[7] else None,
            "merged_by":         r[8],
            "rolled_back":       bool(r[9]),
        })
    return rows


def _recent_person_merges(cur, limit: int = 20) -> list[dict]:
    cur.execute(
        """
        SELECT merge_id, from_person_id, to_person_id, reason, method,
               entries_moved, merged_at, merged_by, rolled_back
          FROM person_merge_log
         ORDER BY merged_at DESC
         LIMIT %s
        """,
        (limit,),
    )
    rows = []
    for r in cur.fetchall():
        rows.append({
            "merge_id":      r[0],
            "from_person_id": r[1],
            "to_person_id":   r[2],
            "reason":         r[3],
            "method":         r[4],
            "entries_moved":  r[5],
            "merged_at":      r[6].isoformat() if r[6] else None,
            "merged_by":      r[7],
            "rolled_back":    bool(r[8]),
        })
    return rows


# ---------------------------------------------------------------------------
# Top-level collectors used by web/app.py
# ---------------------------------------------------------------------------


_CATEGORY_FUNCS = {
    "a":           (_category_a,           "Polluted old ST horses (Cat A)"),
    "b":           (_category_b,           "ST-guest ↔ ATG-synth pairs (Cat B)"),
    "c_pedigree":  (_category_c_pedigree,  "Pedigree-triangulation groups (Cat C)"),
    "c_loose":     (_category_c_loose,     "Loose name dupes — manual review (Cat C)"),
    "d":           (_category_d,           "ST orphan registry horses (Cat D)"),
    "e":           (_category_e,           "UELN / reg_number dupes (Cat E)"),
    "f":           (_category_f,           "Race duplicates by track+date+number (Cat F)"),
    "g":           (_category_g,           "Person synth/real pairs (Cat G)"),
    "h":           (_category_h,           "Track duplicates (Cat H)"),
    "i":           (_category_i,           "ATG-only foreign horses — leave alone (Cat I)"),
    "same_row":    (_category_same_row,    "Same-row st_id + x: atg_id (Face Time Bourbon)"),
}


def collect_health(conn) -> dict:
    """Run every category query + recent merges; return JSON-ready dict."""
    out: dict = {"categories": {}, "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
    with conn.cursor() as cur:
        for key, (fn, label) in _CATEGORY_FUNCS.items():
            try:
                res = fn(cur)
            except Exception as exc:
                conn.rollback()
                res = {"count": None, "entries": None, "error": str(exc)}
            res["label"] = label
            res["key"] = key
            out["categories"][key] = res
        out["recent_horse_merges"]  = _recent_merges(cur)
        out["recent_person_merges"] = _recent_person_merges(cur)
    return out


# ---------------------------------------------------------------------------
# Browse-category queries — top 50 affected rows per category
# ---------------------------------------------------------------------------


def browse_category(conn, category: str, *, limit: int = 50) -> list[dict]:
    """Return up to `limit` representative affected rows for a category."""
    with conn.cursor() as cur:
        if category == "a":
            cur.execute(
                """
                SELECT h.horse_id, h.name, h.date_of_birth, h.birth_country,
                       h.st_id, h.atg_id, h.primary_source,
                       MIN(r.race_date)::text AS first_race,
                       COUNT(*) AS entries
                  FROM horse h
                  JOIN entry e ON e.horse_id = h.horse_id
                  JOIN race  r ON r.race_id  = e.race_id
                 WHERE h.atg_id LIKE 'x:%%'
                   AND h.st_id IS NOT NULL
                   AND h.date_of_birth IS NOT NULL
                 GROUP BY h.horse_id
                HAVING (MIN(r.race_date) - h.date_of_birth) > 6570
                 ORDER BY COUNT(*) DESC
                 LIMIT %s
                """,
                (limit,),
            )
        elif category == "b":
            cur.execute(
                """
                SELECT real.horse_id, real.name, real.birth_country,
                       real.st_id, real.atg_id,
                       synth.horse_id, synth.atg_id,
                       (SELECT COUNT(*) FROM entry e WHERE e.horse_id = real.horse_id)  AS real_entries,
                       (SELECT COUNT(*) FROM entry e WHERE e.horse_id = synth.horse_id) AS synth_entries
                  FROM horse synth
                  JOIN horse real
                    ON real.atg_id ~ '^[0-9]+$'
                   AND real.atg_id::int <> 0
                   AND synth.atg_id LIKE 'x:%%'
                   AND v2_normalize_name(real.name) = v2_normalize_name(synth.name)
                   AND v2_normalize_name(real.name) <> ''
                   AND (real.birth_country = synth.birth_country
                        OR real.birth_country IS NULL OR synth.birth_country IS NULL)
                 WHERE synth.horse_id <> real.horse_id
                 ORDER BY (SELECT COUNT(*) FROM entry e WHERE e.horse_id = synth.horse_id) DESC
                 LIMIT %s
                """,
                (limit,),
            )
            rows = []
            for r in cur.fetchall():
                rows.append({
                    "from_horse_id":   r[5],  # synth source
                    "to_horse_id":     r[0],  # real dest
                    "name":            r[1],
                    "birth_country":   r[2],
                    "st_id":           r[3],
                    "atg_id":          r[4],
                    "synth_atg_id":    r[6],
                    "real_entries":    int(r[7] or 0),
                    "synth_entries":   int(r[8] or 0),
                    "suggested":       "merge synth → real (Cat B)",
                })
            return rows
        elif category == "c_pedigree":
            cur.execute(
                """
                WITH g AS (
                  SELECT v2_normalize_name(name)      AS nname,
                         EXTRACT(year FROM date_of_birth)::int AS yr,
                         v2_normalize_name(sire_name) AS sn,
                         v2_normalize_name(dam_name)  AS dn,
                         COUNT(*) AS n,
                         array_agg(horse_id ORDER BY horse_id) AS ids,
                         MAX(name) AS sample_name
                    FROM horse
                   WHERE name IS NOT NULL AND date_of_birth IS NOT NULL
                     AND sire_name IS NOT NULL AND dam_name IS NOT NULL
                   GROUP BY 1,2,3,4
                  HAVING COUNT(*) > 1
                )
                SELECT sample_name, yr, n, ids
                  FROM g
                 ORDER BY n DESC, sample_name
                 LIMIT %s
                """,
                (limit,),
            )
            rows = []
            for r in cur.fetchall():
                rows.append({
                    "name": r[0], "birth_year": r[1],
                    "group_size": r[2], "horse_ids": list(r[3] or []),
                    "suggested": "merge into lowest id with most entries",
                })
            return rows
        elif category == "c_loose":
            cur.execute(
                """
                WITH g AS (
                  SELECT v2_normalize_name(name) AS nname,
                         COUNT(*) AS n,
                         array_agg(horse_id ORDER BY horse_id) AS ids,
                         MAX(name) AS sample_name
                    FROM horse
                   WHERE atg_id LIKE 'x:%%'
                     AND v2_normalize_name(name) <> ''
                   GROUP BY 1
                  HAVING COUNT(*) > 1
                )
                SELECT sample_name, n, ids
                  FROM g
                 ORDER BY n DESC
                 LIMIT %s
                """,
                (limit,),
            )
            rows = []
            for r in cur.fetchall():
                rows.append({
                    "name": r[0], "group_size": r[1], "horse_ids": list(r[2] or []),
                    "suggested": "manual review (no pedigree)",
                })
            return rows
        elif category == "f":
            cur.execute(
                """
                SELECT track_id, race_date::text, race_number, COUNT(*) AS n,
                       array_agg(race_id ORDER BY race_id) AS ids
                  FROM race
                 WHERE track_id IS NOT NULL AND race_number IS NOT NULL
                 GROUP BY track_id, race_date, race_number
                HAVING COUNT(*) > 1
                 ORDER BY n DESC, race_date DESC
                 LIMIT %s
                """,
                (limit,),
            )
            rows = []
            for r in cur.fetchall():
                rows.append({
                    "track_id": r[0], "race_date": r[1], "race_number": r[2],
                    "group_size": r[3], "race_ids": list(r[4] or []),
                    "suggested": "merge races (Cat F)",
                })
            return rows
        elif category == "g":
            cur.execute(
                """
                SELECT real.person_id, real.name, real.atg_id, real.st_id,
                       synth.person_id, synth.atg_id
                  FROM person synth
                  JOIN person real
                    ON real.atg_id ~ '^[0-9]+$'
                   AND real.atg_id::int <> 0
                   AND synth.atg_id LIKE 'x:%%'
                   AND upper(real.name) = upper(synth.name)
                 WHERE synth.person_id <> real.person_id
                 LIMIT %s
                """,
                (limit,),
            )
            rows = []
            for r in cur.fetchall():
                rows.append({
                    "from_person_id": r[4], "to_person_id": r[0],
                    "name": r[1], "atg_id": r[2], "st_id": r[3],
                    "synth_atg_id": r[5],
                    "suggested": "merge synth → real (Cat G)",
                })
            return rows
        elif category == "same_row":
            cur.execute(
                """
                SELECT h.horse_id, h.name, h.birth_country, h.st_id, h.atg_id,
                       (SELECT COUNT(*) FROM entry e WHERE e.horse_id = h.horse_id) AS entries
                  FROM horse h
                 WHERE h.st_id IS NOT NULL
                   AND h.atg_id LIKE 'x:%%'
                 ORDER BY entries DESC
                 LIMIT %s
                """,
                (limit,),
            )
        elif category == "i":
            cur.execute(
                """
                SELECT h.horse_id, h.name, h.birth_country, h.atg_id,
                       (SELECT COUNT(*) FROM entry e WHERE e.horse_id = h.horse_id) AS entries
                  FROM horse h
                 WHERE h.atg_id LIKE 'x:%%' AND h.st_id IS NULL
                   AND EXISTS (SELECT 1 FROM entry e WHERE e.horse_id = h.horse_id)
                 ORDER BY entries DESC
                 LIMIT %s
                """,
                (limit,),
            )
        elif category == "h":
            cur.execute(
                """
                SELECT lower(name), country, COUNT(*) AS n,
                       array_agg(track_id ORDER BY track_id) AS ids
                  FROM track
                 GROUP BY lower(name), country
                HAVING COUNT(*) > 1
                 ORDER BY n DESC
                 LIMIT %s
                """,
                (limit,),
            )
            return [
                {"name": r[0], "country": r[1], "group_size": r[2], "track_ids": list(r[3] or [])}
                for r in cur.fetchall()
            ]
        elif category == "e":
            cur.execute(
                """
                SELECT 'reg' AS kind, registration_number AS val,
                       array_agg(horse_id ORDER BY horse_id) AS ids
                  FROM horse
                 WHERE registration_number IS NOT NULL
                 GROUP BY registration_number
                HAVING COUNT(*) > 1
                 UNION ALL
                SELECT 'ueln', ueln_number,
                       array_agg(horse_id ORDER BY horse_id)
                  FROM horse
                 WHERE ueln_number IS NOT NULL
                 GROUP BY ueln_number
                HAVING COUNT(*) > 1
                 LIMIT %s
                """,
                (limit,),
            )
            return [
                {"kind": r[0], "value": r[1], "horse_ids": list(r[2] or [])}
                for r in cur.fetchall()
            ]
        elif category == "d":
            cur.execute(
                """
                SELECT horse_id, name, birth_country, st_id, atg_id
                  FROM horse h
                 WHERE st_id IS NOT NULL AND atg_id IS NULL
                   AND NOT EXISTS (SELECT 1 FROM entry e WHERE e.horse_id = h.horse_id)
                 ORDER BY horse_id DESC
                 LIMIT %s
                """,
                (limit,),
            )
        else:
            return [{"error": f"unknown category {category!r}"}]

        cols = [d.name for d in cur.description]
        out: list[dict] = []
        for r in cur.fetchall():
            out.append({
                c: (v if not hasattr(v, "isoformat") else v.isoformat())
                for c, v in zip(cols, r)
            })
        return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _log_job(conn, rid: int, msg: str) -> None:
    if rid is None:
        return
    with conn.cursor() as cur:
        cur.execute(
            "UPDATE job_run SET log = COALESCE(log,'') || %s WHERE job_run_id = %s",
            (msg, rid),
        )
    conn.commit()


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--json", action="store_true", help="emit full JSON")
    p.add_argument("--execute", action="store_true",
                   help="ignored — audit is read-only (accepted for runner uniformity)")
    p.add_argument("--job-run-id", type=int, default=None,
                   help="if set, append progress to that job_run row")
    args = p.parse_args()

    conn = get_connection()
    rid = args.job_run_id
    try:
        _log_job(conn, rid, "[audit] starting\n")
        data = collect_health(conn)

        if args.json:
            print(json.dumps(data, indent=2, default=str))
        else:
            print(f"Matching health audit — {data['generated_at']}\n")
            for k, cat in data["categories"].items():
                count = cat.get("count")
                ents = cat.get("entries")
                ent_s = f"  ents={ents:,}" if isinstance(ents, int) and ents > 0 else ""
                err = cat.get("error")
                err_s = f"  ERROR: {err}" if err else ""
                cnt_s = f"{count:>7,}" if isinstance(count, int) else f"{str(count):>7}"
                print(f"  {k:<13} {cnt_s}{ent_s}  {cat['label']}{err_s}")
            rm = data["recent_horse_merges"]
            print(f"\nRecent horse merges (last {len(rm)}):")
            for m in rm:
                tag = "[ROLLED BACK] " if m["rolled_back"] else ""
                print(f"  {tag}#{m['merge_id']:>5}  {m['from_horse_id']} → {m['to_horse_id']}  "
                      f"{m['method']:<12} moved={m['entries_moved']:>3} ({m['reason'][:50]})")

        _log_job(conn, rid, f"[audit] complete — {len(data['categories'])} categories\n")
        if rid is not None:
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE job_run SET status='success', finished_at=NOW(), "
                    "summary = %s WHERE job_run_id = %s",
                    (json.dumps({k: v.get('count') for k, v in data['categories'].items()}), rid),
                )
            conn.commit()
        return 0
    except Exception as exc:
        if rid is not None:
            _log_job(conn, rid, f"[audit] FAILED: {exc!r}\n")
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE job_run SET status='failed', finished_at=NOW() WHERE job_run_id=%s",
                    (rid,),
                )
            conn.commit()
        raise
    finally:
        conn.close()


if __name__ == "__main__":
    sys.exit(main())
