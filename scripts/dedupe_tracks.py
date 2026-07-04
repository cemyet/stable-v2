"""
Dedupe + normalize tracks in stable_v2.

Two cleanups in one pass:

  1. **Same physical track, multiple rows** (e.g. ENGHIEN vs Enghien with
     different atg_track_ids). We group by (lower(name), country) and merge
     all rows in a group into the lowest track_id. Race FK references are
     repointed; per-source ids on the duplicates are stashed under
     `source_data.<source>.aliases` of the canonical row.

  2. **ALL CAPS or all-lowercase names** are title-cased (Enghien, Gelsenkirchen).

The matching upgrade in `etl.matching.upsert_track` prevents *new* duplicates
from being created via cross-source name+country fallback.

Usage
-----
    python3 -m scripts.dedupe_tracks           # dry run
    python3 -m scripts.dedupe_tracks --execute # apply
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from core.db import get_connection  # noqa: E402

_SOURCE_ID_COLS = ("st_code", "atg_track_id", "usta_id", "letrot_id", "hvt_id")


def _load_tracks(cur) -> list[dict]:
    cur.execute(
        """
        SELECT track_id, name, country, sport, st_code, atg_track_id,
               usta_id, letrot_id, hvt_id, primary_source, source_data
          FROM track
         ORDER BY track_id
        """
    )
    cols = [d.name for d in cur.description]
    return [dict(zip(cols, r)) for r in cur.fetchall()]


def _normalize_name(name: str | None) -> str | None:
    if not name:
        return name
    s = name.strip()
    if not s:
        return s
    if s == s.upper() or s == s.lower():
        return s.title()
    return s


def _group_dupes(tracks: list[dict]) -> dict[tuple, list[dict]]:
    """Group by (lower(name), country). Returns groups with >1 row.

    Also folds NULL-country rows into the same-name group when there's
    exactly one known country for that name. Two NULL+NULL rows with the
    same name are also grouped (e.g. v1's two SLOVENIEN rows with different
    st_codes).
    """
    by_name: dict[str, list[dict]] = defaultdict(list)
    for t in tracks:
        if not t["name"]:
            continue
        by_name[t["name"].strip().lower()].append(t)

    groups: dict[tuple, list[dict]] = {}
    for nm, rows in by_name.items():
        countries = {r["country"] for r in rows if r["country"]}
        if len(countries) == 0:
            # All rows have NULL country -- single group, country=None.
            if len(rows) > 1:
                groups[(nm, None)] = rows
            continue
        if len(countries) == 1:
            # All rows share one country (NULLs included). Merge them.
            ctry = next(iter(countries))
            if len(rows) > 1:
                groups[(nm, ctry)] = rows
            continue
        # Multiple distinct known countries -- different physical tracks.
        # Group separately by country, and orphan the NULLs (don't merge).
        by_country: dict[str, list[dict]] = defaultdict(list)
        for r in rows:
            if r["country"]:
                by_country[r["country"]].append(r)
        for ctry, sub in by_country.items():
            if len(sub) > 1:
                groups[(nm, ctry)] = sub
    return groups


def _race_count_for_tracks(cur, track_ids: list[int]) -> int:
    if not track_ids:
        return 0
    cur.execute(
        "SELECT COUNT(*) FROM race WHERE track_id = ANY(%s)",
        (track_ids,),
    )
    return cur.fetchone()[0]


_SOURCE_FOR_COL = {
    "st_code":      "st",
    "atg_track_id": "atg",
    "usta_id":      "usta",
    "letrot_id":    "letrot",
    "hvt_id":       "hvt",
}


def _merge_aliases(canonical: dict, dups: list[dict]) -> dict:
    """Build a merged source_data dict that stashes duplicate source_ids as
    aliases under their respective source blocks, and folds in any source
    blocks the canonical doesn't already have.
    """
    sd = dict(canonical.get("source_data") or {})
    for d in dups:
        for col, source_key in _SOURCE_FOR_COL.items():
            d_val = d.get(col)
            if d_val is None or d_val == canonical.get(col):
                continue
            block = dict(sd.get(source_key) or {})
            aliases = list(block.get("aliases") or [])
            if d_val not in aliases:
                aliases.append(d_val)
            block["aliases"] = aliases
            sd[source_key] = block
        for src, block in (d.get("source_data") or {}).items():
            if src not in sd:
                sd[src] = block
    return sd


def _coalesce_field(canonical: dict, dups: list[dict], col: str):
    """Return the first non-null value across canonical+dups for `col`.

    Used to lift things like country / sport from a duplicate onto the
    canonical row when the canonical itself is missing them.
    """
    if canonical.get(col) is not None:
        return canonical[col]
    for d in dups:
        if d.get(col) is not None:
            return d[col]
    return None


def _coalesce_source_id(canonical: dict, dups: list[dict], col: str):
    """First non-null source_id across canonical+dups (canonical wins)."""
    if canonical.get(col) is not None:
        return canonical[col]
    for d in dups:
        if d.get(col) is not None:
            return d[col]
    return None


def _names_to_normalize(tracks: list[dict]) -> list[tuple[int, str, str]]:
    """Tracks where current name differs from normalized."""
    out = []
    for t in tracks:
        nm = t.get("name")
        if not nm:
            continue
        norm = _normalize_name(nm)
        if norm and norm != nm:
            out.append((t["track_id"], nm, norm))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--execute", action="store_true",
                    help="Apply changes (default is dry-run).")
    args = ap.parse_args()

    conn = get_connection()
    try:
        with conn.cursor() as cur:
            print("=" * 60)
            print(f"track dedupe — {'EXECUTE' if args.execute else 'DRY-RUN'}")
            print("=" * 60)

            tracks = _load_tracks(cur)
            print(f"\ntotal tracks: {len(tracks)}")

            groups = _group_dupes(tracks)
            print(f"duplicate groups (same lower(name), country): {len(groups)}")

            total_race_repoints = 0
            total_dups_to_delete = 0

            merge_plan: list[dict] = []
            for key, rows in sorted(groups.items()):
                # Lowest track_id wins (oldest = most likely to have FKs already).
                rows = sorted(rows, key=lambda r: r["track_id"])
                canonical = rows[0]
                dups = rows[1:]

                dup_ids = [d["track_id"] for d in dups]
                n_repoint = _race_count_for_tracks(cur, dup_ids)
                total_race_repoints += n_repoint
                total_dups_to_delete += len(dups)

                merge_plan.append({
                    "key": key,
                    "canonical": canonical,
                    "dups": dups,
                    "race_repoints": n_repoint,
                })

                name, country = key
                merged_country = _coalesce_field(canonical, dups, "country")
                print(f"\n  {country or '??'} {canonical['name']!r}  "
                      f"({len(rows)} rows)  -> country={merged_country or 'NULL'}")
                print(f"    -> CANONICAL track_id={canonical['track_id']:<5} "
                      f"name={canonical['name']!r} country={canonical['country'] or 'NULL'} "
                      f"src={canonical['primary_source']}")
                for d in dups:
                    print(f"       merge   track_id={d['track_id']:<5} "
                          f"name={d['name']!r} country={d['country'] or 'NULL'} "
                          f"src={d['primary_source']}")
                if n_repoint:
                    print(f"       race FK repoints: {n_repoint:,}")

            renames = _names_to_normalize(tracks)
            print(f"\nnames to title-case (e.g. ENGHIEN -> Enghien): {len(renames)}")
            for tid, old, new in renames[:15]:
                print(f"  track_id={tid:<5} {old!r} -> {new!r}")
            if len(renames) > 15:
                print(f"  ... and {len(renames) - 15} more")

            print(f"\nsummary:")
            print(f"  duplicate rows to delete:  {total_dups_to_delete}")
            print(f"  race FK rows to repoint:   {total_race_repoints:,}")
            print(f"  track names to title-case: {len(renames)}")

            if not args.execute:
                print("\n[dry-run] no changes made. re-run with --execute to apply.")
                return 0

            print("\nexecuting...")

            for plan in merge_plan:
                canonical = plan["canonical"]
                dups = plan["dups"]
                dup_ids = [d["track_id"] for d in dups]

                new_sd = _merge_aliases(canonical, dups)

                upd: dict = {
                    "source_data": json.dumps(new_sd, default=str),
                    "country":     _coalesce_field(canonical, dups, "country"),
                    "sport":       _coalesce_field(canonical, dups, "sport"),
                }
                # Lift any per-source id the canonical is missing onto it
                # (e.g. canonical is the ST row, dup is the HVT row → keep
                #  st_code on canonical AND adopt hvt_id from the dup).
                for col in _SOURCE_FOR_COL:
                    val = _coalesce_source_id(canonical, dups, col)
                    if val is not None and canonical.get(col) is None:
                        upd[col] = val

                # Prefer a non-uppercase name if one of the dups has it.
                cur_name = canonical.get("name") or ""
                if cur_name and cur_name == cur_name.upper():
                    for d in dups:
                        n = d.get("name") or ""
                        if n and n != n.upper():
                            upd["name"] = n
                            break

                set_parts = [f"{k} = %s::jsonb" if k == "source_data" else f"{k} = %s"
                             for k in upd.keys()]
                set_parts.append("last_updated_at = NOW()")
                params = list(upd.values()) + [canonical["track_id"]]
                cur.execute(
                    f"UPDATE track SET {', '.join(set_parts)} WHERE track_id = %s",
                    params,
                )
                cur.execute(
                    "UPDATE race SET track_id = %s WHERE track_id = ANY(%s)",
                    (canonical["track_id"], dup_ids),
                )
                cur.execute(
                    "DELETE FROM track WHERE track_id = ANY(%s)",
                    (dup_ids,),
                )

            print(f"  merged {total_dups_to_delete} duplicate rows, "
                  f"repointed {total_race_repoints:,} races")

            for tid, _old, new in renames:
                cur.execute(
                    "UPDATE track SET name = %s, last_updated_at = NOW() "
                    " WHERE track_id = %s",
                    (new, tid),
                )
            print(f"  title-cased {len(renames)} names")

        conn.commit()
        print("\ncommitted.")
    finally:
        conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
