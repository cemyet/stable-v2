"""
Repair galopp placements: classified gait-breakers that were stored with
`placement_text = 'g'` even though they have a real numeric `placement`.

The ATG importer used to overwrite `placement_text` with bare 'g' whenever
`result.galloped` was true, discarding the actual finishing position (e.g.
Owen Bros finished 2nd but showed as 'g' and sorted to the bottom). The
importer now keeps the numeric position and relies on the `entry.galopp`
boolean for the gait-break marker. This one-time pass fixes existing rows.

Usage
-----

    python -m scripts.fix_galopp_placement              # dry-run
    python -m scripts.fix_galopp_placement --execute    # apply
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts._merge_helpers import build_argparser, script_runner  # noqa: E402


_SELECT = """
    SELECT COUNT(*) FROM entry
     WHERE placement_text = 'g'
       AND placement IS NOT NULL
       AND placement >= 1
"""

_UPDATE = """
    UPDATE entry
       SET placement_text = placement::text,
           galopp = TRUE,
           last_updated_at = NOW()
     WHERE placement_text = 'g'
       AND placement IS NOT NULL
       AND placement >= 1
"""


def main() -> int:
    parser = build_argparser("fix_galopp_placement")
    args = parser.parse_args()

    with script_runner("fix_galopp_placement", args) as (conn, log, summary):
        log(f"[fix_galopp_placement] execute={args.execute}")
        with conn.cursor() as cur:
            cur.execute(_SELECT)
            n = cur.fetchone()[0]
        log(f"  classified galopp entries with bare 'g' placement_text: {n:,}")
        summary["candidates"] = n

        if not args.execute:
            log("\nDRY-RUN — no DB writes. Pass --execute to apply.")
            return 0

        with conn.cursor() as cur:
            cur.execute(_UPDATE)
            summary["merged"] = cur.rowcount
        conn.commit()
        log(f"  rewrote {summary['merged']:,} placement_text 'g' -> numeric")
        log("\nCommitted.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
