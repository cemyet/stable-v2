"""
Noise diagnostics for the trainer-form signal.

Measures how well the leading variants predict per-start outperformance
(`y_outperf`, the odds-rank residual) ACROSS data slices, to find conditions
that add noise or skew: gait (auto/volt), monté vs sulky, trainer volume,
track, season/month, and field size. Spearman correlation is the yardstick
(rank-based, robust). Read-only; runs in .venv-ml.

    .venv-ml/bin/python -m scripts.analyze_trainer_form
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pandas as pd  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

from core.db import get_connection  # noqa: E402

LEAD = ["tf_posbeat_last100", "tf_posbeat_last50", "tf_beatrate_last50",
        "tf_top3_last50"]


def rho(df, v, y="y_outperf"):
    s = df[[v, y]].dropna()
    if len(s) < 300:
        return None, len(s)
    r, _ = spearmanr(s[v], s[y])
    return r, len(s)


def line(df, label):
    parts = []
    for v in LEAD:
        r, n = rho(df, v)
        parts.append(f"{v.replace('tf_',''):>16}={'' if r is None else f'{r:+.3f}'}")
    print(f"  {label:<26} n={len(df):>7,}  " + "  ".join(parts))


def main() -> int:
    conn = get_connection()
    sql = """
        SELECT x.*,
               r.start_method,
               (r.heading ILIKE '%mont%') AS is_monte,
               r.track_id,
               t.name AS track,
               EXTRACT(MONTH FROM x.race_date)::int AS mon,
               fs.n_field
        FROM trainer_form_exp x
        JOIN entry e ON e.entry_id = x.entry_id
        JOIN race r  ON r.race_id  = e.race_id
        JOIN track t ON t.track_id = r.track_id
        JOIN (
            SELECT race_id, COUNT(*) FILTER (
                     WHERE odds IS NOT NULL AND odds > 1
                       AND NOT COALESCE(withdrawn,false)) AS n_field
            FROM entry GROUP BY race_id
        ) fs ON fs.race_id = e.race_id
        WHERE x.race_date >= '2024-01-01'
          AND x.tf_n_last50 >= 20
    """
    print("loading test slice (2024+, gated)…", flush=True)
    df = pd.read_sql(sql, conn)
    conn.close()
    df["is_monte"] = df["is_monte"].fillna(False).astype(bool)
    df["mon"] = df["mon"].fillna(0).astype(int)
    print(f"rows: {len(df):,}\n", flush=True)

    print("OVERALL (baseline):")
    line(df, "all")

    print("\nBY GAIT:")
    line(df[df.start_method == "A"], "auto (A)")
    line(df[df.start_method == "V"], "volt (V)")

    print("\nMONTÉ vs SULKY:")
    line(df[df.is_monte], "monté")
    line(df[~df.is_monte], "sulky (non-monté)")

    print("\nBY FIELD SIZE:")
    line(df[df.n_field <= 6], "small (<=6)")
    line(df[(df.n_field >= 7) & (df.n_field <= 10)], "medium (7-10)")
    line(df[df.n_field >= 11], "large (>=11)")

    print("\nBY TRAINER VOLUME (last 180d starts):")
    line(df[df.tf_n_180 < 30], "low (<30)")
    line(df[(df.tf_n_180 >= 30) & (df.tf_n_180 < 100)], "mid (30-100)")
    line(df[df.tf_n_180 >= 100], "high (>=100)")

    print("\nBY SEASON (month):")
    for grp, mons in [("winter (12-2)", [12, 1, 2]), ("spring (3-5)", [3, 4, 5]),
                      ("summer (6-8)", [6, 7, 8]), ("autumn (9-11)", [9, 10, 11])]:
        line(df[df.mon.isin(mons)], grp)

    print("\nTOP 12 TRACKS BY VOLUME:")
    top = df.track.value_counts().head(12).index
    for tr in top:
        line(df[df.track == tr], tr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
