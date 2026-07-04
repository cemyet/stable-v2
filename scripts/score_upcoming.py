"""Score upcoming-race entries with galopp-risk models.

Writes two targets per entry into ml_prediction so the race view can toggle
between track-specific and general model scores without loading the ML stack:
  - y_gal_general — best general-scope model (kvikk, etc.)
  - y_gal_track   — best track model for that track, or general when none

Run under the ML venv:
    .venv-ml/bin/python -m scripts.score_upcoming
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pandas as pd
import joblib

from core.db import get_connection


def best_general_model(cur, target="y_gal"):
    cur.execute("""
        SELECT model_id, name, artifact_path,
               (metrics->'test'->>'roc_auc')::float AS auc
        FROM ml_model WHERE target = %s AND scope = 'general'
          AND COALESCE(slice_def->>'method', 'any') = 'any'
        ORDER BY auc DESC NULLS LAST, created_at DESC
        LIMIT 1
    """, (target,))
    return cur.fetchone()


def best_track_models(cur, target="y_gal"):
    """Best (highest-AUC) track-specific model per track_id."""
    cur.execute("""
        SELECT DISTINCT ON (track_id)
               track_id, model_id, name, artifact_path,
               (metrics->'test'->>'roc_auc')::float AS auc
        FROM ml_model
        WHERE target = %s AND scope = 'track' AND track_id IS NOT NULL
        ORDER BY track_id, auc DESC NULLS LAST, created_at DESC
    """, (target,))
    return {r[0]: r[1:] for r in cur.fetchall()}


def _load(artifact_path):
    bundle = joblib.load(_ROOT / artifact_path)
    return bundle["model"], bundle["features"], bundle["numeric"], bundle["categorical"]


def _score_subset(conn, entry_ids, artifact_path, model_id, target):
    """Score entry_ids with one model; returns rows for ml_prediction insert."""
    if not entry_ids:
        return []
    model, features, numeric, categorical = _load(artifact_path)
    cols = ", ".join(f"ef.{c}" for c in features)
    with conn.cursor() as cur:
        cur.execute(f"""
            SELECT ef.entry_id, {cols}
            FROM entry_features ef
            WHERE ef.entry_id = ANY(%s)
        """, (list(entry_ids),))
        names = [d[0] for d in cur.description]
        data = cur.fetchall()
    if not data:
        return []
    df = pd.DataFrame(data, columns=names)
    for c in numeric:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("float64")
    for c in categorical:
        df[c] = df[c].astype("category")
    probs = model.predict_proba(df[features])[:, 1]
    return [(int(eid), target, model_id, float(p))
            for eid, p in zip(df["entry_id"].tolist(), probs)]


def main() -> int:
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            gen = best_general_model(cur)
            if not gen:
                print("no general y_gal model registered", file=sys.stderr)
                return 2
            gen_id, gen_name, gen_path, gen_auc = gen
            track_models = best_track_models(cur)
            print(f"general: #{gen_id} '{gen_name}' (auc={gen_auc})")
            print(f"track models: " + (", ".join(
                f"{tid}->{m[1]}({m[3]:.4f})" for tid, m in track_models.items()) or "none"))

            cur.execute("""
                SELECT ef.entry_id, ef.track_id
                FROM entry_features ef
                JOIN race r ON r.race_id = ef.race_id
                WHERE r.race_date >= CURRENT_DATE
            """)
            upcoming = cur.fetchall()
        if not upcoming:
            print("no upcoming entries to score")
            return 0

        all_ids = [eid for eid, _ in upcoming]
        records = _score_subset(conn, all_ids, gen_path, gen_id, "y_gal_general")

        track_assignment = {}  # (model_id, path) -> [entry_ids]
        for entry_id, track_id in upcoming:
            if track_id in track_models:
                mid, _name, path, _auc = track_models[track_id]
                track_assignment.setdefault((mid, path), []).append(entry_id)

        for (mid, path), eids in track_assignment.items():
            records.extend(_score_subset(conn, eids, path, mid, "y_gal_track"))

        # Tracks without a dedicated model: track score = general score.
        track_only_ids = {eid for eids in track_assignment.values() for eid in eids}
        for entry_id, _track_id in upcoming:
            if entry_id not in track_only_ids:
                # copy general prob for this entry into y_gal_track
                gen_prob = next((p for eid, tgt, mid, p in records
                                   if eid == entry_id and tgt == "y_gal_general"), None)
                if gen_prob is not None:
                    records.append((entry_id, "y_gal_track", gen_id, gen_prob))

        print(f"scoring {len(upcoming):,} upcoming entries "
              f"({len(track_assignment)} track model(s))")

        with conn.cursor() as cur:
            from psycopg2.extras import execute_values
            execute_values(cur, """
                INSERT INTO ml_prediction (entry_id, target, model_id, prob)
                VALUES %s
                ON CONFLICT (entry_id, target)
                DO UPDATE SET model_id = EXCLUDED.model_id, prob = EXCLUDED.prob,
                              updated_at = NOW()
            """, records)
        conn.commit()
        print("done")
    finally:
        conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
