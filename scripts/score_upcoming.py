"""Score upcoming-race entries with the race-view models.

Writes these targets per entry into ml_prediction so the race view can read
them without loading the ML stack:
  - y_gal_general — best general-scope galopp model (kvikk, etc.)
  - y_gal_track   — best track galopp model for that track, or general when none
  - y_lead        — xlead: share of being in front after 100–150 m (SE, per method)
  - y_front       — xfront: share of being in front at 700 m (SE, per method)
  - xgal          — xgal: chance of a gallop, the chain's own (an input to xwin / xedge)
  - xodds         — xodds: share of the win pool the starter "should" draw
  - xwin          — xwin: chance of winning, grown from xodds
  - xedge         — xedge's tilt on the market: multiply by the win odds' shares
                    and renormalise per race for its chances (core/xstack.tilted)

Run under the ML venv:
    .venv-ml/bin/python -m scripts.score_upcoming
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import datetime as dt
import traceback

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


def _latest(cur, target, method=None):
    cur.execute("""
        SELECT model_id, name, artifact_path FROM ml_model
        WHERE target = %s AND (%s::text IS NULL OR slice_def->>'method' = %s)
        ORDER BY created_at DESC LIMIT 1
    """, (target, method, method))
    return cur.fetchone()


def score_race_models(conn):
    """Both passes up to today, then xlead / xfront / xgal -> xodds -> xwin,
    xedge (core/xstack.py)."""
    from core import xstack as XS
    with conn.cursor() as cur:
        lead = {m: _latest(cur, "y_lead", m) for m in ("auto", "volt")}
        front = {m: _latest(cur, "y_front", m) for m in ("auto", "volt")}
        single = {t: _latest(cur, t) for t in ("xgal", "xodds", "xwin", "xedge", "rating")}
    if not any(lead.values()) and not single["xodds"]:
        print("race models: nothing registered")
        return []

    def art(m):
        return joblib.load(_ROOT / m[2]) if m else None

    out = XS.score(
        conn, today=dt.date.today(),
        lead_bundles={m: art(v) for m, v in lead.items()},
        front_bundles={m: art(v) for m, v in front.items()},
        gal_art=art(single["xgal"]), xodds_art=art(single["xodds"]),
        xwin_art=art(single["xwin"]), xedge_art=art(single["xedge"]),
        log=lambda msg: print(f"race models: {msg}"))
    method = out["lead_method"]
    records = []
    for target, mids in (("y_lead", lead), ("y_front", front)):
        mid = {m: v[0] for m, v in mids.items() if v}
        records += [(e, target, mid[method[e]], p) for e, p in out[target].items() if method.get(e) in mid]
    for target, m in single.items():
        if m:
            records += [(e, target, m[0], p) for e, p in out[target].items()]
    for target in ("y_lead", "y_front", *single):
        print(f"race models: {sum(1 for r in records if r[1] == target):,} {target}")
    return records


def score_gal(conn):
    with conn.cursor() as cur:
        gen = best_general_model(cur)
        if not gen:
            print("no general y_gal model registered", file=sys.stderr)
            return []
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
        return []

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
    return records


def _write(conn, records):
    from psycopg2.extras import execute_values
    with conn.cursor() as cur:
        execute_values(cur, """
            INSERT INTO ml_prediction (entry_id, target, model_id, prob)
            VALUES %s
            ON CONFLICT (entry_id, target)
            DO UPDATE SET model_id = EXCLUDED.model_id, prob = EXCLUDED.prob,
                          updated_at = NOW()
        """, records)
    conn.commit()


def main() -> int:
    conn = get_connection()
    failed = 0
    try:
        for name, scorer in (("gal", score_gal), ("race models", score_race_models)):
            try:
                records = scorer(conn)
                if records:
                    _write(conn, records)
            except Exception:
                conn.rollback()
                failed += 1
                print(f"{name}: scoring failed", file=sys.stderr)
                traceback.print_exc()
        print("done" + (f" ({failed} scorer(s) failed)" if failed else ""))
    finally:
        conn.close()
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
