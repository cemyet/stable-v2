"""Train a classifier on entry_features and register it in ml_model.

Slice-driven: reads a saved slice (ml_slice) for filters / features / target,
or takes them as CLI args. Designed to be launched either by hand or by the
training page (which shells out to the ML venv).

Models are auto-named with a single lowercase word borrowed from a random horse
in the DB (e.g. "cash", "sisu") so repeated runs on the same track/target stay
distinguishable. Names are unique across ml_model.

Run under the ML venv:
    .venv-ml/bin/python -m scripts.train_model --track-id 64 --since 2012 \
        --test-since 2024 --scope track --monotonic-galrate
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import pandas as pd
from psycopg2.extras import Json

from core.db import get_connection

# Default galopp-risk feature set. NaN is meaningful (handled natively by the
# histogram gradient booster), so "no prior history" stays informative.
DEFAULT_NUMERIC = [
    "is_auto", "post", "is_springspar", "is_bakspar",
    "distance_m", "distance_added_m", "age",
    "horse_starts", "horse_winrate", "horse_galrate",
    "horse_galrate_auto", "horse_galrate_volt",
    "horse_galrate_auto_volt_delta", "horse_galrate_barefoot",
    "gal_pre", "days_since_last_start", "season_debut",
    "is_barefoot_all", "first_shoes_off",
    "trainer_galrate", "trainer_winrate", "driver_galrate", "driver_winrate",
    "starters", "post_galrate_auto", "post_galrate_volt",
    "earnings_pct_in_race",
    # phase-3/4 engineered signals (recency, momentum, instability, rank)
    "gal_recent_5", "field_avg_galrate",
    "gal_streak", "clean_streak", "galrate_method", "gal_rank_in_field",
    "method_switch", "barefoot_change", "distance_delta_vs_last",
    # phase-5 signals (recent-form rate, workload, driver instability)
    "galrate_recent_10", "starts_90d", "driver_switch",
    # horse-combination as-of gal rates (+ sample sizes) and seasonality
    "horse_trainer_galrate", "horse_trainer_starts",
    "horse_galrate_track", "horse_track_starts",
    "race_month",
]
DEFAULT_CATEGORICAL = ["sex", "breed_code"]

# Columns whose risk relationship is monotone-increasing: a higher historical
# gait-break rate (horse / breeding / post) should never *lower* predicted
# risk. Applied when --monotonic-galrate is set. gal_pre (broke last start) is
# a boolean but still monotone +.
MONOTONIC_INCREASING = {
    "horse_galrate", "horse_galrate_auto", "horse_galrate_volt",
    "horse_galrate_barefoot", "gal_pre",
    "sire_galrate", "dam_galrate",
    "post_galrate_auto", "post_galrate_volt",
    "post_galrate_auto_track", "post_galrate_volt_track",
    "gal_recent_5", "field_avg_galrate",
    "gal_streak", "galrate_method", "gal_rank_in_field",
    "galrate_recent_10",
    "horse_trainer_galrate", "horse_galrate_track",
}
# Risk falls as these rise (more clean starts in a row -> steadier).
MONOTONIC_DECREASING = {
    "clean_streak",
}
# Known categorical columns (everything else is treated numeric; NaN-aware).
KNOWN_CATEGORICAL = {"sex", "breed_code", "shoe_code"}
ARTIFACT_DIR = _ROOT / "web" / "ml_artifacts"


def pick_model_name(conn) -> str:
    """A single lowercase word from a random horse name, unique in ml_model."""
    with conn.cursor() as cur:
        cur.execute("SELECT LOWER(name) FROM ml_model")
        used = {r[0] for r in cur.fetchall()}
        for _ in range(40):
            cur.execute("""
                SELECT name FROM horse
                WHERE name ~ '^[A-Za-zÅÄÖåäöÉÜ. ]+$' AND length(name) >= 3
                ORDER BY random() LIMIT 60
            """)
            for (full,) in cur.fetchall():
                for word in str(full).split():
                    w = "".join(ch for ch in word.lower() if ch.isalpha())
                    if len(w) >= 3 and w not in used:
                        return w
    # Fallback: numbered.
    n = 1
    while f"model{n}" in used:
        n += 1
    return f"model{n}"


def build_where(filters: dict):
    where = ["NOT e.withdrawn"] if filters.get("starters_only", True) else []
    params: list = []
    tracks = filters.get("track_ids") or ([filters["track_id"]] if filters.get("track_id") else [])
    if tracks:
        where.append("ef.track_id = ANY(%s)")
        params.append(list(tracks))
    if filters.get("since"):
        where.append("ef.race_date >= make_date(%s,1,1)")
        params.append(int(filters["since"]))
    if filters.get("year_to"):
        where.append("ef.race_date < make_date(%s,1,1)")
        params.append(int(filters["year_to"]) + 1)
    method = (filters.get("method") or "").lower()
    if method == "auto":
        where.append("ef.is_auto")
    elif method == "volt":
        where.append("ef.is_auto = false")
    if filters.get("complete"):
        where.append("ef.race_complete")
    for col in filters.get("notnull", []):
        if col.replace("_", "").isalnum():
            where.append(f"ef.{col} IS NOT NULL")
    return (" AND ".join(where) if where else "TRUE"), params


def load_frame(conn, features, target, filters):
    cols = ", ".join(f"ef.{c}" for c in features)
    where, params = build_where(filters)
    sql = f"""
        SELECT ef.entry_id, ef.race_date, ef.{target}::int AS y, {cols}
        FROM entry_features ef
        JOIN entry e ON e.entry_id = ef.entry_id
        WHERE {where} AND ef.{target} IS NOT NULL
        ORDER BY ef.race_date
    """
    with conn.cursor() as cur:
        cur.execute(sql, params)
        names = [d[0] for d in cur.description]
        rows = cur.fetchall()
    return pd.DataFrame(rows, columns=names)


def coerce(df, numeric, categorical):
    for c in numeric:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("float64")
    for c in categorical:
        df[c] = df[c].astype("category")
    df["y"] = df["y"].astype(int)
    return df


def evaluate(model, X, y):
    from sklearn.metrics import (roc_auc_score, average_precision_score, log_loss,
                                 brier_score_loss, accuracy_score, confusion_matrix)
    p = model.predict_proba(X)[:, 1]
    base = float(y.mean())
    out = {
        "n": int(len(y)), "base_rate": round(base, 4),
        "roc_auc": round(float(roc_auc_score(y, p)), 4),
        "pr_auc": round(float(average_precision_score(y, p)), 4),
        "log_loss": round(float(log_loss(y, p)), 4),
        "brier": round(float(brier_score_loss(y, p)), 4),
        "accuracy_0.5": round(float(accuracy_score(y, (p >= 0.5).astype(int))), 4),
    }
    tn, fp, fn, tp = confusion_matrix(y, (p >= 0.5).astype(int)).ravel()
    out["confusion_0.5"] = {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}
    dfp = pd.DataFrame({"p": p, "y": y.values})
    dfp["bin"] = pd.qcut(dfp["p"].rank(method="first"), 10, labels=False)
    cal = dfp.groupby("bin").agg(pred=("p", "mean"), actual=("y", "mean"), n=("y", "size")).reset_index()
    out["calibration"] = [{"decile": int(r.bin) + 1, "pred": round(float(r.pred), 4),
                           "actual": round(float(r.actual), 4), "n": int(r.n)} for r in cal.itertuples()]
    out["top_decile_lift"] = round(float(cal.iloc[-1]["actual"] / base), 3) if base else None
    return out


class NotEnoughData(Exception):
    """Raised when a slice has too few rows for a forward test split."""


def train_one(conn, *, scope="track", track_id=None, track_name=None,
              since=2012, test_since=2024, target="y_gal",
              monotonic_galrate=False, max_iter=400, learning_rate=0.06,
              max_leaf_nodes=31, method="", features="", drop="",
              extra_features="", slice_id=None, shuffle=False,
              label="xgal (expected galopp)", log=print) -> dict:
    """Train + register a single model. Returns a summary dict. Raises
    NotEnoughData if the slice can't support a forward test split, and
    ValueError for a bad slice id. The caller owns the connection."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.inspection import permutation_importance

    method_label = method if method else "any"
    filters = {"starters_only": True, "since": since, "method": method or None}
    numeric = list(DEFAULT_NUMERIC)
    categorical = list(DEFAULT_CATEGORICAL)

    if slice_id:
        with conn.cursor() as cur:
            cur.execute("SELECT name, filters, target, features FROM ml_slice WHERE slice_id=%s",
                        (slice_id,))
            row = cur.fetchone()
        if not row:
            raise ValueError(f"slice {slice_id} not found")
        _, sl_filters, sl_target, sl_features = row
        filters.update(sl_filters or {})
        target = target or sl_target or "y_gal"
        if sl_features:
            feats = [f for f in sl_features if not f.startswith("y_")]
            numeric = [f for f in feats if f not in DEFAULT_CATEGORICAL]
            categorical = [f for f in feats if f in DEFAULT_CATEGORICAL]
        tracks = filters.get("track_ids") or []
        if len(tracks) == 1:
            track_id = tracks[0]
            scope = "track"

    # Explicit features override the default/slice feature set entirely.
    explicit = [x.strip() for x in features.split(",") if x.strip()] if isinstance(features, str) else list(features or [])
    if explicit:
        numeric = [f for f in explicit if f not in KNOWN_CATEGORICAL]
        categorical = [f for f in explicit if f in KNOWN_CATEGORICAL]

    if scope == "track" and track_id:
        filters["track_ids"] = [track_id]
    if scope == "general" and "track_id" not in numeric:
        numeric.append("track_id")  # let the tree specialize per track

    extra = extra_features.split(",") if isinstance(extra_features, str) else list(extra_features or [])
    for f in [x.strip() for x in extra if (x.strip() if isinstance(x, str) else x)]:
        if f in KNOWN_CATEGORICAL and f not in categorical:
            categorical.append(f)
        elif f not in numeric and f not in categorical:
            numeric.append(f)

    drop_set = {x.strip() for x in (drop.split(",") if isinstance(drop, str) else (drop or [])) if x}
    if drop_set:
        numeric = [f for f in numeric if f not in drop_set]
        categorical = [f for f in categorical if f not in drop_set]

    feats = numeric + categorical
    df = load_frame(conn, feats, target, filters)
    df = coerce(df, numeric, categorical)
    cut = pd.Timestamp(f"{test_since}-01-01").date()
    if shuffle:
        from sklearn.model_selection import train_test_split
        frac = float((df["race_date"] >= cut).mean()) or 0.15
        train, test = train_test_split(df, test_size=frac, random_state=42, stratify=df["y"])
        log(f"[SHUFFLE] loaded {len(df):,} | train {len(train):,} | test {len(test):,} "
            f"(random {frac:.0%}) | {target} rate train {train['y'].mean():.3f} "
            f"test {test['y'].mean():.3f}")
    else:
        train = df[df["race_date"] < cut]
        test = df[df["race_date"] >= cut]
        log(f"loaded {len(df):,} | train {len(train):,} (<{cut}) | test {len(test):,} "
            f"| {target} rate train {train['y'].mean():.3f} test {test['y'].mean():.3f}")
    if len(test) < 500 or train["y"].nunique() < 2:
        raise NotEnoughData(f"only {len(test)} test rows / {len(train)} train rows")

    Xtr, ytr, Xte, yte = train[feats], train["y"], test[feats], test["y"]

    mono = None
    if monotonic_galrate:
        mono = {f: (1 if f in MONOTONIC_INCREASING else
                    -1 if f in MONOTONIC_DECREASING else 0) for f in feats}

    model = HistGradientBoostingClassifier(
        categorical_features="from_dtype",
        learning_rate=learning_rate, max_iter=max_iter,
        max_leaf_nodes=max_leaf_nodes, l2_regularization=1.0,
        min_samples_leaf=80, early_stopping=True, validation_fraction=0.12,
        monotonic_cst=mono, random_state=42,
    )
    model.fit(Xtr, ytr)
    train_m, test_m = evaluate(model, Xtr, ytr), evaluate(model, Xte, yte)
    log(f"TEST roc_auc={test_m['roc_auc']} pr_auc={test_m['pr_auc']} "
        f"logloss={test_m['log_loss']} lift={test_m['top_decile_lift']} iters={model.n_iter_}")

    rng = np.random.RandomState(42)
    idx = rng.choice(len(Xte), size=min(15000, len(Xte)), replace=False)
    perm = permutation_importance(model, Xte.iloc[idx], yte.iloc[idx], scoring="roc_auc",
                                  n_repeats=4, random_state=42, n_jobs=-1)
    importances = sorted([{"feature": f, "importance": round(float(m), 5)}
                          for f, m in zip(feats, perm.importances_mean)],
                         key=lambda d: d["importance"], reverse=True)
    log(f"top: {[d['feature'] for d in importances[:8]]}")

    if shuffle:
        log("[SHUFFLE] diagnostic run — model NOT registered or saved.")
        return {"skipped": "shuffle diagnostic", "test_auc": test_m["roc_auc"]}

    name = pick_model_name(conn)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    import joblib
    artifact = ARTIFACT_DIR / f"{target}_{name}.joblib"
    joblib.dump({"model": model, "features": feats,
                 "numeric": numeric, "categorical": categorical}, artifact)

    slice_def = {
        "track_id": track_id, "track_name": track_name,
        "years": [since, int(df["race_date"].max().year)],
        "test_since": test_since, "n_total": int(len(df)),
        "n_train": int(len(train)), "n_test": int(len(test)),
        "starters_only": True, "features": feats, "n_features": len(feats),
        "monotonic_galrate": bool(monotonic_galrate), "method": method_label,
        "hyperparams": {"learning_rate": learning_rate, "max_iter": max_iter,
                        "max_leaf_nodes": max_leaf_nodes},
        "label": label,
    }
    metrics = {"train": train_m, "test": test_m, "importances": importances,
               "n_iter": int(model.n_iter_)}
    with conn.cursor() as cur:
        cur.execute("""
            INSERT INTO ml_model (name, scope, track_id, track_name, target, algo,
                                  slice_def, metrics, artifact_path)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s) RETURNING model_id
        """, (name, scope, track_id, track_name, target, "HistGradientBoosting",
              Json(slice_def), Json(metrics), str(artifact.relative_to(_ROOT))))
        mid = cur.fetchone()[0]
    conn.commit()
    log(f"registered ml_model #{mid}: '{name}' ({scope}, {target})")
    return {"model_id": mid, "name": name, "scope": scope, "track_id": track_id,
            "track_name": track_name, "target": target,
            "test_auc": test_m["roc_auc"], "train_auc": train_m["roc_auc"],
            "n_total": int(len(df)), "n_train": int(len(train)),
            "n_test": int(len(test)), "n_features": len(feats)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--slice-id", type=int, default=None)
    ap.add_argument("--track-id", type=int, default=None)
    ap.add_argument("--track-name", default=None)
    ap.add_argument("--since", type=int, default=2012)
    ap.add_argument("--test-since", type=int, default=2024)
    ap.add_argument("--target", default="y_gal")
    ap.add_argument("--scope", default="track", choices=["track", "general"])
    ap.add_argument("--algo", default="histgbm", choices=["histgbm"])
    ap.add_argument("--max-iter", type=int, default=400)
    ap.add_argument("--learning-rate", type=float, default=0.06)
    ap.add_argument("--max-leaf-nodes", type=int, default=31)
    ap.add_argument("--monotonic-galrate", action="store_true",
                    help="constrain risk to rise with any historical galrate / gal_pre")
    ap.add_argument("--features", default="",
                    help="explicit comma-separated feature list (overrides defaults)")
    ap.add_argument("--drop", default="",
                    help="comma-separated features to remove from the active set")
    ap.add_argument("--extra-features", default="", help="comma-separated cols to append")
    ap.add_argument("--method", default="", choices=["", "auto", "volt"],
                    help="filter to a start method (volt/auto); empty = all methods")
    ap.add_argument("--shuffle", action="store_true",
                    help="DIAGNOSTIC ONLY: random train/test split instead of time-based "
                         "(inflates AUC; not representative of live forecasting). Not registered.")
    ap.add_argument("--label", default="xgal (expected galopp)", help="human description suffix")
    args = ap.parse_args()

    conn = get_connection()
    try:
        train_one(
            conn, scope=args.scope, track_id=args.track_id, track_name=args.track_name,
            since=args.since, test_since=args.test_since, target=args.target,
            monotonic_galrate=args.monotonic_galrate, max_iter=args.max_iter,
            learning_rate=args.learning_rate, max_leaf_nodes=args.max_leaf_nodes,
            method=args.method, features=args.features, drop=args.drop,
            extra_features=args.extra_features, slice_id=args.slice_id,
            shuffle=args.shuffle, label=args.label, log=print,
        )
    except ValueError as e:
        print(str(e), file=sys.stderr)
        return 2
    except NotEnoughData as e:
        print(f"not enough data for a forward test split ({e})", file=sys.stderr)
        return 3
    finally:
        conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
