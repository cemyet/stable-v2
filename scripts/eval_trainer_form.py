"""
Evaluate the candidate TRAINER-FORM variants in `trainer_form_exp` (v2) and
crown a gold-standard column. Runs in the ML venv (.venv-ml).

What changed vs the first attempt (per user feedback)
-----------------------------------------------------
1. No `expw` in X. The market is NOT a feature here — it would dominate and
   stop the model from "looking closely" at the form variants. Instead the
   market is baked into the TARGET (we predict beating the market).
2. SE-only population, odds-gated, with a MIN-STARTS gate on the trainer's
   recent history (--min-starts, default 20 over the last 50 starts) to cut
   small-sample noise.
3. Every target entry is guaranteed to have odds (enforced in the builder).
4. New targets that ask "did the horse beat expectations?":
     y_beat_exp  (bool)  — finished ahead of its odds rank
     y_outperf   (real)  — how far above/below expectation (pos_beat_norm)
   Plus y_top3 / y_win (from entry_features) for continuity, still WITHOUT
   expw in X.

Outputs, per target: univariate strength of each variant + permutation
importance in a combined model. Trained models are SAVED to ml_model with
gal-model-style random names (pick_model_name).

Usage
-----
    .venv-ml/bin/python -m scripts.eval_trainer_form
    .venv-ml/bin/python -m scripts.eval_trainer_form --min-starts 20 --test-since 2024
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from psycopg2.extras import Json  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402
from sklearn.ensemble import (HistGradientBoostingClassifier,  # noqa: E402
                              HistGradientBoostingRegressor)
from sklearn.inspection import permutation_importance  # noqa: E402
from sklearn.metrics import r2_score, roc_auc_score  # noqa: E402

from core.db import get_connection  # noqa: E402
from scripts.train_model import pick_model_name  # noqa: E402

ARTIFACT_DIR = _ROOT / "web" / "ml_artifacts"

# Market-RESIDUAL family (edge / value — "how much they beat the market")
MARKET_VARIANTS = [
    "tf_posbeat_90", "tf_posbeat_180", "tf_posbeat_last50", "tf_posbeat_last100",
    "tf_posbeat_last150", "tf_posbeat_last200",
    "tf_posbeat_decay120", "tf_posbeat_decay365", "tf_posbeat_last100_shrunk",
    "tf_beatrate_90", "tf_beatrate_180", "tf_beatrate_last50",
    "tf_winresid_90", "tf_winresid_180", "tf_winresid_last50",
]
# Market-AGNOSTIC "actual form" family (how high horses finish lately)
PERF_VARIANTS = [
    "tf_perf_30", "tf_perf_90", "tf_perf_180", "tf_perf_last50", "tf_perf_last100",
    "tf_perf_decay120", "tf_perf_decay365",
    "tf_winrate_30", "tf_winrate_90", "tf_winrate_last50", "tf_top3rate_90",
    "tf_top3_career", "tf_top3_last50", "tf_win_last50",
]
VARIANTS = MARKET_VARIANTS + PERF_VARIANTS

# (target, kind): kind drives classifier vs regressor + metric
TARGETS = [
    ("y_beat_exp", "clf"),   # market-residual: beat the odds rank
    ("y_outperf",  "reg"),   # market-residual: how far above expectation
    ("y_top3",     "clf"),   # raw outcome, for continuity
    ("y_win",      "clf"),
]


def load_frame(conn, min_starts: int) -> pd.DataFrame:
    cols = ", ".join(f"x.{c}" for c in VARIANTS)
    sql = f"""
        SELECT {cols},
               x.y_beat_exp, x.y_outperf, x.race_date,
               ef.y_top3::int AS y_top3,
               ef.y_win::int  AS y_win
        FROM trainer_form_exp x
        JOIN entry_features ef ON ef.entry_id = x.entry_id
        WHERE x.tf_n_last50 >= {int(min_starts)}
          AND x.tf_posbeat_last50 IS NOT NULL
    """
    print(f"loading frame (min_starts={min_starts})…", flush=True)
    df = pd.read_sql(sql, conn)
    df["race_date"] = pd.to_datetime(df["race_date"])
    print(f"  rows: {len(df):,}", flush=True)
    return df


def uni_clf(test, target, v):
    s, y = test[v], test[target]
    m = s.notna()
    if m.sum() < 500 or y[m].nunique() < 2:
        return None
    return roc_auc_score(y[m], s[m])


def uni_reg(test, target, v):
    s = test[[v, target]].dropna()
    if len(s) < 500:
        return None
    rho, _ = spearmanr(s[v], s[target])
    return rho


def _family(v: str) -> str:
    return "market" if v in MARKET_VARIANTS else "perf"


def univariate(test, target, kind):
    rows = []
    for v in VARIANTS:
        if kind == "clf":
            auc = uni_clf(test, target, v)
            rows.append({"variant": v, "fam": _family(v), "auc": auc,
                         "strength": abs(auc - 0.5) if auc is not None else None})
        else:
            rho = uni_reg(test, target, v)
            rows.append({"variant": v, "fam": _family(v), "spearman": rho,
                         "strength": abs(rho) if rho is not None else None})
    return (pd.DataFrame(rows)
            .sort_values("strength", ascending=False, na_position="last"))


def fit_model(train, test, target, kind):
    Xtr, ytr = train[VARIANTS], train[target]
    Xte, yte = test[VARIANTS], test[target]
    common = dict(learning_rate=0.06, max_iter=400, max_leaf_nodes=31,
                  l2_regularization=1.0, min_samples_leaf=80,
                  early_stopping=True, validation_fraction=0.12, random_state=42)
    if kind == "clf":
        model = HistGradientBoostingClassifier(**common)
        model.fit(Xtr, ytr)
        score = roc_auc_score(yte, model.predict_proba(Xte)[:, 1])
        scoring, metric_name = "roc_auc", "AUC"
    else:
        model = HistGradientBoostingRegressor(**common)
        model.fit(Xtr, ytr)
        pred = model.predict(Xte)
        score = r2_score(yte, pred)
        rho, _ = spearmanr(pred, yte)
        scoring, metric_name = "r2", f"R2 (spearman {rho:.3f})"

    n = min(120_000, len(Xte))
    idx = np.random.RandomState(42).choice(len(Xte), n, replace=False)
    perm = permutation_importance(model, Xte.iloc[idx], yte.iloc[idx],
                                  scoring=scoring, n_repeats=4,
                                  random_state=42, n_jobs=-1)
    imp = (pd.DataFrame({"feature": VARIANTS, "importance": perm.importances_mean})
           .sort_values("importance", ascending=False))
    return model, score, metric_name, imp


def save_model(conn, target, kind, model, score, imp, n_train, n_test, min_starts):
    name = pick_model_name(conn)
    algo = ("HistGradientBoostingClassifier" if kind == "clf"
            else "HistGradientBoostingRegressor")
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    artifact = ARTIFACT_DIR / f"{target}_{name}.joblib"
    joblib.dump({"model": model, "features": VARIANTS,
                 "numeric": VARIANTS, "categorical": []}, artifact)
    slice_def = {
        "label": "trainer-form gold-standard experiment (r3: +perf family)",
        "source_table": "trainer_form_exp",
        "kind": kind, "features": VARIANTS, "n_features": len(VARIANTS),
        "market_features": MARKET_VARIANTS, "perf_features": PERF_VARIANTS,
        "min_starts_last50": min_starts, "scope_country": "SE",
        "note": "two families: PERF (market-agnostic finish percentile / win "
                "rate = 'actual form') vs MARKET (odds-rank outperformance = "
                "edge/value). NO market odds feature in X. Finding: perf "
                "predicts winning (y_win/y_top3); market predicts beating the "
                "market (y_beat_exp/y_outperf) and anti-correlates with winning.",
    }
    metrics = {
        "score": round(float(score), 5),
        "n_train": int(n_train), "n_test": int(n_test),
        "importances": [{"feature": r.feature,
                         "importance": round(float(r.importance), 6)}
                        for r in imp.itertuples()],
    }
    with conn.cursor() as cur:
        cur.execute("""
            INSERT INTO ml_model (name, scope, target, algo, slice_def, metrics, artifact_path)
            VALUES (%s,'general',%s,%s,%s,%s,%s) RETURNING model_id
        """, (name, target, algo, Json(slice_def), Json(metrics),
              str(artifact.relative_to(_ROOT))))
        mid = cur.fetchone()[0]
    conn.commit()
    print(f"  saved model '{name}' (model_id={mid}) -> {artifact.name}", flush=True)
    return name


def main() -> int:
    ap = argparse.ArgumentParser("eval_trainer_form")
    ap.add_argument("--test-since", type=int, default=2024)
    ap.add_argument("--min-starts", type=int, default=20,
                    help="min trainer starts over last 50 (default 20)")
    ap.add_argument("--save", action="store_true", default=True)
    ap.add_argument("--no-save", dest="save", action="store_false")
    args = ap.parse_args()

    conn = get_connection()
    df = load_frame(conn, args.min_starts)
    cut = pd.Timestamp(f"{args.test_since}-01-01")
    train = df[df["race_date"] < cut]
    test = df[df["race_date"] >= cut]
    print(f"train {len(train):,}  test {len(test):,}\n", flush=True)

    for target, kind in TARGETS:
        # Drop rows where the target is undefined (entry_features y_top3/y_win
        # are NULL for DNF/DQ). y_beat_exp / y_outperf are always defined.
        tr = train.dropna(subset=[target]).copy()
        te = test.dropna(subset=[target]).copy()
        if kind == "clf":
            tr[target] = tr[target].astype(int)
            te[target] = te[target].astype(int)

        print("=" * 74)
        if kind == "clf":
            print(f"TARGET: {target} [classification]  base rate(test)="
                  f"{te[target].mean():.3f}")
        else:
            print(f"TARGET: {target} [regression]  mean(test)="
                  f"{te[target].mean():.4f} sd={te[target].std():.3f}")
        print("=" * 74)

        uni = univariate(te, target, kind)
        print("\n-- Univariate strength (test) --")
        fmts = {"auc": lambda v: f"{v:.4f}" if v == v else "  n/a",
                "spearman": lambda v: f"{v:+.4f}" if v == v else "  n/a",
                "strength": lambda v: f"{v:.4f}" if v == v else "  n/a"}
        print(uni.to_string(index=False, formatters=fmts))

        model, score, metric_name, imp = fit_model(tr, te, target, kind)
        print(f"\ncombined model {metric_name} = {score:.4f}")
        print("-- Permutation importance (marginal over the other variants) --")
        print(imp.to_string(index=False,
              formatters={"importance": lambda v: f"{v:+.6f}"}))

        if args.save:
            save_model(conn, target, kind, model, score, imp,
                       len(tr), len(te), args.min_starts)
        print()

    conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
