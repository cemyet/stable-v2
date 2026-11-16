"""What the race models would have returned in real ATG pools.

Reads each model's out-of-sample predictions (ml_backtest, written when
scripts/build_race_models.py fits it) plus the live predictions it has made
since (ml_prediction), settles them against the stored pools
(scrapers/atg_games.py) with core/pools.py, and writes the result to
ml_model.metrics['pools'] — the whole period, and the live part on its own
under 'live'. Runs nightly after the pool scrape (jobs/update.py).

Only models whose output is a win probability are scored: the rating pass,
xodds, xwin and xedge. xlead's, xfront's and xgal's value reaches the pools
through xwin and xedge. xedge's live predictions are tilts on the market,
so they are settled with the closing odds they would have been played on.

    python -m scripts.eval_pools               # the live rating / xodds / xwin / xedge
    python -m scripts.eval_pools --model-id 41 42
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from psycopg2.extras import Json

from core import pools
from core.db import get_connection

TARGETS = ('rating', 'xodds', 'xwin', 'xedge')
SINCE = '2021-01-01'


def live_models(cur):
    out = []
    for t in TARGETS:
        cur.execute("SELECT model_id, target, name FROM ml_model WHERE target = %s "
                    "ORDER BY created_at DESC LIMIT 1", (t,))
        row = cur.fetchone()
        if row:
            out.append(row)
    return out


def predictions(cur, model_id, target):
    cur.execute("SELECT entry_id, prob FROM ml_backtest WHERE model_id = %s", (model_id,))
    backtest = {int(e): float(p) for e, p in cur.fetchall()}
    cur.execute("SELECT entry_id, prob FROM ml_prediction WHERE model_id = %s AND target = %s "
                "AND prob IS NOT NULL", (model_id, target))
    live = {int(e): float(p) for e, p in cur.fetchall() if int(e) not in backtest}
    return backtest, live


def store_coupons(cur, model_id, target, coupons):
    """Replace the settled systems of this model and of older ones of its target."""
    import io
    import json
    cur.execute("DELETE FROM ml_pool_coupon WHERE model_id IN "
                "(SELECT model_id FROM ml_model WHERE target = %s) OR model_id = %s", (target, model_id))
    buf = io.StringIO()
    for gid, src, rule, rows, cost, ret, right, picks in coupons:
        buf.write(f"{model_id}\t{gid}\t{rule}\t{src}\t{rows}\t{cost:.2f}\t{ret:.2f}\t{right}\t{json.dumps(picks)}\n")
    buf.seek(0)
    cur.copy_expert("COPY ml_pool_coupon (model_id, atg_game_id, rule, src, rows, cost, ret, legs_right, picks) "
                    "FROM STDIN", buf)


def score_models(conn, models, since=SINCE, log=print):
    """Pool results of (model_id, target, name) models, into their metrics."""
    t0 = time.time()
    st, games = pools.load(conn, since)
    log(f"[pools] {len(st):,} starters, {len(games):,} marking games ({time.time() - t0:.0f}s)")
    for mid, target, name in models:
        with conn.cursor() as cur:
            backtest, live = predictions(cur, mid, target)
        if target == 'xedge' and live:
            live = pools.tilted(st, live)
        if not backtest and not live:
            log(f"[pools] #{mid} {name} ({target}): no predictions to score")
            continue
        coupons = []
        m = pools.evaluate(st, games, {**live, **backtest}, coupons=coupons)
        if not m:
            log(f"[pools] #{mid} {name} ({target}): no overlap with the stored pools")
            continue
        m['evaluated_at'] = datetime.now().isoformat(timespec='seconds')
        m['n_backtest'], m['n_live'] = len(backtest), len(live)
        if live:
            m['live'] = pools.evaluate(st, games, live) or None
        with conn.cursor() as cur:
            cur.execute("UPDATE ml_model SET metrics = metrics || jsonb_build_object('pools', %s::jsonb) "
                        "WHERE model_id = %s", (Json(m), mid))
            store_coupons(cur, mid, target, coupons)
        conn.commit()
        v4 = next((r for r in (m.get('V4') or {}).get('model', []) if r['rule'] == 'cover 70%'), None)
        log(f"[pools] #{mid} {name} ({target}): {m['races']:,} races {m['from']} .. {m['to']}"
            + (f", V4 cover 70% {v4['roi']:+.1%}" if v4 else '')
            + f" ({time.time() - t0:.0f}s)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model-id', type=int, nargs='*', help='models to score (default: the live ones)')
    ap.add_argument('--since', default=SINCE, help='first pool day read')
    args = ap.parse_args()
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            if args.model_id:
                cur.execute("SELECT model_id, target, name FROM ml_model WHERE model_id = ANY(%s)",
                            (args.model_id,))
                models = cur.fetchall()
            else:
                models = live_models(cur)
        score_models(conn, models, args.since)
    finally:
        conn.close()
    return 0


if __name__ == '__main__':
    sys.exit(main())
