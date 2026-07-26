#!/bin/bash
# Friendly live progress for the weekend catchup.
# Usage:
#   bash scripts/catchup_live.sh          # one snapshot
#   bash scripts/catchup_live.sh --watch  # refresh every 15s (Ctrl-C to stop)

set -euo pipefail
cd /Users/jakob/Dev/stable-v2
PY="/Library/Developer/CommandLineTools/Library/Frameworks/Python3.framework/Versions/3.9/Resources/Python.app/Contents/MacOS/Python"
[[ -x "$PY" ]] || PY="$(command -v python3)"

snapshot() {
  clear 2>/dev/null || true
  echo "========================================"
  echo "  CATCHUP PROGRESS  $(date '+%H:%M:%S')"
  echo "========================================"
  echo

  if pgrep -f 'run_catchup_loop' >/dev/null 2>&1; then
    ETIME=$(ps -o etime= -p "$(pgrep -f 'run_catchup_loop' | head -1)" | tr -d ' ')
    echo "Status:     RUNNING  (uptime $ETIME)"
  else
    echo "Status:     STOPPED"
  fi

  "$PY" - <<'PY'
from datetime import datetime, timezone
from core.db import get_connection

START = datetime(2026, 7, 23, 21, 1, 0)  # launchd catchup start (local naive)
conn = get_connection()
cur = conn.cursor()

# Scrapes this session
cur.execute(
    "SELECT COUNT(*), MIN(scraped_at), MAX(scraped_at) "
    "FROM st_horse_scrape_log WHERE scraped_at >= %s AND http_status = 200",
    (START,),
)
n, first, last = cur.fetchone()
n = n or 0

cur.execute(
    "SELECT COUNT(*) FROM st_horse_scrape_log "
    "WHERE scraped_at > NOW() - INTERVAL '5 minutes' AND http_status = 200"
)
recent = cur.fetchone()[0]

# Backlogs (365d lookback matches the running job)
from etl.import_st import (
    discover_shallow_st_horse_ids,
    discover_raceday_ids_from_horse_raw,
)
shallow = len(discover_shallow_st_horse_ids(conn, lookback_days=365))
foreign = len(discover_raceday_ids_from_horse_raw(conn))

# Rate / ETA from last 15 minutes if possible
cur.execute(
    "SELECT COUNT(*) FROM st_horse_scrape_log "
    "WHERE scraped_at > NOW() - INTERVAL '15 minutes' AND http_status = 200"
)
n15 = cur.fetchone()[0]
per_hour = (n15 / 15.0) * 60.0 if n15 else 0.0
eta_h = (shallow / per_hour) if per_hour > 0 else None

# Batch progress: each shallow batch is 400; infer from session scrapes
batch_size = 400
in_batch = n % batch_size
batch_num = (n // batch_size) + 1
pct_batch = 100.0 * in_batch / batch_size if batch_size else 0

bar_w = 28
filled = int(bar_w * in_batch / batch_size) if batch_size else 0
bar = "█" * filled + "░" * (bar_w - filled)

print()
print(f"Phase:      Healing horse passports (shallow)")
print(f"This batch: [{bar}] {in_batch}/{batch_size}  ({pct_batch:.0f}%)")
print(f"Batch #:    ~{batch_num}")
print()
print(f"Done tonight:     {n:,} passports scraped")
print(f"Last 5 minutes:   {recent}")
print(f"Rate:             {per_hour:.0f} horses/hour" if per_hour else "Rate:             (warming up)")
print(f"Latest scrape:    {last}")
print()
print(f"Still to do:")
print(f"  Passports left: {shallow:,}")
print(f"  Foreign racedays left: {foreign:,}")
if eta_h is not None:
    print(f"  ETA for passports: ~{eta_h:.1f} hours at current rate")
print()
print("Tip: leave this running with:  bash scripts/catchup_live.sh --watch")
print("Good = Status RUNNING and Last 5 minutes > 0")
PY
}

if [[ "${1:-}" == "--watch" || "${1:-}" == "-w" ]]; then
  while true; do
    snapshot
    sleep 15
  done
else
  snapshot
fi
