#!/usr/bin/env bash
# Watches the catchup log and unloads the LaunchAgent the moment the passport
# phase drains, so the loop never enters the foreign-raceday phase.
#
# Needed because the currently-running loop was started before --stop-after-shallow
# existed; a running Python process won't pick up the new flag. Unload (not kill)
# is required: KeepAlive/SuccessfulExit=false would restart a killed process.
set -uo pipefail

ROOT="/Users/jakob/Dev/stable-v2"
LOG="$ROOT/logs/catchup_launchd.out"
PLIST="$HOME/Library/LaunchAgents/com.stable-v2.catchup.plist"
MARK="$ROOT/logs/catchup_boundary_stop.log"

echo "[$(date '+%F %T')] boundary watcher started" >>"$MARK"

while true; do
  if ! pgrep -f run_catchup_loop >/dev/null 2>&1; then
    echo "[$(date '+%F %T')] loop no longer running; watcher exiting" >>"$MARK"
    exit 0
  fi

  # Either signal means shallow is finished and foreign is imminent/starting.
  if tail -40 "$LOG" 2>/dev/null | grep -qE 'shallow backlog drained|=== foreign batch|foreign-drain —'; then
    echo "[$(date '+%F %T')] phase boundary reached — unloading LaunchAgent" >>"$MARK"
    launchctl unload "$PLIST" >>"$MARK" 2>&1
    sleep 3
    pkill -f run_catchup_loop >/dev/null 2>&1
    echo "[$(date '+%F %T')] stopped. remaining procs: $(pgrep -fc run_catchup_loop 2>/dev/null || echo 0)" >>"$MARK"
    exit 0
  fi

  sleep 3
done
