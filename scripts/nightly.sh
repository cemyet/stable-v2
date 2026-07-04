#!/bin/bash
# Nightly stable-v2 pipeline: scrape + derive + ML score, then publish the
# serving set to Supabase. Invoked by launchd (local) at 01:00 CET.
#
# Fully v1-independent (USE_V1_BRIDGE defaults to False). The publish step runs
# only if SUPABASE_DATABASE_URL is set (in the repo .env or the environment).
set -euo pipefail

REPO="/Users/jakob/Dev/stable-v2"
PYTHON="/usr/bin/python3"
cd "$REPO"

mkdir -p logs
STAMP="$(date +%Y%m%d)"
LOG="logs/nightly_${STAMP}.log"

echo "==== nightly start $(date -Iseconds) ====" >>"$LOG"
# --mode all: native ATG + native ST + kmtid + letrot + cleanup, then --publish
# pushes the serving set to Supabase.
"$PYTHON" -m jobs.update --mode all --publish >>"$LOG" 2>&1
echo "==== nightly end $(date -Iseconds) exit=$? ====" >>"$LOG"
