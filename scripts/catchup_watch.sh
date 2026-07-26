#!/bin/bash
# Quick overnight checkup for the catchup LaunchAgent.
cd /Users/jakob/Dev/stable-v2
echo "=== launchctl ==="
launchctl list | grep stable-v2.catchup || echo "(not loaded)"
echo
echo "=== process ==="
pgrep -fl 'run_catchup_loop' || echo "STOPPED — no run_catchup_loop process"
echo
echo "=== status.json ==="
cat logs/catchup_status.json 2>/dev/null || echo "(missing)"
echo
echo "=== log (last 25) ==="
tail -25 logs/catchup_launchd.out 2>/dev/null
echo
echo "=== errors (last 15) ==="
tail -15 logs/catchup_launchd.err 2>/dev/null || echo "(none)"
echo
echo "=== scrapes last 15 min ==="
/Library/Developer/CommandLineTools/Library/Frameworks/Python3.framework/Versions/3.9/Resources/Python.app/Contents/MacOS/Python -c "
from core.db import get_connection
c=get_connection(); cur=c.cursor()
cur.execute(\"SELECT COUNT(*) FROM st_horse_scrape_log WHERE scraped_at>NOW()-INTERVAL '15 minutes'\")
print(cur.fetchone()[0])
cur.execute('SELECT horse_id, http_status, scraped_at FROM st_horse_scrape_log ORDER BY scraped_at DESC LIMIT 3')
for r in cur.fetchall(): print(r)
"
