#!/usr/bin/env bash
# Unattended one-shot MTP n_max sweep: stop the interactive flash server (kills
# the assistant session — run when Michael is done talking to the assistant),
# run mtp-tune.sh, bring the server back, and append results to the wiki log
# so the next assistant session picks the decision up.
#
# Usage: nohup ./run-mtp-tune.sh >> run-mtp-tune.log 2>&1 &   (or from terminal)
set -uo pipefail  # deliberately NOT -e: a failed config must still restart the server

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

log() { echo "[$(date '+%H:%M:%S')] $*"; }

log "stopping flash-next-mtp (assistant session dies as designed)"
./serve stop flash-next-mtp || true
sleep 5

STATUS="ok"
./mtp-tune.sh || STATUS="sweep-FAILED-rc$?"

log "starting flash-next-mtp back up (status: $STATUS)"
./serve start flash-next-mtp || STATUS="$STATUS;restart-FAILED"

# record results for the next session
python3 - "$STATUS" <<'PYEOF'
import csv, datetime, subprocess, sys
status = sys.argv[1]
try:
    rows = list(csv.reader(open("/Users/michael/localllm/mtp-tune.csv")))
except FileNotFoundError:
    rows = [["(no csv produced)"]]
entry = (
    f"\n## [{datetime.date.today()}] work | MTP n_max sweep ran unattended: {status}\n"
    "mtp-tune.csv:\n" + "\n".join("  " + ",".join(r) for r in rows)
    + "\nNEXT: pick n_max vs vanilla per ctx tier; stage 2 = scripted `opencode run` "
    "real-agent sessions on the winning config; consider large-ctx tier.\n"
)
open("/Users/michael/wiki/wiki/log.md", "a").write(entry)
subprocess.run(["git", "-C", "/Users/michael/wiki", "add", "wiki/log.md"], check=True)
subprocess.run(["git", "-C", "/Users/michael/wiki", "commit", "-q",
                "-m", f"log: mtp-tune sweep results ({status})"], check=True)
print("results appended to wiki log and committed")
PYEOF

log "done ($STATUS)"
