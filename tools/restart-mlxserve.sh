#!/usr/bin/env bash
# Restart an mlxserve server without cutting off a live agent session.
#
# ./serve has no `restart` subcommand, and an mlxserve server is usually the thing
# answering the very session that would run it — so this detaches, waits for the
# engine to go quiet, then stops/starts it and records the boot lines that prove the
# new flags took effect. Run it with nohup and walk away:
#
#   nohup tools/restart-mlxserve.sh mlx-flash >/dev/null 2>&1 &
#   tail -f ~/.localllm/restart-mlx-flash-*.log
#
# Flags:
#   --now    skip the idle wait and restart on the next sample
#   --tries <n>  give up after n busy samples (default 240 = 20 min at 5 s)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
REPO="$SCRIPT_DIR"
SERVER="${1:-mlx-flash}"
IDLE_SAMPLES=3
TRY_LIMIT=240
NOW=0

log() { echo "[$(date '+%H:%M:%S')] $*"; }

while [ $# -gt 0 ]; do
  case "${1:-}" in
    --now) NOW=1 ;;
    --tries) shift; TRY_LIMIT="${1:-240}" ;;
  esac
  shift || true
done

CONF="$REPO/servers.jsonc"
PORT="$(SERVER="$SERVER" CONF="$CONF" python3 -c "
import json, os, re
s = re.sub(r'^\s*//.*$', '', open(os.environ['CONF']).read(), flags=re.M)
print(json.loads(s)['servers'][os.environ['SERVER']]['port'])
")"
METRICS="http://127.0.0.1:${PORT}/metrics.json"
TS="$(date '+%Y%m%d-%H%M%S')"
LOGFILE="${HOME}/.localllm/restart-${SERVER}-${TS}.log"
exec > >(tee -a "$LOGFILE") 2>&1

log "armed: restart $SERVER (port $PORT); log $LOGFILE"

# --- wait for quiet: no decoder, no prefill, N samples running (a single 0 can be a
# gap between two turns of the same agentic loop, which is not a safe moment).
quiet=0
tries=0
while [ "$tries" -lt "$TRY_LIMIT" ]; do
  read -r running prefilling <<<"$(curl -sf --max-time 3 "$METRICS" 2>/dev/null | python3 -c "
import json,sys
g = json.load(sys.stdin).get('gauges', {})
print(g.get('requests_running', 0), g.get('requests_prefilling', 0))
" 2>/dev/null || echo '0 0')"
  if [ "$NOW" = "1" ]; then
    log "--now: restarting at once (running=$running)"
    break
  fi
  if [ "$running" = "0" ] && [ "$prefilling" = "0" ]; then
    quiet=$((quiet + 1))
    log "quiet $quiet/$IDLE_SAMPLES"
    [ "$quiet" -ge "$IDLE_SAMPLES" ] && break
  else
    quiet=0
    log "busy (running=$running prefilling=$prefilling) — waiting"
  fi
  tries=$((tries + 1))
  sleep 5
done
if [ "$tries" -ge "$TRY_LIMIT" ]; then
  log "gave up after $TRY_LIMIT busy samples; nothing was restarted"
  exit 1
fi

# --- restart. mlxserve stops get a 50 s grace so --shutdown-persist-secs can flush
# the SSD prefix tier; a SIGKILL there loses the sessions we are trying to preserve.
log "stopping $SERVER"
cd "$REPO"
./serve stop "$SERVER" || log "stop returned non-zero (continuing)"
until ! curl -sf --max-time 2 "$METRICS" >/dev/null 2>&1; do sleep 1; done
log "port $PORT is down; starting"
./serve start "$SERVER"

# --- verify: the boot banner is the receipt for every memory flag.
ready=0
for _ in $(seq 1 180); do
  if curl -sf --max-time 3 "$METRICS" >/dev/null 2>&1; then ready=1; break; fi
  sleep 2
done
if [ "$ready" = "0" ]; then
  log "FAILED: server did not answer within 6 min — read the tail of ~/localllm/${SERVER}.log"
  exit 1
fi
log "server is answering again"
log "boot receipt (memory plan as the engine read it):"
grep -E "\[wired\]|\[hot-cache\]|Hot prefix cache|Prefill chunk|Concurrency|Context size|\[admission\]|Model ready" \
  "$REPO/${SERVER}.log" | tail -12 | while IFS= read -r line; do log "  $line"; done
log "done — watch refusals: grep -c 'verdict=refuse' ~/localllm/${SERVER}.log"
