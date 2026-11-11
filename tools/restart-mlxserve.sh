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
#
# Env:
#   MLXSERVE_BIN   which engine binary to boot — this is how a version bump is armed
#                  (./serve:136 reads it; unset = the pinned default)
#   READY_SAMPLES  boot wait in 2-second units (default 180; raise it when the SSD
#                  prefix tier is big enough that its boot scan needs longer)
#
#   # example: arm a bump plus a config change on the next quiet moment
#   MLXSERVE_BIN=~/.localllm/mlx-serve-26.11.2/mlx-serve-macos-arm64/mlx-serve \
#     READY_SAMPLES=300 nohup tools/restart-mlxserve.sh mlx-flash >/dev/null 2>&1 &

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

# --- the "what changed?" baseline. ~/localllm/mlx-flash.log keeps exactly one
# rotation, so a log-derived before/after is worth ~2 runs. /metrics.json counters are
# lifetime: prompt_tokens_total/requests_success_total is the real mean depth, and
# requests_rejected_total counts admission failures (NOT `[admission] verdict=refuse`,
# which is a cold-plan decline the warm path usually serves anyway). Snapshot it here
# so the A/B after a flag change compares two full counters, not surviving log lines.
#
# /props is the one that decomposes it: weights_bytes, kv_cache_bytes, cache_bytes,
# active_bytes, peak_bytes, available_bytes. `active - weights - kv - cache` is memory
# with NO OWNER, which is the whole test for the 26.11.1 leak (#1071/#1091). Measured
# controls, both engines on the same pack and the same `[preflight] weights ~70.13 GB`:
#   26.10.2, two sessions live, 2026-11-05: active 83.05 = weights 75.30 + kv 8.61 +
#          cache 0.91  ->  unowned -1.8 GB (zero, within gauge timing)
#   26.11.1, ~960 requests in, 2026-11-05: active 102.9 GB with the hot cache billing
#          6.4 GB      ->  unowned ~21 GB, which is upstream's number exactly
# PASS after the fix ships: unowned <= ~1 GB at a quiet moment and available_bytes back
# near 32 GB. FAIL: unowned > ~5 GB, or the hourly max of active_bytes ratchets without
# plateauing (#1079's discriminator). So snapshot it here, at the natural boundary.
PROPS_SNAP="${HOME}/.localllm/pre-restart-props-${SERVER}-$(date '+%Y%m%d-%H%M%S').json"
if curl -sf --max-time 5 "${METRICS%/metrics.json}/props" -o "$PROPS_SNAP"; then
  python3 - "$PROPS_SNAP" <<'PY'
import json, sys
m = json.load(open(sys.argv[1])).get("memory", {})
G = 1e9
try:
    unowned = (m["active_bytes"] - m["weights_bytes"] - m["kv_cache_bytes"] - m["cache_bytes"]) / G
    print("[leak-test] saved %s" % sys.argv[1])
    print("[leak-test] active %.2f = weights %.2f + kv %.2f + cache %.2f -> UNOWNED %+.2f GB (control: 26.10.2 = -1.8, leaked 26.11.1 = +21) | available %.2f GB | peak %.2f GB"
          % (m["active_bytes"]/G, m["weights_bytes"]/G, m["kv_cache_bytes"]/G, m["cache_bytes"]/G, unowned,
             m["available_bytes"]/G, m["peak_bytes"]/G))
except KeyError as e:
    print("[leak-test] /props has no %s -- engine too old or field renamed" % e)
PY
else
  log "could not read /props for the leak baseline (continuing)"
fi

SNAP="${HOME}/.localllm/pre-restart-metrics-${SERVER}-$(date '+%Y%m%d-%H%M%S').json"
if curl -sf --max-time 5 "$METRICS" -o "$SNAP"; then
  python3 - "$SNAP" <<'PY'
import json, sys
d = json.load(open(sys.argv[1])); c = d.get("counters", {}); g = d.get("gauges", {})
n = c.get("requests_success_total", 0) or 1
print("[baseline] saved %s" % sys.argv[1])
print("[baseline] mean prompt depth %d tok over %d requests | rejected %d | hits %d/%d | footprint %s MB | active %.1f GB" % (
    c.get("prompt_tokens_total", 0) // n, n, c.get("requests_rejected_total", 0),
    c.get("prefix_cache_hits_total", 0), c.get("prefix_cache_queries_total", 0) or 1,
    g.get("memory_mb", "?"), g.get("mlx_active_bytes", 0) / 1e9))
PY
else
  log "could not read $METRICS for the baseline snapshot (continuing)"
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
# The wait is sized for a cold boot that has to SCAN the SSD prefix tier before it
# answers: 26.11.x logs `[disk-cache] scanned N persisted entries (~200 GB)`, and a
# 200 GB scan plus a 70 GB weight load can pass 6 min, which would print a FAILED that
# is really just a slow boot. READY_SAMPLES is in 2-second units (default 180 = 6 min).
ready=0
for _ in $(seq 1 "${READY_SAMPLES:-180}"); do
  if curl -sf --max-time 3 "$METRICS" >/dev/null 2>&1; then ready=1; break; fi
  sleep 2
done
if [ "$ready" = "0" ]; then
  log "FAILED: server did not answer within 6 min — read the tail of ~/localllm/${SERVER}.log"
  log "two things to try, in this order:"
  log "  1. a cache-format mismatch (a newer engine wrote ~/.mlx-serve/kv-cache) is the"
  log "     likely cause on a version rollback. Move the tier aside -- deleting it loses"
  log "     nothing but warmth, every miss is handled as a cold prefill:"
  log '       mv ~/.mlx-serve/kv-cache ~/.mlx-serve/kv-cache.bak-<YYYYMMDD> && ./serve start '"$SERVER"
  log '  2. or go back to the engine that was serving (26.10.2 is the current pin):'
  # shellcheck disable=SC2016  # $HOME is deliberate: this line is a suggestion to paste, not to expand
  log '       MLXSERVE_BIN=$HOME/.localllm/mlx-serve-26.10.2/mlx-serve-macos-arm64/mlx-serve ./serve start '"$SERVER"
  log '     NOT 26.11.1: that build carries the #1091 batched-decode logits leak (~21 GB'
  log '     surviving unload on this model). 26.11.2 is the first release with the fix.'
  exit 1
fi
log "server is answering again"
# --- the floor. What the process holds with NO session loaded is the fixed cost, and
# it is the number that actually moved across the 26.11.1 bump: mlx_active_bytes was
# 81.2 GB on 26.10.2 (see ~/.localllm/pre-bump-metrics-26.10.2.json) against 102.9 GB
# after, with identical weights, an identical 32 GB n-gram table and the same
# `[wired] mode=max limit=110100 MB`. Sample it once the server is up AND idle, so a
# first request does not pollute the reading.
for _ in $(seq 1 12); do
  busy="$(curl -sf --max-time 3 "$METRICS" 2>/dev/null | python3 -c "
import json, sys
try: g = json.load(sys.stdin).get('gauges', {})
except Exception: g = {}
print(int(g.get('requests_running', 0) or 0) + int(g.get('requests_prefilling', 0) or 0))
" 2>/dev/null || echo 1)"
  [ "$busy" = "0" ] && break
  sleep 5
done
curl -sf --max-time 5 "$METRICS" 2>/dev/null | python3 -c "
import json, sys
try: d = json.load(sys.stdin)
except Exception: raise SystemExit('[floor] no metrics to read')
g = d.get('gauges', {})
act = g.get('mlx_active_bytes', 0) / 1e9
print('[floor] idle with %s requests running: mlx_active_bytes %.1f GB | memory_mb %s | cache %.2f GB | ngram %.1f GB'
      % (g.get('requests_running', '?'), act, g.get('memory_mb', '?'),
         g.get('mlx_cache_bytes', 0) / 1e9, g.get('ngram_warm_bytes', 0) / 1e9))
print('[floor] 26.10.2 reference: 81.2 GB active / 82,628 MB footprint -- but that was a'
      ' MID-TRAFFIC reading saved at the cutover, NOT an idle floor (Michael), so treat'
      ' any delta as a ceiling on the leak, not the leak. Upstream #1079 is right that'
      ' the discriminator is the hourly MIN of mlx_active_bytes over 24-48 h of real'
      ' traffic (~2 GiB/h accumulation on 26.11.1); one boot cannot see it.')
" || log "floor: metrics unreadable"
log "boot receipt (memory plan as the engine read it):"
grep -E "\[wired\]|\[hot-cache\]|Hot prefix cache|Prefill chunk|Concurrency|Context size|\[admission\]|Model ready" \
  "$REPO/${SERVER}.log" | tail -12 | while IFS= read -r line; do log "  $line"; done
# --- and the same reading once the new engine is up and idle: that is the starting
# line the next [leak-test] above will be compared against.
curl -sf --max-time 5 "${METRICS%/metrics.json}/props" 2>/dev/null | python3 -c "
import json, sys
try: m = json.load(sys.stdin).get('memory', {})
except Exception: raise SystemExit('[floor] /props unreadable')
G = 1e9
if 'active_bytes' in m and 'weights_bytes' in m:
    print('[floor] unowned at idle = %+.2f GB (active %.2f - weights %.2f - kv %.2f - cache %.2f) | available %.2f GB'
          % ((m['active_bytes']-m['weights_bytes']-m.get('kv_cache_bytes',0)-m.get('cache_bytes',0))/G,
             m['active_bytes']/G, m['weights_bytes']/G, m.get('kv_cache_bytes',0)/G,
             m.get('cache_bytes',0)/G, m.get('available_bytes',0)/G))
" || log "floor: /props unreadable"
log "done — watch refusals: grep -c 'verdict=refuse' ~/localllm/${SERVER}.log"
