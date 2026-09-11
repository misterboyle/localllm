#!/usr/bin/env bash
set -euo pipefail

# probe-glm53.sh — Metal memory probe for GLM-5.3-Flash UD-IQ3_XXS (120.4GB
# weights on a 128GB machine), round 2: MANUAL MoE offload ladder.
# MUST BE RUN DETACHED — it stops flash-next-mtp (which disconnects any live
# opencode sessions ~45s after launch), runs the ladder, kills each probe
# server, and restarts flash-next-mtp. Everything is recorded to RESULT, so
# the session loss is cosmetic.
#
#   nohup ./probe-glm53.sh > ~/.localllm/logs/glm53-probe.out 2>&1 & disown
#
# Why r2: round 1 (2026-09-11) showed -fit on lies for this arch — it
# estimated 108.2G used / 1.3G free but the request-time need was 114.1G vs
# a 110.1G MTL0 budget (offloaded only 6.3G to CPU; ~112G of weights landed
# on the GPU) and decode OOMed at token ~20. The fix is manual control:
# keep all layers on GPU EXCEPT the first N layers' MoE experts
# (--n-cpu-moe N, ~2.4G of pageable CPU RAM per layer), which trims the GPU
# weight set below the Metal wired ceiling with honest margin.
#
# Ladder: attempts run INDEPENDENTLY (a FAIL does not abort the rest) —
# one safe baseline first, then progressively less offload / more context,
# recording decode rate (t/s) on every PASS so we learn whether running
# this model is *worth* it, not just possible.
#
# Every server gets -np 1 (single slot), --ctx-checkpoints 1 (r1 made 32
# 145MB checkpoints/slot available) and --cache-ram 1024 (default 8G prompt
# cache) to keep hidden allocations off the ledger.
#
# Test hooks (all overridable for stub runs, e.g. the tests/test-probe-glm53.sh
# harness): SERVER_BIN MODEL SERVE_BIN PORT ATTEMPTS LOG_DIR RESULT LOAD_TIMEOUT
# PING_TIMEOUT START_GRACE DRAIN_SECONDS ASSIST ASSIST_PORT RESTORE_TIMEOUT.
# STUB_DIE / STUB_DIE_NCMOE are consumed by the stub server in the tests.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SERVER_BIN="${SERVER_BIN:-$HOME/llama.cpp-glm5/build/bin/llama-server}"
MODEL="${MODEL:-$HOME/.cache/huggingface/hub/models--unsloth--GLM-5.3-Flash-GGUF/snapshots/621d456e93e926e4b52f85cff5f634358c1828f9/UD-IQ3_XXS/GLM-5.3-Flash-UD-IQ3_XXS-00001-of-00004.gguf}"
SERVE_BIN="${SERVE_BIN:-$SCRIPT_DIR/serve}"
PORT="${PORT:-30099}"
LOG_DIR="${LOG_DIR:-$HOME/.localllm/logs}"
RESULT="${RESULT:-$LOG_DIR/glm53-probe-result.md}"
LOAD_TIMEOUT="${LOAD_TIMEOUT:-900}"   # seconds to wait for /health ok
PING_TIMEOUT="${PING_TIMEOUT:-300}"   # seconds for the decode ping
START_GRACE="${START_GRACE:-45}"      # delay before killing the assistant
DRAIN_SECONDS="${DRAIN_SECONDS:-120}" # max wait for wired memory to drop
read -r -a LADDER <<< "${ATTEMPTS:-nc12-c32k:12:32768 nc06-c32k:6:32768 nc04-c32k:4:32768 nc12-c131k:12:131072 nc06-c131k:6:131072}"
ASSIST="${ASSIST:-flash-next-mtp}"
ASSIST_PORT="${ASSIST_PORT:-30085}"
RESTORE_TIMEOUT="${RESTORE_TIMEOUT:-600}"

log() { echo "[$(date '+%H:%M:%S')] $*"; }

free_wired() { top -l 1 -n 0 2>/dev/null | grep PhysMem | sed 's/^PhysMem: //'; }

# wait_ready <port> <pid> -> 0 healthy, 1 process died, 2 timeout
wait_ready() {
  local port="$1" pid="$2" elapsed=0
  while [ "$elapsed" -lt "$LOAD_TIMEOUT" ]; do
    if ! kill -0 "$pid" 2>/dev/null; then
      log "server pid $pid is dead (exit early) — aborting wait after ${elapsed}s"
      return 1
    fi
    if curl -s -m 3 "http://127.0.0.1:${port}/health" 2>/dev/null | grep -q '"ok"'; then
      log "healthy after ${elapsed}s"
      return 0
    fi
    sleep 5; elapsed=$((elapsed + 5))
  done
  return 2
}

# stop_server <pid> — TERM, grace, then KILL; wait for the process to go
# through so the next attempt starts against drained memory (Metal release
# plus page-cache-friendly ordering).
stop_server() {
  local pid="$1" n=0
  kill -TERM "$pid" 2>/dev/null || true
  while kill -0 "$pid" 2>/dev/null && [ "$n" -lt 6 ]; do sleep 2; n=$((n + 1)); done
  kill -KILL "$pid" 2>/dev/null || true
  wait "$pid" 2>/dev/null || true
  sleep 10
}

# record <heading...> / append block to RESULT
record() { printf '### %s\n' "$*" >> "$RESULT"; }

# attempt <label> <n_cpu_moe> <ctx> -> 0 pass / nonzero fail
attempt() {
  local label="$1" ncmoe="$2" ctx="$3"
  local logf="$LOG_DIR/glm53-probe-${label}.log"
  : > "$logf"
  log "attempt ${label}: n-cpu-moe=${ncmoe} ctx=${ctx}"

  local args=(--model "$MODEL" --port "$PORT" --host 127.0.0.1 --no-webui
    --fit off -ngl 99 -np 1 --ctx-checkpoints 1 --cache-ram 1024
    -c "$ctx"
    --temp 1.0 --top-p 0.95
    --chat-template-kwargs '{"reasoning_effort":"max"}')
  [ "$ncmoe" -gt 0 ] && args+=(--n-cpu-moe "$ncmoe")

  "$SERVER_BIN" "${args[@]}" > "$logf" 2>&1 &
  local pid=$!

  local rc=0
  wait_ready "$PORT" "$pid" || rc=$?
  if [ "$rc" -ne 0 ]; then
    {
      record "FAIL: ${label} (server never became healthy, rc=${rc})"
      echo "- PhysMem: $(free_wired)"
      echo "- log tail:"; echo '```'; tail -20 "$logf" || true; echo '```'
    } >> "$RESULT"
    stop_server "$pid"
    return 1
  fi
  log "attempt ${label}: health ok — decode ping"

  # decode is where Metal OOMs actually surface (flash-next Q4_K_XL lesson);
  # require an HTTP 200 and a live process afterwards. 64 tokens so the
  # eval-rate average is meaningful.
  local http
  http=$(curl -s -m "$PING_TIMEOUT" -o "/tmp/glm53-ping-${label}.json" -w "%{http_code}" \
    -H "Content-Type: application/json" \
    -d '{"model":"glm53","messages":[{"role":"user","content":"Reply with the single word: pong"}],"max_tokens":64,"temperature":1.0}' \
    "http://127.0.0.1:${PORT}/v1/chat/completions" 2>/dev/null || echo "000")
  sleep 2
  local alive=0; kill -0 "$pid" 2>/dev/null && alive=1

  if [ "$http" = "200" ] && [ "$alive" = "1" ]; then
    {
      record "PASS: ${label} (n-cpu-moe=${ncmoe}, ctx=${ctx})"
      echo "- server up + 64-token decode completed"
      echo "- PhysMem at steady state: $(free_wired)"
      echo "- memory layout lines:"
      echo '```'
      grep -Ei "offloaded|Mapped|RS buffer|KV buffer|compute buffer|context checkpoints|cache.ram|n_ctx" "$logf" |
        grep -viE "slot" | tail -25 || true
      echo '```'
      echo "- eval rate:"
      echo '```'
      grep -E "print_timing|eval time|eval rate|t/s|ms per token" "$logf" | tail -8 || true
      echo '```'
    } >> "$RESULT"
    stop_server "$pid"
    return 0
  fi

  {
    record "FAIL: ${label} (http=${http} alive=${alive})"
    echo "- PhysMem: $(free_wired)"
    echo "- log tail:"; echo '```'; tail -40 "$logf" || true; echo '```'
  } >> "$RESULT"
  stop_server "$pid"
  return 1
}

main() {
  mkdir -p "$LOG_DIR"
  {
    echo "# glm53 Metal probe r2 (manual MoE offload) — $(date)"
    echo ""
    echo "model: GLM-5.3-Flash UD-IQ3_XXS (120.4GB weights, 46 layers, ~2.4G experts/layer)"
    echo "MTL0 ceiling (r1 measurement): 110100 MiB; r1 actual need: 114117 MiB at ctx 4096"
    echo "ladder: ${LADDER[*]}"
    echo "free at start: $(free_wired)"
    echo ""
    echo "assistant will be disconnected in ${START_GRACE}s — save anything open."
    echo ""
  } > "$RESULT"

  log "grace period ${START_GRACE}s, then stopping ${ASSIST} to free ~94GB"
  sleep "$START_GRACE"

  "$SERVE_BIN" stop "$ASSIST" >> "$LOG_DIR/glm53-probe-serve.log" 2>&1 || true
  # wait for memory to actually drain (DRAIN_SECONDS=0 to skip, used in tests)
  local n=0 used
  while [ "$n" -lt "$((DRAIN_SECONDS / 10))" ]; do
    used=$(free_wired | awk '{gsub("G","",$1); print $1}')
    [ "${used:-999}" -lt 40 ] && break
    sleep 10; n=$((n + 1))
  done
  echo "after ${ASSIST} stop: $(free_wired)" >> "$RESULT"
  echo "" >> "$RESULT"

  local pass_count=0 first_pass="" label ncmoe ctx entry
  for entry in "${LADDER[@]}"; do
    IFS=: read -r label ncmoe ctx <<< "$entry"
    if attempt "$label" "$ncmoe" "$ctx"; then
      pass_count=$((pass_count + 1))
      [ -z "$first_pass" ] && first_pass="$label"
      log "attempt ${label}: PASS"
    else
      log "attempt ${label}: FAIL (ladder continues)"
    fi
  done

  log "probe done (${pass_count} pass) — killing probe server, restoring assistant"
  pkill -TERM -f "llama-server.*--port ${PORT}" 2>/dev/null || true
  sleep 10

  "$SERVE_BIN" start "$ASSIST" >> "$LOG_DIR/glm53-probe-serve.log" 2>&1 || true
  if wait_health_port "$ASSIST_PORT" "$RESTORE_TIMEOUT"; then
    echo "" >> "$RESULT"; echo "${ASSIST}: RESTORED and healthy" >> "$RESULT"
  else
    echo "" >> "$RESULT"
    echo "${ASSIST}: RESTORE FAILED — manual: ./serve start ${ASSIST}" >> "$RESULT"
  fi

  if [ "$pass_count" -gt 0 ]; then
    { echo ""; echo "## Verdict: pass — ${pass_count} config(s) usable, first good: ${first_pass}"; } >> "$RESULT"
  else
    { echo ""; echo "## Verdict: fail — no config on the ladder survived decode"; } >> "$RESULT"
  fi
  log "probe finished (${pass_count} pass). results: $RESULT"
}

# wait_health_port <port> <timeout> — plain health poll, no process tracking
wait_health_port() {
  local port="$1" timeout="$2" elapsed=0
  while [ "$elapsed" -lt "$timeout" ]; do
    if curl -s -m 3 "http://127.0.0.1:${port}/health" 2>/dev/null | grep -q '"ok"'; then
      return 0
    fi
    sleep 5; elapsed=$((elapsed + 5))
  done
  return 1
}

main "$@"
