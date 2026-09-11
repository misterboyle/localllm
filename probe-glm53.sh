#!/usr/bin/env bash
set -euo pipefail

# probe-glm53.sh — Metal memory probe for GLM-5.3-Flash UD-IQ3_XXS (120.4GB
# weights on a 128GB machine). MUST BE RUN DETACHED — it stops
# flash-next-mtp (which disconnects any live opencode sessions ~45s after
# launch), runs a context ladder against llama.cpp's -fit auto-fitter, kills
# the probe, and restarts flash-next-mtp. Everything is recorded to RESULT
# below, so the session loss is cosmetic.
#
#   nohup ./probe-glm53.sh > ~/.localllm/logs/glm53-probe.out 2>&1 & disown
#
# Ladder (stops at first FAIL): "fit" = auto-fit chooses ctx + ngl,
# then fixed contexts with the fitter still choosing the GPU layer count.
# A dead server process aborts its attempt immediately (no 15-min health
# wait — learned 2026-09-11 when a bad flag looked like a hang).
#
# Test hooks (all overridable for stub runs, e.g. the tests/test-probe-glm53.sh
# harness): SERVER_BIN MODEL SERVE_BIN PORT CTXS LOG_DIR RESULT LOAD_TIMEOUT
# PING_TIMEOUT START_GRACE DRAIN_SECONDS ASSIST ASSIST_PORT RESTORE_TIMEOUT.

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
read -r -a CTXS <<< "${CTXS:-fit 32768 131072}"
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

# record <heading...> / append block to RESULT
record() { printf '### %s\n' "$*" >> "$RESULT"; }

# attempt <label> [extra llama-server args...] -> 0 pass / nonzero fail
attempt() {
  local label="$1"; shift
  local logf="$LOG_DIR/glm53-probe-${label}.log"
  : > "$logf"
  log "attempt ${label}: starting server"
  "$SERVER_BIN" --model "$MODEL" --port "$PORT" --host 127.0.0.1 --no-webui \
    --fit on -lv 4 --temp 1.0 --top-p 0.95 \
    --chat-template-kwargs '{"reasoning_effort":"max"}' \
    "$@" > "$logf" 2>&1 &
  local pid=$!

  local rc=0
  wait_ready "$PORT" "$pid" || rc=$?
  if [ "$rc" -ne 0 ]; then
    {
      record "FAIL: ${label} (server never became healthy, rc=${rc})"
      echo "- PhysMem: $(free_wired)"
      echo "- log tail:"; echo '```'; tail -20 "$logf" || true; echo '```'
    } >> "$RESULT"
    kill -TERM "$pid" 2>/dev/null || true
    return 1
  fi
  log "attempt ${label}: health ok — decode ping"

  # decode is where Metal OOMs actually surface (flash-next Q4_K_XL lesson);
  # require an HTTP 200 and a live process afterwards.
  local http
  http=$(curl -s -m "$PING_TIMEOUT" -o "/tmp/glm53-ping-${label}.json" -w "%{http_code}" \
    -H "Content-Type: application/json" \
    -d '{"model":"glm53","messages":[{"role":"user","content":"Reply with the single word: pong"}],"max_tokens":16,"temperature":1.0}' \
    "http://127.0.0.1:${PORT}/v1/chat/completions" 2>/dev/null || echo "000")
  sleep 2
  local alive=0; kill -0 "$pid" 2>/dev/null && alive=1

  if [ "$http" = "200" ] && [ "$alive" = "1" ]; then
    {
      record "PASS: ${label}"
      echo "- server up + 16-token decode completed"
      echo "- PhysMem at steady state: $(free_wired)"
      echo "- fit/ctx/KV/offload log lines:"
      echo '```'
      grep -Ei "fit[ :(]|n_ctx|context_size|\bkv\b|offload|GPU[0-9]|buffer size|n_gpu_layers" "$logf" |
        grep -viE "slot" | tail -25 || true
      echo '```'
      echo "- eval rate:"
      echo '```'
      grep -E "eval rate|ms per token|t/s" "$logf" | tail -6 || true
      echo '```'
    } >> "$RESULT"
    kill -TERM "$pid" 2>/dev/null || true
    wait "$pid" 2>/dev/null || true
    sleep 3
    return 0
  fi

  {
    record "FAIL: ${label} (http=${http} alive=${alive})"
    echo "- PhysMem: $(free_wired)"
    echo "- log tail:"; echo '```'; tail -40 "$logf" || true; echo '```'
  } >> "$RESULT"
  kill -TERM "$pid" 2>/dev/null || true
  wait "$pid" 2>/dev/null || true
  return 1
}

main() {
  mkdir -p "$LOG_DIR"
  {
    echo "# glm53 Metal probe — $(date)"
    echo ""
    echo "model: GLM-5.3-Flash UD-IQ3_XXS (120.4GB weights, 46 layers, ~2.6GB/layer)"
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

  local overall=pass
  for c in "${CTXS[@]}"; do
    if [ "$c" = "fit" ]; then
      if ! attempt "fit-autofit"; then overall=fail; log "ladder stopped at ${c}"; break; fi
    else
      if ! attempt "c${c}" "-c" "$c"; then overall=fail; log "ladder stopped at ctx ${c}"; break; fi
    fi
  done

  log "probe done (${overall}) — killing probe server, restoring assistant"
  pkill -TERM -f "llama-server.*--port ${PORT}" 2>/dev/null || true
  sleep 10

  "$SERVE_BIN" start "$ASSIST" >> "$LOG_DIR/glm53-probe-serve.log" 2>&1 || true
  if wait_health_port "$ASSIST_PORT" "$RESTORE_TIMEOUT"; then
    echo "" >> "$RESULT"; echo "${ASSIST}: RESTORED and healthy" >> "$RESULT"
  else
    echo "" >> "$RESULT"
    echo "${ASSIST}: RESTORE FAILED — manual: ./serve start ${ASSIST}" >> "$RESULT"
  fi

  { echo ""; echo "## Verdict: ${overall}"; } >> "$RESULT"
  log "probe finished (${overall}). results: $RESULT"
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
