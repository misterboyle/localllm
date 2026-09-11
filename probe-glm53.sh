#!/usr/bin/env bash
set -euo pipefail

# probe-glm53.sh — Metal memory probe for GLM-5.3-Flash UD-IQ3_XXS (120.4GB
# weights on a 128GB machine). MUST BE RUN DETACHED — it stops
# flash-next-mtp (which disconnects any live opencode session), runs a
# context ladder against llama.cpp's -fit auto-fitter, kills the probe, and
# restarts flash-next-mtp. Everything is recorded to RESULT below, so the
# session loss is cosmetic.
#
#   nohup ./probe-glm53.sh > ~/.localllm/logs/glm53-probe.out 2>&1 & disown
#
# Ladder (stops at first FAIL): "fit" = auto-fit chooses ctx + ngl,
# then fixed contexts with the fitter still choosing the GPU layer count.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SERVER_BIN="$HOME/llama.cpp-glm5/build/bin/llama-server"
MODEL="$HOME/.cache/huggingface/hub/models--unsloth--GLM-5.3-Flash-GGUF/snapshots/621d456e93e926e4b52f85cff5f634358c1828f9/UD-IQ3_XXS/GLM-5.3-Flash-UD-IQ3_XXS-00001-of-00004.gguf"
LOG_DIR="$HOME/.localllm/logs"
RESULT="$LOG_DIR/glm53-probe-result.md"
PORT=30099
LOAD_TIMEOUT="${LOAD_TIMEOUT:-900}"   # seconds to wait for /health ok
PING_TIMEOUT="${PING_TIMEOUT:-300}"   # seconds for the decode ping
CTXS=("fit" "32768" "131072")

log() { echo "[$(date '+%H:%M:%S')] $*"; }

free_wired() { top -l 1 -n 0 2>/dev/null | grep PhysMem | sed 's/^PhysMem: //'; }

wait_health() {
  local port="$1" deadline elapsed=0
  deadline=$(( $(date +%s) + LOAD_TIMEOUT ))
  while [ "$(date +%s)" -lt "$deadline" ]; do
    if curl -s -m 3 "http://127.0.0.1:${port}/health" 2>/dev/null | grep -q '"ok"'; then
      return 0
    fi
    sleep 5; elapsed=$((elapsed + 5))
  done
  return 1
}

# attempt <label> [extra llama-server args...] -> 0 pass / 1 fail
attempt() {
  local label="$1"; shift
  local logf="$LOG_DIR/glm53-probe-${label}.log"
  : > "$logf"
  log "attempt ${label}: starting server"
  "$SERVER_BIN" --model "$MODEL" --port "$PORT" --host 127.0.0.1 --no-webui \
    --fit on --fitp on --temp 1.0 --top-p 0.95 \
    --chat-template-kwargs '{"reasoning_effort":"max"}' \
    "$@" > "$logf" 2>&1 &
  local pid=$!

  if ! wait_health "$PORT"; then
    log "attempt ${label}: health timeout (${LOAD_TIMEOUT}s)"
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
      echo "### PASS: ${label}"
      echo "- server up + 16-token decode completed"
      echo "- PhysMem at steady state: $(free_wired)"
      echo "- fit/ctx/KV/offload log lines:"
      echo '```'
      grep -E "fit|n_ctx =|context_size|kv|KV|offload|GPU[0-9]| buffer size|n_gpu_layers" "$logf" \
        | grep -viE "slot|debug" | tail -25 || true
      echo '```'
      echo "- eval rate:"
      echo '```'
      grep -E "eval rate|prompt eval rate|ms per token|t/s" "$logf" | tail -6 || true
      echo '```'
    } >> "$RESULT"
    kill -TERM "$pid" 2>/dev/null || true
    return 0
  fi

  {
    echo "### FAIL: ${label}"
    echo "- http=${http} alive=${alive} PhysMem: $(free_wired)"
    echo "- log tail:"
    echo '```'
    tail -40 "$logf" || true
    echo '```'
  } >> "$RESULT"
  kill -TERM "$pid" 2>/dev/null || true
  return 1
}

mkdir -p "$LOG_DIR"
{
  echo "# glm53 Metal probe — $(date)"
  echo ""
  echo "model: GLM-5.3-Flash UD-IQ3_XXS (120.4GB weights, 46 layers, ~2.6GB/layer)"
  echo "free at start: $(free_wired)"
  echo ""
} > "$RESULT"

log "stopping assistant server (flash-next-mtp) to free ~94GB — this disconnects live sessions"
"$SCRIPT_DIR/serve" stop flash-next-mtp >> "$LOG_DIR/glm53-probe-serve.log" 2>&1 || true
sleep 10
echo "after flash-next-mtp stop: $(free_wired)" >> "$RESULT"
echo "" >> "$RESULT"

overall=pass
for c in "${CTXS[@]}"; do
  if [ "$c" = "fit" ]; then
    if ! attempt "fit-autofit"; then overall=fail; log "probe ladder stopped at ${c}"; break; fi
  else
    if ! attempt "c${c}" "-c" "$c"; then overall=fail; log "probe ladder stopped at ctx ${c}"; break; fi
  fi
done

log "probe done (${overall}) — killing probe server, restoring assistant"
pkill -TERM -f "llama-server.*port ${PORT}" 2>/dev/null || true
sleep 10

"$SCRIPT_DIR/serve" start flash-next-mtp >> "$LOG_DIR/glm53-probe-serve.log" 2>&1 || true
if wait_health 30085; then
  echo "" >> "$RESULT"; echo "flash-next-mtp: RESTORED and healthy" >> "$RESULT"
else
  echo "" >> "$RESULT"; echo "flash-next-mtp: RESTORE FAILED — manual: ./serve start flash-next-mtp" >> "$RESULT"
fi

{ echo ""; echo "## Verdict: ${overall}"; } >> "$RESULT"
log "probe finished (${overall}). results: $RESULT"
