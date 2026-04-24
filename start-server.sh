#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LLAMA_SERVER="$HOME/llama.cpp-turbo/build/bin/llama-server"
MODEL_DIR="$SCRIPT_DIR/models"
DENSE_MODEL="$MODEL_DIR/Qwen3.6-27B-UD-Q6_K_XL.gguf"
MOE_MODEL="$MODEL_DIR/Qwen3.6-35B-A3B-UD-Q6_K_XL.gguf"

DENSE_PORT=30080
MOE_PORT=30081

DENSE_PID_FILE="$MODEL_DIR/.dense.pid"
MOE_PID_FILE="$MODEL_DIR/.moe.pid"

log() {
  echo "[$(date '+%H:%M:%S')] $*"
}

stop_servers() {
  log "Stopping servers..."
  [ -f "$DENSE_PID_FILE" ] && kill "$(cat "$DENSE_PID_FILE")" 2>/dev/null && rm "$DENSE_PID_FILE"
  [ -f "$MOE_PID_FILE" ] && kill "$(cat "$MOE_PID_FILE")" 2>/dev/null && rm "$MOE_PID_FILE"
  log "Stopped."
  exit 0
}

trap stop_servers EXIT INT TERM

check_model() {
  local name="$1"
  local path="$2"
  if [ ! -f "$path" ]; then
    log "ERROR: $name not found at $path"
    log "Download from HuggingFace or copy to $MODEL_DIR/"
    return 1
  fi
  return 0
}

check_model "dense" "$DENSE_MODEL" || exit 1
check_model "moe" "$MOE_MODEL" || exit 1

# Start dense server (port 30080)
log "Starting dense server on port $DENSE_PORT..."
nohup $LLAMA_SERVER \
  --model "$DENSE_MODEL" \
  --port "$DENSE_PORT" \
  -ctk q8_0 -ctv turbo4 \
  --flash-attn on \
  --jinja \
  --gpu-layers 99 -c 262144 -np 1 \
  --host 127.0.0.1 \
  --api-key sk-local \
  > "$MODEL_DIR/dense.log" 2>&1 &
echo $! > "$DENSE_PID_FILE"
log "Dense server PID: $(cat $DENSE_PID_FILE)"

# Start moe server (port 30081)
log "Starting moe server on port $MOE_PORT..."
nohup $LLAMA_SERVER \
  --model "$MOE_MODEL" \
  --port "$MOE_PORT" \
  -ctk q8_0 -ctv turbo4 \
  --flash-attn on \
   --jinja \
   --gpu-layers 99 -c 1048576 -np 4 \
   --host 127.0.0.1 \
   --api-key sk-local \
   > "$MODEL_DIR/moe.log" 2>&1 &
echo $! > "$MOE_PID_FILE"
log "Moe server PID: $(cat $MOE_PID_FILE)"

log "Waiting for servers to be ready..."
for i in $(seq 1 60); do
  if curl -sf http://localhost:$DENSE_PORT/health > /dev/null 2>&1 && \
     curl -sf http://localhost:$MOE_PORT/health > /dev/null 2>&1; then
    log "Both servers ready."
    log "  Dense:  http://localhost:$DENSE_PORT/v1"
    log "  Moe:    http://localhost:$MOE_PORT/v1"
    log "Press Ctrl+C to stop."
    wait
    break
  fi
  sleep 1
done

log "Timed out waiting for servers. Check logs:"
log "  Dense: $MODEL_DIR/dense.log"
log "  Moe:   $MODEL_DIR/moe.log"
exit 1
