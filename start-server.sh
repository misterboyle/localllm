#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CONF_FILE="$SCRIPT_DIR/models.conf"

# Load model config
if [ -f "$CONF_FILE" ]; then
  # shellcheck source=models.conf
  . "$CONF_FILE"
fi

LLAMA_SERVER="$HOME/llama.cpp-turbo/build/bin/llama-server"
DENSE_MODEL="$MODEL_DIR/$DENSE_MODEL"
MOE_MODEL="$MODEL_DIR/$MOE_MODEL"

DENSE_PORT=30080
MOE_PORT=30081

DENSE_PID_DIR="$HOME/.localllm/pids"
MOE_PID_DIR="$HOME/.localllm/pids"

log() {
  echo "[$(date '+%H:%M:%S')] $*"
}

stop_servers() {
  log "Stopping servers..."
  [ -f "$DENSE_PID_DIR/dense.pid" ] && kill "$(cat "$DENSE_PID_DIR/dense.pid")" 2>/dev/null && rm "$DENSE_PID_DIR/dense.pid"
  [ -f "$DENSE_PID_DIR/moe.pid" ] && kill "$(cat "$DENSE_PID_DIR/moe.pid")" 2>/dev/null && rm "$DENSE_PID_DIR/moe.pid"
  log "Stopped."
  exit 0
}

trap stop_servers EXIT INT TERM

check_model() {
  local name="$1"
  local path="$2"
  if [ ! -f "$path" ]; then
    log "ERROR: $name model not found at $path"
    log "Expected in: $MODEL_DIR"
    log "Download from HuggingFace or copy there"
    return 1
  fi
  return 0
}

check_model "dense" "$DENSE_MODEL" || exit 1
check_model "moe" "$MOE_MODEL" || exit 1

mkdir -p "$DENSE_PID_DIR"

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
  > "$HOME/.localllm/dense.log" 2>&1 &
echo $! > "$DENSE_PID_DIR/dense.pid"
log "Dense server PID: $(cat $DENSE_PID_DIR/dense.pid)"

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
   > "$HOME/.localllm/moe.log" 2>&1 &
echo $! > "$DENSE_PID_DIR/moe.pid"
log "Moe server PID: $(cat $DENSE_PID_DIR/moe.pid)"

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
log "  Dense: $HOME/.localllm/dense.log"
log "  Moe:   $HOME/.localllm/moe.log"
exit 1
