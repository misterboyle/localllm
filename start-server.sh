#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CONF_FILE="$HOME/.localllm/models.conf"

# Load model config (per-user, not in git)
if [ -f "$CONF_FILE" ]; then
  # shellcheck source=models.conf
  . "$CONF_FILE"
fi

MODEL_DIR="${MODEL_DIR:-$HOME/.localllm/models}"
DENSE_MODEL="${DENSE_MODEL:-Qwen3.6-27B-UD-Q6_K_XL.gguf}"
MOE_MODEL="${MOE_MODEL:-Qwen3.6-35B-A3B-UD-Q6_K_XL.gguf}"

LLAMA_SERVER="${LLAMA_SERVER:-$HOME/llama.cpp-turbo/build/bin/llama-server}"
DENSE_MODEL="$MODEL_DIR/$DENSE_MODEL"
MOE_MODEL="$MODEL_DIR/$MOE_MODEL"

DENSE_PORT="${DENSE_PORT:-30080}"
MOE_PORT="${MOE_PORT:-30081}"

DENSE_PID_DIR="${DENSE_PID_DIR:-$HOME/.localllm/pids}"
MOE_PID_DIR="${MOE_PID_DIR:-$HOME/.localllm/pids}"

# Defaults (overridden by models.conf if present)
DENSE_ENABLED=${DENSE_ENABLED:-0}  # Set to 1 in models.conf to enable dense server
DENSE_THREADS=${DENSE_THREADS:-q8_0}
DENSE_VARIANT=${DENSE_VARIANT:-turbo4}
DENSE_GPULAYERS=${DENSE_GPULAYERS:-99}
DENSE_PER_SLOT_CONTEXT=${DENSE_PER_SLOT_CONTEXT:-262144}  # 262K per slot
DENSE_SLOTS=${DENSE_SLOTS:-2}
DENSE_CONTEXT=$((DENSE_PER_SLOT_CONTEXT * DENSE_SLOTS))

# KV cache tuning (see KV-CACHE-ANALYSIS.md for details)
DENSE_CACHE_RAM=${DENSE_CACHE_RAM:-65536}       # 64GB prompt cache RAM
DENSE_CTX_CHECKPOINTS=${DENSE_CTX_CHECKPOINTS:-32}  # Max checkpoints per slot
DENSE_CHECKPOINT_EVERY_NT=${DENSE_CHECKPOINT_EVERY_NT:-8192}  # Checkpoint every N tokens during prefill
DENSE_CACHE_REUSE=${DENSE_CACHE_REUSE:-128}  # Min chunk size for KV cache reuse

MOE_ENABLED=${MOE_ENABLED:-1}
MOE_THREADS=${MOE_THREADS:-q8_0}
MOE_VARIANT=${MOE_VARIANT:-turbo4}
MOE_GPULAYERS=${MOE_GPULAYERS:-99}
MOE_PER_SLOT_CONTEXT=${MOE_PER_SLOT_CONTEXT:-262144}
MOE_SLOTS=${MOE_SLOTS:-8}
MOE_CONTEXT=$((MOE_PER_SLOT_CONTEXT * MOE_SLOTS))

# MoE-specific KV cache tuning
MOE_CACHE_RAM=${MOE_CACHE_RAM:-65536}        # 64GB prompt cache RAM
MOE_CTX_CHECKPOINTS=${MOE_CTX_CHECKPOINTS:-32}  # Max checkpoints per slot
MOE_CHECKPOINT_EVERY_NT=${MOE_CHECKPOINT_EVERY_NT:-8192}  # Checkpoint every N tokens during prefill
MOE_CACHE_REUSE=${MOE_CACHE_REUSE:-128}       # Min chunk size for KV cache reuse

log() {
  echo "[$(date '+%H:%M:%S')] $*"
}

stop_servers() {
  log "Stopping servers..."
  [ -f "$DENSE_PID_DIR/dense.pid" ] && kill "$(cat "$DENSE_PID_DIR/dense.pid")" 2>/dev/null && rm "$DENSE_PID_DIR/dense.pid"
  [ -f "$MOE_PID_DIR/moe.pid" ] && kill "$(cat "$MOE_PID_DIR/moe.pid")" 2>/dev/null && rm "$MOE_PID_DIR/moe.pid"
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

if [ "$DENSE_ENABLED" = "1" ]; then
  check_model "dense" "$DENSE_MODEL" || exit 1
fi
if [ "$MOE_ENABLED" = "1" ]; then
  check_model "moe" "$MOE_MODEL" || exit 1
fi

mkdir -p "$DENSE_PID_DIR"
mkdir -p "$MOE_PID_DIR"

# Start dense server (port 30080) if enabled
if [ "$DENSE_ENABLED" = "1" ]; then
  log "Starting dense server on port $DENSE_PORT (per-slot context=$DENSE_PER_SLOT_CONTEXT, slots=$DENSE_SLOTS, total context=$DENSE_CONTEXT, cache_ram=$DENSE_CACHE_RAM, checkpoints=$DENSE_CTX_CHECKPOINTS, checkpoint_every=$DENSE_CHECKPOINT_EVERY_NT, cache_reuse=$DENSE_CACHE_REUSE)..."
  nohup $LLAMA_SERVER \
    --model "$DENSE_MODEL" \
    --port "$DENSE_PORT" \
    -ctk "$DENSE_THREADS" -ctv "$DENSE_VARIANT" \
    --flash-attn on \
    --jinja \
    --gpu-layers $DENSE_GPULAYERS \
    -c $DENSE_CONTEXT -np $DENSE_SLOTS \
    --kv-unified \
    --cache-ram $DENSE_CACHE_RAM \
    --ctx-checkpoints $DENSE_CTX_CHECKPOINTS \
    --checkpoint-every-n-tokens $DENSE_CHECKPOINT_EVERY_NT \
    --cache-reuse $DENSE_CACHE_REUSE \
    --host 127.0.0.1 \
    > >(sed 's/^/[DENSE] /' >> "$HOME/.localllm/dense.log") 2>&1 &
  echo $! > "$DENSE_PID_DIR/dense.pid"
  log "Dense server PID: $(cat $DENSE_PID_DIR/dense.pid)"
fi

# Start moe server (port 30081) if enabled
if [ "$MOE_ENABLED" = "1" ]; then
  log "Starting moe server on port $MOE_PORT (per-slot context=$MOE_PER_SLOT_CONTEXT, slots=$MOE_SLOTS, total context=$MOE_CONTEXT, cache_ram=$MOE_CACHE_RAM, checkpoints=$MOE_CTX_CHECKPOINTS, checkpoint_every=$MOE_CHECKPOINT_EVERY_NT, cache_reuse=$MOE_CACHE_REUSE)..."
  nohup $LLAMA_SERVER \
    --model "$MOE_MODEL" \
    --port "$MOE_PORT" \
    -ctk "$MOE_THREADS" -ctv "$MOE_VARIANT" \
    --jinja \
    --flash-attn on \
    --gpu-layers $MOE_GPULAYERS \
    -c $MOE_CONTEXT -np $MOE_SLOTS \
    --kv-unified \
    --cache-ram $MOE_CACHE_RAM \
    --ctx-checkpoints $MOE_CTX_CHECKPOINTS \
    --checkpoint-every-n-tokens $MOE_CHECKPOINT_EVERY_NT \
    --cache-reuse $MOE_CACHE_REUSE \
    --host 127.0.0.1 \
    > >(sed 's/^/[MOE] /' >> "$HOME/.localllm/moe.log") 2>&1 &
  echo $! > "$MOE_PID_DIR/moe.pid"
  log "Moe server PID: $(cat $MOE_PID_DIR/moe.pid)"
fi

# Build health check and log file list based on enabled servers
HEALTH_CHECK=""
LOG_FILES=""
if [ "$DENSE_ENABLED" = "1" ]; then
  HEALTH_CHECK="${HEALTH_CHECK}curl -sf http://localhost:$DENSE_PORT/health > /dev/null 2>&1"
  LOG_FILES="$LOG_FILES $HOME/.localllm/dense.log"
fi
if [ "$MOE_ENABLED" = "1" ]; then
  [ -n "$HEALTH_CHECK" ] && HEALTH_CHECK="$HEALTH_CHECK && "
  HEALTH_CHECK="${HEALTH_CHECK}curl -sf http://localhost:$MOE_PORT/health > /dev/null 2>&1"
  LOG_FILES="$LOG_FILES $HOME/.localllm/moe.log"
fi

log "Waiting for server(s) to be ready..."
for i in $(seq 1 60); do
  if eval "$HEALTH_CHECK"; then
    log "Servers ready."
    [ "$DENSE_ENABLED" = "1" ] && log "  Dense: http://localhost:$DENSE_PORT/v1"
    [ "$MOE_ENABLED" = "1" ] && log "  Moe:   http://localhost:$MOE_PORT/v1"
    log "Press Ctrl+C to stop."
    tail -f $LOG_FILES --pid=$$ 2>/dev/null || true
    exit 0
  fi
  sleep 1
done

log "Timed out waiting for server(s). Check logs:"
[ "$DENSE_ENABLED" = "1" ] && log "  Dense: $HOME/.localllm/dense.log"
[ "$MOE_ENABLED" = "1" ] && log "  Moe:   $HOME/.localllm/moe.log"
exit 1
