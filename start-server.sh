#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CONF_FILE="$HOME/.localllm/models.jsonc"

# Server selection: moe, dense, both, or all (default)
SERVER_FILTER="${1:-all}"
case "$SERVER_FILTER" in
  moe|dense|both|all) ;;
  *) echo "ERROR: unknown server '$SERVER_FILTER'. Use: moe, dense, both, all"; exit 1 ;;
esac

# Activate Python venv
VENV_DIR="${VENV_DIR:-$HOME/.localllm/venv}"
if [ ! -f "$VENV_DIR/bin/activate" ]; then
  echo "ERROR: venv not found at $VENV_DIR"
  echo "Run: python3.14 -m venv $VENV_DIR && source $VENV_DIR/bin/activate && pip install -e $HOME/mlx-lm-turbo"
  exit 1
fi
# shellcheck disable=SC1091
. "$VENV_DIR/bin/activate"

# Ensure mlx-lm is installed from local repo (editable, tracks updates)
MLX_LM_REPO="${MLX_LM_REPO:-$HOME/mlx-lm-turbo}"
if [ -f "$MLX_LM_REPO/setup.py" ]; then
  pip install -e "$MLX_LM_REPO" --quiet 2>/dev/null || true
fi

# Ensure turboquant-mlx is installed from local repo (editable, tracks updates)
TQ_MLX_REPO="${TQ_MLX_REPO:-$HOME/turboquant-mlx}"
if [ -f "$TQ_MLX_REPO/pyproject.toml" ]; then
  pip install -e "$TQ_MLX_REPO" --quiet 2>/dev/null || true
fi

if [ ! -f "$CONF_FILE" ]; then
  echo "ERROR: config not found at $CONF_FILE"
  echo "Copy models.jsonc.example there and edit."
  exit 1
fi

# Parse JSONC into flat shell variables via Python.
# Writes NAME=VALUE lines to a temp file for safe eval with shlex quoting.
CONF_ENV=$(mktemp)

python3 -c "
import json, os, sys, re, shlex

raw = open('$CONF_FILE').read()
raw = re.sub(r'//.*?$', '', raw, flags=re.MULTILINE)
raw = re.sub(r'/\*.*?\*/', '', raw, flags=re.DOTALL)
cfg = json.loads(raw)

home = os.environ['HOME']

def v(k, default=None):
    return cfg.get(k, default)

def expand(s):
    return s.replace('\$HOME', home) if isinstance(s, str) else s

def p(k, val):
    print(f'{k}={shlex.quote(str(val))}')

pid_dir = expand(v('pidDir', home + '/.localllm/pids'))
cache_dir = expand(v('cacheDir', home + '/.localllm/prompt_cache'))
log_dir = expand(v('logDir', home + '/.localllm'))
model_dir = expand(v('modelDir', home + '/.localllm/models'))

d = v('defaults', {})
p('CONF_KV_QUANT', f'{d[\"kvQuant\"][0]},{d[\"kvQuant\"][1]}')
p('CONF_KV_QUANT_START', d['kvQuantStart'])
p('CONF_PROMPT_CACHE_SIZE', d['promptCacheSize'])
p('CONF_PROMPT_CACHE_BYTES', d.get('promptCacheBytes', ''))
p('CONF_PROMPT_CACHE_DISK_SIZE', d['promptCacheDiskSize'])
p('CONF_DECODE_CONCURRENCY', d['decodeConcurrency'])
p('CONF_PREFILL_CONCURRENCY', d['prefillConcurrency'])
p('CONF_PREFILL_STEP_SIZE', d['prefillStepSize'])
p('CONF_TEMP', d['temperature'])
p('CONF_MAX_TOKENS', d['maxTokens'])
p('CONF_CHAT_TEMPLATE_ARGS', json.dumps(d['chatTemplateArgs']))

for name, srv in cfg.get('servers', {}).items():
    n = name.upper()
    p(f'{n}_ENABLED', 1 if srv['enabled'] else 0)
    p(f'{n}_MODEL', srv['model'])
    p(f'{n}_PORT', srv['port'])
    p(f'{n}_LOG', srv['logFile'])
    for key, jkey in [('kvQuant', 'KV_QUANT'), ('kvQuantStart', 'KV_QUANT_START'),
                        ('promptCacheSize', 'PROMPT_CACHE_SIZE'),
                        ('promptCacheBytes', 'PROMPT_CACHE_BYTES'),
                        ('decodeConcurrency', 'DECODE_CONCURRENCY'),
                        ('prefillConcurrency', 'PREFILL_CONCURRENCY'),
                       ('prefillStepSize', 'PREFILL_STEP_SIZE'),
                       ('temperature', 'TEMP'), ('maxTokens', 'MAX_TOKENS')]:
        if key in srv:
            val = srv[key]
            if key == 'kvQuant':
                val = f'{val[0]},{val[1]}'
            p(f'{n}_{jkey}', val)

p('CONF_PID_DIR', pid_dir)
p('CONF_CACHE_DIR', cache_dir)
p('CONF_LOG_DIR', log_dir)
p('CONF_MODEL_DIR', model_dir)
" > "$CONF_ENV" 2>&1

if [ ! -s "$CONF_ENV" ]; then
  echo "ERROR: Failed to parse $CONF_FILE"
  cat "$CONF_ENV"
  exit 1
fi
# shellcheck disable=SC1090
source "$CONF_ENV"

log() {
  echo "[$(date '+%H:%M:%S')] $*"
}

to_upper() {
  echo "$1" | tr '[:lower:]' '[:upper:]'
}

# Disable hf_xet (crashes on Python 3.14 due to multiprocessing semaphore leak)
export HF_HUB_ENABLE_XET=0

PID_DIR="$CONF_PID_DIR"
CACHE_DIR="$CONF_CACHE_DIR"
LOG_DIR="$CONF_LOG_DIR"

# Resolve per-server values (override or fallback to defaults)
resolve() {
  local var="$1" default_var="$2"
  eval "local val=\${${var}:-}"
  if [ -n "$val" ]; then
    echo "$val"
  else
    eval "echo \${${default_var}}"
  fi
}

stop_servers() {
  log "Stopping servers..."
  for name in dense moe; do
    local pidfile="$PID_DIR/${name}.pid"
    [ -f "$pidfile" ] && kill "$(cat "$pidfile")" 2>/dev/null && rm -f "$pidfile"
  done
  rm -f "$CONF_ENV"
  log "Stopped."
  exit 0
}

trap stop_servers EXIT INT TERM

mkdir -p "$PID_DIR" "$CACHE_DIR"

# Build server args for a given server name
build_args() {
  local name="$1"
  local upper
  upper=$(to_upper "$name")

  local model kv qstart csize cbytes dconc pconc pstep temp maxtok chat_args port logf
  model=$(resolve "${upper}_MODEL" "error")
  # Resolve model to local path if it exists under modelDir
  if [ -d "$CONF_MODEL_DIR/$model" ]; then
    model="$CONF_MODEL_DIR/$model"
  elif [ -d "$CONF_MODEL_DIR/$(basename "$model")" ]; then
    model="$CONF_MODEL_DIR/$(basename "$model")"
  elif [ -d "$CONF_MODEL_DIR/${model#*/}" ]; then
    model="$CONF_MODEL_DIR/${model#*/}"
  fi
  port=$(resolve "${upper}_PORT" "error")
  kv=$(resolve "${upper}_KV_QUANT" "CONF_KV_QUANT")
  qstart=$(resolve "${upper}_KV_QUANT_START" "CONF_KV_QUANT_START")
  csize=$(resolve "${upper}_PROMPT_CACHE_SIZE" "CONF_PROMPT_CACHE_SIZE")
  cbytes=$(resolve "${upper}_PROMPT_CACHE_BYTES" "CONF_PROMPT_CACHE_BYTES")
  dconc=$(resolve "${upper}_DECODE_CONCURRENCY" "CONF_DECODE_CONCURRENCY")
  pconc=$(resolve "${upper}_PREFILL_CONCURRENCY" "CONF_PREFILL_CONCURRENCY")
  pstep=$(resolve "${upper}_PREFILL_STEP_SIZE" "CONF_PREFILL_STEP_SIZE")
  temp=$(resolve "${upper}_TEMP" "CONF_TEMP")
  maxtok=$(resolve "${upper}_MAX_TOKENS" "CONF_MAX_TOKENS")
  chat_args="$CONF_CHAT_TEMPLATE_ARGS"
  logf="$CONF_LOG_DIR/$(resolve "${upper}_LOG" "${name}.log")"

  local enabled_var="${upper}_ENABLED"
  eval "local enabled=\${${enabled_var:-0}}"

  if [ "$enabled" != "1" ]; then
    return 0
  fi

  log "Starting $name on port $port (model=$model, kv_quant=$kv, cache=$csize, cbytes=${cbytes:-unlimited}, decode=$dconc)..."

  local server_args=(
    --model "$model"
    --port "$port"
    --host 127.0.0.1
    --kv-cache-quantization "$kv"
    --quantized-kv-start "$qstart"
    --prompt-cache-size "$csize"
    --prompt-cache-dir "$CACHE_DIR"
    --prompt-cache-disk-size "$CONF_PROMPT_CACHE_DISK_SIZE"
    --decode-concurrency "$dconc"
    --prompt-concurrency "$pconc"
    --prefill-step-size "$pstep"
    --temp "$temp"
    --max-tokens "$maxtok"
    --chat-template-args "$chat_args"
  )

  if [ -n "$cbytes" ]; then
    server_args+=(--prompt-cache-bytes "$cbytes")
  fi

  nohup mlx_lm.server "${server_args[@]}" >> "$logf" 2>&1 &

  echo $! > "$PID_DIR/${name}.pid"
  log "$name PID: $(cat "$PID_DIR/${name}.pid")"
}

# Start selected servers
case "$SERVER_FILTER" in
  moe)    build_args moe ;;
  dense)  build_args dense ;;
  both)   build_args dense; build_args moe ;;
  all)    build_args dense; build_args moe ;;
esac

# Build health checks and log tail list for selected servers
HEALTH_CHECK=""
LOG_FILES=""
case "$SERVER_FILTER" in
  moe)    SERVERS="moe" ;;
  dense)  SERVERS="dense" ;;
  both|all) SERVERS="dense moe" ;;
esac

for name in $SERVERS; do
  upper=$(to_upper "$name")
  enabled_var="${upper}_ENABLED"
  eval "enabled=\${${enabled_var:-0}}"
  [ "$enabled" != "1" ] && continue

  port_var="${upper}_PORT"
  eval "port=\${${port_var}}"

  [ -n "$HEALTH_CHECK" ] && HEALTH_CHECK="$HEALTH_CHECK && "
  HEALTH_CHECK="${HEALTH_CHECK}curl -sf http://localhost:$port/health > /dev/null 2>&1"

  logf="$CONF_LOG_DIR/$(resolve "${upper}_LOG" "${name}.log")"
  LOG_FILES="$LOG_FILES $logf"
done

log "Waiting for server(s) to be ready..."
for i in $(seq 1 120); do
  if eval "$HEALTH_CHECK"; then
    log "Servers ready."
    for name in $SERVERS; do
      upper=$(to_upper "$name")
      enabled_var="${upper}_ENABLED"
      eval "enabled=\${${enabled_var:-0}}"
      [ "$enabled" != "1" ] && continue
      port_var="${upper}_PORT"
      eval "port=\${${port_var}}"
      log "  $(to_upper "$name"): http://localhost:$port/v1/chat/completions"
    done
    log "Press Ctrl+C to stop."
    tail -f $LOG_FILES --pid=$$ 2>/dev/null || true
    exit 0
  fi
  sleep 1
done

log "Timed out. Check logs in $CONF_LOG_DIR/"
exit 1
