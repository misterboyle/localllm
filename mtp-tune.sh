#!/usr/bin/env bash
# Sweep --spec-draft-n-max for flash-next-mtp. Unsloth guidance (docs ask-API,
# 2026-09-08): n_max=2 is the "best starting point, works best in most setups";
# sweep 1..6 since fastest is hardware-dependent; they caution >2 because
# acceptance drops 83%->50% at n=4 (greedy, short ctx, GPU). Their own
# qwen3.8-next llama-server example ships n_max=5 — docs contradict; 5 is in
# the ladder to test their advertised config. Runs llama-server DIRECTLY on a
# private port with synthetic prompts — this session's server is NOT involved.
#
# RUN ONLY WITH NO PRODUCTION FLASH SERVER UP (09-01 memory-incident rule):
#   ./serve stop flash-next-mtp   # kills the assistant session — do it from terminal
#   ./mtp-tune.sh
#   ./serve start flash-next-mtp  # or ./switch-flashserver.sh flash-next --yes
#
# Output: stdout table + ~/localllm/mtp-tune.csv
set -euo pipefail

BIN="$HOME/llama.cpp-mtp-src/build-mtp/bin/llama-server"
MODEL="$HOME/.cache/huggingface/hub/models--unsloth--Qwen3.8-Flash-Next-GGUF/snapshots/c8b5954a88c2775c546b92593eda40ea041d3176/UD-IQ4_XS/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf"
MMPROJ="$HOME/.cache/huggingface/hub/models--unsloth--Qwen3.8-Flash-Next-GGUF/snapshots/c8b5954a88c2775c546b92593eda40ea041d3176/mmproj-F16.gguf"
HEAD="$HOME/localllm/mtp-heads/MTP/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf"
LOG_DIR="$(cd "$(dirname "$0")" && pwd)"
PROMPT_DIR="${TMPDIR:-/tmp}/mtp-tune"
PORT=30087
CSV="$LOG_DIR/mtp-tune.csv"
NMAXES="${MTP_TUNE_NMAXES:-1 2 3 4 5}"
MAXTOKENS="${MTP_TUNE_MAXTOKENS:-400}"

log() { echo "[$(date '+%H:%M:%S')] $*"; }

cleanup() {
  if [ -n "${SRV_PID:-}" ] && kill -0 "$SRV_PID" 2>/dev/null; then
    kill "$SRV_PID" 2>/dev/null || true
    wait "$SRV_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

# ---- build synthetic prompts (code-heavy, agent-like) -------------------------
mkdir -p "$PROMPT_DIR"
if [ ! -f "$PROMPT_DIR/prompt-small.txt" ]; then
  {
    cat "$LOG_DIR/serve" "$LOG_DIR"/ple-sample.sh 2>/dev/null || true
    cat "$HOME/llama.cpp-mtp-src/common/speculative.cpp" 2>/dev/null | head -400
  } | head -c 9000 > "$PROMPT_DIR/prompt-small.txt"
fi
# ~40k tokens: pad with numbered code chunks so prefix-cache misses on every run
if [ ! -f "$PROMPT_DIR/prompt-big.txt" ]; then
  { for i in $(seq 1 12); do
      echo "=== block $i ==="; head -c 9000 "$PROMPT_DIR/prompt-small.txt"; echo
    done; } > "$PROMPT_DIR/prompt-big.txt"
fi
for f in prompt-small.txt prompt-big.txt; do
  log "prompt $f: $(wc -c < "$PROMPT_DIR/$f") bytes"
done

# ---- server lifecycle ----------------------------------------------------------
start_server() {  # $1=ctx $2=nmax ("none"=vanilla control)
  local ctx="$1" nmax="$2" logf="$LOG_DIR/mtp-tune-server.log"
  local args=(-m "$MODEL" --mmproj "$MMPROJ" --host 127.0.0.1 --port "$PORT"
              -np 1 -ngl 99 -c "$ctx"
              --temp 1.0 --top-p 0.95 --top-k 20 --min-p 0.0 -lv 4)
  if [ "$nmax" != "none" ]; then
    args+=(-md "$HEAD" --spec-type draft-mtp --spec-draft-n-max "$nmax")
  fi
  [ -f "$logf" ] && mv -f "$logf" "${logf}.1"
  log "starting llama-server ctx=$ctx n_max=$nmax ..."
  "$BIN" "${args[@]}" > "$logf" 2>&1 &
  SRV_PID=$!
  local waited=0
  until curl -sf "http://127.0.0.1:$PORT/health" > /dev/null 2>&1; do
    kill -0 "$SRV_PID" 2>/dev/null || { log "server died at start — see $logf"; return 1; }
    sleep 2; waited=$((waited+2))
    [ "$waited" -ge 300 ] && { log "server health timeout — see $logf"; return 1; }
  done
  log "server up ($((waited))s)"
}

stop_server() {
  if [ -n "${SRV_PID:-}" ] && kill -0 "$SRV_PID" 2>/dev/null; then
    kill "$SRV_PID" 2>/dev/null || true
    wait "$SRV_PID" 2>/dev/null || true
    sleep 3
  fi
  SRV_PID=""
}

# ---- one measurement ------------------------------------------------------------
measure() {  # $1=ctx $2=nmax $3=prompt-file
  local label
  label="ctx=$1 n_max=$2 prompt=$(basename "$3")"
  local body
  body=$(python3 - "$3" "$MAXTOKENS" <<'PYEOF'
import json, sys
prompt = open(sys.argv[1], encoding="utf-8", errors="ignore").read()
prompt += "\n// Explain and extend this test harness. Think step by step.\n"
print(json.dumps({"prompt": prompt, "max_tokens": int(sys.argv[2]),
                  "temperature": 1.0, "top_p": 0.95, "seed": 42,
                  "ignore_eos": True}))
PYEOF
)
  local t0 secs
  t0=$(date +%s)
  curl -sf "http://127.0.0.1:$PORT/v1/completions" -d "$body" > /dev/null || { log "bench request failed ($label)"; return 1; }
  secs=$(( $(date +%s) - t0 ))
  sleep 1
  local tg acc
  tg=$(grep -E "tg = " "$LOG_DIR/mtp-tune-server.log" | tail -1 | grep -oE "tg_3s = *[0-9.]+" | grep -oE "[0-9.]+" || echo "")
  [ -z "$tg" ] && tg=$(grep -E "tg = " "$LOG_DIR/mtp-tune-server.log" | tail -1 | sed -nE "s/.*tg = *([0-9.]+).*/\1/p")
  acc=$(grep -oE "mean acc len = *[0-9.]+" "$LOG_DIR/mtp-tune-server.log" | tail -1 | grep -oE "[0-9.]+$" || echo "")
  local wall_tps
  wall_tps=$(python3 -c "print(round($MAXTOKENS/$secs,1))" 2>/dev/null || echo "")
  printf "  %-28s tg=%s tok/s  (wall %s tok/s over %ss incl prefill)  mean_acc_len=%s\n" \
    "$label" "${tg:-?}" "${wall_tps:-?}" "$secs" "${acc:-n/a}"
  echo "$2,$1,${tg:-},${wall_tps:-},${acc:-}" >> "$CSV"
}

# ---- sweep ----------------------------------------------------------------------
echo "n_max,ctx,tg_tps,wall_tps,mean_acc_len" > "$CSV"
for pctx in "8192 prompt-small.txt" "49152 prompt-big.txt"; do
  # shellcheck disable=SC2086
  set -- $pctx; ctx="$1"; pfile="$PROMPT_DIR/$2"
  for n in none $NMAXES; do
    log "== config: ctx=$ctx n_max=$n =="
    stop_server
    start_server "$ctx" "$n" || continue
    measure "$ctx" "$n" "$pfile"
  done
  stop_server
done

log "done. results:"
column -s, -t "$CSV"
