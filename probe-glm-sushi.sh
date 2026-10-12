#!/usr/bin/env bash
set -euo pipefail

# probe-glm-sushi.sh — is GLM-5.3-Flash (Sushi 2.4bpw EXL3 pack) usable as a driver on this
# M5 Max 128 GB, measured resident at the settings Sushi publishes for a 128 GB Mac?
#
# MUST BE RUN DETACHED — it stops the assistant server (which disconnects any live opencode
# session ~45 s after launch), owns the GPU for the whole run, measures, then restores the
# assistant. Everything lands in RESULT, so the session loss is cosmetic.
#
#   nohup ./probe-glm-sushi.sh > ~/.localllm/logs/glm-sushi-probe.out 2>&1 & disown
#
# Why a new probe and not probe-glm53.sh: glm53 was llama.cpp + UD-IQ3_XXS (120.4 GB), a
# quant that cannot fit the pool, so its ladder was forced into --n-cpu-moe and the box paged.
# 0.95 t/s is that configuration's number, not the model's. Sushi is a detached fork of
# mlx-serve with native EXL3 decode kernels and the DFlash2 assistant, and it ships a pack
# sized for this machine (97.8 GB on disk, 90.0 GiB of weights, 101.0 GiB for a full 1M
# context — its own table). Its published M5 Max numbers are decode 44-52 t/s at prefill
# 645-859 t/s: the incumbent's decode speed at roughly half its prefill speed. So the open
# questions are not "can it load" but:
#
#   1. what a long prompt actually costs us (their TTFT at 128k is 203 s — that IS the trade
#      for agentic sessions, and it is the number the driver decision turns on);
#   2. whether a 1M-ctx load passes with the AUTO pool (110,100 MiB) or needs iogpu.
#      wired_limit_mb raised to Sushi's documented 120000 for a 128 GB Mac;
#   3. whether tool calls and thinking survive end to end (the sibling ds4 engine served GLM
#      GGUFs that wrote past their answer until mlx-serve 26.10.2 — wiki glm-5-3-flash);
#   4. how much the prefix cache rescues depth (Sushi keeps seen prefixes on the SSD).
#
# One load serves the whole depth ladder: ctx-size bounds the server, not the request, so
# re-wiring 90 GiB per depth would measure patience, not the model.
#
# Test hooks (all overridable for stub runs): SUSHI_BIN PACK SERVE_BIN PORT CTX ASSIST
# ASSIST_PORT LOG_DIR RESULT PROMPT_DIR DEPTHS GEN_TOKENS LOAD_TIMEOUT READY_TIMEOUT
# DRAIN_SECONDS PREFIX_CACHE_DISK SUSHI_EXTRA BENCH PY SKIP_STOP.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SUSHI_BIN="${SUSHI_BIN:-/opt/homebrew/bin/sushi}"
PACK="${PACK:-$HOME/.sushi/models/beamster/GLM-5.3-Flash-Sushi-2.4bpw}"
SERVE_BIN="${SERVE_BIN:-$SCRIPT_DIR/serve}"
PORT="${PORT:-30087}"
CTX="${CTX:-524288}"
LOG_DIR="${LOG_DIR:-$HOME/.localllm/logs}"
RESULT="${RESULT:-$LOG_DIR/glm-sushi-probe-result.md}"
PROMPT_DIR="${PROMPT_DIR:-$LOG_DIR/probe-prompts}"
BENCH="${BENCH:-$SCRIPT_DIR/server-metrics-bench.py}"
PY="${PY:-python3}"
LOAD_TIMEOUT="${LOAD_TIMEOUT:-2400}"   # seconds to wait for /health ok (90 GiB wire + warmup)
READY_TIMEOUT="${READY_TIMEOUT:-1800}" # seconds for one measured request to come back
DRAIN_SECONDS="${DRAIN_SECONDS:-240}"  # max wait for the assistant's wired pool to be released
PREFIX_CACHE_DISK="${PREFIX_CACHE_DISK:-20GB}"
read -r -a DEPTHS <<< "${DEPTHS:-2048 32768 131072}"
read -r -a GEN_TOKENS <<< "${GEN_TOKENS:-256}"
ASSIST="${ASSIST:-mlx-flash}"
ASSIST_PORT="${ASSIST_PORT:-30085}"
SUSHI_EXTRA="${SUSHI_EXTRA:-}"
SKIP_STOP="${SKIP_STOP:-0}"
SUSHI_PID=""
MODEL_ID=""

log() { echo "[$(date '+%H:%M:%S')] $*"; }
phys() { top -l 1 -n 0 2>/dev/null | grep PhysMem | sed 's/^PhysMem: //'; }
wired_g() { printf '%s' "$(phys)" | sed -n 's/.*(\([0-9]*\)G wired.*/\1/p'; }
pool_mb() { sysctl -n iogpu.wired_limit_mb 2>/dev/null || echo 0; }
api() { curl -fsS --max-time "${3:-15}" "$1" 2>/dev/null || true; }   # api <url> [ignore] [secs]

# wait_ready -> 0 healthy, 1 process died, 2 timeout
wait_ready() {
  local elapsed=0
  while [ "$elapsed" -lt "$LOAD_TIMEOUT" ]; do
    if ! kill -0 "$SUSHI_PID" 2>/dev/null; then
      log "sushi pid $SUSHI_PID exited early — tail of $LOG_DIR/glm-sushi.log:"
      tail -20 "$LOG_DIR/glm-sushi.log" 2>/dev/null || true
      return 1
    fi
    if curl -fsS --max-time 5 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
      log "ready after ${elapsed}s"
      return 0
    fi
    sleep 10; elapsed=$((elapsed + 10))
  done
  log "timed out after ${LOAD_TIMEOUT}s waiting for /health"
  return 2
}

# start_sushi -> sets SUSHI_PID, waits for /health
start_sushi() {
  read -r -a extra_flags <<< "$SUSHI_EXTRA"   # space-separated extra flags, deliberately simple
  # ${arr[@]+"${arr[@]}"} because /bin/bash is 3.2.57 here and an empty array under set -u
  # is an "unbound variable" exit in that version.
  local args=(--model "$PACK" --serve --host 127.0.0.1 --port "$PORT" --ctx-size "$CTX"
              --kv-quant 8 --metrics --prefix-cache-disk "$PREFIX_CACHE_DISK"
              ${extra_flags[@]+"${extra_flags[@]}"})
  log "starting: $SUSHI_BIN ${args[*]}"
  nohup "$SUSHI_BIN" "${args[@]}" >> "$LOG_DIR/glm-sushi.log" 2>&1 &
  SUSHI_PID=$!
  log "sushi pid $SUSHI_PID"
  wait_ready
}

# model_id -> whatever the server calls itself (never guess: Sushi names the pack by folder)
model_id() {
  local id
  id="$(api "http://127.0.0.1:$PORT/v1/models" | python3 -c "
import json,sys
try:
    d = json.load(sys.stdin); print((d.get('data') or [{}])[0].get('id', ''))
except Exception:
    print('')
")"
  MODEL_ID="${id:-default}"
  log "model id: $MODEL_ID"
}

# ask_json <body-file> <out-file> -> curl a JSON body built from a file (args may be long:
# the vision smoke carries a base64 image)
ask_json() {
  curl -sS --max-time "$READY_TIMEOUT" "http://127.0.0.1:$PORT/v1/chat/completions" \
    -H 'Content-Type: application/json' -d @"$1" -o "$2" \
    2>> "$LOG_DIR/glm-sushi-probe-errors.log" || log "request failed (see $LOG_DIR/glm-sushi-probe-errors.log)"
}

# tool_smoke — a tool call must come back as a tool call. The failure that mattered on the
# sibling ds4 path was not bad arguments but the model writing past its answer, so this
# records finish_reason and the whole content as well as the call.
tool_smoke() {
  local body="$LOG_DIR/glm-sushi-toolcall-body.json" resp="$LOG_DIR/glm-sushi-toolcall.json"
  python3 -c "
import json, sys
json.dump({
  'model': sys.argv[1], 'max_tokens': 400,
  'messages': [{'role': 'user', 'content': 'What is 27 times 43? Use the calculator tool.'}],
  'tools': [{'type': 'function', 'function': {
      'name': 'calculator', 'description': 'Multiply two integers.',
      'parameters': {'type': 'object',
        'properties': {'a': {'type': 'integer'}, 'b': {'type': 'integer'}},
        'required': ['a', 'b']}}}]}, open(sys.argv[2], 'w'))
" "$MODEL_ID" "$body"
  ask_json "$body" "$resp"
  python3 - "$resp" >> "$RESULT" <<'PYX'
import json, sys
try:
    d = json.load(open(sys.argv[1]))
    ch = (d.get("choices") or [{}])[0]
    msg = ch.get("message") or {}
    tc = msg.get("tool_calls") or []
    usage = d.get("usage") or {}
    content = (msg.get("content") or "").strip().replace("\n", " ")
    if tc:
        fn = tc[0].get("function") or {}
        print(f"| tool call | OK — `{fn.get('name')}({fn.get('arguments')})`, "
              f"finish={ch.get('finish_reason')}, {usage.get('prompt_tokens')} in / "
              f"{usage.get('completion_tokens')} out |")
    else:
        print(f"| tool call | **no tool_calls** — finish={ch.get('finish_reason')}, "
              f"content: {content[:100]!r} |")
except Exception as exc:
    print(f"| tool call | unparseable response ({exc}) |")
PYX
}

# vision_smoke — a 64x64 PNG built with stdlib only; GLM-2.4bpw advertises image and video
# input, and a streamed/vision load that cannot decode a picture is a driver bug.
vision_smoke() {
  local png="$PROMPT_DIR/probe.png" body="$LOG_DIR/glm-sushi-vision-body.json"
  local resp="$LOG_DIR/glm-sushi-vision.json"
  mkdir -p "$PROMPT_DIR"
  python3 -c "
import struct, sys, zlib
w = h = 64
row = b''.join(bytes([i % 256, 128, 255 - i % 256]) for i in range(w))
raw = b''.join(b'\x00' + row for _ in range(h))
def chunk(t, d):
    return struct.pack('>I', len(d)) + t + d + struct.pack('>I', zlib.crc32(t + d) & 0xFFFFFFFF)
open(sys.argv[1], 'wb').write(b'\x89PNG\r\n\x1a\n'
    + chunk(b'IHDR', struct.pack('>IIBBBBB', w, h, 8, 2, 0, 0, 0))
    + chunk(b'IDAT', zlib.compress(raw)) + chunk(b'IEND', b''))
" "$png" 2>> "$LOG_DIR/glm-sushi-probe-errors.log" || { log "vision smoke: png write failed"; return 0; }
  python3 -c "
import base64, json, sys
uri = 'data:image/png;base64,' + base64.b64encode(open(sys.argv[1], 'rb').read()).decode()
json.dump({'model': sys.argv[2], 'max_tokens': 150, 'messages': [{'role': 'user', 'content': [
    {'type': 'text', 'text': 'One sentence: what is in this image?'},
    {'type': 'image_url', 'image_url': {'url': uri}}]}]}, open(sys.argv[3], 'w'))
" "$png" "$MODEL_ID" "$body"
  ask_json "$body" "$resp"
  python3 - "$resp" >> "$RESULT" <<'PYX'
import json, sys
try:
    d = json.load(open(sys.argv[1]))
    txt = ((d.get("choices") or [{}])[0].get("message") or {}).get("content") or ""
    if isinstance(txt, list):
        txt = " ".join(p.get("text", "") for p in txt if isinstance(p, dict))
    txt = txt.strip().replace("\n", " ")
    print(f"| vision | {'OK — ' + txt[:110] if txt else '**empty answer**'} |")
except Exception as exc:
    print(f"| vision | unparseable ({exc}) |")
PYX
}

# think_smoke — GLM's thinking levels (low|high|max) are a per-request knob; an agent that
# cannot turn thinking off is an agent paying tokens it did not ask for.
think_smoke() {
  local body="$LOG_DIR/glm-sushi-think-body.json" resp="$LOG_DIR/glm-sushi-think.json"
  python3 -c "
import json, sys
json.dump({'model': sys.argv[1], 'max_tokens': 200, 'chat_template_kwargs': {'thinking': 'off'},
  'messages': [{'role': 'user', 'content': 'Reply with exactly OK.'}]}, open(sys.argv[2], 'w'))
" "$MODEL_ID" "$body"
  ask_json "$body" "$resp"
  python3 - "$resp" >> "$RESULT" <<'PYX'
import json, sys
try:
    d = json.load(open(sys.argv[1]))
    ch = (d.get("choices") or [{}])[0]
    msg = ch.get("message") or {}
    think = msg.get("reasoning_content") or msg.get("reasoning") or ""
    u = d.get("usage") or {}
    print(f"| thinking off | content={(msg.get('content') or '').strip()[:40]!r}, "
          f"reasoning_chars={len(think)}, finish={ch.get('finish_reason')}, "
          f"out_tokens={u.get('completion_tokens')} |")
except Exception as exc:
    print(f"| thinking off | unparseable ({exc}) |")
PYX
}

cleanup() {
  local rc=$?
  if [ -n "$SUSHI_PID" ] && kill -0 "$SUSHI_PID" 2>/dev/null; then
    log "stopping sushi pid $SUSHI_PID"
    kill "$SUSHI_PID" 2>/dev/null || true
  fi
  if [ "$SKIP_STOP" != "1" ]; then
    log "restoring the assistant ($ASSIST)"
    "$SERVE_BIN" start "$ASSIST" >> "$LOG_DIR/glm-sushi-probe.out" 2>&1 \
      || log "WARN: restore failed — start it by hand: ./serve start $ASSIST"
  fi
  echo "" >> "$RESULT"
  echo "- memory at exit (rc=$rc): \`$(phys)\`" >> "$RESULT"
  echo "- logs: $LOG_DIR/glm-sushi.log, $LOG_DIR/glm-sushi-bench-*.json" >> "$RESULT"
  log "done rc=$rc — verdict in $RESULT"
}
trap cleanup EXIT INT TERM

# ------------------------------------------------------------------ preflight
[ -x "$SUSHI_BIN" ] || { log "FATAL: sushi not at $SUSHI_BIN (brew install beamivalice/tap/sushi)"; exit 1; }
[ -d "$PACK" ] || { log "FATAL: pack missing at $PACK (sushi pull glm-5.3-flash)"; exit 1; }
[ -f "$BENCH" ] || { log "FATAL: bench script missing: $BENCH"; exit 1; }
command -v timeout >/dev/null 2>&1 || { log "FATAL: timeout(1) missing (coreutils)"; exit 1; }

PACK_GB=$(( $(du -sk "$PACK" | cut -f1) * 1024 / 1000000000 ))
log "pack: $PACK ($PACK_GB GB on disk)"
log "gpu pool: iogpu.wired_limit_mb=$(pool_mb) MB (Sushi documents 120000 for a 128 GB Mac; auto here is 110100 MiB)"
log "memory at start: $(phys)"
[ "$SKIP_STOP" = "1" ] || log "this probe stops $ASSIST (:$ASSIST_PORT); live sessions die ~45 s in"

{ echo "# GLM-5.3-Flash (Sushi 2.4bpw) probe — $(date '+%Y-%m-%d %H:%M')"
  echo ""
  echo "- pack $PACK_GB GB, ctx request $CTX, pool \`iogpu.wired_limit_mb=$(pool_mb)\` MB"
  echo "- memory at start: \`$(phys)\`"
} > "$RESULT"

# ------------------------------------------------- stop the assistant, drain memory
if [ "$SKIP_STOP" != "1" ]; then
  log "stopping $ASSIST"
  "$SERVE_BIN" stop "$ASSIST" >> "$LOG_DIR/glm-sushi-probe.out" 2>&1 || true
  # The assistant's wired pool is released when its process exits; watch PhysMem's wired
  # figure stop falling, and never start GLM while the old pool is still above 30 G.
  prev=""; waited=0
  while [ "$waited" -lt "$DRAIN_SECONDS" ]; do
    now="$(wired_g)"
    [ -z "$now" ] && break
    if [ "$now" -lt 30 ]; then break; fi
    if [ -n "$prev" ] && [ "$now" = "$prev" ]; then break; fi
    prev="$now"; sleep 10; waited=$((waited + 10))
  done
  log "memory after draining $ASSIST: $(phys)"
  echo "- memory after draining the assistant: \`$(phys)\`" >> "$RESULT"
fi

# ------------------------------------------------------------------ load
log "loading GLM ($((PACK_GB - 8)) GiB of weights + warmup)"
if ! start_sushi; then
  log "retrying with --skip-mem-preflight (Sushi's preflight is conservative: macOS reclaims"
  log "file cache as MLX allocates; a real over-commit crashes the server instead)"
  SUSHI_PID=""
  SUSHI_EXTRA="$SUSHI_EXTRA --skip-mem-preflight"
  if ! start_sushi; then
    { echo ""; echo "**LOAD FAILED** — see $LOG_DIR/glm-sushi.log"; } >> "$RESULT"
    exit 1
  fi
  echo "- loaded only with \`--skip-mem-preflight\`" >> "$RESULT"
fi
model_id
log "memory with GLM resident: $(phys)"
echo "- memory with GLM resident: \`$(phys)\`" >> "$RESULT"
api "http://127.0.0.1:$PORT/metrics.json" > "$LOG_DIR/glm-sushi-metrics.json" 2>/dev/null || true

# ------------------------------------------------------------- depth ladder
echo "" >> "$RESULT"
echo "## Depth ladder (one load; decode / TTFT / prefill as measured)" >> "$RESULT"
echo '```' >> "$RESULT"
# Rates come from sushi's OWN counters/histograms (it is an mlx-serve fork, same
# /metrics.json): prefill_time_seconds / decode_time_seconds / time_to_first_token
# deltas around exactly our requests, with a POLLUTED verdict if the server says any
# foreign request completed inside the window. Nothing is re-derived client-side.
timeout $((LOAD_TIMEOUT + 600)) "$PY" "$BENCH" --port "$PORT" --model "$MODEL_ID" \
  --depths "${DEPTHS[*]}" --reps 2 --gen-tokens "${GEN_TOKENS[0]:-256}" --timeout "$READY_TIMEOUT" \
  --out "$LOG_DIR/glm-sushi-bench.json" 2>&1 | tee -a "$RESULT"
echo '```' >> "$RESULT"
log "memory after the depth ladder: $(phys)"
echo "- memory after the depth ladder: \`$(phys)\`" >> "$RESULT"

# ------------------------------------------------------------------ smokes
echo "" >> "$RESULT"
echo "| smoke | result |" >> "$RESULT"
echo "| --- | --- |" >> "$RESULT"
tool_smoke
vision_smoke
think_smoke

log "probe complete"
