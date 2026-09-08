#!/usr/bin/env bash
# Real-agent DOE for MTP n_max: scripted `opencode run` bug-fix sessions against
# an exclusive-window flash server. Experimental unit = one full agent session
# (wall time + peak ctx + success); server-side tg/acc harvested per run via
# mtp-harvest.py. Arms (vanilla / n_max 1 / n_max 2) rotate order across 3
# rounds. Kills the assistant server on start — do not talk to the assistant
# while it runs. opencode finds the server via its default model (provider
# llama-flash -> port 30085), so no -m flag here.
#
# Results: mtp-doe/mtp-doe-opencode.csv (+ harvest into mtp-experiments.csv);
# summary auto-committed to the wiki log.
set -uo pipefail  # NOT -e: every exit path restores the production server

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"
DOE="mtp-doe"
RUNS="$DOE/runs"
mkdir -p "$RUNS"
exec > >(tee -a "$RUNS/run-doe.log") 2>&1

BIN="$HOME/llama.cpp-mtp-src/build-mtp/bin/llama-server"
MODEL="$HOME/.cache/huggingface/hub/models--unsloth--Qwen3.8-Flash-Next-GGUF/snapshots/c8b5954a88c2775c546b92593eda40ea041d3176/UD-IQ4_XS/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf"
MMPROJ="$HOME/.cache/huggingface/hub/models--unsloth--Qwen3.8-Flash-Next-GGUF/snapshots/c8b5954a88c2775c546b92593eda40ea041d3176/mmproj-F16.gguf"
HEAD="$HOME/localllm/mtp-heads/MTP/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf"
PORT=30085
CSV="$DOE/mtp-doe-opencode.csv"
PROMPT='This Python project has exactly two bugs, one per module under ledger/. The tests in tests/ define correct behavior — do NOT modify anything under tests/. Read every module and test file fully before editing. Fix the bugs, then run: python3 -m unittest discover -s tests until everything passes. When done, reply DONE as your final line.'

log() { echo "[$(date '+%H:%M:%S')] $*"; }

SRV_PID=""
restore() {
  if [ -n "$SRV_PID" ]; then
    kill "$SRV_PID" 2>/dev/null || true
    wait "$SRV_PID" 2>/dev/null || true
    SRV_PID=""
  fi
  ./serve start flash-next-mtp || log "RESTORE FAILED — start flash-next-mtp manually"
}
trap restore EXIT INT TERM

start_arm() {  # $1=arm (none|N) $2=logfile
  local args=(-m "$MODEL" --mmproj "$MMPROJ" --host 127.0.0.1 --port "$PORT"
              -np 1 -ngl 99 -c 262144
              --temp 1.0 --top-p 0.95 --top-k 20 --min-p 0.0 -lv 4
              --image-min-tokens 1024
              --chat-template-kwargs '{"reasoning_effort":"xhigh","preserve_thinking":true}')
  [ "$1" != "none" ] && args+=(-md "$HEAD" --spec-type draft-mtp --spec-draft-n-max "$1")
  log "starting arm=$1 (log $2)"
  "$BIN" "${args[@]}" > "$2" 2>&1 &
  SRV_PID=$!
  local w=0
  until curl -sf "http://127.0.0.1:$PORT/health" > /dev/null 2>&1; do
    kill -0 "$SRV_PID" 2>/dev/null || { log "server died at start (arm $1)"; return 1; }
    sleep 3; w=$((w + 3))
    [ "$w" -ge 420 ] && { log "health timeout (arm $1)"; return 1; }
  done
  log "server up arm=$1 after ${w}s"
}

stop_arm() {
  if [ -n "$SRV_PID" ]; then
    kill "$SRV_PID" 2>/dev/null || true
    wait "$SRV_PID" 2>/dev/null || true
    SRV_PID=""
    sleep 6
  fi
}

session() {  # wd title outfile timeout prompt  -> exits with opencode rc (124=timeout)
  # NOTE: -m must be explicit — `opencode run` splits provider/model on the FIRST
  # slash, so the configured default model (ID is an absolute path) arrives
  # mangled (ProviderModelNotFoundError). "llama-flash/" + path works.
  python3 - "$1" "$2" "$3" "$4" "$5" <<'PY'
import subprocess, sys
wd, title, out, tout, prompt = sys.argv[1:6]
model = "llama-flash/" + "/Users/michael/.cache/huggingface/hub/models--unsloth--Qwen3.8-Flash-Next-GGUF/snapshots/c8b5954a88c2775c546b92593eda40ea041d3176/UD-IQ4_XS/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf"
try:
    p = subprocess.run(["opencode", "run", "-m", model, "--dir", wd, "--auto", "--print-logs",
                        "--title", title, prompt],
                       capture_output=True, text=True, timeout=int(tout), start_new_session=True)
    open(out, "w").write(p.stdout + "\n--STDERR--\n" + p.stderr)
    sys.exit(p.returncode)
except subprocess.TimeoutExpired:
    open(out, "a").write("\n--TIMEOUT--\n")
    sys.exit(124)
PY
}

warmup_server() {  # first real request compiles Metal graphs — do it via curl, not opencode
  log "warmup: first chat completion (graph compile) ..."
  local w=0
  until curl -sf -m 180 http://127.0.0.1:$PORT/v1/chat/completions \
        -H 'Content-Type: application/json' \
        -d '{"messages":[{"role":"user","content":"ping"}],"max_tokens":4}' \
        | python3 -c "import json,sys; d=json.load(sys.stdin); sys.exit(0 if d.get('choices') else 1)"; do
    sleep 5; w=$((w + 5))
    [ "$w" -ge 300 ] && { log "warmup failed — server rejects chat completions"; return 1; }
  done
  log "warmup done (${w}s)"
}

log "=== real-agent DOE start; stopping production flash-next-mtp ==="
./serve stop flash-next-mtp || true
[ -f "$CSV" ] || echo "round,arm,wall_s,rc,success,max_ctx,turns" > "$CSV"

ORDERS=("none 1 2" "1 2 none" "2 none 1")
SMOKED=0
for R in 1 2 3; do
  read -ra ORDER <<< "${ORDERS[$((R - 1))]}"
  for ARM in "${ORDER[@]}"; do
    LOGF="$RUNS/server-r${R}-${ARM}.log"
    log "-- run: round=$R arm=$ARM --"
    if ! start_arm "$ARM" "$LOGF"; then
      echo "$R,$ARM,,,,-1,-1,-1" >> "$CSV"
      stop_arm
      continue
    fi
    if [ "$SMOKED" = 0 ]; then  # gate: scripted opencode works against this server
      warmup_server || { log "aborting: warmup"; exit 2; }
      mkdir -p /tmp/doe-smoke-gate
      GATE_OK=0
      for try in 1 2 3; do
        session /tmp/doe-smoke-gate "doe-smoke-$try" /tmp/doe-smoke.out 300 \
          "Reply with exactly one word: READY"
        if grep -q READY /tmp/doe-smoke.out; then GATE_OK=1; break; fi
        log "smoke attempt $try failed; retrying in 15s"
        sleep 15
      done
      if [ "$GATE_OK" != 1 ]; then
        log "SMOKE FAILED x3 — aborting. smoke output:"
        tail -40 /tmp/doe-smoke.out
        log "last arm-server lines:"
        tail -20 "$LOGF"
        exit 2
      fi
      log "smoke passed (attempt $try)"
      SMOKED=1
    fi
    WORK="$RUNS/work"
    rm -rf "$WORK"
    cp -R "$DOE/task" "$WORK"
    T0=$(date +%s)
    session "$WORK" "doe-r${R}-${ARM}" "$RUNS/out-r${R}-${ARM}.txt" 1200 "$PROMPT"
    RC=$?
    WALL=$(( $(date +%s) - T0 ))
    ( cd "$WORK" && python3 -m unittest discover -s tests ) > "$RUNS/probe-r${R}-${ARM}.txt" 2>&1 && S=1 || S=0
    MAXCTX=$(grep -oE "task\.n_tokens = [0-9]+" "$LOGF" | tail -1 | grep -oE "[0-9]+" || echo 0)
    TURNS=$(grep -c "new prompt" "$LOGF" || true)
    log "result: r$R arm=$ARM wall=${WALL}s rc=$RC success=$S peak_ctx=${MAXCTX:-0} turns=${TURNS:-0}"
    echo "$R,$ARM,$WALL,$RC,$S,${MAXCTX:-0},${TURNS:-0}" >> "$CSV"
    stop_arm
    python3 mtp-harvest.py "$LOGF" --source "oc-doe-r${R}-${ARM}" --build d1a9235 --nmax "$ARM" || true
  done
done

log "=== DOE complete — restoring production (n_max from servers.jsonc) ==="
trap - EXIT INT TERM
restore

python3 - <<'PY'
import csv, statistics, subprocess, datetime
rows = list(csv.DictReader(open("mtp-doe/mtp-doe-opencode.csv")))
lines = []
for arm in ("none", "1", "2"):
    sel = [r for r in rows if r["arm"] == arm and r["rc"] == "0"]
    if sel:
        med = statistics.median(int(r["wall_s"]) for r in sel)
        succ = sum(int(r["success"]) for r in sel)
        lines.append(f"  arm {arm:>4}: median wall {med:6.0f}s  n={len(sel)}  success {succ}/{len(sel)}")
entry = (f"\n## [{datetime.date.today()}] work | MTP real-agent DOE (opencode run) complete\n"
         + "\n".join(lines)
         + "\nNEXT: read mtp-doe/mtp-doe-opencode.csv + mtp-experiments.csv sources oc-doe-*;"
           " compare wall-time ranking vs sweep's n_max=1 verdict at meaningful ctx;"
           " finalize servers.jsonc and close the DOE thread.\n")
open("/Users/michael/wiki/wiki/log.md", "a").write(entry)
subprocess.run(["git", "-C", "/Users/michael/wiki", "add", "wiki/log.md"], check=True)
subprocess.run(["git", "-C", "/Users/michael/wiki", "commit", "-q", "-m", "log: real-agent MTP DOE results"], check=True)
print("wiki summary committed")
PY

git add "$CSV" "$RUNS/run-doe.log" && git commit -q -m "doe: real-agent opencode sweep results" || true
log "ALL DONE"
