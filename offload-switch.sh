#!/usr/bin/env bash
# Exclusive-window switch: production flash-next (lazy-mmap) -> build-direct (--lazy-mode on-direct).
# THIS STOPS PORT 30085 — the llama server serving the current assistant session.
# Run from a real terminal, never hand it to an agent session.
set -uo pipefail
cd ~/localllm

MODEL="$HOME/.cache/huggingface/hub/models--unsloth--Qwen3.8-Flash-Next-GGUF/snapshots/c8b5954a88c2775c546b92593eda40ea041d3176/UD-IQ4_XS/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf"
MMPROJ="$HOME/.cache/huggingface/hub/models--unsloth--Qwen3.8-Flash-Next-GGUF/snapshots/c8b5954a88c2775c546b92593eda40ea041d3176/mmproj-F16.gguf"
BIN="$HOME/llama.cpp-turbo/build-direct/bin/llama-server"
LOG="$HOME/localllm/flash-next-offload.log"

# refuse if offload server already up (no doubles, per 2026-09-01 rule)
if pgrep -f 'llama-server.*30085' >/dev/null && [ -f pids/flash-next-offload.pid ]; then
  echo "switch script already ran? delete pids/flash-next-offload.pid to redo"; exit 1
fi
"$BIN" --version | head -2 | sed 's/^/test binary: /'
[ "${1:-}" = "--yes" ] || { echo "re-run with --yes to proceed (stops 30085, kills assistant session)"; exit 1; }

echo "== stopping production flash-next =="
pkill -f 'llama-server.*30085' 2>/dev/null
while pgrep -f 'llama-server.*30085' >/dev/null; do sleep 2; done
rm -f pids/flash-next.pid
sleep 5
vm_stat | awk '/wired/{printf "wired after stop: %.1f GB\n", $2*16384/2^30}'

echo "== launching offload server on 30085 (log: flash-next-offload.log) =="
nohup "$BIN" \
  -m "$MODEL" --mmproj "$MMPROJ" \
  --port 30085 --host 127.0.0.1 \
  -np 1 -ngl 99 -c 262144 \
  --temp 1.0 --top-p 0.95 --top-k 20 --min-p 0.0 \
  --image-min-tokens 1024 \
  --chat-template-kwargs '{"reasoning_effort":"xhigh","preserve_thinking":true}' \
  --lazy-mode on-direct -lv 4 \
  >> "$LOG" 2>&1 &
echo $! > pids/flash-next.pid
echo "pid $(cat pids/flash-next.pid); waiting for model load (111 GB shards, expect ~2-3 min)..."

for i in $(seq 1 120); do
  sleep 5
  grep -q 'PLE direct read' "$LOG" && { echo "== PLE direct-read marker seen: $(grep -m1 'PLE direct read' "$LOG" | tail -c 100)"; break; }
done
for i in $(seq 1 120); do
  curl -sf -m 2 http://127.0.0.1:30085/health >/dev/null && { echo "== HEALTHY. Now:"; break; }
  sleep 5
done
cat <<'EOF'
== NEXT (from terminals) ==
  1. kill the old baseline sampler, then capture with a NEW LABEL to its own file:
       kill 66293   # or Ctrl-C in the sampler window
       cd ~/localllm && ./ple-sample.sh offload | tee ~/localllm/samples-offload.csv
  2. restart opencode — it will land on the offload server; ask the new session
     for a memory report + ~500-word generation, 3x, same as baseline method.
  3. grade: wired ceiling during decode (baseline peaked ~103), prefill rate in
     flash-next-offload.log, server_rss, shard_resident plateau.
== ROLLBACK ==
  pkill -f 'llama-server.*30085'; rm -f pids/flash-next-offload.pid
  cd ~/localllm && ./serve start flash-next
EOF
