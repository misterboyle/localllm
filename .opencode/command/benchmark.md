---
description: Run full benchmark suite against mtp-vision and save results
model: opencode/kimi-k2.5
---

Run the benchmark suite against `mtp-vision` (the production dense server).
Context-length levels simulate agent conversations growing in depth; each
concurrency level simulates simultaneous conversations.

## Steps

1. Read port and model from `servers.jsonc` (this repo). Verify the server is
   up:
   ```bash
   curl http://localhost:30082/health
   ```
   If it's down, tell Michael — do not start it silently (restarting mtp-vision
   kills the active opencode session).

2. Run context-length benchmarks (concurrency 1 — single-user agentic decode):
   ```bash
   # Sibling repo — absolute paths required
   PY=$HOME/mlx-lm-turbo/venv/bin/python3
   BENCH=$HOME/mlx-lm-turbo/benchmarks/server_benchmark.py
   URL="http://localhost:30082/v1/chat/completions"

   for CTX in baseline 10k 50k 100k; do
     PROMPT_FILE="/tmp/prompt-${CTX}.txt"   # generate first per skills/benchmark/SKILL.md
     "$PY" "$BENCH" \
       --url "$URL" \
       --model "Qwen3.8-27B-UD-Q4_K_XL" \
       --prompt-file "$PROMPT_FILE" \
       --max-tokens 256 --concurrency 1 --total-requests 3 \
       --output "benchmarks/bench-mtp-vision-${CTX}-conc1.json"
     echo "---"
   done
   ```
   (Check `server_benchmark.py --help` for the exact prompt-file flag name if
   it has changed.)

3. Run concurrency scaling at short context (1, 4, 8):
   ```bash
   for CONC in 1 4 8; do
     "$PY" "$BENCH" \
       --url "$URL" \
       --model "Qwen3.8-27B-UD-Q4_K_XL" \
       --max-tokens 256 \
       --concurrency "$CONC" \
       --total-requests $((CONC * 4)) \
       --output "benchmarks/bench-mtp-vision-conc${CONC}.json"
     echo "---"
   done
   ```

4. Summarize results:
   - TG (tokens/sec) at each context depth — degradation curve
   - TTFT (prompt processing) at each context depth
   - Aggregate vs per-request TG across concurrency
   - Spot regressions against the latest previous run in `benchmarks/`

5. Save summary to `benchmarks/summary-<date>.md`.
