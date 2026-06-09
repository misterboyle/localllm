---
name: benchmark
description: Run server benchmarks to measure decode speed, TTFT, and context-length degradation
---

Run benchmarks against live servers to measure decode speed, time-to-first-token (TTFT), and how these degrade with growing context length.

Two benchmark modes, depending on what you're testing:

- **Dense servers** (single-user agentic coding): context-length benchmarks at concurrency 1
- **MoE servers** (multi-agent factory): concurrency scaling benchmarks

## Server Management

Servers are defined in `servers.jsonc` and managed with the `serve` script:

```bash
# Start/stop individual servers
./serve start mtp-vision
./serve stop mlx-dense-mtp

# Check status
./serve status
```

Key servers for the dense comparison:

| Name | Backend | Model | Port |
|------|---------|-------|------|
| `mtp-vision` | llama.cpp | Qwen3.6-27B-UD-Q4_K_XL GGUF | 30082 |
| `mlx-dense-mtp` | optiq (MLX) | Qwen3.6-27B-OptiQ-4bit | 30091 |

## Context-Length Benchmarks (Dense)

For comparing dense servers on agentic coding workloads — single-user decode speed at growing context depths.

### 1. Generate prompts

```bash
python3 /tmp/gen_prompts.py
```

Generated files: `/tmp/prompt-10k.txt`, `prompt-50k.txt`, `prompt-100k.txt`

### 2. Run benchmarks

```bash
PY=~/mlx-lm-turbo/venv/bin/python3
BENCH=~/mlx-lm-turbo/benchmarks/server_benchmark.py

# Short context baseline
"$PY" "$BENCH" \
  --url "http://localhost:<port>/v1/chat/completions" \
  --api-key "sk-optiq-local" \   # only needed for optiq servers
  --model "<model-name>" \
  --max-tokens 256 --concurrency 1 --total-requests 3 \
  --output benchmarks/bench-<name>-conc1.json

# 10K context
"$PY" "$BENCH" \
  --url "..." --api-key "..."
  --model "<model-name>" \
  --prompt-file /tmp/prompt-10k.txt \
  --max-tokens 256 --concurrency 1 --total-requests 3 \
  --output benchmarks/bench-<name>-10k-conc1.json

# 50K context
"$PY" "$BENCH" \
  --url "..." --model "<model-name>" \
  --prompt-file /tmp/prompt-50k.txt \
  --max-tokens 256 --concurrency 1 --total-requests 3 \
  --output benchmarks/bench-<name>-50k-conc1.json

# 100K context
"$PY" "$BENCH" \
  --url "..." --model "<model-name>" \
  --prompt-file /tmp/prompt-100k.txt \
  --max-tokens 256 --concurrency 1 --total-requests 3 \
  --output benchmarks/bench-<name>-100k-conc1.json
```

### 3. Read results

Key metrics for each context depth:

- **Decode speed (per-req tok/s)**: how fast tokens come once generation starts — degrades with context length
- **TTFT first request**: full prompt processing time (cold cache) — this is the pain point at 80K+ depth
- **TTFT cached**: prompt processing with prefix cache hit — measures how well KV cache is preserved
- **Prompt processing speed (implied)**: prompt_tokens / first_ttft — how fast the server digests context

## Concurrency Benchmarks (MoE)

For testing batch throughput on MoE servers:

```bash
# Single-threaded baseline
"$PY" "$BENCH" \
  --url "http://localhost:<port>/v1/chat/completions" \
  --model "<model>" \
  --max-tokens 256 --concurrency 1 --total-requests 5

# Multi-threaded
"$PY" "$BENCH" \
  --url "http://localhost:<port>/v1/chat/completions" \
  --model "<model>" \
  --max-tokens 256 --concurrency 4 --total-requests 20 \
  --output benchmarks/bench-<name>-conc4.json

# High concurrency
"$PY" "$BENCH" \
  --url "http://localhost:<port>/v1/chat/completions" \
  --model "<model>" \
  --max-tokens 256 --concurrency 8 --total-requests 32 \
  --output benchmarks/bench-<name>-conc8.json
```

## Analyzing Results

```bash
# Compare all results
python3 benchmarks/compare.py

# Filter by model name
python3 benchmarks/compare.py mtp-vision
python3 benchmarks/compare.py optiq
```

The comparison tool shows:
- Aggregate throughput (tok/s) across concurrency levels
- Per-request decode speed
- TTFT averages and p95
- Scaling efficiency (actual vs ideal)

## Notes

- **Run benchmarks sequentially**, one at a time. Concurrent benchmarks against the same server = compound stress test, not clean data.
- The `mlx-lm-turbo` repo is a sibling at `~/mlx-lm-turbo/` — use absolute paths.
- OptiQ servers require `--api-key sk-optiq-local`; llama.cpp servers accept no auth header.
- Generated prompt files use synthetic conversation text. Token estimates are rough (chars/4).
