---
description: Benchmark engines head-to-head (llama.cpp mtp-vision vs MLX)
model: opencode/kimi-k2.5
---

Run the same prompts through both dense engines and compare. Default pairing:
`mtp-vision` (llama.cpp, Qwen3.8) vs `mlx-dense` (MLX, Qwen3.6).

**Caveat:** the engines are on different model generations (Qwen3.8 vs
Qwen3.6) and quant formats — compare the *systems* (engine + stack), and say
so in the writeup. For a strict same-model comparison, both servers must run
the same model — check `servers.jsonc` before claiming model parity.

## Setup

Read ports/models from `servers.jsonc`. Make sure both servers are running:

```bash
curl http://localhost:30082/health   # mtp-vision (llama.cpp)
curl http://localhost:30090/health   # mlx-dense (MLX)
```

Do **not** silently start mtp-vision (it hosts the opencode assistant).

## Procedure

1. Pick 3–5 representative prompts (code gen, debugging, reasoning, multi-step)
2. Send each prompt to both engines with identical sampling (Qwen thinking
   defaults: temp 1.0, top_p 0.95, top_k 20)
3. Compare:
   - Output quality (correctness, completeness, style)
   - TG (tokens/sec) decode speed
   - TTFT (prompt processing) at a medium context (~16–32K)
   - Response length / any systematic differences

## Example

```bash
LLAMA_URL="http://localhost:30082/v1/chat/completions"
MLX_URL="http://localhost:30090/v1/chat/completions"

for URL in "$LLAMA_URL" "$MLX_URL"; do
  curl -s "$URL" \
    -H "Content-Type: application/json" \
    -d '{"messages":[{"role":"user","content":"Write a Python function to merge two sorted arrays"}],"max_tokens":512,"temperature":1.0,"top_p":0.95,"top_k":20}'
done
```

4. Save comparison to `benchmarks/comparison-<date>.md`.
