---
description: Benchmark MoE vs 27B on the same tasks
model: opencode/kimi-k2.5
---

Run the same prompts through both models and compare outputs.

## Setup

Read ports from `~/.localllm/models.jsonc`. Make sure both servers are running:

```bash
# Replace ports with values from models.jsonc
curl http://localhost:<moe-port>/health
curl http://localhost:<dense-port>/health
```

## Procedure

1. Pick 3-5 representative prompts (code gen, debugging, reasoning, multi-step)
2. Send each prompt to both models with identical parameters
3. Compare:
   - Output quality (correctness, completeness, style)
   - Response length
   - Any systematic differences

## Example

Read model paths and ports from `~/.localllm/models.jsonc`:

```bash
MOE_PORT=30083
DENSE_PORT=30090
MOE_MODEL=~/.localllm/models/Qwen3.6-35B-A3B-UD-MLX-4bit
DENSE_MODEL=~/.localllm/models/Qwen3.6-27B-UD-MLX-4bit

# MoE response
curl -s "http://localhost:$MOE_PORT/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d "{\"model\":\"$MOE_MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"Write a Python function to merge two sorted arrays\"}],\"max_tokens\":512,\"temperature\":0.0}"

# Dense response
curl -s "http://localhost:$DENSE_PORT/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d "{\"model\":\"$DENSE_MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"Write a Python function to merge two sorted arrays\"}],\"max_tokens\":512,\"temperature\":0.0}"
```

4. Save comparison to `benchmarks/comparison-<date>.md`
