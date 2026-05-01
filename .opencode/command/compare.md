---
description: Benchmark MoE vs 27B on the same tasks
model: opencode/kimi-k2.5
---

Run the same prompts through both models and compare outputs.

## Setup

Make sure both servers are running:
- MoE: `curl http://localhost:30083/health`
- 27B: `curl http://localhost:30090/health`

## Procedure

1. Pick 3-5 representative prompts (code gen, debugging, reasoning, multi-step)
2. Send each prompt to both models with identical parameters
3. Compare:
   - Output quality (correctness, completeness, style)
   - Response length
   - Any systematic differences

## Example

```bash
# MoE response
curl -s http://localhost:30083/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"/Users/michael/.localllm/models/Qwen3.6-35B-A3B-UD-MLX-4bit","messages":[{"role":"user","content":"Write a Python function to merge two sorted arrays"}],"max_tokens":512,"temperature":0.0}'

# 27B response
curl -s http://localhost:30090/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"/Users/michael/.localllm/models/Qwen3.6-27B-UD-MLX-4bit","messages":[{"role":"user","content":"Write a Python function to merge two sorted arrays"}],"max_tokens":512,"temperature":0.0}'
```

4. Save comparison to `benchmarks/comparison-<date>.md`
