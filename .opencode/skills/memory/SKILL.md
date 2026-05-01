---
name: memory
description: Analyze memory usage and budget for MLX models
---

Use the memory budget calculator to understand model memory characteristics.

## Memory Budget Calculator

```bash
# MoE analysis
python3 memory-budget.py --model moe

# 27B dense analysis
python3 memory-budget.py --model 27b

# Side-by-side comparison
python3 memory-budget.py --model both
```

## Memory Snapshot

```bash
# Check running server memory
./snapshot-memory.sh running

# Full memory report
./snapshot-memory.sh
```

## Key Numbers (MoE 35B-A3B)

- Weights: 21.6 GB (4-bit safetensors)
- Fixed overhead: 39 GB (weights 26 + compute 8 + Python/MLX 5)
- KV cache: 20 KB/token active (fp16), 6.9 KB/token cached (K8,V2)
- KV budget: 74 GB of 128 GB system
- Max 13 concurrent conversations at 262K context

## Key Numbers (27B Dense)

- Weights: 26.2 GB (4-bit safetensors)
- Fixed overhead: 44 GB (weights 31 + compute 8 + Python/MLX 5)
- KV cache: 64 KB/token active (fp16), 22 KB/token cached (K8,V2)
- KV budget: 69 GB of 128 GB system
- Max 3 concurrent conversations at 262K context

## Rules

- Always run `memory-budget.py --model both` before making concurrency changes
- If total memory exceeds ~110 GB, reduce concurrency or cache size
- The MoE model can handle ~4x more concurrent conversations than 27B dense
- KV quantization is K8+V2 (2.9x compression) — safe for all workloads
