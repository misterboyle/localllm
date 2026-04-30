# Memory Budget Formula — Qwen3.6-27B Dense Server

> **Historical note:** This formula was derived from llama.cpp testing. The MLX
> server has different memory characteristics (no checkpoints, KV cache
> quantization via turboquant-mlx, different concurrency model). Kept as
> reference for understanding the migration.

**Validated:** 2026-04-29, live testing, 2-slot config
**Model:** Qwen3.6-27B-UD-Q6_K_XL
**Server:** --kv-unified, --ctx-checkpoints 32, --cache-ram 65536

---

## The Formula

```
RSS = FIXED + KV_FAULTED(active_tokens × 26B) + CACHE(cold conversations, ≤ cache-ram) + CHECKPOINTS(active_slots × count × 149 MiB) + OVERHEAD(~16 GB)
```

| Component | Cost | Variable? | Notes |
|-----------|------|-----------|-------|
| Fixed (weights Q6_K + compute buffers) | ~35 GB | No | Per process, independent of slots |
| KV buffer (faulted) | active_tokens × 26 bytes | Yes | Only faults pages for active tokens, bounded by `-c` |
| Prompt cache | Sum of cold conversation sizes | Yes | Capped by `--cache-ram`, serializes KV + checkpoints |
| In-slot checkpoints | slots × count × 149 MiB | Yes | Each checkpoint is 149 MiB regardless of token count |
| Overhead | ~16 GB | No | CUDA allocations, macOS memory management, page alignment |

---

## Architecture A vs B Budget

### Architecture A: 27B + MoE (2 processes)

| Component | 27B | MoE | Total |
|-----------|-----|-----|-------|
| Fixed (weights + buffers) | 35 GB | 35 GB | **70 GB** |
| KV faulted (peak) | 8 GB (130K tokens) | 1.5 GB (10K tokens) | 9.5 GB |
| In-slot checkpoints (1 slot each) | 4.7 GB | 4.7 GB | 9.4 GB |
| Prompt cache (8 workers) | 2 GB | 10 GB | 12 GB |
| Overhead | 16 GB | 16 GB | 32 GB |
| **Total peak** | | | **~133 GB** |

### Architecture B: 27B × 2 slots (1 process)

| Component | Value |
|-----------|-------|
| Fixed (weights + buffers) | **35 GB** |
| KV faulted (both slots) | 9.5 GB |
| In-slot checkpoints (2 slots) | 4.7 GB |
| Prompt cache (8 workers) | 10 GB |
| Overhead | 16 GB |
| **Total peak** | **~75 GB** |

---

## Key Findings

1. **Checkpoints are the hidden cost** — 32 × 149 MiB = 4.7 GB per active slot, plus serialized to cache (70-75% of cache entry size). With `--ctx-checkpoints 0`, save ~9 GB (2 slots).

2. **Prompt cache grows fast** — 3 conversations already use 14 GB (checkpoint serialization dominates). 8 worker conversations ≈ 10 GB with `--ctx-checkpoints 0`.

3. **Architecture A is tight** — 133 GB peak vs 128 GB limit. Need `--ctx-checkpoints 0` to reduce by ~9 GB → 124 GB. Or reduce worker context.

4. **Architecture B has headroom** — 75 GB vs 128 GB = 53 GB headroom. Comfortable with `--ctx-checkpoints 0` → 66 GB.

---

## Optimization: `--ctx-checkpoints 0`

With high LCP (>0.8), the prompt cache handles everything. In-slot checkpoints provide marginal value. Setting `--ctx-checkpoints 0`:

| Architecture | Before | After | Saved |
|--------------|--------|-------|-------|
| A (27B + MoE) | 133 GB | 124 GB | 9 GB |
| B (27B × 2) | 75 GB | 66 GB | 9 GB |

---

*Last updated: 2026-04-29*
