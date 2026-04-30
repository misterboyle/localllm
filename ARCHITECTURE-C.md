# Architecture C: MoE-Only, Pinned Slots, No Cache

> **Historical note:** This architecture was designed for llama.cpp's llama-server.
> The current MLX setup uses a different slot/cache model. The concepts still apply
> as design goals but the implementation details differ.

**Date:** 2026-04-29

---

## Concept

Single MoE model with pinned slots to specific conversations, minimal prompt cache. Maximizes parallel throughput by leveraging the MoE's 3B active params (9x fewer FLOPs than 27B per decode step).

| Model | Slots | Pin | Cache | Per-slot ctx |
|-------|-------|-----|-------|-------------|
| MoE 35B-A3B | 4-6 slots | Each slot pinned to one conversation | Minimal (just cold-start resume) | 262K (full) |

---

## Rationale

1. **Decode speed:** MoE's 3B active params decode ~5-9x faster than 27B (less compute per token)
2. **No cache thrashing:** Pinned slots mean conversations never migrate → no prompt cache saves/evicts → predictable memory
3. **Bounded memory:** Known slots × known max context × 26 bytes = deterministic upper bound
4. **Simpler config:** One process, no migration logic, no LCP matching complexity

---

## Memory Budget

| Component | Cost | Notes |
|-----------|------|-------|
| Fixed (MoE weights + buffers) | ~35 GB | Per process |
| KV buffer (6 slots × 64K × 26B) | ~10 GB | Smaller context for workers |
| KV faulted (6 slots active, avg 20K tokens) | ~3 GB | 120K active tokens × 26B |
| Checkpoints (6 slots × 0 × 149 MiB) | 0 GB | `--ctx-checkpoints 0` (pinned = no need) |
| Prompt cache | ~0 GB | Minimal (no migration) |
| Overhead | ~16 GB | CUDA, macOS |
| **Total** | **~64 GB** | |

---

## Tradeoffs

### Advantages
- **Fast decode:** 3B active params = higher throughput per conversation
- **Predictable memory:** Pinned slots + no cache = fixed budget
- **No contention:** Each slot owns its conversation, no migration overhead
- **Simple:** One process, no LCP matching, no checkpoint thrashing

### Disadvantages
- **Reasoning quality:** MoE may not match 27B for complex reasoning
- **Less context per conversation:** 64K limit (vs 262K for 27B)
- **No elasticity:** Pinned slots can't be repurposed if a conversation dies
- **No conversation migration:** If a pinned slot crashes, that conversation is lost (no cache backup)

---

## When Architecture C Makes Sense

- Workload is predominantly short, focused tasks (tool use, code generation, analysis)
- Conversations don't need >64K context
- Priority is throughput and predictability over reasoning depth
- Willing to accept MoE reasoning quality vs 27B

---

*Last updated: 2026-04-29*
