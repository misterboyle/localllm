# KV Cache Multi-Slot Memory Scaling Test — Progress Report

> **Historical note:** This test was run against llama.cpp's llama-server. The
> current MLX setup uses a different memory model (no checkpoints, different KV
> cache quantization). Kept as reference for migration decisions.

**Date:** 2026-04-29
**Model:** Qwen3.6-27B-UD-Q6_K_XL (dense)
**Server config:** 2 slots, 262K per-slot, 524K total context, --kv-unified, --cache-ram 65536, --ctx-checkpoints 32, --checkpoint-every-n-tokens 8192

---

## Test Objective

Validate the **KV buffer page faulting effect** documented in KV-CACHE-ANALYSIS.md: that the unified KV buffer (52GB mmap'd) only faults pages in when written, causing memory to scale with active slots rather than total capacity.

**Hypothesis:** 1 active slot → ~42-44 GB RSS. 2 active slots → ~55-60 GB RSS. (Smaller jump than the 64→107 GB observed with 8 slots, proportional to the smaller total buffer.)

---

## Test Setup

**Slot 0** (SUBJECT): Building `logrotate-lite` — bash CLI tool for log rotation
- Prompt 1: Initial script + test harness ✓
- Follow-up 1: Config file parsing ✓
- Follow-up 2: (pending) JSON report generation

**Slot 1** (SUBJECT): Building Python data analysis pipeline
- Initial prompt: (pending — will start after Slot 0 reaches 50K tokens)

**Orchestration:** Manual, via opencode CLI with `-s` session names

---

## Current State (Multi-Slot Active)

| Metric | Slot 0 (A: logrotate-lite) | Slot 1 (B: data pipeline) |
|--------|---------------------------|--------------------------|
| Tokens | 156,216 (59%) | 70,278 (27%) |
| Checkpoints | 22/32 | 32/32 (full) |
| Server RSS | 63.1 GB (both idle, cache loaded) | |
| Prompt cache | 3 entries, 14 GB (cold conversations) | |

**Memory jump:** +20 GB from single-slot baseline (43.1 → 63.1 GB). Breakdown:
- KV faulted: +5.9 GB (226K active tokens × 26B)
- Prompt cache: +13.5 GB (3 cold conversations, checkpoint serialization dominates)
- Overhead: ~16 GB (CUDA, macOS page alignment)

---

## Key Discoveries

### 1. Slot Count ≠ Max Conversations — It's Max Parallel Decoders

**Slots = concurrent decoding threads.** The prompt cache is the idle store. With 2 slots, you can have 2 conversations generating simultaneously; the rest queue in the prompt cache. The real throughput question is: **how many slots maximize parallel generation without tanking prompt eval time, decode speed, or system memory?**

**Proposed follow-up test:** Characterize 2, 4, and 8 slot configs measuring:
- Prompt eval time (ms/token) with 1, 2, 4, 8 concurrent active conversations
- Decode speed (tokens/sec) at each concurrency level
- RSS/Mem at each configuration
- Page faulting behavior as slots activate

### 2. Per-Turn Checkpoint Creation Bug

`--checkpoint-every-n-tokens 8192` creates checkpoints at 8K intervals **within a single prefill**, but **also creates one at the end of every prefill regardless of interval**. With LCP matching (f_keep ~0.99), each turn only processes 500-2K delta tokens, yet still gets a checkpoint. Result: 32 checkpoints fill in ~32 turns (~50K tokens), not ~256K as intended.

**Impact:** 4.7 GB per slot burned on checkpoints that provide marginal value when LCP is high.

### 3. Checkpoints Serialize to Prompt Cache

When a slot is saved to the prompt cache, all checkpoints are serialized along with the KV data. From dense.log.bkup2 (8-slot session):

| Saved slot | Tokens | Checkpoints | Checkpoint cost | KV cost |
|------------|--------|-------------|-----------------|---------|
| 7 | 89,907 | 22 | 5,682 MiB | ~2,390 MiB |
| 3 | 102,112 | 26 | 6,584 MiB | ~2,694 MiB |
| 0 | 136,521 | 32 | 8,339 MiB | ~3,551 MiB |

Checkpoints account for **70-75% of prompt cache entry size** for large conversations.

### 4. Memory Budget (2 slots)

| Component | Cost | Notes |
|-----------|------|-------|
| Model weights (Q6_K) | ~24 GB | Fixed |
| KV buffer (524K cells) | ~10.6 GB mmap'd, ~2 GB faulted | Variable by active tokens |
| Compute buffers | ~10 GB | Fixed |
| Recurrent state | ~1.2 GB | Fixed |
| Checkpoints (slot 0) | ~3.1 GB (21 × 149 MiB) | Growing toward 4.7 GB |
| Prompt cache | 496 MiB | Minimal (only system prompt) |
| **Total RSS** | **~42.4 GB** | Observed |

---

## Next Steps

1. **Continue Slot 0** through remaining follow-ups (2-8) to grow context toward 100K tokens
2. **Start Slot 1** when Slot 0 reaches ~50K tokens (current state: idle, system prompt loaded)
3. **Memory snapshots** at:
   - Slot 0: 50K → snapshot
   - Slot 0: 100K → snapshot
   - Slot 1: 50K → snapshot (measure the multi-slot jump)
4. **Validate** whether memory scales as predicted: 42 GB → ~55-60 GB with 2 active slots
5. **Run memory snapshot script** after each milestone

---

## Expected vs Actual

| Phase | Expected RSS | Actual RSS | Delta | Notes |
|-------|-------------|------------|-------|-------|
| Baseline (1 slot, 130K tokens) | ~43 GB | 43.1 GB | ✓ | Single slot, flat |
| Slot 1 activated (14K tokens) | ~55-60 GB | 49.1 GB | +6.0 GB | Initial activation, B just started |
| Both growing (143K + 23K) | ~55-60 GB | **53.7 GB** | +10.6 GB | Both at 32/32 checkpoints |
| **Key finding** | | | | Jump is +10.6 GB, not +43 GB (8-slot data) |

**Smaller jump than 8-slot test:** 8-slot config had 2M total cells (52 GB mmap), 2-slot has 524K cells (13.6 GB mmap). With 2 slots at ~166K combined tokens, ~10 GB faulted in. Proportional to buffer size.

---

*Last updated: 2026-04-29*
*Test agent: Slot 0 (logrotate-lite), Slot 1 (data pipeline) — pending*
