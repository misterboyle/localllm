# Dual-Model Architecture Proposal

**Date:** 2026-04-29
**Context:** Characterizing optimal slot/cache configuration for agent workflows

---

## Current Understanding

From KV-CACHE-ANALYSIS.md and live testing:

- **Slots** = parallel decoding channels (not "max conversations")
- **Prompt cache** = cold storage for idle conversations
- With high LCP (>0.8), the prompt cache handles everything — checkpoints are redundant
- Page faulting is the dominant memory cost: ~43 GB RSS for 1 active slot, scales with concurrent active decoders

## The Observation

Agent workflows have a natural two-tier structure:

1. **Main reasoner** — reads context, plans, makes decisions (needs big context window, runs complex reasoning)
2. **Worker** — executes specific tasks (tool calls, code generation, analysis — often shorter, more focused prompts)

These two tiers don't run simultaneously. The reasoner thinks, then spawns the worker, then waits for the result. They take turns on the same inference channel.

## Proposal: 2 Models, 1 Slot Each

### Architecture A: 27B + MoE Hybrid

| Model | Slots | Per-slot ctx | Cache | Role |
|-------|-------|-------------|-------|------|
| 27B Dense | 1 slot | **128K** (default), 262K (on demand) | 8 conversations | Main reasoner (complex reasoning, big context) |
| 35B MoE A3B | 1 slot | **64K** (reduced) | 8 conversations | Worker (tool use, code, analysis) |

**Rationale:** The 27B has deeper reasoning for the main agent loop. The MoE is fast and cheap for worker tasks. The 8-cache on the MoE preserves all the worker conversations across turns, avoiding reprocessing when the worker re-enters.

**Reduced 27B context (128K default):** Live testing shows we generated substantial work (full codebase with configs, tests, reports) at only 44% of 262K. Most sessions won't accumulate 200K+ tokens. 128K provides ample headroom with:
- KV buffer halved (~5.3 GB mmap'd vs ~10.6 GB) → less page faulting, faster decode
- Can bump to 262K for deep analysis sessions by restarting the server

**Reduced MoE context (64K):** Worker conversations are focused and short — a tool call + result + response rarely exceeds 10-20K tokens. 64K provides 3-6x headroom. Benefits:
- Smaller KV buffer → less page faulting → faster decode (better memory locality)
- Less memory overhead for the MoE process
- Leaves more headroom for the 27B's large context windows

**Memory budget (rough):**
- 27B: ~43 GB RSS (1 slot, 102K tokens observed)
- MoE (64K ctx): ~40 GB RSS (1 slot, reduced buffer) + ~2 GB prompt cache (8 conversations × small)
- Total: **~85 GB** (fits in 128 GB with 43 GB headroom)

**Both generate simultaneously.** Independent `llama-server` processes on different ports (30080/30081). Same parallelism as Architecture B (2 slots in one process can also decode simultaneously). The advantage of Architecture A is **worker decode speed** — the MoE only activates 3B of 35B params per step, so worker responses should be faster than the 27B's 27B active params.

### Architecture B: 27B Only, 2 Slots

| Model | Slots | Cache | Role |
|-------|-------|-------|------|
| 27B Dense | 2 slots, 8 cache | Slot 0 = main reasoner, Slot 1 = worker |

**Rationale:** Single model, simpler config. Both slots share the same 8-slot prompt cache for conversation persistence. Worker uses slot 1, reasoner uses slot 0.

**Memory budget (rough):**
- 27B: ~43 GB RSS (1 slot active) → ~55 GB (2 slots active, page faulting)
- Prompt cache: ~5 GB (8 conversations)
- Total: ~60 GB (comfortable in 128 GB)

**Tradeoff:** No model specialization. Worker tasks get the same reasoning power as main agent (which is fine — the 27B is strong at both). But worker decode will be slower than the MoE equivalent. Simpler config (one process, one port). Lower memory cost (~60 GB vs ~85 GB).

---

## Benchmark Plan

Measure for both architectures:

1. **Prompt eval latency** (ms/token) with growing context (10K, 50K, 100K, 200K tokens)
2. **Decode speed** (tokens/sec) for short responses (worker) vs long responses (reasoner)
3. **Memory behavior** at each context size (RSS, page faulting)
4. **LCP hit rate** across turns — how often does cache save us from reprocessing?
5. **End-to-end agent turn time** — reasoner → worker → reasoner loop

### Architecture A Benchmark (27B + MoE)

```bash
# Start 27B on port 30080
DENSE_SLOTS=1 DENSE_PER_SLOT_CONTEXT=262144 ./start-server.sh

# Start MoE on port 30081
MOE_SLOTS=1 MOE_PER_SLOT_CONTEXT=262144 MOE_CACHE_RAM=32768 ./start-server.sh

# Test: reasoner (27B) grows to 100K tokens, measure latency
# Test: worker (MoE) runs 8 concurrent conversations, measure cache hit rate
# Test: switch between models, measure overhead
```

### Architecture B Benchmark (27B × 2)

```bash
# Start 27B with 2 slots
DENSE_SLOTS=2 DENSE_PER_SLOT_CONTEXT=262144 ./start-server.sh

# Test: slot 0 (reasoner) grows to 100K tokens
# Test: slot 1 (worker) processes requests simultaneously
# Test: measure page faulting jump (single → dual active)
# Test: compare to Architecture A latencies
```

---

## Key Questions to Answer

1. **Is the 27B's reasoning advantage worth the 43GB cost vs MoE's ~45GB?**
   - For agent workflows, reasoning quality matters more than raw speed
   - But the MoE's 3B active params may be *fast enough* for reasoning too

2. **Does the MoE's 8-cache actually help, or is 1 cache sufficient?**
   - If the worker always returns to the same conversation, 1 cache suffices
   - If the worker branches across multiple tool calls, 8 cache preserves them all

3. **What's the memory overhead of running 2 processes vs 1?**
   - Each process loads its own model weights (24 GB each)
   - 2 processes = 48 GB in weights vs 24 GB for single process
   - But with unified memory, duplicated weights may page-share at the OS level

---

*Last updated: 2026-04-29*
