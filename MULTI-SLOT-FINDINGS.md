# Multi-Slot Test Findings — Qwen3.6-27B Dense Server

> **Historical note:** These findings are from llama.cpp testing. The current MLX
> setup uses a different slot/cache architecture (no LCP matching, no checkpoints,
> different prompt cache behavior). Kept as reference for migration decisions.

**Date:** 2026-04-29
**Test config:** 2 slots, 262K per-slot, 524K total context, --kv-unified, --cache-ram 65536, --ctx-checkpoints 32
**Conversations:** A (logrotate-lite, conversational), B (data pipeline, file-heavy)

---

## 1. Memory Scaling

| Phase | RSS | Delta | Notes |
|-------|-----|-------|-------|
| Baseline (1 slot, 130K tokens) | 43.1 GB | — | Single slot, flat |
| Slot 1 activated (14K tokens) | 49.1 GB | +6.0 GB | Initial activation |
| Both growing (143K + 23K) | 53.7 GB | +10.6 GB | Both at 32/32 checkpoints |
| Both at peak (185K + 131K) | 51.3 GB | +8.2 GB | Some page reclaim, cache loaded |

**Memory formula validated:** `RSS = 35GB(fixed) + active_tokens×26B(KV) + cache(cold convos) + checkpoints(slots×count×149MB) + 16GB(overhead)`

---

## 2. Checkpoint Behavior

- **Per-turn checkpoint creation:** Each conversation turn creates 1 checkpoint at end of prefill (in addition to 8K-interval checkpoints during full prefill). 32-checkpoint FIFO ring fills in ~32 turns.
- **Checkpoint size:** Fixed at 149 MiB regardless of token count (captures full model state at that position)
- **Checkpoint ring eviction:** New checkpoints evict oldest (FIFO). When n_past rolls back beyond oldest checkpoint, no matching checkpoint → full reprocessing.
- **Checkpoint ↔ prompt cache:** Checkpoints are serialized to prompt cache on slot save. 70-75% of cache entry size is checkpoint data.

---

## 3. Prompt Cache Behavior

- **Cache holds cold conversations** when slot gets evicted. 64 GB soft cap, but appears to hold ~3 prompt entries max (regardless of size).
- **Cache thrashing observed:** 3 conversations competing for 2 slots → cache alternates between 2-3 entries (6.6 GB ↔ 14-15 GB).
- **No eviction by size:** Cache entries aren't evicted for size reasons (only 15 GB used of 64 GB cap). Eviction appears to be entry-count based.
- **Stale entries:** Cache can hold stale entries that don't match current conversation state, causing checkpoint mismatches on restore.

---

## 4. LCP Matching

- **High LCP (>0.99):** Conversational sessions (Conv A) maintain near-perfect matches turn-to-turn
- **Variable LCP (0.91-1.0):** File-heavy sessions (Conv B) show more variation. Lower matches correlate with turns that generated large file outputs.
- **f_keep < 1.0 causes reprocessing:** When f_keep is 0.754, only 75% of the conversation matches. The remaining 25% (23K tokens) gets reprocessed.
- **n_past rollback:** Conv B's n_past rolled back from ~131K to ~99K (32K gap), causing checkpoint mismatch and full reprocessing. Conv A (185K) had no rollbacks.

---

## 5. Full Reprocessing Events

- **1 documented event:** Conv B at ~70K tokens, n_past rolled back from 131K to 99K, no checkpoint matched (oldest at 109K) → 82K-token full reprocessing (151 seconds)
- **Root cause:** n_past rollback + checkpoint ring eviction. The per-turn checkpoint bug filled the 32-checkpoint ring with checkpoints at 109K-131K, evicting the checkpoint at 99K.
- **Conv A avoided this:** Higher LCP (>0.99), no n_past rollback, checkpoints matched consistently

---

## 6. Decode Speed Contention

- **Single slot:** 14.5 tokens/sec (Conv A), 58.6 tokens/sec (Conv B short prompt)
- **Both slots active:** 9.89 tokens/sec (Conv B) — 32% slower due to shared GPU compute
- **Both architectures (A and B) share this limitation:** Single GPU means shared compute regardless of process count

---

## 7. Slot Migration Behavior

- **LCP matching picks slot:** Each new request matches against both slots' cached prompts. Higher similarity wins.
- **LRU fallback:** When no LCP match, LRU picks the least recently used slot.
- **Migration cost:** Save to cache (3-4 GB for 130K conversation) → clear slot → restore on next request
- **No slot pinning:** Conversations can migrate between slots on any request

---

## 8. Architecture Implications

### Architecture A (27B + MoE, 2 processes)
- **Advantage:** MoE's 3B active params decode 5-9x faster than 27B for worker tasks
- **No cache thrashing:** Separate processes = separate caches
- **Peak memory:** ~133 GB (tight for 128 GB, need `--ctx-checkpoints 0`)

### Architecture B (27B × 2 slots, 1 process)
- **Advantage:** Simpler config, lower memory (~75 GB)
- **Disadvantage:** Cache thrashing, decode contention, migration overhead
- **Peak memory:** ~75 GB (comfortable with headroom)

### Architecture C (MoE, pinned slots, no cache)
- **Advantage:** No cache thrashing, no migration, predictable memory, fastest decode
- **Disadvantage:** No conversation migration, pinned slots can't be repurposed
- **Peak memory:** ~64 GB (most efficient)

### Single-slot + cache hypothesis
- 1 slot, large cache for 8+ conversations
- No page faulting contention, no decode speed penalty
- Cache handles all cold storage
- Worth testing for agent workflows where conversations take turns

---

## 9. Root Cause: opencode Compaction (KEY FINDING)

**opencode compaction is the root cause of Conv B's n_past rollbacks and full reprocessing.**

opencode's default compaction settings:
- `DEFAULT_TAIL_TURNS = 2` (keep 2 recent turns verbatim)
- `MAX_PRESERVE_RECENT_TOKENS = 8,000` (max tokens to preserve)
- `MIN_PRESERVE_RECENT_TOKENS = 2,000`
- `preserve_recent_budget = usable * 0.25` (25% of usable context, clamped to 2K-8K)

When Conv B's conversation grows beyond 8K preserved tokens, the next turn's n_past rolls back to the preserved boundary. The checkpoint ring has no checkpoint at that position (evicted by newer checkpoints) → full reprocessing.

**Conv A (conversational, smaller outputs):** Fits in 8K budget → LCP >0.99 → no rollback
**Conv B (file-heavy, large outputs):** Exceeds 8K budget → LCP 0.75-0.85 → n_past rollback → checkpoint mismatch

**Configurable via `~/.config/opencode/opencode.jsonc`:**
```json
{
  "compaction": {
    "auto": false,
    "prune": false
  }
}
```
Both `auto` (automatic compaction) and `prune` (trimming old tool outputs) need to be disabled. `prune` was likely the primary culprit — it trims tool outputs between turns, changing the conversation history and causing n_past rollbacks.

**Recommendation for local testing:** Disable compaction entirely (`"auto": false`) or increase `preserve_recent_tokens` to 16K+. This eliminates n_past rollbacks, checkpoint mismatches, and full reprocessing events.

---

## 10. Open Questions

1. **Cache entry count limit:** Is there a hard limit on prompt entries? (observed max 3)
2. **Cache eviction policy:** LRU? By size? By similarity?
3. **Single-slot + cache performance:** Does 1 slot + large cache outperform 2 slots for turn-taking workflows?

---

## 11. Recommendations

1. **Disable opencode compaction for local testing:** Set `"compaction": {"auto": false}` in `~/.config/opencode/opencode.jsonc`. This eliminates n_past rollbacks, checkpoint mismatches, and full reprocessing events.
2. **Test `--ctx-checkpoints 0`:** Eliminates 9.4 GB (2 slots) and prevents checkpoint ring eviction
3. **Test single-slot + cache:** 1 slot, large cache for 8+ conversations
4. **Consider pinned slots:** Eliminates migration, cache thrashing, and n_past rollback issues

---

*Last updated: 2026-04-29*
