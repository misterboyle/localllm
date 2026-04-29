# KV Cache Deep Dive: Qwen3.6-35B-A3B on M5 Max 128GB

## Overview

This documents the KV cache mechanics of llama.cpp's llama-server for the Qwen3.6-35B-A3B MoE model with 8 slots at 262K context, validated against actual production logs (moe.log, 12,932 lines).

**Data sources:**
- `moe.log` (12,932 lines) — full log with 7 documented full reprocessing events
- Qwen3.6-27B dense model — **not logged / not tested**, discussed as speculation below

---

## 1. KV Cache Quantization: K vs V (Not Model Weights)

llama.cpp stores KV **cache** (not model weights) in quantized form. The `-ctk` and `-ctv` flags control this independently:

- `-ctk q8_0` -- K-cache stored at Q8_0 quantization
- `-ctv turbo4` -- V-cache stored at Turbo4 quantization

### How KV Cache Works

During attention computation, for each token in the sequence:
- **K (Key)**: A 128-dim vector per KV-head per layer
- **V (Value)**: A 128-dim vector per KV-head per layer

These are cached across all tokens to avoid recomputation.

### Quantization Type Breakdown

| Type | Block Size | Bytes/Block | Bytes/Element | Compression vs F16 |
|---|---|---|---|---|
| **F16** | 1 | 2.0 | 2.0 | baseline |
| **Q8_0** | 32 | 34 | ~1.06 | ~47% of F16 |
| **Turbo4** | 128 | 68 | ~0.53 | ~26% of F16 |
| Turbo3 | 128 | 14 | ~0.11 | ~5% of F16 |
| Turbo2 | 128 | 10 | ~0.08 | ~4% of F16 |

### Your Configuration: Mixed K/V Quantization

Your setup: `-ctk q8_0 -ctv turbo4`

For Qwen3.6-35B-A3B (34 layers, ~32 KV heads, 128 head dim):

```
K per token = 34 × 128 × 1.06 B = ~4.61 KB
V per token = 34 × 128 × 0.53 B = ~2.31 KB
Total per token = ~6.92 KB

2M tokens × 6.92 KB = ~13.5 GB total KV cache
```

This is well within 128GB. The bottleneck was never total memory -- it was allocation strategy and checkpoint management.

---

## 2. Slot Architecture: Non-Unified vs Unified

### Non-Unified (current default)

Without `--kv-unified`, each slot gets its own ring buffer:

```
8 slots × 256K tokens × 6.92 KB = ~13.5 GB (fully reserved upfront)
```

Even if only 1 slot is active, all 8 buffers exist. This causes:
- No memory sharing between active and idle slots
- Decode failures when any single slot's ring buffer fills
- Forced eviction of idle slot KV data

### Unified (`--kv-unified`)

With `--kv-unified`, all slots share a single ring buffer:

```
2M tokens × 6.92 KB = ~13.5 GB (allocated once, shared)
```

Only active slots consume KV cache slots. This is the single biggest optimization.

---

## 3. Root Cause: The 7 Full Reprocessing Events

We found exactly **7 full prompt reprocessing events** in the 12,932-line moe.log. Each follows the same fatal pattern. Here's the breakdown:

### Event Timeline

| # | Slot | Task | n_past (resume point) | Checkpoint Range | Checkpoints Checked | Checkpoints Erased |
|---|------|------|----------------------|-----------------|---------------------|-------------------|
| 1 | 6 | 5254 | 18,909 | 48K-55K | ~30 | ~15 |
| 2 | 6 | 17753 | 10,595 | 16K-86K | ~20 | ~30+ |
| 3 | 6 | 60607 | 16,225 | 67K-113K | ~28 | ~15 |
| 4 | 2 | 85597 | 525 | 7K-7.5K | 2 | 2 |
| 5 | 1 | 95555 | 4,315 | 74K-110K | ~15 | ~18 |
| 6 | 1 | 102365 | (similar) | -- | -- | -- |
| 7 | 0 | 109507 | 175,291 | 181K-187K | ~18 | ~30+ |

### Event 1 -- Detailed Trace (task 5254, slot 6)

```
Line 1252: slot release: id 6 | task 5003 | stop processing: n_tokens = 56092
Line 1255: selected slot by LCP similarity, sim_best = 0.347, f_keep = 0.337
Line 1257: prompt_save:  - saving prompt with length 56092, total state size = 500.390 MiB
Line 1260: prompt 0x889774290:   56092 tokens, checkpoints: 32,  2510.413 MiB
Line 1264: new prompt, task.n_tokens = 54501
Line 1265: n_past = 18909, slot.prompt.tokens.size() = 56092
Lines 1266-1297: Checking checkpoint with [55839] against 18909...
                 Checking checkpoint with [55665] against 18909...
                 ... (all 30 checkpoints have pos_min > 18909)
Line 1298: FORCING FULL PROMPT RE-PROCESSING
Lines 1299-1309: ERASED invalidated context checkpoint (pos_min = 48476, size = 62.813 MiB)
                 ... (15 checkpoints erased, 945 MiB wiped)
```

### Event 2 -- Slot Token Accumulation (task 17753, slot 6)

```
Line 2490: new prompt, task.n_tokens = 17597
Line 2491: n_past = 10595, slot.prompt.tokens.size() = 87104  <-- 87K accumulated tokens
Lines 2492-2510: Checking checkpoint with [86916] against 10595...
                 Checking checkpoint with [16383] against 10595...
                 (all checkpoints have pos_min > 10595)
Line 2511: FORCING FULL PROMPT RE-PROCESSING
Lines 2512-2519: ERASED invalidated context checkpoint (pos_min = 16383, size = 62.813 MiB)
                 ... (all checkpoints erased)
```

### Event 5 -- Smallest Prompt, Still Reprocessed (task 85597, slot 2)

```
Line 6841: selected slot by LCP similarity, sim_best = 0.840, f_keep = 0.057
Line 6847: prompt 0x8887e8290:  135804 tokens, checkpoints: 26,  2755.371 MiB
Line 6853: new prompt, task.n_tokens = 625  <-- tiny prompt!
Line 6854: n_past = 525, slot.prompt.tokens.size() = 9263
Lines 6855-6857: Checking checkpoint with [7531] against 525...
                 Checking checkpoint with [7019] against 525...
Line 6857: FORCING FULL PROMPT RE-PROCESSING
```

Even a **625-token prompt** triggers full reprocessing because the 2 checkpoints at positions 7019 and 7531 are above n_past=525.

### Event 7 -- Massive Slot Accumulation (task 109507, slot 0)

```
Lines 11010-11026: Checking checkpoint with [187785] against 175291...
                 Checking checkpoint with [181115] against 175291...
                 (all 18 checkpoints have pos_min > 175291)
Line 11027: FORCING FULL PROMPT RE-PROCESSING
Lines 11028-11039+: ERASED invalidated context checkpoint (pos_min = 181115, size = 62.813 MiB)
                 ... (~30 checkpoints erased, ~1.9 GiB wiped)
```

---

## 4. The Fatal Pattern

Every reprocessing event follows this exact sequence:

```
1. Conversation A ends on slot X (e.g., 56K tokens processed)
2. Checkpoints created at positions: 10K, 18K, 26K, 34K, ... (every 8K tokens)
3. Slot X accumulates tokens across multiple conversations (56K → 87K → 106K → 187K)
4. New conversation arrives, assigned to slot X via LCP matching (same system prompt)
5. New conversation needs to resume at n_past = 10K-18K (its own context)
6. llama.cpp searches all 32 checkpoints for one with pos_min < 10K
7. ALL checkpoints have pos_min >> 10K (they were created for the accumulated conversations)
8. NO matching checkpoint found → full prompt reprocessing
9. ALL checkpoints erased (945 MiB - 1.9 GiB wiped per event)
```

### Why Checkpoints Always Fail

The checkpoints are position-dependent. They're created during prompt processing at controlled intervals (default: every 8192 tokens, via `--checkpoint-every-n-tokens` from PR #20087). When a slot accumulates tokens across multiple conversations (via LCP matching on system prompts), the checkpoint positions reflect the **accumulated** conversation, not the **new** conversation's resume point.

```
Slot accumulation over time:
  Conversation 1: processed tokens 0-18K, checkpoints at 10K, 18K
  Conversation 2: processed tokens 18K-36K, checkpoints at 26K, 34K
  Conversation 3: processed tokens 36K-56K, checkpoints at 48K, 55K
  New Conversation 4: needs resume at n_past=18K
    → Checkpoint search: all 32 checkpoints have pos_min > 18K → FAIL
```

The checkpoint at position 18K from conversation 1 would have matched, but it was already **erased** by the FIFO limit (32 max) when conversation 3 added more checkpoints.

---

## 5. Prompt Cache Overload

From the logs, the prompt cache was saturated:

```
Line 6845: total cache state: 4 prompts, 7028.599 MiB (limits: 8192.000 MiB)
Line 6867: total cache state: 5 prompts, 7610.077 MiB (limits: 8192.000 MiB)
```

| Prompt | Tokens | Checkpoints | Size |
|---|---|---|---|
| 0x889774290 | 56,092 | 32 | 2,510 MiB |
| 0x8887e8290 | 135,804 | 26 | 2,755 MiB |
| 0x889775c90 | 23,463 | 20 | 1,502 MiB |
| 0x8887e9310 | 9,263 | 2 | 261 MiB |
| **Total** | -- | -- | **7,028 MiB** |

**Each prompt slot carries ~62.8 MiB × 32 checkpoints = ~2 GiB of checkpoints.** The checkpoints dominate the cache memory (75-85% of each prompt entry). This is the core waste: checkpoints are position-bound to a single conversation, yet they consume massive amounts of RAM that could hold more useful prompt states.

---

## 6. Mitigation Strategy

### Fix 1: Control Checkpoint Frequency (`--checkpoint-every-n-tokens`)

**PR #20087** (merged upstream March 2026, present in our fork) added the `--checkpoint-every-n-tokens` flag.

**What it does:** Creates a checkpoint every N tokens during prefill. Upstream default: 8192 tokens. The value is **tokens**, not batches. At typical ~512-token batch sizes, 8192 means roughly one checkpoint every 16 batches.

**Flag name:** `--checkpoint-every-n-tokens` (short: `-cpent`). There is no `--checkpoint-every-nb` flag — that was a misunderstanding of the PR.

**How it works (code: `server-context.cpp:2692-2703`):**
```
For each batch during prefill:
  - Check if checkpoint_every_nt > 0
  - Find last checkpoint position
  - If (current_tokens - last_checkpoint) >= checkpoint_every_nt, create checkpoint
```

**Interaction with `--ctx-checkpoints`:** `--ctx-checkpoints` is the **maximum number** of checkpoints retained per slot (FIFO eviction). `--checkpoint-every-n-tokens` controls **how often** checkpoints are created. Both are needed: a high checkpoint count with a low token interval creates too many; a low count with a high interval may not capture enough.

**Why we moved from 0 to 32 checkpoints:** Our original analysis concluded `--ctx-checkpoints 0` was necessary because hybrid model checkpoints are position-dependent and break across conversations. However, checkpoints **do help within a single conversation** when a long prompt is split across multiple batches. The `--checkpoint-every-n-tokens` flag gives us control over the tradeoff:
- At 8192 (upstream default), a 56K-token prompt creates ~7 checkpoints
- A 262K-token prompt creates ~32 checkpoints (hits the `--ctx-checkpoints` cap)
- Within a single long conversation, these checkpoints allow resuming from the nearest saved point
- **Cross-conversation, they still break** — the fundamental problem from our analysis remains

### Fix 2: Increase Prompt Cache RAM (`--cache-ram 32768`)

**Why:** The 8192 MiB default is insufficient. The logs show the cache at 7-8 GiB (96-100% full) with only 4-5 prompt states. With 32 GiB, the cache holds many more prompt states, improving LCP matching hit rates.

**Effect:** More prompt states in cache → higher LCP matching hit rate → slots find cached KV data instead of reprocessing.

### Fix 3: Enable Unified KV Buffer (`--kv-unified`)

**Why:** Without unified mode, each slot reserves its own ring buffer even when idle. With 8 slots at 256K each, ~13.5 GB is reserved upfront.

**Effect:** Memory scales with active slots only. Eliminates decode failures from ring buffer overflow.

### Fix 4: Increase Cache Reuse Threshold (`--cache-reuse 128`)

**Why:** Default 64 tokens is too small. With the same system prompt across conversations, the LCP similarity matching already handles slot selection. Increasing the reuse threshold means more tokens can be matched and reused via KV shifting.

**Effect:** When a new prompt shares 128+ tokens with cached data, those tokens' KV cache is reused via shifting instead of full reprocessing.

---

## 7. Updated Configuration

```bash
MOE_PER_SLOT_CONTEXT=262144
MOE_SLOTS=8
MOE_CONTEXT=$((MOE_PER_SLOT_CONTEXT * MOE_SLOTS))

# Optimized llama-server launch flags:
--kv-unified                    # Single shared KV buffer
--cache-ram 32768               # 32GB prompt cache RAM
--ctx-checkpoints 32            # Max checkpoints per slot (upstream default)
--checkpoint-every-n-tokens 8192  # Checkpoint every 8192 tokens during prefill
--cache-reuse 128               # Reuse cached chunks >= 128 tokens
--cache-idle-slots              # Keep idle slots cached (default, explicit)
--slot-prompt-similarity 0.1    # Keep LCP matching enabled
--flash-attn on                 # Flash attention (required)
--jinja                         # Jinja templates
```

### Trade-off Analysis

| Change | Before | After | Impact |
|---|---|---|---|
| Checkpoints | 32, created at every ubatch boundary | 32, created every 8192 tokens | Same max, controlled spacing via `--checkpoint-every-n-tokens` |
| Cache RAM | 8192 MiB | 32768 MiB | 4x more prompt states stored |
| KV buffer | Per-slot reservation | Shared unified | No decode failures, memory scales with usage |
| Cache reuse | 64 tokens | 128 tokens | More KV shifting hits |

### Expected Behavior After Fix

1. **Controlled checkpoint creation** -- checkpoints created every 8192 tokens during prefill, up to 32 per slot
2. **Full prompt reprocessing events should decrease** -- with 32 GB cache RAM and unified KV, more slots retain useful cached state for LCP matching
3. **Prompt cache holds more states** -- 32 GB RAM can store many more prompt entries than 8 GB
4. **Intra-conversation resumption works** -- checkpoints at 8192-token intervals allow resuming within long conversations
5. **Cross-conversation erasure still occurs** -- hybrid model checkpoints remain position-dependent and will be erased when a new conversation with different resume point is assigned to the slot

### What Full Reprocessing Cost Looks Like

From the logs (task 2013, slot 6):
```
prompt eval time =    2243.13 ms /  2737 tokens
```

That's 2.2 seconds of pure MoE computation re-processing ~19K tokens through 34 layers. If this happens 7 times per session, that's ~15 seconds of wasted compute.

---

## 8. Log Analysis Guide

### After Applying Fix -- What You Should See

```
[MOE] slot get_availabl: id  6 | task -1 | selected slot by LCP similarity
[MOE] slot launch_slot_: id  6 | task 5254 | processing task
[MOE] slot update_slots: id  6 | task 5254 | prompt processing done, n_tokens = 18911
[MOE] slot print_timing: id  6 | task 5254 | prompt eval time = 1200.45 ms
```

Clean prompt processing. Checkpoint creation visible at 8192-token intervals during long prefill.

### Cross-Conversation Checkpoint Erasure (Expected)

When a new conversation is assigned to a slot with existing checkpoints from a different conversation:
```
[MOE] Checking checkpoint with [48476] against 18909...
[MOE] FORCING FULL PROMPT RE-PROCESSING
[MOE] ERASED invalidated context checkpoint (pos_min = 48476, size = 62.813 MiB)
```

This is **expected and unavoidable** for hybrid models. The checkpoints are position-dependent and won't match the new conversation's resume point. The `--checkpoint-every-n-tokens` flag does not change this behavior — it only controls checkpoint spacing.

### If Issues Persist -- Check For

```
[MOE] purging slot %d with %zu tokens     <- still happens if KV ring buffer full
[MOE] all slots are idle                   <- healthy, no forced evictions
```

The "purging slot" message indicates the KV ring buffer is still too small for peak concurrent usage. Fix: reduce `MOE_SLOTS` or increase `MOE_PER_SLOT_CONTEXT`.

---

## 9. Checkpoint Strategy: Enabled with Controlled Spacing

**Current stance: `--ctx-checkpoints 32 --checkpoint-every-n-tokens 8192`**

We originally recommended `--ctx-checkpoints 0` because hybrid model checkpoints are position-dependent and break across conversations. However, with the `--checkpoint-every-n-tokens` flag (PR #20087, March 2026), we have finer control:

1. **Intra-conversation value:** Checkpoints help when a single long conversation is interrupted and resumes. At 8192-token intervals, a 262K prompt gets ~32 checkpoints covering the full range.
2. **Cross-conversation problem remains:** When a new conversation is assigned to a slot via LCP matching, the slot's existing checkpoints are position-bound to the old conversation. They still won't match and will still be erased.
3. **Memory cost is manageable:** 62.8 MiB × 32 = ~2 GiB per slot is acceptable given the 32 GB cache RAM and unified KV buffer.
4. **Controlled creation:** `--checkpoint-every-n-tokens 8192` ensures checkpoints are spaced sensibly (not too frequent, not too sparse).

### When to disable (`--ctx-checkpoints 0`)

- If you're seeing checkpoint erasure storms in logs and cache RAM is tight
- If your workload is exclusively short conversations with high turnover (no long-running sessions)
- On the MoE server specifically, where hybrid memory + rapid slot turnover amplifies the problem

### When to keep enabled (`--ctx-checkpoints 32`)

- If you have long-running conversations that benefit from intra-prompt resumption
- If cache RAM is sufficient (32 GB with unified KV leaves plenty of headroom)
- On the dense server, where slot turnover may be lower

---

## 10. Architecture Notes: Qwen3.6-35B-A3B Hybrid Memory

The Qwen3.6-35B-A3B uses **hybrid memory** (recurrent delta-net + standard attention), which is why:

- `n_swa = 0` in all log entries (no sliding window attention)
- Checkpoints are needed because `ctx_seq_rm_type != FULL` for this model
- Hybrid memory doesn't support partial sequence removal reliably
- `pos_min` threshold checks fail because checkpoint positions don't align with conversation boundaries

This is fundamentally different from standard Transformer models where KV cache management is simpler.

---

## 11. Quick Reference

| Flag | Default | Optimized | Reason |
|---|---|---|---|
| `--kv-unified` | auto | **true** | Shared KV buffer |
| `--cache-ram` | 8192 | **32768** | 4x prompt cache capacity |
| `--ctx-checkpoints` | 32 | **32** | Keep intra-conversation checkpoints |
| `--checkpoint-every-n-tokens` | 8192 | **8192** | Controlled checkpoint spacing (PR #20087) |
| `--cache-reuse` | 0 | **128** | More KV shifting hits |
| `--cache-idle-slots` | true | **true** | Keep idle slots cached |
| `--slot-prompt-similarity` | 0.1 | 0.1 | LCP matching |
| `--flash-attn` | auto | **on** | Required |

### Checkpoint Flags Explained

- `--ctx-checkpoints N`: Maximum number of checkpoints retained per slot (FIFO eviction). Default: 32.
- `--checkpoint-every-n-tokens N`: Create a checkpoint every N tokens during prefill. Default: 8192. Set to -1 to disable.
- These are **tokens**, not batches. At ~512-token batch sizes, 8192 ≈ 16 batches.
- **Known issue:** Hybrid model checkpoints are position-dependent and break across conversations. Within a single conversation they work correctly. Cross-conversation, they will be erased when the new conversation's resume point doesn't match.

---

## 12. Dense 27B vs MoE 35B-A3B: Same Bug, Different Speed

### Architecture comparison

Both models use the **same hybrid Transformer** architecture with delta-net recurrent layers:

| | Dense 27B | MoE 35B-A3B |
|---|---|---|
| Source file | `qwen35.cpp` | `qwen35moe.cpp` |
| Memory init | `build_inp_mem_hybrid()` | `build_inp_mem_hybrid()` |
| Recurrent layers | `hparams.is_recurrent(il)` → `build_layer_attn_linear()` | Same |
| Standard attention | `build_layer_attn()` | Same |
| n_swa | hybrid model (no SWA in this model, but hybrid memory) | hybrid model (n_swa=0, but hybrid memory) |
| Checkpoint problem | **Same** | **Same** |
| Active params/step | 27B | 3B (of 35B total) |
| Prompt eval speed | Faster | Slower |

**Both models share the same bug.** The checkpoint failure is in `llama.cpp`'s hybrid memory handling (`llama_memory_seq_rm` doesn't work reliably on hybrid models), not in the model weights. Both models use `build_inp_mem_hybrid()` which triggers the same checkpoint creation path.

### The dense model *might* be less painful

The 27B dense model would face the same full reprocessing events, but:

1. **Faster prompt eval** — 27B standard attention is simpler than MoE FFN with expert routing. A full reprocessing event on the 27B would be noticeably faster than on the MoE.
2. **Easier to fit 8 slots** — Less memory pressure per slot, less chance of decode failures from ring buffer overflow.

**But it's untested.** We don't have log data from the 27B dense server to confirm whether it avoids the checkpoint issue. It won't — same hybrid architecture. It might just be *less painful* due to faster eval and lower memory pressure.

If you want to test this, the same optimized config (`--kv-unified --ctx-checkpoints 32 --checkpoint-every-n-tokens 8192 --cache-ram 32768`) should work for both. Watch for the same "forcing full prompt re-processing" message in the dense log.

---

*Last updated: 2026-04-29*
*Based on llama.cpp source at: ~/llama.cpp-turbo/*
*Validated against: moe.log (12,932 lines, 7 full reprocessing events documented)*
