# MLX-LM Gastown Architecture Evaluation

> **Date:** 2026-05-01
> **Goal:** Determine which architecture can run gastown locally on M5 Max 128GB
> **Server:** mlx-lm-turbo fork (`misterboyle/mlx-lm`), branch `feature/turboquant-kv-cache`
> **Hardware:** M5 Max, 128 GB unified memory

---

## Step 0: Local LLM Server Options

Before evaluating architectures, we need to pick the right server. Here's the landscape for Mac/M5 Max:

| Server | Platform | Multi-model | KV cache control | Customization | Agent-friendly |
|--------|----------|-------------|-----------------|---------------|----------------|
| **llama.cpp** | Cross-platform | ✅ | ✅ (fine-grained) | ✅ (C++ source) | ✅ (OpenAI API) |
| **MLX (mlx-lm)** | **Apple Silicon only** | ✅ | ✅ (TurboQuant, disk cache) | ✅ (Python source) | ✅ (OpenAI API) |
| **Ollama** | Cross-platform | ✅ | ❌ (black box) | ❌ (no source access) | ⚠️ (limited API) |
| **llamafile** | Cross-platform | ❌ (single model/binary) | ✅ (llama.cpp internals) | ⚠️ (compile-time) | ✅ (OpenAI API) |
| **vLLM** | Linux/AMD GPU | ✅ | ✅ | ✅ (Python) | ❌ (no Mac) |
| **SGLang** | Linux | ✅ | ✅ | ✅ (Python) | ❌ (no Mac) |
| **TensorRT-LLM** | NVIDIA only | ✅ | ✅ | ✅ | ❌ (no Mac) |

### Eliminated

- **vLLM, SGLang, TensorRT-LLM** — Linux/NVIDIA only, don't run on Apple Silicon
- **Ollama** — black box, no KV cache control, no TurboQuant, limited API, can't customize cache behavior
- **llamafile** — single model per binary, compile-time config, same slot/checkpoint problems as llama.cpp

### llama.cpp: Why It Failed Us

llama.cpp was attractive for easy setup and cross-platform support. But once we hit real gastown workloads, the architecture became a liability:

1. **Slot-based concurrency** — fixed number of slots, each with reserved KV buffer. No dynamic sharing.
2. **Checkpoint invalidation** — per-turn checkpoint creation fills the FIFO ring, cross-conversation checkpoints always mismatch, full reprocessing storms.
3. **LCP matching** — slot migration is unpredictable, causes cache thrashing and checkpoint erasure.
4. **mmap page faulting** — memory usage is unpredictable (2 GB → 52 GB depending on which pages are touched).
5. **No disk cache** — prompt cache is RAM-only, limited by `--cache-ram`.
6. **No TurboQuant** — KV cache quantization is limited to Q8_0/Turbo4, no polar quantization.
7. **C++ codebase** — hard to customize, hard to debug, hard to contribute fixes.

We spent weeks characterizing and working around these problems (KV-CACHE-ANALYSIS.md, MULTI-SLOT-FINDINGS.md). The fixes were incremental and never fully resolved the core issues.

### MLX: Why It's the Right Fit

1. **Native Apple Silicon** — MLX is designed for unified memory, no mmap page faulting, GPU-managed memory.
2. **No slots** — conversations are independent cache entries, no migration, no LCP matching, no checkpoint invalidation.
3. **TurboQuant** — polar quantization for KV cache (3-bit: 4.6x compression, negligible quality loss for V).
4. **Disk cache** — persist prompt caches to disk, survive restarts, RAM-efficient.
5. **Python source** — easy to customize, debug, and contribute. We already have a working fork.
6. **OpenAI-compatible API** — drop-in replacement for llama.cpp server, opencode works unchanged.
7. **Batched decode** — `decode_concurrency` queue handles concurrent requests without slot reservation.

**Tradeoff:** Mac-only. But gastown is on Mac, so this is a feature, not a bug.

### Conclusion

MLX is the only option that gives us:
- Native Apple Silicon performance
- Fine-grained KV cache control (TurboQuant, disk cache)
- No slot/checkpoint/LCP problems
- Python source for customization
- OpenAI-compatible API for agent integration

The remaining question is **which architecture** (27B only, MoE only, or both) works best on MLX. That's what the tests below answer.

---

## Architecture Ranking

Desired architectures, in order:

| Rank | Architecture | Rationale |
|------|-------------|-----------|
| **(1)** | **27B only** | Simplest, best quality, single process |
| **(2)** | **MoE only** | Fastest decode (3B active), lower memory |
| **(3)** | **27B + MoE** | Best of both, but GPU contention + complexity |

**Decision tree:**
1. Can 27B decode fast enough at gastown concurrency? → **(1) wins**
2. If not, is MoE smart enough for agent tasks? → **(2) wins**
3. If not, can 27B + MoE coexist without OOM? → **(3) or fallback**

---

## Test 1: 27B Decode Speed at Gastown Concurrency

**Question:** Can a single 27B process handle 3-4 concurrent conversations at acceptable speed?

**Setup:**
```bash
mlx_lm.server --model unsloth/Qwen3.6-27B-UD-MLX-4bit \
  --kv-cache-quantization 8,4 \
  --decode-concurrency 32 --prompt-concurrency 8 \
  --prompt-cache-size 10 --prompt-cache-bytes 17179869184
```

**Procedure:**
1. Start 1 conversation, measure decode speed (tokens/sec)
2. Start 2 concurrent conversations, measure per-conversation speed
3. Scale to 3, 4, 6 concurrent conversations
4. At each step, record: per-convo tokens/sec, aggregate throughput, peak memory

**Acceptance criteria for (1):**
- ≥ 10 tokens/sec per conversation at 3 concurrent
- ≥ 5 tokens/sec per conversation at 4 concurrent
- Peak memory < 100 GB (leave headroom for macOS)

**Kill criteria:**
- < 5 tokens/sec at 3 concurrent → 27B is too slow, move to Test 2

---

## Test 2: MoE Quality for Agent Tasks

**Question:** Is MoE smart enough for opencode workflows (reasoning, debugging, code generation)?

**Setup:**
```bash
mlx_lm.server --model unsloth/Qwen3.6-35B-A3B-UD-MLX-4bit \
  --kv-cache-quantization 8,4 \
  --decode-concurrency 32 --prompt-concurrency 8
```

**Procedure:**
Run the same agent tasks through MoE and 27B, compare outputs:
1. **Code generation:** "Build a Python CLI tool for X"
2. **Debugging:** "Why does this code fail? [paste code]"
3. **Reasoning:** "Explain the architecture of Y and suggest improvements"
4. **Multi-step:** Full opencode session (file reads, edits, test runs)

**Acceptance criteria for (2):**
- MoE output quality ≥ 90% of 27B (subjective, but clear)
- No systematic failures on reasoning or debugging tasks
- Decode speed ≥ 2x faster than 27B (the tradeoff must be worth it)

**Kill criteria:**
- MoE consistently produces lower quality on reasoning/debugging → move to Test 3

---

## Test 3: 27B + MoE Coexistence

**Question:** Can both processes run simultaneously without OOM or severe contention?

**Setup:**
```bash
# Process 1: 27B (agent tasks)
mlx_lm.server --model unsloth/Qwen3.6-27B-UD-MLX-4bit \
  --kv-cache-quantization 8,4 --port 8000 &

# Process 2: MoE (worker tasks)
mlx_lm.server --model unsloth/Qwen3.6-35B-A3B-UD-MLX-4bit \
  --kv-cache-quantization 8,4 --port 8001 &
```

**Procedure:**
1. Start both processes, record baseline memory
2. Run 1 conversation on each simultaneously
3. Scale to 2 conversations per process
4. Record: peak memory, decode speed per process, GPU contention

**Memory budget (estimated):**
| Component | 27B process | MoE process | Total |
|-----------|------------|-------------|-------|
| Model weights (4-bit) | ~14 GB | ~20 GB | 34 GB |
| Compute buffers | ~8 GB | ~8 GB | 16 GB |
| KV cache (2 convos each) | ~4 GB | ~4 GB | 8 GB |
| **Total** | **~26 GB** | **~32 GB** | **~58 GB** |

**Acceptance criteria for (3):**
- Combined peak memory < 100 GB
- Each process maintains ≥ 80% of single-process decode speed
- No OOM or GPU memory errors

**Kill criteria:**
- Combined memory > 110 GB → too tight for 128 GB system
- Decode speed drops > 50% with both running → GPU contention too severe

---

## Supporting Tests (as needed)

### Disk Cache + Many Conversations
If architecture (1) or (2) wins, verify one process can handle 8+ idle conversations:
- Enable `--prompt-cache-dir /tmp/kv_cache`
- Run 8 conversations to ~50K tokens each
- Let all go idle, measure RAM usage
- Resume each, measure cache hit rate and restore time

### Compaction Behavior
If cache misses are frequent, test with opencode compaction disabled:
- `"compaction": {"auto": false, "prune": false}`
- Compare cache hit rate and reprocess frequency

---

## Server Configurations

```bash
# 27B (Test 1)
mlx_lm.server --model unsloth/Qwen3.6-27B-UD-MLX-4bit \
  --kv-cache-quantization 8,4 \
  --decode-concurrency 32 --prompt-concurrency 8 \
  --prompt-cache-size 10 --prompt-cache-bytes 17179869184

# MoE (Test 2)
mlx_lm.server --model unsloth/Qwen3.6-35B-A3B-UD-MLX-4bit \
  --kv-cache-quantization 8,4 \
  --decode-concurrency 32 --prompt-concurrency 8

# Both (Test 3)
mlx_lm.server --model unsloth/Qwen3.6-27B-UD-MLX-4bit \
  --kv-cache-quantization 8,4 --port 8000 &
mlx_lm.server --model unsloth/Qwen3.6-35B-A3B-UD-MLX-4bit \
  --kv-cache-quantization 8,4 --port 8001 &
```

---

## Memory Snapshot

```bash
#!/bin/bash
# snapshot-mlx-memory.sh
echo "=== MLX Memory Snapshot ==="
echo "Date: $(date)"
for pid in $(pgrep -f mlx_lm.server); do
  port=$(lsof -Pan -p $pid -i4TCP 2>/dev/null | grep LISTEN | awk '{print $9}' | cut -d: -f2)
  echo "PID $pid (port $port):"
  echo "  RSS: $(ps -o rss= -p $pid | awk '{printf "%.1f GB", $1/1024}')"
  echo "  VIRT: $(ps -o vsize= -p $pid | awk '{printf "%.1f GB", $1/1024/1024}')"
done
```

---

## Expected Deliverables

- `MLX-GASTOWN-REPORT.md` — which architecture wins, with data
- Updated `start-server.sh` with winning config
- Memory snapshot script

---

*Last updated: 2026-05-01*
*Based on: mlx-lm-turbo fork, branch feature/turboquant-kv-cache*
*Reference: KV-CACHE-ANALYSIS.md, MULTI-SLOT-FINDINGS.md*
