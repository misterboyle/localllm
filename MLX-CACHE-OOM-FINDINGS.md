# MLX Cache OOM Findings

## Status: RESOLVED ✅

The fix has been implemented in `start-server.sh` and `models.jsonc.example`.

## What Was Fixed

Added `--prompt-cache-bytes` flag to the server launch command, enforcing a hard memory ceiling on the prompt cache.

### Before (vulnerable to OOM)
```
--prompt-cache-size 20
--prompt-cache-disk-size 100
```
No byte cap — a single 70K-token cache entry could consume several GB in RAM and cause OOM.

### After (bounded memory)
```
--prompt-cache-size 20
--prompt-cache-disk-size 100
--prompt-cache-bytes 8589934592   # 8 GB hard cap on 48 GB machines
```

## How It Works

The `LRUPromptCache` class tracks total bytes via `_n_bytes` and evicts entries (LRU order) when:
1. Entry count exceeds `--prompt-cache-size` (existing behavior)
2. Total bytes exceed `--prompt-cache-bytes` (new behavior)

After each response, the server calls `prompt_cache.trim_to(n_bytes=total - active)` to enforce the cap. Evicted entries are also deleted from disk if using `DiskBackedPromptCache`.

## Memory Budget — M4 Max 48GB

| Component | Size | Notes |
|-----------|------|-------|
| MoE weights (35B-A3B, 4-bit) | ~12-15 GB | All experts loaded, 3B active per step |
| Compute buffers | ~3-5 GB | Activation buffers, temp allocations |
| macOS system + display | ~6-8 GB | Desktop, background processes, GPU display surface |
| **Prompt cache budget** | **8 GB** | Enforced by `--prompt-cache-bytes` |
| **KV cache (active sessions)** | **~4-8 GB** | Variable, depends on concurrent usage |
| **Headroom** | **~2-4 GB** | Safety margin — don't squeeze this to zero |

### Tuning Guide

| Machine | RAM | Recommended `promptCacheBytes` | Concurrency |
|---------|-----|-------------------------------|-------------|
| M4 Max | 48 GB | 8 GB (`8589934592`) | 24/4/1024 |
| M5 Max | 96 GB | 24 GB (`25769803776`) | 32/8/2048 |
| M5 Max | 128 GB | 32 GB (`34359738368`) | 32/8/2048 |

To override, set in `models.jsonc` under `defaults` or per-server:

```json
"defaults": {
  "promptCacheBytes": 25769803776
}
```

To adjust for 48 GB machines, edit `models.jsonc`:
```json
"defaults": {
  "promptCacheBytes": 8589934592
}
```

Or per-server:
```json
"moe": {
  "promptCacheBytes": 8589934592
}
```

## Additional Optimizations for 48GB

- **decodeConcurrency**: 32 → 24 (fewer concurrent slots = less KV cache pressure)
- **prefillConcurrency**: 8 → 4 (prefill is memory-intensive)
- **prefillStepSize**: 2048 → 1024 (smaller steps = lower peak memory during prefill)
- **kvQuant**: [8, 2] (K8+V2) — ~20% memory savings, no quality loss
