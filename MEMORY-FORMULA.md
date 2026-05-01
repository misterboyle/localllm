# Memory Budget Formula — M4 Max 48GB

**Model:** Qwen3.6-35B-A3B (MoE, 4-bit quantized)
**Server:** mlx-lm-turbo (forked mlx_lm.server)
**KV Cache:** TurboQuant K8+V2 (~20% savings vs fp16)
**Prompt Cache:** Bounded by `--prompt-cache-bytes` (8 GB default on 48 GB machines)

---

## The Formula (MLX, MoE)

```
RSS = WEIGHTS + BUFFERS + KV_ACTIVE + CACHE_EVICTED + OVERHEAD
```

| Component | Cost | Variable? | Notes |
|-----------|------|-----------|-------|
| Weights (35B-A3B, 4-bit) | ~12 GB | No | All experts loaded, 3B active per step |
| Compute buffers | ~3-5 GB | No | Activation buffers, temp allocations |
| KV (active tokens) | ~26 bytes/token | Yes | Only for active sessions, quantized K8+V2 |
| Prompt cache (evicted) | ≤ 8 GB | Yes | Capped by `--prompt-cache-bytes` |
| Overhead | ~4-6 GB | No | Python, macOS memory management |

---

## 48GB Budget Breakdown

| Component | Size | % of 48GB |
|-----------|------|-----------|
| Weights + buffers | ~17 GB | 35% |
| macOS system + display | ~6 GB | 13% |
| Prompt cache cap | 8 GB | 17% |
| KV cache (active) | ~5-8 GB | 12-16% |
| Headroom | ~2-4 GB | 5-8% |
| **Total** | **~46 GB** | **~96%** |

---

## Tuning for Memory Pressure

If you see OOM or swap usage:

1. **Reduce `promptCacheBytes`** — e.g. from 8 GB to 4 GB
2. **Reduce `decodeConcurrency`** — e.g. from 24 to 16
3. **Reduce `promptCacheSize`** — e.g. from 20 to 12
4. **Disable the dense server** if enabled (saves ~28 GB)

If you have headroom and want more cache:

1. **Increase `promptCacheBytes`** — e.g. to 12 GB (only on machines with >48 GB RAM)
2. **Increase `decodeConcurrency`** — e.g. to 32

---

## Multi-Server Considerations

Running both dense (27B) and MoE (35B-A3B) simultaneously:

| Component | Dense (27B) | MoE (35B-A3B) | Total |
|-----------|-------------|---------------|-------|
| Weights | ~14 GB | ~12 GB | 26 GB |
| Buffers | ~4 GB | ~4 GB | 8 GB |
| Cache cap | 8 GB | 8 GB | 16 GB |
| **Total** | **~26 GB** | **~24 GB** | **50 GB** |

Running both servers on 48GB is tight — reduce cache caps to 6-8 GB each.

*Note: Both servers are disabled/enabled independently in `models.jsonc`.*

## Machine-Specific Defaults

| Machine | RAM | `promptCacheBytes` (each server) | Concurrency |
|---------|-----|----------------------------------|-------------|
| M4 Max | 48 GB | 8 GB (`8589934592`) | 24/4/1024 |
| M5 Max | 96 GB | 24 GB (`25769803776`) | 32/8/2048 |
| M5 Max | 128 GB | 32 GB (`34359738368`) | 32/8/2048 |

To override, set in `models.jsonc` under `defaults` or per-server:

```json
"defaults": {
  "promptCacheBytes": 34359738368
}
```
