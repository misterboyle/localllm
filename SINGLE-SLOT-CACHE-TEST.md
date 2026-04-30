# Single-Slot + Cache Test Plan

**Date:** 2026-04-29
**Goal:** Characterize cache behavior with 4 conversations competing for 1 slot

---

## Test Config

- `DENSE_SLOTS=1`
- `DENSE_PER_SLOT_CONTEXT=262144`
- `DENSE_CACHE_RAM=65536` (64 GB)
- `--ctx-checkpoints 32`
- opencode compaction disabled (`"auto": false`)

---

## Test Steps

1. Update `~/.config/opencode/opencode.jsonc` to disable compaction
2. Restart server with 1 slot config
3. Start 4 conversations (different workspaces, different topics)
4. Grow each conversation to ~50K tokens
5. Take memory snapshots at each milestone

---

## Metrics to Measure

| Metric | Expected (1 slot) | Notes |
|--------|-------------------|-------|
| RSS (1 active) | ~43 GB | Same as before |
| RSS (cache full) | ~43 GB + cache size | Cache is separate from RSS |
| LCP match | >0.99 for all | No compaction = no rollbacks |
| Decode speed | 14.5 tokens/sec | No contention |
| Cache entries | Up to 3 cold | 1 active + 3 cold |

---

## Hypotheses

1. **No n_past rollbacks** with compaction disabled → no full reprocessing
2. **Cache holds 3 cold conversations** (~15 GB) within 64 GB limit
3. **Memory stays flat** at ~43 GB + cache overhead (no multi-slot page faulting)
4. **LCP >0.99 for all conversations** (no compaction = full history preserved)

---

*Last updated: 2026-04-29*
