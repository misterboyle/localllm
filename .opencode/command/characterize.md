---
description: Characterize mtp-vision performance and memory (the daily driver)
model: opencode/kimi-k2.5
---

Run a full characterization of `mtp-vision` (Qwen3.8-27B + MTP spec decode,
llama.cpp, port 30082): memory, speed, and accept rate. This determines
whether the daily driver is holding up as the context lengths grow.

## Steps

1. **Memory**
   ```bash
   ./snapshot-memory.sh baseline
   ```
   Note: RSS under-reports Metal memory — use PhysMem via
   `top -l 1 -stats pid | grep PhysMem` for the process's real footprint.

2. **Decode speed at growing context** (concurrency 1)
   Read port/model from `servers.jsonc`. Use the context-length suite in
   `skills/benchmark/SKILL.md` (baseline / 10k / 50k / 100k), via
   `~/mlx-lm-turbo/benchmarks/server_benchmark.py` with
   `~/mlx-lm-turbo/venv/bin/python3`.

3. **MTP accept rate** — check the server log (`mtp-vision.log`) for the
   spec-decode accept stats during the bench runs; MTP is the main TG lever,
   so report acceptance alongside raw TG.

4. **Quality spot-check** — send 2–3 representative agentic prompts and
   verify output quality is acceptable (thinking mode on, reasoning_effort
   xhigh per servers.jsonc).

5. **Document findings** — save to `benchmarks/characterization-<date>.md`
   with: memory profile (RSS + PhysMem, KV estimate), TG/TTFT per context
   depth, MTP acceptance, any anomalies. Compare against the previous
   characterization.
