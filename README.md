# localllm

Local compute engine for multi-agent software development. Runs the best local
models on Apple Silicon (M5 Max 128GB) entirely on-device — powers agent
orchestration (gastown.ai, gas city) and, since the Qwen3.8 upgrade, the
opencode assistant itself (mtp-vision server).

## Servers: `servers.jsonc` + `./serve`

One source of truth: **`servers.jsonc`** in this repo. The `./serve` launcher
reads only that file (`start` / `stop` / `status` [server...]).

**`~/.localllm` is data-only** — models, pids, logs, prompt cache, venvs.
Never put config there (the old `~/.localllm/models.jsonc` flow was removed
2026-08-21).

| Server | Backend | Model | Port | Status |
|--------|---------|-------|------|--------|
| `mtp-vision` | llama.cpp (`~/llama.cpp-turbo`, `mtp-vanilla`) | Qwen3.8-27B-UD-Q4_K_XL + mmproj-F16, MTP spec decode | 30082 | **Production** — the opencode assistant runs here |
| `dense` | llama.cpp | Qwen3.6-27B-UD-Q6_K_XL | 30080 | parked |
| `moe` | llama.cpp | Qwen3.6-35B-A3B-UD-Q6_K_XL | 30081 | parked — model not on disk |
| `mlx-dense` | MLX (`~/mlx-lm-turbo`) | Qwen3.6-27B-UD-MLX-4bit | 30090 | parked — MLX layer on standby |
| `mlx-moe` | MLX (`~/mlx-lm-turbo`) | Qwen3.6-35B-A3B-UD-MLX-4bit | 30083 | parked |
| `mlx-moe-8bit` | MLX (`~/mlx-lm-turbo`) | Qwen3.6-35B-A3B-MLX-8bit | 30084 | parked |
| `mlx-dense-mtp` | optiq (MLX + MTP) | Qwen3.6-27B-OptiQ-4bit | 30091 | parked (model not fully downloaded) |

Decision (2026-08-21): llama.cpp is the daily driver. The MLX layer migrates
to Qwen3.8 only if its use resumes.

## Daily use

```bash
./serve status                       # what's up
./serve start mtp-vision             # start one (or: ./serve start = all)
curl http://localhost:30082/health   # health

# NOTE: mtp-vision has no restart subcommand. Restart =
./serve stop mtp-vision && ./serve start mtp-vision
# — restarting it kills the opencode session running on it. Do it at session end.
```

## Configuration

- **`servers.jsonc`** — all server definitions (model, port, args). For
  `mtp-vision`, sampling is set explicitly to the official Qwen3.8
  thinking-mode defaults (see `AGENTS.md` → Sampling Parameters; llama.cpp's
  silent defaults silently differ).
- **MLX KV-quant knobs** — when the MLX layer runs, they are CLI flags on the
  `mlx-*` server entries (`--turbo-kv-bits`, `--kv-cache-quantization`,
  ...). Reference config variants: `configs/{turboQuant,mixedQuant,vanilla}/`.
- **opencode client config** — `~/.config/opencode/opencode.jsonc`; example in
  `opencode.jsonc.example` (MLX providers only; needed only if the MLX layer
  runs).

## Benchmarking

See `.opencode/skills/benchmark/SKILL.md`. The benchmark script lives in the
sibling repo `~/mlx-lm-turbo/benchmarks/server_benchmark.py` and works against
any OpenAI-compatible endpoint — run it with the MLX venv python
(`~/mlx-lm-turbo/venv/bin/python3`) for its deps. Results live in
`benchmarks/`.

## Tools

- `./snapshot-memory.sh <phase-label>` — memory + health snapshot
  (llama.cpp and MLX processes, system memory, server ports, prompt cache);
  appends to `~/.localllm/memory-snapshots.log`. RSS under-reports Metal
  memory — treat as a floor.
- `memory-budget.py` — KV/memory budget calculator for the parked MLX models
  (planning tool for if the MLX layer is reactivated).
- `tests/` + `make lint` / `make test` / `make check` — quality gates
  (`check` verifies the llama-server binary, `servers.jsonc`, and every
  configured model file).

## Historical docs

Analysis docs from the llama.cpp evaluation and MLX characterization eras are
kept with "Historical note" headers: `KV-CACHE-ANALYSIS.md`,
`MULTI-SLOT-FINDINGS.md`, `MLX-CACHE-OOM-FINDINGS.md`,
`MLX-CHARACTERIZATION-PLAN.md`, `TEST-2SLOT-REPORT.md`,
`SINGLE-SLOT-CACHE-TEST.md`, `ARCHITECTURE-C.md`, `MEMORY-FORMULA.md`,
`DUAL-MODEL-PROPOSAL.md`. They describe prior eras, not current operations.
