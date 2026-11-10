# Fine-Tuning on This Box — What We Can Actually Do

Status: **analysis, nothing trained yet** (2026-11-10). Cross-project reasoning and the
decision law live in the personal wiki (`~/wiki/wiki/concepts/when-to-fine-tune.md`);
this page is the project record: our files, our measurements, our run book.

Every number below is either measured on this machine (command shown) or a labelled
estimate with its source. Nothing here is a claim that a run happened.

## 1. Box reality (measured 2026-11-10)

```
Apple M5 Max, 18 cores (6P+12E), 128 GB unified, 1.9 TB disk (947 GB free)
mlx-flash (mlx-serve 26.10.2, :30085) resident: phys_footprint 80 GB, peak 100 GB
PhysMem: 124G used (81G wired, 4G compressor), 3.6 GB unused
```

There is **no room for a training job while a server is up**, and the server hosts the
agent doing the work. Every measurement below is header/SQLite-only and did not need the
GPU. The first real training run therefore starts with `./serve stop mlx-flash`, which
ends the current opencode session — schedule it at a session boundary.

## 2. Toolchain census (our venv, verified 2026-11-10)

`venv/` has `mlx 0.31.2`, `mlx-lm 0.31.3`, `transformers 5.8.1`, `datasets 4.8.5`.
`mlx_lm.lora` supports `--fine-tune-type {lora,dora,full}`, `--optimizer
{adam,adamw,muon,sgd,adafactor}`, `--grad-checkpoint`, `--grad-accumulation-steps`,
`--mask-prompt`, `--max-seq-length`.

What it can and cannot load (checked against `mlx_lm/models/` and upstream `main`):

| Our model | `model_type` | mlx-lm training support |
| --- | --- | --- |
| `Qwen3.8-27B-bf16-MTP-MLX` | `qwen3_5` | **yes** (with the hazard in §3) |
| `Qwen3.8-Flash-Next` (daily driver) | `qwen4_exp` | **no** — no `qwen4_exp.py` in 0.31.3 and none in upstream main |
| small dense Qwen3 / `qwen3_5_moe` / `qwen3_next` | — | yes |

MoE training itself is supported in principle: `mlx_lm/tuner/lora.py` has
`LoRASwitchLinear` (per-expert low-rank adapters over `SwitchLinear`/
`QuantizedSwitchLinear`). The gap on Flash-Next is an architecture port, not a framework
limit. mlx-serve (Zig) and mlx-lm (Python) are separate codebases — engine support has
never implied training support.

Throughput dependency: mlx-lm PR **1389** "Add chunk-parallel gated delta ops for
training" is **closed and unmerged** (checked via the GitHub API). It is reported to be the
difference between ~50 t/s and 111-150 t/s on exactly our `qwen3_5` architecture
(`models/gated_delta.py`, plus lowering `SUB_BLOCK` 16→8). Our install has no `SUB_BLOCK`
symbol, i.e. the patch is not applied. PR **1661** (honor `seed=0` in `iterate_batches`)
merged upstream after the 0.31.3 release, so we lack that too. `MLX_DISABLE_COMPILE=1` is
mandatory for training (compile-cache shape retention is a measured OOM source).

## 3. Verified hazard: the 27B we hold is a poisoned training base

```
$ python3 tools/finetune-budget.py --census ~/.localllm/models/Qwen3.8-27B-bf16-MTP-MLX
  51.7 GiB in 1199 tensors … INLINE MTP BYTES: 0.79 GiB
```

15 tensors carry `mtp.` inside the main shards, and `mlx_lm/models/qwen3_5.py` does:

```python
has_mtp_weights = any("mtp." in k for k in weights)
should_shift_norm_weights = has_mtp_weights or has_unsanitized_conv1d
weights = {k: v for k, v in weights.items() if "mtp." not in k}
… for norm tensors: weights[k] = v + 1.0        # 161 backbone norm tensors here
```

Our file's final norm already averages **1.9441** — it is stored in the post-shift
convention (the bf16 source value is ~0.944), so mlx-lm would add 1.0 **again** to 161
tensors: a silently corrupted model, and the corruption would be baked into whatever the
adapter learns. mlx-serve is unaffected (different loader), so production is fine; the trap
is only on the training path. Fix before any run: build a training base with `mtp.*` moved
to a sidecar file the loader never reads, then assert the check prints zero:

```
count(k for k in index["weight_map"] if "mtp." in k)   # must be 0
mean(model.norm.weight)                                  # ~0.94 = unshifted, ~1.94 = shifted once
```

## 4. Memory budget (`python3 tools/finetune-budget.py`, seq 4096 batch 1)

`peak ≈ base_resident + 16 B/trainable + 2.5 × batch·seq·vocab·2 + reserve`

| target | rank / layers | trainable | state | logits×2.5 | base | ~peak | 128 GB | 512 GB |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen3-4B bf16 | 32 / 36 | 61.6 M | 0.9 G | 2.9 G | 8.0 G | 14.3 G | fit | fit |
| Qwen3-8B 4-bit | 32 / 36 | 87.3 M | 1.3 G | 2.9 G | 4.6 G | 11.3 G | fit | fit |
| 27B bf16 (ours) | 32 / 48 | 154.9 M | 2.3 G | 2.9 G | 51.7 G | 62.9 G | tight\* | fit |
| 27B 8-bit base | 128 / 48 | 619.7 M | 9.2 G | 2.9 G | 28.0 G | 42.6 G | fit\* | fit |
| Flash-Next 125B, attn-only | 32 / 48 | 40.3 M | 0.6 G | 4.7 G | 70.1 G | 81.4 G | tight\* | fit |
| Flash-Next 125B, +per-expert | 8 / 48×512 | 1 898 M | 28.3 G | 4.7 G | 70.1 G | 109.1 G | **NO** | fit |

\* "fit"/"tight" is capacity on an otherwise idle machine. With `mlx-flash` resident at
80 GB, every row above 20 GB is out, and the small ones would crawl.

Three things this table says out loud:

- **Base residency, not adapter size, is the binding constraint.** The adapter state of a
  rank-128 27B LoRA is 9 GB; the base is 28-52 GB.
- **The vocab term is our specific problem.** Flash-Next's 248,320-entry vocab costs
  4.7 GiB per batch-1 sequence at 4k and 9.5 GiB at 8k, before any activation. Fused
  cross-entropy (never materialise logits) is the largest single memory win available for
  agent data, and sequence packing is the largest throughput win (~3× on short-example
  instruction sets: IonDen/mlx-train-perf, measured on an M1 Max).
- **Full fine-tuning is not on the menu here.** The 27B full-FT plan in the field sits at
  324-432 GB — plausible on 512 GB only with 8-bit optimizer states or layerwise rounds
  (bounding backward to K layers at a time; the published 745B case OOMs at step 0 on
  512 GB with all layers in the graph, and runs at ~334 GB peak with K=8).

## 5. Our training corpus already exists (measured)

`~/.local/share/opencode/opencode.db` (45 GB):

```
533 sessions · 47,438 messages (44,640 assistant / 2,798 user) · 284,612 parts
45,707 tool calls: bash 32,282 · read 5,932 · edit 5,612 · write 725 · grep 445 ·
                   todowrite 235 · glob 203 · webfetch 101 · task 22 …
45,324 completed vs 377 error tool results
step-finish tokens: 123.7 M input, 32.9 M output · 99 compaction parts
```

`tools/export-trajectories.py` turns that into mlx-lm chat JSONL. Verified run:

```
$ python3 tools/export-trajectories.py --stats --limit-sessions 40
sessions 40  rows 7,227  ~20.04M est tokens  mean prefix 11,089 chars (~2,772 tok)
$ python3 tools/export-trajectories.py --limit-sessions 20 --out …/data
wrote train.jsonl (2,840 rows) / valid.jsonl (79 rows)
read-back: rows=2,919 not-ending-on-assistant=0 rows whose loss contains an OBSERVATION=0
```

Two construction rules are load-bearing, both forced by the reader (`mlx_lm/tuner/datasets.py`):

- `--mask-prompt` masks everything before the **last** message, so a whole session as one
  row trains on only its final turn → **one row per assistant step**, ending on the
  assistant message.
- Tool outputs must **not** sit in the assistant message, or the model learns to invent
  tool results. They become `OBSERVATION` turns in the *next* row's prefix. The read-back
  above asserts zero leakage.

Scaling to all 533 sessions: ~96k rows, ~266 M full-sequence tokens per epoch. The tool
accumulates rows in memory, so run the full export in `--limit-sessions` chunks (or point
`--cwd-filter` at one project) — a single all-sessions pass holds ~1 GB of strings.

## 6. Wall clock is the real ceiling

Masking does not reduce compute: every row re-pays its full prefix, and our mean prefix is
2,772 tokens against a few hundred loss-carrying tokens. At one epoch = 266 M tokens
(`finetune-budget.py`, same arithmetic):

| shape | full corpus | 5% curated (~13 M tok) |
| --- | --- | --- |
| 27B r32/16L, 113 t/s (measured elsewhere, unmerged patch applied) | 654 h | 33 h |
| 27B stock mlx-lm, ~50 t/s | 1,478 h | 74 h |
| 27B on M5 Ultra, ×3.5 compute (estimate) | 187 h | 9 h |

So: **the whole-corpus LoRA is not a laptop job at any rank.** The workable move is a
curated subset (5% ≈ 5k steps — still 10× the "500 good examples" floor the field uses) or
a small base, and the M5 Ultra's contribution is cadence — turning a 33-hour run into a
9-hour one, i.e. a nightly rebuild per base-model upgrade rather than a weekend project.
The ×3.5 is an inference from Apple's own M5-vs-M3 Ultra benchmark (up to 4.6× on prefill
TTFT via Neural Accelerators, 1.6× on bandwidth-bound decode; training is a prefill-like
compute-bound workload). It is an estimate and the §7 calibration run exists to replace it.

## 7. Run book — the calibration that makes every number above ours

At a session boundary, box idle:

```bash
cd ~/localllm && ./serve stop mlx-flash
python3 tools/export-trajectories.py --limit-sessions 40 --out ~/finetune/data-pilot   # ~2 min
MLX_DISABLE_COMPILE=1 venv/bin/mlx_lm.lora --model mlx-community/Qwen3-4B-4bit \
  --train --data ~/finetune/data-pilot --mask-prompt --num-layers 16 --rank 16 \
  --batch-size 1 --max-seq-length 2048 --iters 100 --save-every 25 --adapter-path ~/finetune/a4b
```

Log per run: wall seconds/iters, `mx.get_peak_memory()` (see
`~/wiki/wiki/concepts/applesilicon-memory-monitoring.md` — RSS lies here), swap delta, and
whether the val loss turned upward (it turns upward at ~7 epochs on corpora this small;
**copy the step-50 checkpoint out on purpose**, the trainer's last-two rule deletes it).
Then the same 100 iterations on the 27B training base to see whether our bf16 folder is
even trainable at 2k/4k sequences without the unmerged patch — the open question the
source itself flags as unverified.

## 8. First real experiment, chosen deliberately

Not a behaviour LoRA. **Quant recovery** — tune the served quant's scales/biases (optionally
with small LoRA error-compensators) against the bf16 teacher we already hold on disk.
The tool is **already in the venv** and is upstream mlx-lm, not a fork:
`venv/bin/mlx_lm.dwq --help` (verified 2026-11-10) exposes `--model` (teacher),
`--quantized-model` (the existing student to improve), `--targets-only` + `--target-dir`
(dump the teacher's top-k logits to disk once so the teacher is never resident during
training), `--num-samples`, `--max-seq-length`, `--grad-checkpoint`, `--pipeline`,
`--mlx-path`. Two-phase means the peak is student-shaped, not teacher+student:

```bash
MLX_DISABLE_COMPILE=1 venv/bin/mlx_lm.dwq --model ~/.localllm/models/Qwen3.8-27B-bf16-MTP-MLX \
  --quantized-model <a 4-bit build of the same weights> --targets-only \
  --target-dir ~/finetune/dwq-targets --num-samples 145 --max-seq-length 512
# teacher exits, never comes back; then train the student against the dump
```

Caveat: DWQ loads through the same `sanitize()` path as the trainer, so the §3 hazard
applies to the teacher too — point `--model` at an mtp-stripped copy (or the unconverted
bf16 source), not at the file mlx-serve ships from.

Why this one first:

- "dataset" = ~145 calibration sequences, not a corpus.
- metric = KL / token-flip-rate against the teacher, which we already compute (see
  `benchmarks/` and the `turbo3` KV post-mortem: 165.6 perplexity at 27× f16 KV — we have
  been burned by quant quality and have the measurement reflex to prove a fix).
- bounded memory by layerwise rounds, so it is a Max-class job for the 27B and an
  Ultra-class job for Flash-Next.
- rule inherited from the field: never fuse adapters into a quantized base (fusing
  re-quantises and can erase the adapter); gate the merge with KL against base+adapter.

A behaviour LoRA on our own traces is the second experiment, after (a) the tool schema has
been frozen for a month — post-training against a moving harness collapses under tool
shift, and a good harness bolted on afterwards is worse than one present during training —
and (b) a gate exists that can reject a round: clean-set loss delta ≤ 0.010 plus a
general-capability probe, not "it felt better in the transcript".
