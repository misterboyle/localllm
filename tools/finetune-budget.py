#!/usr/bin/env python3
"""Fine-tuning feasibility arithmetic for this box: what fits, what it costs in hours.

Two modes:
  --census PATH      walk a local MLX model folder's safetensors headers and report
                     resident bytes per tensor class (reads headers only, no weights).
  (default)          print the LoRA/QLoRA memory budget table and the wall-clock
                     estimate for a given corpus size.

Memory model (stated so it can be argued with):
  peak ~= base_resident + adapter_state + 2.5 * logits + reserve
  * adapter_state = 16 bytes per trainable parameter (fp32 adapter + fp32 grad +
    two Adam moments), which is mlx-lm's default trainer shape.
  * logits = batch * seq * vocab * 2 bytes, materialised once forward and read once
    backward. With a 150k-250k vocab this term, not the adapter, dominates long-seq
    training unless the loss is fused (never build logits).
  * reserve is a flat activation/allocator allowance, pinned to the one measured
    point available: 27B r32 over 16 of 64 layers ran at 31.7 GB active on a 28 GiB
    8-bit base (M3 Ultra 96 GB, leanzero, 2026-09).

Wall clock is dominated by the FULL sequence (prompt + completion), not by the
loss-carrying tokens: --mask-prompt saves gradient signal size, not compute.

Usage:
  python3 tools/finetune-budget.py --census ~/.localllm/models/Qwen3.8-27B-bf16-MTP-MLX
  python3 tools/finetune-budget.py --tokens 156e6 --tps 113
Stdlib only.
"""

import argparse
import collections
import json
import os
import struct

ITEMSIZE = {"F32": 4, "F64": 8, "F16": 2, "BF16": 2, "F8_E4M3": 1, "F8_E8M0": 1,
            "I8": 1, "U8": 1, "I32": 4, "I64": 8, "BOOL": 1, "F4": 0.5, "F6_E3M2": 0.75}

# label, base_GiB_resident, layers, vocab, hidden, mlp_inter, experts (0 = dense)
TARGETS = [
    ("Qwen3-4B dense bf16",        8.0,  36, 151_936, 2560,  9728, 0),
    ("Qwen3-8B dense 4bit",        4.6,  36, 151_936, 4096, 12288, 0),
    ("27B qwen3_5 bf16 (on disk)", 51.7, 48, 151_936, 5120, 17408, 0),
    ("27B 8-bit base",             28.0, 48, 151_936, 5120, 17408, 0),
    ("Flash-Next 125B MoE (disk)", 70.1, 48, 248_320, 2560,   640, 512),
]
BYTES_PER_TRAINABLE = 16


def census(path):
    path = os.path.expanduser(path)
    per_class, dt_bytes = collections.defaultdict(float), collections.defaultdict(float)
    n_tensors = 0
    for f in sorted(os.listdir(path)):
        if not f.endswith(".safetensors"):
            continue
        with open(os.path.join(path, f), "rb") as fh:
            hdr_len = struct.unpack("<Q", fh.read(8))[0]
            hdr = json.loads(fh.read(hdr_len))
        for k, v in hdr.items():
            if k == "__metadata__":
                continue
            n_tensors += 1
            nb = v["data_offsets"][1] - v["data_offsets"][0]
            dt_bytes[v["dtype"]] += nb
            kl = k.lower()
            if "mtp" in kl:
                cls = "MTP head"
            elif "experts" in kl:
                cls = "routed experts"
            elif "shared_expert" in kl:
                cls = "shared expert"
            elif "embed" in kl or "lm_head" in kl:
                cls = "embed/head"
            elif "vision" in kl or "visual" in kl:
                cls = "vision tower"
            elif "ngram" in kl or ".ple" in kl:
                cls = "n-gram PLE"
            else:
                cls = "attn/mlp projections"
            per_class[cls] += nb
    total = sum(per_class.values())
    g = lambda x: x / 2**30  # noqa: E731
    print(f"{path}\n  {g(total):8.1f} GiB in {n_tensors} tensors")
    for cls, b in sorted(per_class.items(), key=lambda kv: -kv[1]):
        print(f"    {cls:22s} {g(b):8.1f} GiB ({100*b/total:5.1f}%)")
    mtp = sum(v for k, v in per_class.items() if k == "MTP head")
    print(f"  dtypes: " + ", ".join(f"{k}={g(v):.1f}G" for k, v in dt_bytes.items()))
    print(f"  INLINE MTP BYTES: {g(mtp):.2f} GiB  -> mlx-lm sanitize() will +1.0 every "
          f"backbone norm when these are present in the shards (double-shift if the "
          f"conversion already applied it). Check the final-norm mean before training.")


def lora_params(hidden, inter, r, experts=0):
    """Trainable params of one layer with LoRA on q,k,v,o + mlp (and experts)."""
    attn = sum(r * (hidden + d) for d in (hidden, hidden // 4, hidden // 4, hidden))
    mlp = 2 * r * (hidden + inter) + r * (inter + hidden)
    if experts:
        mlp += experts * (2 * r * (hidden + inter) + r * (inter + hidden))
    return attn + mlp


def budget(seq, batch):
    g = lambda x: x / 2**30  # noqa: E731
    print(f"{'target':28s} {'r':>4s} {'layers':>7s} {'trainable':>10s} {'state':>7s} "
          f"{'logits':>7s} {'base':>7s} {'~peak':>7s} {'128G':>5s} {'512G':>5s}")
    for label, base, layers, vocab, hidden, inter, experts in TARGETS:
        shapes = [(16, min(16, layers), 0), (32, layers, 0), (128, layers, 0)]
        if experts:
            shapes.append((8, layers, experts))
        for r, l, exp in shapes:
            p = lora_params(hidden, inter if not exp else inter, r, exp) * l
            state = g(p * BYTES_PER_TRAINABLE)
            logits = g(batch * seq * vocab * 2)
            reserve = 6.0 if base > 40 else 2.5
            peak = base + state + 2.5 * logits + reserve
            f128 = "fit" if peak < 55 else ("tight" if peak < 95 else "NO")
            f512 = "fit" if peak < 380 else ("tight" if peak < 460 else "NO")
            tag = "exp" if exp else ""
            print(f"{label:28s} {r:4d} {str(l)+tag:>7s} {p/1e6:9.1f}M {state:6.1f}G "
                  f"{2.5*logits:6.1f}G {base:6.1f}G {peak:6.1f}G {f128:>5s} {f512:>5s}")
    print("\nlogits term alone (batch 1):")
    for vocab in (151_936, 248_320):
        for s in (2048, 4096, 8192):
            print(f"   vocab {vocab:7d} seq {s:5d}: fwd {g(s*vocab*2):5.2f} GiB, "
                  f"with backward ~{2.5*g(s*vocab*2):5.2f} GiB")


def clock(tokens, tps, note=""):
    print(f"   {tokens/1e6:8.1f}M tok at {tps:5.0f} t/s = {tokens/tps/3600:7.1f} h{note}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--census", default="", help="path to a local MLX model folder")
    ap.add_argument("--seq", type=int, default=4096)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--tokens", type=float, default=156e6,
                    help="full-sequence tokens per epoch (default: our measured corpus)")
    ap.add_argument("--tps", type=float, default=113, help="measured tokens/s for the shape")
    args = ap.parse_args()
    if args.census:
        census(args.census)
        return
    budget(args.seq, args.batch)
    print(f"\nwall clock at seq {args.seq} (full-sequence cost; masking does not reduce it):")
    for frac, why in ((1.0, "full corpus"), (0.05, "5% curated subset")):
        for t, label in ((args.tps, "measured 27B r32/16L @113 t/s"),
                         (50, "stock mlx-lm, no chunk-parallel gated-delta patch"),
                         (args.tps * 3.5, "M5 Ultra, prefill-like x3.5 (estimate)")):
            clock(args.tokens * frac, t, f"  [{label}: {why}]")


if __name__ == "__main__":
    main()
