#!/usr/bin/env python3
"""Memory budget calculator for MLX models.

Reads model config to get architecture params, then computes KV cache costs
for various concurrency scenarios. Model definitions come from the default
config in models.jsonc.example — override with --models-dir if different.

Usage:
    python3 memory-budget.py                    # 27B (default)
    python3 memory-budget.py --model moe        # MoE 35B-A3B
    python3 memory-budget.py --model both       # both models side by side
    python3 memory-budget.py --ram 96           # adjust for your machine
    python3 memory-budget.py --active 2 --active-ctx 262144
"""

import argparse
import json
import os
import sys

# --- Config ---
MODELS_DIR = os.path.expanduser("~/.localllm/models")
SYSTEM_RAM_GB = 128  # override with --ram
MACOS_HEADROOM_GB = 15  # macOS needs this for IOMemoryDescriptor, compression, etc.

# Model definitions from default config (models.jsonc.example).
# If your config uses different models, point --models-dir accordingly.
MODEL_DEFS = {
    "27b": {
        "dir": "Qwen3.6-27B-UD-MLX-4bit",
        "label": "Qwen3.6-27B-UD (dense)",
    },
    "moe": {
        "dir": "Qwen3.6-35B-A3B-UD-MLX-4bit",
        "label": "Qwen3.6-35B-A3B (MoE, 3B active)",
    },
}


def load_model(name):
    """Load model architecture and compute memory costs."""
    md = MODEL_DEFS[name]
    model_dir = os.path.join(MODELS_DIR, md["dir"])
    config_path = os.path.join(model_dir, "config.json")

    with open(config_path) as f:
        cfg = json.load(f)

    tc = cfg.get("text_config", cfg)
    n_layers = tc["num_hidden_layers"]
    n_kv_heads = tc["num_key_value_heads"]
    head_dim = tc["head_dim"]
    layer_types = tc.get("layer_types", [])
    full_attn_count = layer_types.count("full_attention")
    linear_attn_count = layer_types.count("linear_attention")

    # Weight size
    weight_bytes = sum(
        os.path.getsize(os.path.join(model_dir, f))
        for f in os.listdir(model_dir)
        if f.startswith("model-") and f.endswith(".safetensors")
    )
    weight_gb = weight_bytes / 1e9

    # KV per-token costs
    fp16_per_token = 2 * n_kv_heads * head_dim * 2 * full_attn_count

    # K8,V2 quantized (MixedQuantKVCache, group_size=64)
    k8_per_head = head_dim // 4 * 4 + 2 * (head_dim // 64) * 2
    v2_per_head = head_dim // 16 * 4 + 2 * (head_dim // 64) * 2
    quant_per_token = (k8_per_head + v2_per_head) * n_kv_heads * full_attn_count

    # Fixed costs
    weights_ram_gb = weight_gb * 1.2
    compute_buf_gb = 8  # activation buffers
    overhead_gb = 5  # Python, MLX, tokenizer, queues
    fixed_gb = weights_ram_gb + compute_buf_gb + overhead_gb
    kv_budget_gb = SYSTEM_RAM_GB - MACOS_HEADROOM_GB - fixed_gb

    return {
        "name": name,
        "label": md["label"],
        "n_layers": n_layers,
        "full_attn": full_attn_count,
        "linear_attn": linear_attn_count,
        "n_kv_heads": n_kv_heads,
        "head_dim": head_dim,
        "weight_gb": weight_gb,
        "fp16_per_token": fp16_per_token,
        "quant_per_token": quant_per_token,
        "weights_ram_gb": weights_ram_gb,
        "fixed_gb": fixed_gb,
        "kv_budget_gb": kv_budget_gb,
    }


def scenario(m, name, active_n, active_ctx, cached_n=0, cached_ctx=0):
    a_kv = m["fp16_per_token"] * active_ctx * active_n / 1e9
    c_kv = m["quant_per_token"] * cached_ctx * cached_n / 1e9 if cached_n > 0 else 0
    total = m["fixed_gb"] + a_kv + c_kv
    ok = "OK" if total <= SYSTEM_RAM_GB - 3 else "OOM"
    margin = SYSTEM_RAM_GB - total
    return {
        "name": name,
        "active_kv": a_kv,
        "cached_kv": c_kv,
        "total": total,
        "margin": margin,
        "status": ok,
    }


def print_arch(m):
    print(f"=== {m['label']} ===")
    print(f"  Layers: {m['n_layers']} total ({m['full_attn']} full-attn, {m['linear_attn']} linear-attn)")
    print(f"  KV heads: {m['n_kv_heads']}, head dim: {m['head_dim']}")
    print(f"  Weights: {m['weight_gb']:.1f} GB (4-bit on disk)")
    print()
    print(f"  Active KV (fp16):  {m['fp16_per_token']:>8,} B = {m['fp16_per_token']/1024:>6.0f} KB/token")
    print(f"  Cached KV (K8,V2): {m['quant_per_token']:>8,} B = {m['quant_per_token']/1024:>6.1f} KB/token")
    print(f"  Compression:       {m['fp16_per_token']/m['quant_per_token']:.1f}x")
    print()
    print(f"  Fixed: {m['fixed_gb']:.0f} GB (weights {m['weights_ram_gb']:.0f} + compute 8 + overhead 5)")
    print(f"  KV budget: {m['kv_budget_gb']:.0f} GB (of {SYSTEM_RAM_GB} GB system)")
    print()


def print_table(rows):
    print(f"  {'Scenario':<45} {'Active':>7} {'Cached':>7} {'Total':>7} {'Margin':>7}  Status")
    print(f"  {'-'*45} {'-'*7} {'-'*7} {'-'*7} {'-'*7}  {'-'*6}")
    for r in rows:
        print(
            f"  {r['name']:<45} "
            f"{r['active_kv']:>6.0f}G {r['cached_kv']:>6.0f}G "
            f"{r['total']:>6.0f}G {r['margin']:>+6.0f}G  {r['status']}"
        )


def print_workload(m):
    print(f"=== Expected Workload ({m['label']}) ===")
    rows = [
        scenario(m, "Typical: 2L@150K + 2S@50K active, 8 cached",
                 4, 100000, 8, 75000),
        scenario(m, "Light: 2L@100K + 2S@30K active, 6 cached",
                 4, 65000, 6, 60000),
        scenario(m, "Heavy: 2L@200K + 4S@80K active, 4 cached",
                 6, 136000, 4, 100000),
        scenario(m, "Worst: 2L@262K + 4S@128K active",
                 6, 180000),
        scenario(m, "12 total: 4 active @ 100K, 8 cached @ 75K",
                 4, 100000, 8, 75000),
    ]
    print_table(rows)
    print()


def print_limits(m):
    print(f"=== Limits ({m['label']}) ===")
    max_active = int(m["kv_budget_gb"] / (m["fp16_per_token"] * 262144 / 1e9))
    print(f"  Max active @ 262K: {max_active} conversations")
    for n in [1, 2, 4, 8]:
        max_ctx = int(m["kv_budget_gb"] * 1e9 / (m["fp16_per_token"] * n))
        print(f"  Max context @ {n} active: {max_ctx:,} tokens ({max_ctx // 1000}K)")
    print()


def main():
    global SYSTEM_RAM_GB
    parser = argparse.ArgumentParser(description="MLX memory budget calculator")
    parser.add_argument("--model", choices=["27b", "moe", "both"], default="27b",
                        help="Model to analyze (default: 27b)")
    parser.add_argument("--ram", type=int, default=None,
                        help=f"System RAM in GB (default: {SYSTEM_RAM_GB})")
    parser.add_argument("--active", type=int, default=None, help="Active conversations")
    parser.add_argument("--active-ctx", type=int, default=None, help="Active context length")
    parser.add_argument("--cached", type=int, default=None, help="Cached conversations")
    parser.add_argument("--cached-ctx", type=int, default=None, help="Cached context length")
    args = parser.parse_args()
    if args.ram is not None:
        SYSTEM_RAM_GB = args.ram

    if args.model == "both":
        models = [load_model("27b"), load_model("moe")]
    else:
        models = [load_model(args.model)]

    for m in models:
        print_arch(m)

    if args.active is not None:
        for m in models:
            name = f"{args.active} active @ {args.active_ctx//1000}K"
            if args.cached:
                name += f" + {args.cached} cached @ {args.cached_ctx//1000}K"
            r = scenario(m, name, args.active, args.active_ctx, args.cached or 0, args.cached_ctx or 0)
            print(f"  {m['label']}: {r['total']:.0f} GB total, {r['margin']:+.0f} GB margin  {r['status']}")
        return

    for m in models:
        print_workload(m)
        print_limits(m)

    # Comparison summary
    if len(models) == 2:
        m27, mmoe = models
        print("=== Comparison ===")
        print(f"  {'Metric':<35} {'27B':>10} {'MoE':>10}  Ratio")
        print(f"  {'-'*35} {'-'*10} {'-'*10}  {'-'*6}")
        print(f"  {'KV/token active (fp16)':<35} {m27['fp16_per_token']:>10,} {mmoe['fp16_per_token']:>10,}  {m27['fp16_per_token']/mmoe['fp16_per_token']:.1f}x")
        print(f"  {'KV/token cached (K8,V2)':<35} {m27['quant_per_token']:>10,} {mmoe['quant_per_token']:>10,}  {m27['quant_per_token']/mmoe['quant_per_token']:.1f}x")
        print(f"  {'Fixed memory':<35} {m27['fixed_gb']:>10.0f}G {mmoe['fixed_gb']:>10.0f}G  {m27['fixed_gb']/mmoe['fixed_gb']:.1f}x")
        print(f"  {'KV budget':<35} {m27['kv_budget_gb']:>10.0f}G {mmoe['kv_budget_gb']:>10.0f}G  {mmoe['kv_budget_gb']/m27['kv_budget_gb']:.1f}x")
        print(f"  {'Max active @ 262K':<35} {int(m27['kv_budget_gb']/(m27['fp16_per_token']*262144/1e9)):>10} {int(mmoe['kv_budget_gb']/(mmoe['fp16_per_token']*262144/1e9)):>10}")


if __name__ == "__main__":
    main()
