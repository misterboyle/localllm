#!/usr/bin/env python3
"""
Growing-context benchmark: simulates N concurrent conversations that grow
incrementally (like real agent sessions), measuring how TTFT and TPS
degrade as context accumulates.

Realistic design: each channel has its OWN shared prefix (system prompt +
knowledge base), then grows with unique content. Different channels have
different prefixes, so cache hits happen within a channel but not across.

Example:
  Channel 1: prefix_A + tail_A1, prefix_A + tail_A1 + tail_A2, ...
  Channel 2: prefix_B + tail_B1, prefix_B + tail_B1 + tail_B2, ...
  Channel 3: prefix_C + tail_C1, ...
  Channel 4: prefix_D + tail_D1, ...

Cache hits: prefix_A reused within channel 1, prefix_B within channel 2, etc.
Cache misses: prefix_A ≠ prefix_B ≠ prefix_C ≠ prefix_D

Usage:
    python3 growing_context_bench.py --channels 4 --steps 10 --step-size 2000 --shared-size 5000
"""

import argparse
import asyncio
import json
import time
import random
import aiohttp


def generate_channel_prefix(channel_id, num_words):
    """Generate a unique shared prefix for each channel."""
    return " ".join([f"channel-{channel_id}-system-{i}" for i in range(num_words)])


def generate_unique_content(channel_id, step, num_words):
    """Generate unique content for a specific channel/step."""
    return " ".join([f"agent-{channel_id}-step-{step}-word-{i}" for i in range(num_words)])


async def run_channel(
    session: aiohttp.ClientSession,
    channel_id: int,
    url: str,
    api_key: str,
    model: str,
    step_size: int,
    num_steps: int,
    shared_size: int,
    step_event: asyncio.Event,
    round_barrier: asyncio.Event,
) -> dict:
    """Run one growing conversation channel with its own shared prefix.
    
    Channels grow in rounds: all channels do step 0, then all do step 1, etc.
    This keeps the cache warm within each channel across rounds.
    """
    history = []
    step_metrics = []
    shared_words = generate_channel_prefix(channel_id, shared_size).split()

    for step in range(num_steps):
        # Build prompt: channel's shared prefix + accumulated unique content + new step
        context_words = list(shared_words)  # fresh copy each time
        for h in history:
            context_words.extend(h["prompt_words"])
            context_words.extend(h["response_words"])

        # Add new step content
        new_words = generate_unique_content(channel_id, step, step_size).split()
        context_words.extend(new_words)

        prompt = f"Continue this conversation: {' '.join(context_words)} What is the next step?"
        prompt_tokens = len(context_words) + 5  # rough estimate

        # Wait for this round to start
        await step_event.wait()
        step_event.clear()

        start = time.perf_counter()
        ttft = None
        total_tokens = 0
        try:
            async with session.post(
                url,
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 128,
                    "temperature": 0.0,
                },
                headers={"Authorization": f"Bearer {api_key}"},
            ) as resp:
                data = await resp.json()
                ttft = time.perf_counter() - start
                content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
                total_tokens = len(content.split())
                history.append({
                    "prompt_words": new_words,
                    "response_words": content.split(),
                })
        except Exception as e:
            ttft = time.perf_counter() - start
            total_tokens = 0
            print(f"  Channel {channel_id} step {step}: ERROR: {e}")

        elapsed = time.perf_counter() - start
        tps = total_tokens / elapsed if elapsed > 0 and total_tokens > 0 else 0

        step_metrics.append({
            "step": step,
            "context_tokens": prompt_tokens,
            "shared_tokens": shared_size,
            "unique_tokens": prompt_tokens - shared_size - 5,
            "ttft": round(ttft, 4),
            "tps": round(tps, 2),
            "total_time": round(elapsed, 4),
            "tokens_generated": total_tokens,
        })

        # Signal next round to start
        round_barrier.set()

    return {
        "channel_id": channel_id,
        "steps": step_metrics,
    }


async def main():
    parser = argparse.ArgumentParser(description="Growing-context benchmark")
    parser.add_argument("--url", default="http://localhost:30083/v1/chat/completions")
    parser.add_argument("--api-key", default="sk-local")
    parser.add_argument("--model", default="Qwen3.6-35B-A3B-UD-MLX-4bit")
    parser.add_argument("--channels", type=int, default=4, help="Number of concurrent channels")
    parser.add_argument("--steps", type=int, default=10, help="Steps per channel")
    parser.add_argument("--step-size", type=int, default=2000, help="Words added per step")
    parser.add_argument("--shared-size", type=int, default=5000, help="Shared prefix size in words")
    parser.add_argument("--output", default="bench-growing.json")
    args = parser.parse_args()

    model = f"/Users/michael/.localllm/models/{args.model}"
    url = args.url
    api_key = args.api_key

    print(f"Growing-context benchmark: {args.channels} channels, {args.steps} steps, {args.step_size} words/step")
    print(f"Shared prefix: ~{args.shared_size} tokens per channel (unique per channel)")
    print(f"Max context per channel: ~{args.shared_size + args.steps * args.step_size} tokens")
    print(f"Growth pattern: round-robin (all channels step 0, then all step 1, etc.)")
    print()

    async with aiohttp.ClientSession() as session:
        # Run round-robin: each round, all channels send their next step concurrently
        all_results = []
        start = time.perf_counter()

        for step in range(args.steps):
            # Create events for this round
            step_event = asyncio.Event()
            round_barrier = asyncio.Event()

            # Start all channels for this step
            tasks = []
            for ch_id in range(args.channels):
                task = asyncio.create_task(
                    run_channel(session, ch_id, url, api_key, model, args.step_size, 1, args.shared_size, step_event, round_barrier)
                )
                tasks.append(task)

            # Kick off the first channel (they all wait on step_event)
            step_event.set()

            # Wait for all channels in this round to finish
            await asyncio.gather(*tasks)

            # Collect results
            for task in tasks:
                all_results.append(task.result())

        total_time = time.perf_counter() - start

    # Reorganize results by channel
    channels_results = {}
    for r in all_results:
        ch_id = r["channel_id"]
        if ch_id not in channels_results:
            channels_results[ch_id] = {"channel_id": ch_id, "steps": []}
        channels_results[ch_id]["steps"].extend(r["steps"])

    results = list(channels_results.values())

    # Aggregate metrics by unique tail size bucket (every 5K tokens)
    all_steps = []
    for r in results:
        all_steps.extend(r["steps"])

    buckets = {}
    for s in all_steps:
        bucket = (s["unique_tokens"] // 5000) * 5000
        if bucket not in buckets:
            buckets[bucket] = {"ttft": [], "tps": []}
        buckets[bucket]["ttft"].append(s["ttft"])
        buckets[bucket]["tps"].append(s["tps"])

    # Print results
    print(f"{'Unique tail':>12} {'Channels':>8} {'TTFT avg':>10} {'TTFT p95':>10} {'TPS avg':>10} {'TPS min':>10}")
    print("-" * 65)
    for ctx in sorted(buckets.keys()):
        data = buckets[ctx]
        n = len(data["ttft"])
        ttft_avg = sum(data["ttft"]) / n
        ttft_p95 = sorted(data["ttft"])[int(n * 0.95)] if n > 0 else 0
        tps_avg = sum(data["tps"]) / n
        tps_min = min(data["tps"]) if data["tps"] else 0
        print(f"{ctx:>12} {n:>8} {ttft_avg:>10.3f} {ttft_p95:>10.3f} {tps_avg:>10.2f} {tps_min:>10.2f}")

    print(f"\nTotal time: {total_time:.1f}s")

    # Save full results
    output = {
        "config": {
            "channels": args.channels,
            "steps": args.steps,
            "step_size": args.step_size,
            "shared_size": args.shared_size,
            "total_time": round(total_time, 2),
        },
        "channels": results,
        "summary": {
            "buckets": {str(k): {
                "count": len(v["ttft"]),
                "ttft_avg": round(sum(v["ttft"]) / len(v["ttft"]), 4) if v["ttft"] else 0,
                "ttft_p95": round(sorted(v["ttft"])[int(len(v["ttft"]) * 0.95)], 4) if v["ttft"] else 0,
                "tps_avg": round(sum(v["tps"]) / len(v["tps"]), 2) if v["tps"] else 0,
                "tps_min": round(min(v["tps"]), 2) if v["tps"] else 0,
            } for k, v in buckets.items()},
        },
    }

    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Results saved to {args.output}")


if __name__ == "__main__":
    asyncio.run(main())
