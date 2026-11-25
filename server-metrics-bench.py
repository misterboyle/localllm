#!/usr/bin/env python3
"""server-metrics-bench.py — context-depth rates read from the server's own counters.

For the GLM-5.3-Flash (Sushi 2.4bpw) probe and any future driver candidate. Engine-agnostic:
any mlx-serve-family server (/metrics.json + /tokenize + /v1/chat/completions).

Design law: the server already measures itself — its dashboard's decode t/s, prefill t/s and
TTFT all come from these histograms — so we diff its histograms around a batch of requests and
report what IT says, instead of re-deriving rates from client-side arrival times. Client
arithmetic was the old design and Michael killed it ("we don't need to make this stuff up"):
it could also silently absorb another session's work. This version cannot: the window is only
trusted when `Δrequests_success == requests sent` and `Δprefill_time count == requests sent`,
i.e. the server itself confirms exactly our requests completed inside it. A foreign request
that completed mid-window makes the row POLLUTED instead of a false number (measured
2026-10-12: the incumbent read 20.3 t/s through a harness that could not see the second
session sharing the box; exclusive, the same model is 47-50).

Per depth D: exact-count the prompt via /tokenize, fire N single-token requests (prefill
work), then N generation requests, diffing metrics around each group.

    python3 server-metrics-bench.py --port 30087 --depths "2048 32768 131072" \
        --reps 2 --gen-tokens 512 --fresh --out ~/.localllm/bench/glm-sushi.json

    # the same ladder with speculation switched off, request by request (no restart):
    python3 server-metrics-bench.py --port 30085 --model <id> --depths "4096 1040000" \
        --reps 1 --gen-tokens 512 --plain --fresh

Depths are space-separated inside one argument. --fresh is what makes a prefill number cold;
--plain is what makes a decode number a baseline. Worked example of both, with the verdicts:
wiki [[flash-next-m5ultra-measured]].
"""

import argparse
import json
import random
import sys
import threading
import time
import urllib.request

UNIT = (  # agentic filler: code + prose + a CJK line; lorem tokenizes ~2x dense
    "func handle(req Request) (Response, error) {\n"
    "\tif req.ID == 0 {\n\t\treturn Response{}, ErrInvalidID\n\t}\n"
    "\treturn store.Load(ctx, req.ID)\n}\n\n"
    "// Review the failure path: the caller retries three times with backoff and logs\n"
    "// only the last error. 用一句话说明这段代码做什么。\n"
)
TAIL = " Summarise in one sentence what changes if ErrInvalidID is removed.\n"

# A long-form ask, because the row is a decode rate over N tokens and a one-sentence
# answer returns 5-11 of them: measured 2026-11-25, "answer in one short sentence" at
# 3K depth printed 41.8/77.9/94.7/120.1/193.0 t/s for the same prompt size (planner
# warm-up, not throughput), while this ask fills max_tokens every time (finish=length).
LONG_TAIL = (" Write a numbered list of 400 distinct short facts about the ocean, "
             "one line each, and do not stop early.\n")
SOUP = ("river stone light wind morning house garden road iron glass table letter music "
        "window mountain paper shadow door winter silver clock bridge harbour quiet "
        "lantern field timber thread mirror meadow anchor valley copper sentence harvest "
        "ledge silence feather orbit canyon marble fable thistle ember plough kettle").split()


def http_json(url: str, body: dict | None = None, timeout: float = 120) -> dict:
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data,
                                 headers={"Content-Type": "application/json"} if data else {})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


class Sampler(threading.Thread):
    """Polls /metrics.json during a group so we can report max concurrent requests seen."""

    def __init__(self, base: str):
        super().__init__(daemon=True)
        self.base, self.stop, self.peak = base, threading.Event(), 0

    def run(self):
        while not self.stop.is_set():
            try:
                g = http_json(f"{self.base}/metrics.json", timeout=3)["gauges"]
                self.peak = max(self.peak, int(g.get("requests_running") or 0))
            except Exception:
                pass
            self.stop.wait(0.2)


def snapshot(base: str) -> dict:
    d = http_json(f"{base}/metrics.json", timeout=5)
    c, h = d["counters"], d["histograms"]
    return {"success": c["requests_success_total"],
            "prompt_tok": h["prompt_tokens"]["sum"], "prompt_n": h["prompt_tokens"]["count"],
            "out_tok": h["output_tokens"]["sum"], "out_n": h["output_tokens"]["count"],
            "pre_s": h["prefill_time_seconds"]["sum"], "pre_n": h["prefill_time_seconds"]["count"],
            "dec_s": h["decode_time_seconds"]["sum"], "dec_n": h["decode_time_seconds"]["count"],
            "ttft_s": h["time_to_first_token_seconds"]["sum"],
            "ttft_n": h["time_to_first_token_seconds"]["count"]}


def delta(a: dict, b: dict) -> dict:
    return {k: b[k] - a[k] for k in a}


def build_prompt(base: str, target: int, tail: str = TAIL) -> tuple[str, int]:
    text = UNIT * max(1, (target * 3) // 40) + tail
    for _ in range(6):
        n = len(http_json(f"{base}/tokenize", {"content": text, "add_bos": False})["tokens"])
        if abs(n - target) <= 0.01 * target:
            return text, n
        text = UNIT * max(1, round(text.count(UNIT) * target / n)) + tail
    return text, n


def count_tokens(base: str, text: str) -> int:
    return len(http_json(f"{base}/tokenize", {"content": text, "add_bos": False})["tokens"])


def fresh_prompt(base: str, target: int, rng) -> tuple[str, int]:
    """A prompt of ~`target` tokens that no earlier request used, so the prefix cache misses.

    Reps of the same text measure warm-prefix restore, not prefill (see the note in
    [[mlx-serve]]); this is the cold path at depth, which is what an agent turn actually pays.
    Sized in words, because this soup tokenizes at ~1.4 words per token.
    """
    words = [rng.choice(SOUP) for _ in range(int(target * 0.72))]
    text = " ".join(words) + LONG_TAIL
    n = count_tokens(base, text)
    for _ in range(4):
        if abs(n - target) <= 0.01 * target:
            break
        want = max(8, int(len(words) * target / n))
        words = words[:want] if want <= len(words) else words + [rng.choice(SOUP)
                                                                 for _ in range(want - len(words))]
        text = " ".join(words) + LONG_TAIL
        n = count_tokens(base, text)
    return text, n


def mem(base: str) -> dict:
    """The engine's own accounting. Note this is NOT the same instrument as a Reddit-style
    'peak GiB' column, which is whole-Mac wired memory including the engine's pool."""
    try:
        m = http_json(f"{base}/props", timeout=5).get("memory", {})
        return {k: round(v / 1e9, 1) for k, v in m.items()
                if k in ("active_bytes", "peak_bytes", "kv_cache_bytes", "weights_bytes")}
    except Exception as exc:
        return {"error": str(exc)}


def group(base: str, model: str, prompts: list, max_tokens: int, timeout: float,
          thinking: bool = False, mtp: bool = True, pld: bool = True) -> dict:
    """Send the given requests sequentially; return the server's own histogram delta."""
    n = len(prompts)
    pre = snapshot(base)
    m0 = mem(base)
    s = Sampler(base)
    s.start()
    for prompt in prompts:
        http_json(f"{base}/v1/chat/completions",
                  {"model": model, "messages": [{"role": "user", "content": prompt}],
                   "max_tokens": max_tokens, "temperature": 0.0, "thinking": thinking,
                   "enable_mtp": mtp, "enable_pld": pld},
                  timeout=timeout)
    s.stop.set()
    s.join(timeout=3)
    d = delta(pre, snapshot(base))
    d["mem_after"] = m0
    d["mem_peak"] = mem(base)
    d["peak_running"] = s.peak
    exact = d["success"] == n and d["pre_n"] == n and d["dec_n"] == n
    if not exact:
        d["verdict"] = f"POLLUTED (server says {d['success']} succeeded / {d['pre_n']} prefilled / {d['dec_n']} decoded, expected {n})"
        return d
    d["prefill_tps"] = round(d["prompt_tok"] / d["pre_s"], 1) if d["pre_s"] else 0
    d["decode_tps"] = round((d["out_tok"] - n) / (d["dec_s"]), 1) if d["dec_s"] else 0
    d["ttft_s"] = round(d["ttft_s"] / d["ttft_n"], 2)
    d["cache_hit_tok"] = d["prompt_tok"]          # tokens fed to the prompt (pre pass)
    d["verdict"] = "PASS" if d["peak_running"] <= 1 else \
        f"CONTENDED (peak {d['peak_running']} requests in flight)"
    return d


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--base", default="")
    ap.add_argument("--model", required=True)
    ap.add_argument("--depths", default="2048 32768 131072")
    ap.add_argument("--reps", type=int, default=2, help="requests per measured group")
    ap.add_argument("--gen-tokens", type=int, default=256)
    ap.add_argument("--ask", choices=["long", "short"], default="long",
                    help="'short' (one-sentence) answers in 5-49 tokens, which makes the decode "
                         "row a planner-warm-up artefact — see LONG_TAIL. 'long' fills "
                         "--gen-tokens. Default long; short is only for cache-shape probes.")
    ap.add_argument("--thinking", action="store_true", help="send thinking=true (the daily's shape)")
    ap.add_argument("--fresh", action="store_true",
                    help="new randomised prompt per request (cold prefill every rep, no cache hits)")
    ap.add_argument("--no-mtp", action="store_true",
                    help="send enable_mtp:false: PLD still drafts, so this is NOT the serial rate. "
                         "Per-request, so an A/B needs no server restart (verified 2026-11-25: "
                         "4K same prefix, 152.6 on / 81.0 off).")
    ap.add_argument("--plain", action="store_true",
                    help="send enable_mtp:false AND enable_pld:false: the true serial cell, i.e. the "
                         "baseline that prices speculation per depth. The engine's own "
                         "serial_cell field in [spec-stats] is a planner probe and reads the same "
                         "value at every depth, so it cannot answer this (measured 2026-11-25).")
    ap.add_argument("--timeout", type=float, default=1800)
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    base = args.base or f"http://127.0.0.1:{args.port}"

    rng = random.Random(1)
    tail = LONG_TAIL if args.ask == "long" else TAIL
    rows = []
    hdr = f"{'depth':>8} {'decode':>12} {'prefill':>12} {'TTFT':>9} {'peak n':>7}  verdict"
    print(f"# {base} model={args.model} thinking={args.thinking} fresh={args.fresh} "
          f"mtp={not args.no_mtp and not args.plain} pld={not args.plain}\n{hdr}")
    for depth in (int(d) for d in args.depths.split()):
        prompt, exact_tok = (fresh_prompt(base, depth, rng) if args.fresh
                             else build_prompt(base, depth, tail=tail))
        spec_on = not args.no_mtp and not args.plain
        cold = group(base, args.model, [prompt], 1, args.timeout, args.thinking, spec_on,
                     not args.plain)                                    # cold prefix
        prompts = [fresh_prompt(base, depth, rng)[0] for _ in range(args.reps)] if args.fresh \
            else [prompt] * args.reps
        warm = group(base, args.model, prompts, args.gen_tokens, args.timeout, args.thinking,
                     spec_on, not args.plain)
        warm["prompt_tokens_exact"] = exact_tok
        warm["cold_prefill_tps"] = cold.get("prefill_tps")
        rows.append(warm)
        if warm["verdict"] == "PASS":
            print(f"{depth:>8} {warm['decode_tps']:>10} t/s {warm['prefill_tps']:>10} t/s "
                  f"{warm['ttft_s']:>7} s {warm['peak_running']:>7}  PASS "
                  f"(cold prefill {cold.get('prefill_tps')} t/s, "
                  f"engine memory active {warm['mem_peak'].get('active_bytes')} GB, "
                  f"peak-since-boot {warm['mem_peak'].get('peak_bytes')} GB, "
                  f"KV {warm['mem_peak'].get('kv_cache_bytes')} GB)")
        else:
            print(f"{depth:>8} {warm['verdict']}")
        sys.stdout.flush()
    if args.out:
        json.dump({"base": base, "model": args.model, "rows": rows}, open(args.out, "w"), indent=2)
        print(f"wrote {args.out}")
    return 0 if any(r["verdict"] == "PASS" for r in rows) else 1


if __name__ == "__main__":
    sys.exit(main())
