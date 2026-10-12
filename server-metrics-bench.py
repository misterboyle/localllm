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

    python3 benchmarks/server-metrics-bench.py --port 30087 --depths 2048 32768 131072 \
        --out ~/.localllm/logs/glm-sushi-bench.json
"""

import argparse
import json
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


def build_prompt(base: str, target: int) -> tuple[str, int]:
    text = UNIT * max(1, (target * 3) // 40) + TAIL
    for _ in range(6):
        n = len(http_json(f"{base}/tokenize", {"content": text, "add_bos": False})["tokens"])
        if abs(n - target) <= 0.01 * target:
            return text, n
        text = UNIT * max(1, round(text.count(UNIT) * target / n)) + TAIL
    return text, n


def group(base: str, model: str, prompt: str, n: int, max_tokens: int, timeout: float) -> dict:
    """Send n requests sequentially; return the server's own histogram delta for exactly them."""
    pre = snapshot(base)
    s = Sampler(base)
    s.start()
    for _ in range(n):
        http_json(f"{base}/v1/chat/completions",
                  {"model": model, "messages": [{"role": "user", "content": prompt}],
                   "max_tokens": max_tokens, "temperature": 0.0}, timeout=timeout)
    s.stop.set()
    s.join(timeout=3)
    d = delta(pre, snapshot(base))
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
    ap.add_argument("--timeout", type=float, default=1800)
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    base = args.base or f"http://127.0.0.1:{args.port}"

    rows = []
    hdr = f"{'depth':>8} {'decode':>12} {'prefill':>12} {'TTFT':>9} {'peak n':>7}  verdict"
    print(f"# {base} model={args.model}\n{hdr}")
    for depth in (int(d) for d in args.depths.split()):
        prompt, exact_tok = build_prompt(base, depth)
        cold = group(base, args.model, prompt, 1, 1, args.timeout)      # cold prefix
        warm = group(base, args.model, prompt, args.reps, args.gen_tokens, args.timeout)
        warm["prompt_tokens_exact"] = exact_tok
        warm["cold_prefill_tps"] = cold.get("prefill_tps")
        rows.append(warm)
        if warm["verdict"] == "PASS":
            print(f"{depth:>8} {warm['decode_tps']:>10} t/s {warm['prefill_tps']:>10} t/s "
                  f"{warm['ttft_s']:>7} s {warm['peak_running']:>7}  PASS "
                  f"(cold prefill {cold.get('prefill_tps')} t/s)")
        else:
            print(f"{depth:>8} {warm['verdict']}")
        sys.stdout.flush()
    if args.out:
        json.dump({"base": base, "model": args.model, "rows": rows}, open(args.out, "w"), indent=2)
        print(f"wrote {args.out}")
    return 0 if any(r["verdict"] == "PASS" for r in rows) else 1


if __name__ == "__main__":
    sys.exit(main())
