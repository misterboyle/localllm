#!/usr/bin/env python3
"""Sample mlx-serve memory over time and judge the batched-decode leak (upstream #1071/#1091).

The upstream rule is that a pre/post-unload delta is the WRONG test — the leak
accumulates ~2 GiB/h, far slower than any one request, so what discriminates is the
HOURLY MINIMUM of `mlx_active_bytes` over 24-48 h of real traffic (the minimum is the
idle floor; a rising floor is the leak). Nothing on this box recorded a series, so
this tool is the instrument:

    # watch in the background (safe next to live traffic; it only GETs two endpoints)
    nohup python3 tools/mem-watch.py --interval 60 >/dev/null 2>&1 &
    tail -f ~/.localllm/mem-watch.jsonl

    # judge whatever has been recorded
    python3 tools/mem-watch.py --report

Each line records the clock, the engine version, `active/weights/kv/cache/available`
from /props, and — critically — EXPOSURE: `requests_running` and `batched_group_size`
from /metrics.json, plus the generation tokens that accrued while two or more streams
were decoding together. #1091 leaked logits rows only in the batched-decode path, so a
flat series with zero batched exposure proves nothing; --report says so out loud
instead of reporting a clean pass.

Verdict rules (from the wiki's acceptance test for a fixed build):
  PASS  unowned <= ~1 GiB at the idle floor AND available_bytes back near ~32 GiB
  FAIL  unowned > ~5 GiB, or the hourly floor ratchets up without plateauing
  VOID  fewer than ~2 h of samples, or fewer than 20k tokens generated at batch >= 2
"""

import argparse
import datetime
import json
import os
import statistics
import sys
import time
import urllib.request

DEFAULT_OUT = os.path.expanduser("~/.localllm/mem-watch.jsonl")
GIB = 1024 ** 3
MIN_EXPOSURE_TOKENS = 20_000
MIN_HOURS = 2.0


def get(url, timeout=5):
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return json.loads(r.read().decode())
    except Exception as e:                                    # noqa: BLE001 - a probe
        return {"_error": f"{type(e).__name__}: {e}"}         # never kills the sample


def sample(base):
    props = get(f"{base}/props")
    met = get(f"{base}/metrics.json")
    m = props.get("memory", {}) or {}
    g = met.get("gauges", {}) or {}
    c = met.get("counters", {}) or {}
    active = m.get("active_bytes", g.get("mlx_active_bytes", 0))
    w, kv, ca = m.get("weights_bytes", 0), m.get("kv_cache_bytes", 0), m.get("cache_bytes", 0)
    return {
        "ts": int(time.time()),
        "clock": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "version": (props.get("settings") or {}).get("version", "?"),
        "active_gib": round(active / GIB, 3),
        "weights_gib": round(w / GIB, 3),
        "kv_gib": round(kv / GIB, 3),
        "cache_gib": round(ca / GIB, 3),
        "available_gib": round(m.get("available_bytes", 0) / GIB, 3),
        "unowned_gib": round((active - w - kv - ca) / GIB, 3),
        "running": g.get("requests_running", 0),
        "batched": g.get("batched_group_size", 0),
        "gen_tokens": c.get("generation_tokens_total", 0),
        "rejected": c.get("requests_rejected_total", 0),
        "error": met.get("_error") or props.get("_error") or "",
    }


def watch(base, interval, out, max_secs):
    t0 = time.time()
    prev = None
    exposure = 0
    print(f"watching {base} every {interval}s -> {out}", flush=True)
    while time.time() - t0 < max_secs:
        s = sample(base)
        if prev and s["gen_tokens"] and prev["gen_tokens"]:
            delta = s["gen_tokens"] - prev["gen_tokens"]
            if max(s["batched"], prev["batched"]) >= 2 or s["running"] >= 2:
                exposure += max(delta, 0)
        s["exposure_tokens"] = exposure
        with open(out, "a") as fh:
            fh.write(json.dumps(s) + "\n")
        print(f"{s['clock']} v{s['version']} active {s['active_gib']:6.2f} "
              f"unowned {s['unowned_gib']:+6.2f} avail {s['available_gib']:6.2f} "
              f"run {s['running']} batch {s['batched']} exp {exposure:,} tok"
              + (f"  {s['error']}" if s["error"] else ""), flush=True)
        prev = s
        time.sleep(interval)


def epochs(rows, gap_secs):
    """Split the series at engine restarts. Counters reset on boot, so a decrease in
    lifetime `generation_tokens_total` is a restart signature the version alone cannot
    show — 26.11.2 -> 26.11.2 with a new process looks identical in `version`."""
    out, cur = [], []
    for prev, r in zip([None] + rows[:-1], rows):
        new = (prev is None or r["gen_tokens"] < prev["gen_tokens"]
               or r["ts"] - prev["ts"] > gap_secs or r["version"] != prev["version"])
        if new and cur:
            out.append(cur)
            cur = []
        cur.append(r)
    if cur:
        out.append(cur)
    return out


def judge(rows, label):
    hours = (rows[-1]["ts"] - rows[0]["ts"]) / 3600
    print(f"\n--- {label}: {len(rows)} samples over {hours:.1f} h, "
          f"version {rows[0]['version']} ---")
    byhour = {}
    for r in rows:
        byhour.setdefault(datetime.datetime.fromtimestamp(r["ts"]).strftime("%d %H"), []).append(r)
    print(f"{'hour':7} {'n':>4} {'floor active':>12} {'floor unowned':>14} {'avail max':>10} {'exp tok':>9}")
    floors = []
    for h in sorted(byhour, key=lambda k: min(x["ts"] for x in byhour[k])):
        b = min(byhour[h], key=lambda r: r["active_gib"])
        floors.append(b["active_gib"])
        print(f"{h:7} {len(byhour[h]):4} {b['active_gib']:12.2f} {b['unowned_gib']:14.2f} "
              f"{max(r['available_gib'] for r in byhour[h]):10.2f} {b['exposure_tokens']:9,}")

    base = rows[0]["exposure_tokens"]
    exp = rows[-1]["exposure_tokens"] - base          # exposure GAINED in this epoch
    idle = min(rows, key=lambda r: r["active_gib"])
    unowned_last = statistics.median([r["unowned_gib"] for r in rows[-6:]])
    print(f"idle floor: active {idle['active_gib']:.2f} GiB, unowned {idle['unowned_gib']:+.2f} GiB; "
          f"batched exposure gained here: {exp:,} tokens; rejections: {max(r['rejected'] for r in rows)}")
    ratchet = len(floors) >= 3 and floors[-1] - max(floors[:-1]) > 1.0
    if hours < MIN_HOURS:
        print(f"VERDICT: VOID — {hours:.1f} h is under the {MIN_HOURS} h the accumulation rate needs")
    elif exp < MIN_EXPOSURE_TOKENS:
        print(f"VERDICT: VOID — only {exp:,} tokens decoded with 2+ streams together. The leak "
              f"lived in the batched-decode path, so this series could not have seen it. Run two "
              f"sessions generating AT THE SAME TIME (or boot with --prefill-decode-share 0.25 "
              f"and let the train prefill beside an interactive session), then re-report.")
    elif unowned_last > 5 or ratchet:
        print("VERDICT: FAIL — the idle floor carries unowned bytes; keep the #1071 report open")
    elif unowned_last <= 1:
        print(f"VERDICT: PASS — unowned {unowned_last:+.2f} GiB at the floor with {exp:,} "
              f"batched tokens behind it")
    else:
        print(f"VERDICT: WATCH — unowned {unowned_last:+.2f} GiB is neither clean nor damning")


def report(out):
    rows = []
    with open(out) as fh:
        for line in fh:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    if len(rows) < 6:
        sys.exit(f"only {len(rows)} samples in {out} — nothing to judge")

    eps = epochs(rows, 600)
    if len(eps) > 1:
        print(f"{len(eps)} engine epochs in this file (a restart resets the counters); "
              f"judging the current one only")
        for i, e in enumerate(eps[:-1], 1):
            exp = e[-1]["exposure_tokens"] - e[0]["exposure_tokens"]
            print(f"  epoch {i}: {e[0]['clock']} -> {e[-1]['clock']}, v{e[0]['version']}, "
                  f"floor {min(r['active_gib'] for r in e):.2f} GiB, "
                  f"exposure {exp:,} tok, verdict: not judged")
    judge(eps[-1], f"epoch {len(eps)} (current)")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--port", type=int, default=30085)
    ap.add_argument("--interval", type=int, default=60)
    ap.add_argument("--max-secs", type=int, default=60 * 60 * 36, help="stop after this long")
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--report", action="store_true", help="judge the recorded series and exit")
    ap.add_argument("--once", action="store_true", help="take a single sample and print it")
    args = ap.parse_args()
    base = f"http://127.0.0.1:{args.port}"

    if args.report:
        report(args.out)
        return
    if args.once:
        print(json.dumps(sample(base), indent=1))
        return
    watch(base, args.interval, args.out, args.max_secs)


if __name__ == "__main__":
    main()
