#!/usr/bin/env python3
"""Report what speculative decoding is actually earning on a live mlx-serve.

Reads the engine's own `[spec-stats] mode=mtp ...` lines (written by every request
when a model has an MTP head) and answers one question per context bucket: is a
speculative round cheaper per token than a plain decode step?

    python3 tools/mtp-yield.py                       # current run (since last version banner)
    python3 tools/mtp-yield.py --all --top 12        # whole log, 12 widest buckets
    python3 tools/mtp-yield.py --log ~/.mlx-serve/logs/mlx-serve-30085.log

`plain` is the engine's `serial_cell` (ms for one un-speculated decode step at that
depth); `spec` is `round_ms / avg_per_round` (a whole draft+verify round divided by
the tokens it landed). `gain` is plain/spec, so **< 1.0 means speculation is losing**.
The width table is the planner's own measured price of a verify at each row count;
`EV best` is the width that maximises expected tokens per millisecond given the
measured per-draft acceptance. Rates come from the server's accounting, never from a
client-side timer (a second agent on the box is invisible to client timing).
"""

import argparse
import os
import re
import statistics
import sys

BUCKET_ORDER = ["<2k", "2-4k", "4-8k", "8-16k", "16-32k", "32-64k",
                "64-128k", "128-256k", "256k+"]

SAMPLE_RE = re.compile(r"\[spec-stats\] mode=mtp (attempts=\d+ .*)$")
FIELD_RE = re.compile(r"(\w+)=([0-9.]+)")
WIDTH_RE = re.compile(r"w(\d):([0-9.]+)/(\d+)")
BUCKET_RE = re.compile(r"table=(\S+?):w")
VERSION_RE = re.compile(r"^mlx-serve (\S+) ")
NOT_ADOPTED = "mtp history not adopted"
MIN_ATTEMPTS = 20


def parse(path, current_run_only, min_attempts):
    """Return (samples, banner, not_adopted_count). Samples are dicts per spec-stats line."""
    samples, last_version, not_adopted = [], None, 0
    with open(path, "rb") as fh:
        for raw in fh:
            line = raw.decode("utf-8", "replace").rstrip("\n")
            ver = VERSION_RE.match(line.strip())
            if ver:
                last_version = ver.group(1)
                if current_run_only:
                    samples, not_adopted = [], 0   # counts are per run, not per file
                continue
            if NOT_ADOPTED in line:
                not_adopted += 1
            m = SAMPLE_RE.search(line)
            if not m:
                continue
            body = m.group(1)
            fields = dict(FIELD_RE.findall(body))
            attempts = float(fields.get("attempts", 0))
            if attempts < min_attempts:
                continue
            bucket = BUCKET_RE.search(body)
            key = (last_version or "?", bucket.group(1) if bucket else "?")
            widths = {}
            for w, ms, n in WIDTH_RE.findall(body):
                widths.setdefault(int(w), []).append((float(ms), int(n)))
            samples.append({
                "run": key[0], "bucket": key[1], "attempts": attempts,
                "apr": float(fields.get("avg_per_round", 0)),
                "pct": float(fields.get("per_draft_pct", 0)),
                "serial": float(fields.get("serial_cell", 0)),
                "round": float(fields.get("round_ms", 0)),
                "widths": {w: (sum(a * b for a, b in v) / sum(b for _, b in v))
                           for w, v in widths.items()},
            })
    return samples, last_version, not_adopted


def expected_tokens(width, p):
    """Expected committed tokens in a round that drafts `width` rows at acceptance p."""
    if p <= 0:
        return 1.0
    if p >= 1:
        return 1.0 + width
    return (1 - p ** (width + 1)) / (1 - p)


def report(samples, banner, not_adopted, show_widths):
    if not samples:
        print("no [spec-stats] samples with enough attempts — is --mtp on, and is the log "
              "the engine's own (~/.mlx-serve/logs/) rather than ./serve's rotation?")
        return 1

    by = {}
    for s in samples:
        by.setdefault(s["bucket"], []).append(s)

    print(f"run: mlx-serve {banner}   samples: {len(samples)}   "
          f"'{NOT_ADOPTED}': {not_adopted}")
    header = (f"{'depth':9} {'n':>4} {'plain':>6} {'spec':>6} {'gain':>6} "
              f"{'tok/rnd':>7} {'acc%':>5} {'EV best':>8}")
    if show_widths:
        header += "  |  measured round price ms, w1..w6"
    print("\n" + header)
    for bucket in [b for b in BUCKET_ORDER if b in by] + sorted(set(by) - set(BUCKET_ORDER)):
        rs = by[bucket]
        plain_cells = [r["serial"] for r in rs if r["serial"] > 0]
        plain = statistics.median(plain_cells) if plain_cells else 0
        spec = statistics.median(r["round"] / r["apr"] for r in rs if r["apr"] > 0)
        apr = statistics.median(r["apr"] for r in rs)
        acc = statistics.median(r["pct"] for r in rs)
        gain = (plain / spec) if plain and spec else 0
        p = acc / 100.0
        price = {}
        for r in rs:
            for w, ms in r["widths"].items():
                price.setdefault(w, []).append(ms)
        price = {w: statistics.median(v) for w, v in price.items()}
        ev_tps = {w: expected_tokens(w, p) / ms * 1000 for w, ms in price.items() if ms > 0}
        best = max(ev_tps, key=ev_tps.get) if ev_tps else None
        best_str = f"w{best} ({ev_tps[best]:.0f})" if best else "n/a"
        line = (f"{bucket:9} {len(rs):4} {plain:6.2f} {spec:6.2f} {gain:5.2f}x "
                f"{apr:7.2f} {acc:5.1f} {best_str:>8}")
        if show_widths:
            line += "  |  " + " ".join(f"{price[w]:5.1f}" if w in price else "    - "
                                       for w in range(1, 7))
        print(line)

    total = sum(r["attempts"] for r in samples)
    w_apr = sum(r["apr"] * r["attempts"] for r in samples) / total
    w_round = sum(r["round"] * r["attempts"] for r in samples) / total
    cells = [r["serial"] for r in samples if r["serial"] > 0]
    w_plain = (sum(r["serial"] * r["attempts"] for r in samples if r["serial"] > 0)
               / sum(r["attempts"] for r in samples if r["serial"] > 0)) if cells else 0
    spec_tps = 1000 / (w_round / w_apr) if w_apr else 0
    print(f"\nattempt-weighted: {w_apr:.2f} tokens/round at {w_round:.2f} ms "
          f"= {spec_tps:.1f} t/s speculated", end="")
    if w_plain:
        print(f" vs {1000 / w_plain:.1f} t/s plain  ->  **{w_plain / (w_round / w_apr):.2f}x**")
    else:
        print("  (no serial_cell samples: cannot price plain decode)")
    print("gain < 1.00x means the round costs more than the tokens it lands — "
          "lower --mtp-max-depth toward the EV-best width and re-measure.")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--log", default=os.path.expanduser("~/.mlx-serve/logs/mlx-serve-30085.log"),
                    help="engine's own app log (default: the 30085 daily driver)")
    ap.add_argument("--all", action="store_true",
                    help="include every run in the file, not just since the last version banner")
    ap.add_argument("--min-attempts", type=int, default=MIN_ATTEMPTS,
                    help="drop samples from turns with fewer MTP rounds (default 20)")
    ap.add_argument("--top", type=int, default=0,
                    help="show only the N deepest buckets, measured by sample count")
    args = ap.parse_args()

    path = os.path.expanduser(args.log)
    if not os.path.exists(path):
        sys.exit(f"no log at {path}")
    samples, banner, not_adopted = parse(path, not args.all, args.min_attempts)
    if args.top:
        counts = {}
        for s in samples:
            counts[s["bucket"]] = counts.get(s["bucket"], 0) + 1
        keep = set(sorted(counts, key=counts.get, reverse=True)[:args.top])
        samples = [s for s in samples if s["bucket"] in keep]
    sys.exit(report(samples, banner, not_adopted, show_widths=not args.top or args.top > 3))


if __name__ == "__main__":
    main()
