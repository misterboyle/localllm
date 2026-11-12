#!/usr/bin/env python3
"""Report what speculative decoding is actually earning on a live mlx-serve.

Reads the engine's own `[spec-stats] mode=mtp ...` lines and answers one question per
context bucket: is a speculative round cheaper per token than a plain decode step?

    python3 tools/mtp-yield.py                       # current run (since last version banner)
    python3 tools/mtp-yield.py --all                 # every run in the file
    python3 tools/mtp-yield.py --log ~/.mlx-serve/logs/mlx-serve-30085.log

FIELD SEMANTICS — got these wrong once, do not re-derive them by guessing:
  attempts        = speculative rounds in the request
  accepts         = DRAFT tokens accepted across those rounds (NOT counting the token
                    the model commits anyway) — so `avg_per_round` = accepts/attempts is
                    accepted drafts, and TOKENS PER ROUND = 1 + avg_per_round.
  drafted         = draft rows proposed in total; drafted/attempts = the width the
                    planner actually chose, which is far below the `depth` ceiling.
  serial_cell     = ms for one plain (un-speculated) decode step at that depth
  round_ms        = ms for one whole draft+verify round
Dividing round_ms by `avg_per_round` (the tempting reading) understates throughput by
(1+1/apr) — ~44 t/s against a request log that says 72.8. This tool therefore carries a
**self-check**: it also harvests the engine's per-request `decode: N tok/s` lines and
refuses to let the round-derived number drift more than 15% from them.

`gain` = plain/spec, so **> 1.0 means speculation is paying**.
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
REQ_RE = re.compile(r"decode: ([0-9.]+) tok/s\]")
NOT_ADOPTED = "mtp history not adopted"


def tokens_per_round(fields):
    """A round lands its committed token PLUS every accepted draft."""
    return 1.0 + float(fields.get("avg_per_round", 0))


def parse(path, current_run_only, min_attempts):
    """Return (samples, request_decode_rates, banner, not_adopted)."""
    samples, rates, last_version, not_adopted = [], [], None, 0
    with open(path, "rb") as fh:
        for raw in fh:
            line = raw.decode("utf-8", "replace").rstrip("\n")
            ver = VERSION_RE.match(line.strip())
            if ver:
                last_version = ver.group(1)
                if current_run_only:
                    samples, rates, not_adopted = [], [], 0
                continue
            if NOT_ADOPTED in line:
                not_adopted += 1
            m = REQ_RE.search(line)
            if m:
                rates.append(float(m.group(1)))
                continue
            m = SAMPLE_RE.search(line)
            if not m:
                continue
            body = m.group(1)
            f = dict(FIELD_RE.findall(body))
            if float(f.get("attempts", 0)) < min_attempts:
                continue
            bucket = BUCKET_RE.search(body)
            widths = {}
            for w, ms, n in WIDTH_RE.findall(body):
                widths.setdefault(int(w), []).append((float(ms), int(n)))
            samples.append({
                "run": last_version or "?",
                "bucket": bucket.group(1) if bucket else "?",
                "attempts": float(f["attempts"]),
                "apr": tokens_per_round(f),
                "width": float(f.get("drafted", 0)) / float(f["attempts"]),
                "depth": float(f.get("depth", 0)),
                "acc": float(f.get("per_draft_pct", 0)),
                "serial": float(f.get("serial_cell", 0)),
                "round": float(f.get("round_ms", 0)),
                "price": {w: sum(a * b for a, b in v) / sum(b for _, b in v)
                          for w, v in widths.items()},
            })
    return samples, rates, last_version, not_adopted


def expected_tokens(width, p):
    """Committed tokens from a round that drafts `width` rows at per-row acceptance p."""
    if p <= 0:
        return 1.0
    if p >= 1:
        return 1.0 + width
    return (1 - p ** (width + 1)) / (1 - p)


def report(samples, rates, banner, not_adopted, show_widths):
    if not samples:
        print("no [spec-stats] samples with enough attempts — is --mtp on, and is the log "
              "the engine's own (~/.mlx-serve/logs/) rather than ./serve's rotation?")
        return 1

    # instrument self-check: the round math must agree with the request-level rates
    spec_tps = [1000 * s["apr"] / s["round"] for s in samples if s["round"] > 0]
    got = statistics.median(spec_tps) if spec_tps else 0
    logd = statistics.median(rates) if rates else 0
    print(f"run: mlx-serve {banner}   samples: {len(samples)}   "
          f"'{NOT_ADOPTED}': {not_adopted}")
    if got and logd:
        drift = abs(got - logd) / logd * 100
        verdict = "OK" if drift <= 15 else "DIVERGES — do not trust the gain column"
        print(f"instrument check: round-derived {got:.1f} t/s vs the engine's own per-request "
              f"{logd:.1f} t/s -> {drift:.1f}% apart, {verdict}")
    print(f"planner's realised draft width: {statistics.median(s['width'] for s in samples):.2f} "
          f"rows/round against a ceiling of "
          f"{statistics.median(s['depth'] for s in samples):.0f}")

    by = {}
    for s in samples:
        by.setdefault(s["bucket"], []).append(s)
    header = (f"\n{'depth':9} {'n':>4} {'plain':>6} {'spec':>6} {'gain':>6} "
              f"{'tok/rnd':>7} {'rows':>5} {'acc%':>5} {'EV best':>8}")
    if show_widths:
        header += "  |  measured round price ms, w1..w6"
    print(header)
    for bucket in [b for b in BUCKET_ORDER if b in by] + sorted(set(by) - set(BUCKET_ORDER)):
        rs = by[bucket]
        cells = [r["serial"] for r in rs if r["serial"] > 0]
        plain = statistics.median(cells) if cells else 0
        spec = statistics.median(r["round"] / r["apr"] for r in rs if r["apr"] > 0)
        tpr = statistics.median(r["apr"] for r in rs)
        rows = statistics.median(r["width"] for r in rs)
        acc = statistics.median(r["acc"] for r in rs)
        price = {}
        for r in rs:
            for w, ms in r["price"].items():
                price.setdefault(w, []).append(ms)
        price = {w: statistics.median(v) for w, v in price.items()}
        p = acc / 100.0
        ev = {w: expected_tokens(w, p) / ms * 1000 for w, ms in price.items() if ms > 0}
        best = max(ev, key=ev.get) if ev else None
        gain = (plain / spec) if plain and spec else 0
        line = (f"{bucket:9} {len(rs):4} {plain:6.2f} {spec:6.2f} {gain:5.2f}x "
                f"{tpr:7.2f} {rows:5.2f} {acc:5.1f} "
                f"{(f'w{best} ({ev[best]:.0f})' if best else 'n/a'):>8}")
        if show_widths:
            line += "  |  " + " ".join(f"{price[w]:5.1f}" if w in price else "    - "
                                       for w in range(1, 7))
        print(line)

    tot = sum(s["attempts"] for s in samples)
    tok = sum(s["attempts"] * s["apr"] for s in samples)
    ms = sum(s["attempts"] * s["round"] for s in samples)
    cells = [s for s in samples if s["serial"] > 0]
    if cells:
        ct = sum(s["attempts"] for s in cells)
        plain = sum(s["attempts"] * s["serial"] for s in cells) / ct
        spec = ms / tok
        print(f"\nattempt-weighted: {tok/ct if ct else 0:.2f} tokens/round, "
              f"{spec:.2f} ms/token speculated vs {plain:.2f} plain "
              f"-> **{plain/spec:.2f}x** ({1000/spec:.1f} vs {1000/plain:.1f} t/s)")
    else:
        print("\nno serial_cell samples in this window: cannot price plain decode")
    print("EV best = the draft width that maximises expected tokens per ms at the measured "
          "acceptance; realised rows/round is what the planner actually chose.")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--log", default=os.path.expanduser("~/.mlx-serve/logs/mlx-serve-30085.log"),
                    help="engine's own app log (default: the 30085 daily driver)")
    ap.add_argument("--all", action="store_true",
                    help="include every run in the file, not just since the last version banner")
    ap.add_argument("--min-attempts", type=int, default=20,
                    help="drop samples from turns with fewer MTP rounds (default 20)")
    ap.add_argument("--top", type=int, default=0,
                    help="show only the N deepest buckets, measured by sample count")
    args = ap.parse_args()

    path = os.path.expanduser(args.log)
    if not os.path.exists(path):
        sys.exit(f"no log at {path}")
    samples, rates, banner, not_adopted = parse(path, not args.all, args.min_attempts)
    if args.top:
        counts = {}
        for s in samples:
            counts[s["bucket"]] = counts.get(s["bucket"], 0) + 1
        keep = set(sorted(counts, key=counts.get, reverse=True)[:args.top])
        samples = [s for s in samples if s["bucket"] in keep]
    sys.exit(report(samples, rates, banner, not_adopted, show_widths=not args.top or args.top > 3))


if __name__ == "__main__":
    main()
