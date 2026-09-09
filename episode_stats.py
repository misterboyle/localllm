#!/usr/bin/env python3
"""Episode stats for llama-server -lv 4 logs (agentic tg benchmark).

Summarises real-episode decode speed for spec-decode sweeps:
  - token-weighted tg (the nmax-sensitive rate), prefill rate, busy split
  - draft acceptance (mean acc.len, accept/pos) and draft-gen cost share
  - per-request detail with --verbose

Log timestamps are MM.SS.mmm.uuu since server start, so one episode is
the request slice in time; use --window for logs that continue past it.

Usage:
  episode_stats.py LOG [LOG ...] [--window START_MIN END_MIN] [-v]
"""
import argparse
import re

REQ_HDR = re.compile(
    r'^(?P<stamp>[\d.]+) I slot print_timing: id\s+\d+ \| task \d+ \| '
    r'prompt eval time =\s+(?P<p_ms>[\d.]+) ms /\s+(?P<p_tok>\d+) tokens')
EVAL = re.compile(r'\beval time =\s+(?P<d_ms>[\d.]+) ms /\s+(?P<d_tok>\d+) tokens')
ACCEPT = re.compile(
    r'draft acceptance = (?P<rate>[\d.]+) \(\s*(?P<acc>\d+) accepted / '
    r'\s*(?P<gen>\d+) generated\)')
SPEC = re.compile(
    r'statistics\s+draft-\w+: #calls\(b,g,a\) =\s+\d+\s+(?P<drafts>\d+)\s+\d+, .*?'
    r'#acc drafts =\s*(?P<accd>\d+), .*?'
    r'dur\(b,g,a\) = (?P<bd>[\d.]+), (?P<gd>[\d.]+), (?P<ad>[\d.]+) ms')


def stamp_s(stamp: str) -> float:
    """'MM.SS.mmm.uuu' -> seconds since server start."""
    parts = (stamp.rstrip(".") + ".0" * 4).split(".")[:4]
    mm, ss, ms, us = (int(p) for p in parts)
    return mm * 60 + ss + ms / 1e3 + us / 1e6


def parse(path: str, lo: float, hi: float, verbose: bool):
    lines = open(path, errors="ignore").read().splitlines()
    reqs, spec_last = [], None
    i, n = 0, len(lines)
    while i < n:
        m = REQ_HDR.match(lines[i])
        if m:
            # the completion block: eval/draft acceptance/overall lines follow
            blk = lines[i:i + 8]
            d = a = None
            stamp = m.group("stamp")
            for j, ln in enumerate(blk):
                if j > 0 and REQ_HDR.match(ln):
                    break
                if d is None and "eval time" in ln and "prompt" not in ln:
                    d = EVAL.search(ln)
                if a is None and "draft acceptance" in ln:
                    a = ACCEPT.search(ln)
            if d:
                t = stamp_s(stamp)
                if lo <= t <= hi:
                    reqs.append({
                        "t": t,
                        "p_ms": float(m.group("p_ms")), "p_tok": int(m.group("p_tok")),
                        "d_ms": float(d.group("d_ms")), "d_tok": int(d.group("d_tok")),
                        "acc": a,
                    })
            i += 1
            continue
        s = SPEC.search(lines[i])
        if s:
            spec_last = s  # cumulative snapshots: keep the last
        i += 1

    print(f"\n=== {path} ===")
    if not reqs:
        print("no completed decode requests in window")
        return
    d_ms = sum(r["d_ms"] for r in reqs)
    d_tk = sum(r["d_tok"] for r in reqs)
    p_ms = sum(r["p_ms"] for r in reqs)
    p_tk = sum(r["p_tok"] for r in reqs)
    print(f"requests: {len(reqs)}   last completion t={reqs[-1]['t']:.0f}s")
    print(f"decode : {d_tk:>7} tok / {d_ms / 1e3:7.1f} s  "
          f"-> {d_tk / (d_ms / 1e3):6.2f} t/s (token-weighted)")
    print(f"prefill: {p_tk:>7} tok / {p_ms / 1e3:7.1f} s  "
          f"-> {p_tk / (p_ms / 1e3):6.1f} t/s")
    print(f"busy   : {(d_ms + p_ms) / 1e3:7.1f} s   "
          f"(decode {d_ms / (d_ms + p_ms) * 100:.0f}% "
          f"/ prefill {p_ms / (d_ms + p_ms) * 100:.0f}%)")

    with_acc = [r for r in reqs if r["acc"]]
    if with_acc:
        acc_tk = sum(int(r["acc"]["acc"]) for r in with_acc)
        gen_tk = sum(int(r["acc"]["gen"]) for r in with_acc)
        print(f"accept : {acc_tk}/{gen_tk} = {acc_tk / gen_tk:.3f} of draft "
              f"tokens (mean acc.len {1 + acc_tk / gen_tk:.2f})")
    if spec_last:
        drafts = int(spec_last.group("drafts"))
        accd = int(spec_last.group("accd"))
        g_s = float(spec_last.group("gd")) / 1e3
        print(f"spec   : {drafts} drafts, {accd} accepted "
              f"({accd / drafts:.3f}), mean acc.len {1 + accd / drafts:.2f}")
        print(f"         draft-gen {g_s:.1f} s = {g_s / (d_ms / 1e3) * 100:.1f}% "
              f"of decode wall")
    if verbose:
        print("  t(s)  prefill      decode    tg    accept")
        for r in reqs:
            ar = r["acc"]["rate"] if r["acc"] else "-"
            print(f"{r['t']:6.0f}  {r['p_tok']:>6}tok  "
                  f"{r['d_tok']:>6}tok {r['d_tok'] / (r['d_ms'] / 1e3):5.1f}  {ar}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("logs", nargs="+")
    ap.add_argument("--window", nargs=2, type=float, metavar=("START_MIN", "END_MIN"))
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()
    lo, hi = (args.window[0] * 60, args.window[1] * 60) if args.window else (0.0, float("inf"))
    for p in args.logs:
        parse(p, lo, hi, args.verbose)


if __name__ == "__main__":
    main()
