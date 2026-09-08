#!/usr/bin/env python3
"""Summarize mtp-experiments.csv: median tg per context band per source,
plus (for spec runs) median acceptance fields. Usage: python3 mtp-stats.py [csv]."""
import csv
import statistics
import sys

BANDS = [(0, 8000, "0-8k"), (8000, 16000, "8-16k"), (16000, 32000, "16-32k"),
         (32000, 48000, "32-48k"), (48000, 64000, "48-64k"), (64000, 80000, "64-80k"),
         (80000, 100000, "80-100k"), (100000, 120000, "100-120k"), (120000, 10 ** 9, "120k+")]


def band(ctx):
    for lo, hi, name in BANDS:
        if lo <= ctx < hi:
            return name
    return "?"


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "/Users/michael/localllm/mtp-experiments.csv"
    rows = list(csv.DictReader(open(path)))
    sources = sorted({r["source"] for r in rows}, key=lambda s: (s != "mtp-fixed-today", s))
    hdr = f"{'band':8}" + "".join(f"{s[:20]:>22}" for s in sources)
    print(hdr)
    for _, _, name in BANDS:
        line = f"{name:8}"
        for s in sources:
            sel = [r for r in rows if r["source"] == s and band(int(r["ctx"])) == name and r["tg"]]
            if sel:
                med = statistics.median(float(r["tg"]) for r in sel)
                extra = ""
                accs = [float(r["acc"]) for r in sel if r["acc"]]
                if accs:
                    extra = f" acc{statistics.median(accs):.2f}"
                line += f"{med:>15.1f}{len(sel):>2}n{extra:<5}"
            else:
                line += f"{'-':>22}"
        print(line)


if __name__ == "__main__":
    main()
