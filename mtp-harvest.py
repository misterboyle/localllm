#!/usr/bin/env python3
"""Harvest MTP experiment data points from llama-server logs into a CSV row set.

Parses slot print_timing output (tg windows, per-task draft acceptance) and
pairs each task with its context position at generation start:
  new builds: "new prompt ... task.n_tokens = X"
  old builds: "stop processing: n_tokens = T" minus generated tokens

Usage:
  python3 mtp-harvest.py LOGFILE --source TAG --build TAG --nmax N [--csv out.csv]

Rows: source,build,nmax,task,ctx,n_gen,tg,acc,mean_len,pos (pos = "p1,p2,...")
Dedupe: rows keyed by source+task; rerunning harvest on the same source is idempotent.
"""
import argparse
import csv
import os
import re
import sys

RE_RUN = re.compile(r"llama_server: model loaded")
RE_NEW = re.compile(r"task (\d+) \| new prompt.*task\.n_tokens = (\d+)")
RE_REL = re.compile(r"release: id\s+\d+ \| task (\d+) \| stop processing: n_tokens = (\d+)")
RE_TGW = re.compile(r"task (\d+) \| n_gen =\s+(\d+), tg =\s+([\d.]+) t/s")
RE_EV = re.compile(r"task (\d+) \| eval time =\s+[\d.]+ ms /\s+(\d+) tokens "
                   r"\(\s*[\d.]+ ms per token,\s+([\d.]+) tokens per second\)")
RE_ACC = re.compile(r"task (\d+) \| draft acceptance = ([\d.]+) \.*\s*\(\s*\d+ accepted /\s*(\d+) generated\), mean len =\s+([\d.]+)")
RE_POS = re.compile(r"task (\d+) \| acc per pos = \(([^)]*)\)")


def harvest(path):
    run, tasks = 0, {}
    with open(path, errors="ignore") as f:
        for line in f:
            if RE_RUN.search(line):
                run += 1
                continue
            m = RE_NEW.search(line)
            if m:
                tasks[(run, m.group(1))] = {"ctx": int(m.group(2)), "n_gen": 0}
                continue
            m = RE_TGW.search(line)
            if m:
                st = tasks.setdefault((run, m.group(1)), {})
                if int(m.group(2)) >= st.get("n_gen", 0):
                    st["n_gen"], st["tg"] = int(m.group(2)), float(m.group(3))
                continue
            m = RE_EV.search(line)
            if m:
                st = tasks.setdefault((run, m.group(1)), {})
                if int(m.group(2)) >= st.get("n_gen", 0):
                    st["n_gen"], st["tg"] = int(m.group(2)), float(m.group(3))
                continue
            m = RE_ACC.search(line)
            if m:
                st = tasks.setdefault((run, m.group(1)), {})
                st["acc"], st["mean_len"] = float(m.group(2)), float(m.group(4))
                continue
            m = RE_POS.search(line)
            if m:
                st = tasks.setdefault((run, m.group(1)), {})
                st["pos"] = m.group(2).replace(" ", "")
                continue
            m = RE_REL.search(line)
            if m:
                st = tasks.setdefault((run, m.group(1)), {})
                st["ctx_rel"] = int(m.group(2)) - st.get("n_gen", 0)
    for (r, task), st in sorted(tasks.items(), key=lambda kv: (kv[0][0], int(kv[0][1]))):
        ctx = st.get("ctx", st.get("ctx_rel"))
        if st.get("n_gen", 0) >= 150 and ctx:
            yield {"task": f"r{r}t{task}", "ctx": ctx, "n_gen": st["n_gen"],
                   "tg": st.get("tg", ""), "acc": st.get("acc", ""),
                   "mean_len": st.get("mean_len", ""), "pos": st.get("pos", "")}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("logfile")
    ap.add_argument("--source", required=True)
    ap.add_argument("--build", required=True)
    ap.add_argument("--nmax", required=True)
    ap.add_argument("--csv", default=os.path.expanduser("~/localllm/mtp-experiments.csv"))
    args = ap.parse_args()

    cols = ["source", "build", "nmax", "task", "ctx", "n_gen", "tg", "acc", "mean_len", "pos"]
    existing = set()
    if os.path.exists(args.csv):
        with open(args.csv) as f:
            existing = {(r["source"], r["task"]) for r in csv.DictReader(f)}
    new = open(args.csv, "a", newline="") if existing else open(args.csv, "w", newline="")
    w = csv.DictWriter(new, fieldnames=cols, extrasaction="ignore")
    if not existing:
        w.writeheader()
    n = 0
    for row in harvest(args.logfile):
        if (args.source, row["task"]) in existing:
            continue
        row.update(source=args.source, build=args.build, nmax=args.nmax)
        w.writerow(row)
        n += 1
    new.close()
    print(f"{args.source}: {n} new rows -> {args.csv}")


if __name__ == "__main__":
    main()
