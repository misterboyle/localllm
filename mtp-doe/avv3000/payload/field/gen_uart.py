#!/usr/bin/env python3
"""Generate the uart.log telemetry captures in payload/field/.

Two artifacts, one deterministic model (fixed seed — byte-stable):

  uart.log                — affected unit, production image 3.2.1, 3 h.
  uart-image-304.log      — pre-deploy engineering capture of the same unit
                            slot, image 3.0.4 (before the 2 Hz STATUS burst
                            existed), 3 h. Control capture: no bursts, no
                            gaps. The fleet diff that started the B2 trail.

Model (the field model named by svc/telem_svc.h "STATUS BURST"): one
telemetry line per SDR window (2 ms) = 5,400,000 lines over 3 h. On image
3.1.0+ the telemetry service ALSO emits a 100 ms status burst at 2 kHz
every 5 s into the SAME ring/console writer as the window stream. The
service ring (capacity 64) overflows at burst onset when the ring is full
(pre-fix admission policy has no guard), dropping 12..19 consecutive
window lines per overflow (every other burst — the ring drains between).
That is the seeded B2 signature; the summary at the end of uart.log is
the aggregate over the FULL model, not over the excerpt.

The committed uart.log is an EXCERPT: sample lines every 30 s, 5-line
windows around one in four gap events, and the full-capture aggregate
summaries. Body lines are verbatim; only the selection is editorial.

Regenerate: python3 gen_uart.py   (writes both files, byte-stable)
"""
import math
import random

rng = random.Random(0xA7A3007)  # fixed seed — byte-stable output

T_W = 2  # ms per window line
HOURS = 3
T_END = HOURS * 3600 * 1000      # capture end, ms
N_LINES = T_END // T_W           # 5,400,000 window lines
BURST_PERIOD_MS = 5000
BURST_LEN_LINES = 200            # 100 ms @ 2 kHz status burst
RING_CAP = 64
GAP_SPAN = (12, 19)              # dropped consecutive lines per overflow
SAMPLE_S = 300                   # excerpt: one sample line per 300 s
GAP_EVERY = 16                   # excerpt: full window around one in sixteen gaps

def temp_at(ts_ms):
    base = 36.8 + (ts_ms / (3 * 3600 * 1000)) * 1.1  # 36.8 -> 37.9 C
    wave = 0.05 * math.sin(ts_ms / 900000.0)
    return base + wave

def seq_at(ts_ms):
    return ts_ms // T_W + 1

def line(ts_ms):
    s = seq_at(ts_ms)
    t = temp_at(ts_ms)
    ch0 = 41 if ts_ms < 20_000 else (42 + (ts_ms // 200_000) % 4)
    ch1 = ch0 + 1
    return f"{ts_ms},{s},{t:.1f},{ch0},{ch1},"

# ---- gap manifest (burst/gap model — identical sequence on every run) ----
# A burst starts every 5000 ms. The window stream is continuous; the ring
# overflows at burst start when the ring is full (it is full whenever the
# service has been writing >= RING_CAP lines since any drain — i.e.
# essentially always at burst onset under the pre-fix policy). Overflow
# lands on alternate bursts only (the ring drains in between). Drop span
# is deterministic per burst index under the fixed seed.
burst_ts = []
t = BURST_PERIOD_MS
while t <= T_END:
    burst_ts.append(t)
    t += BURST_PERIOD_MS

gap_events = []  # (gap_start_seq, gap_size)
for bi, bt in enumerate(burst_ts):
    if bi % 2 == 0:
        gap_size = rng.randint(*GAP_SPAN)
        gap_events.append((seq_at(bt), gap_size))

# ---- uart.log: affected unit, image 3.2.1 ----
out = []
out.append("# AVV-3000 pump telemetry — uart.log (3 h capture, affected unit, EXCERPT)")
out.append("# Format: ts_ms,seq,t_c,ch0_mmhg,ch1_mmhg,flags  (flags O=occl_det A=alarm H=htemp L=ltemp)")
out.append("# Full capture = 1 line per SDR window (2 ms): 5,400,000 lines / 3 h.")
out.append("# This file keeps: (a) one sample line per 300 s, (b) 5-line windows")
out.append("# around one in sixteen ring-gap events (seq holes),")
out.append("# (c) the aggregate gap summaries at the end (FULL capture, not the")
out.append("# excerpt). Body lines are verbatim; only the selection is editorial.")
out.append("# Control capture: same slot, image 3.0.4 (no STATUS burst): uart-image-304.log")
out.append("ts_ms,seq,t_c,ch0_mmhg,ch1_mmhg,flags")

# sample lines: one per SAMPLE_S seconds
for ts in range(120, T_END, SAMPLE_S * 1000):
    out.append(line(ts))

# gap windows: 5 lines around every GAP_EVERY-th gap (2 before, hole, 2 after)
for gi, (gs, gsz) in enumerate(gap_events):
    if gi % GAP_EVERY != 0:
        continue
    ts0 = (gs - 1) * T_W
    out.append("# -- gap: seq %d..%d missing (ring overflow under 2 Hz burst) --" % (gs, gs + gsz - 1))
    for k in range(2):
        out.append(line(ts0 - (2 - k) * T_W))
    for k in range(2):
        out.append(line(ts0 + (gsz + k) * T_W))

# ---- aggregate summaries over the FULL model ----
out.append("")
out.append("# ---- full-capture gap summary (3 h) ----")
out.append(f"# gap events: {len(gap_events)} total, all inside 2 Hz status-burst windows")
out.append(f"# missing lines total: {sum(s for _, s in gap_events)}")
out.append("# size distribution (missing seq lines per gap event):")
dist = {}
for _, s in gap_events:
    dist[s] = dist.get(s, 0) + 1
for s in sorted(dist):
    out.append(f"#   {s} missing: {dist[s]} events")
out.append("# gaps in NON-burst minutes: 0  => ring overflow is burst-correlated")
out.append("# burst windows with NO drop: 0 (pre-fix policy: burst onset finds the")
out.append("#   ring full every other burst; when it lands, windows are lost)")
out.append("")
out.append("# ---- per-hour breakdown (full model counts, not excerpt) ----")
out.append("# hour,gap_events,missing_lines")
for h in range(HOURS):
    lo, hi = h * 3600 * 1000, (h + 1) * 3600 * 1000
    ev = [(gs, sz) for gs, sz in gap_events if lo <= (gs - 1) * T_W < hi]
    out.append(f"# {h},{len(ev)},{sum(sz for _, sz in ev)}")
out.append("")
out.append("# ---- first/last gap events (full model, seq numbers) ----")
out.append(f"# first: burst #{0} seq {gap_events[0][0]} size {gap_events[0][1]}")
out.append(f"# last:  burst #{2*(len(gap_events)-1)} seq {gap_events[-1][0]} size {gap_events[-1][1]}")
out.append("# NOTE: gaps begin at 5 s (first burst) — this unit has dropped window")
out.append("#   records since boot since image 3.1.0 introduced the burst cadence;")
out.append("#   compare the pre-deploy control capture for the same slot.")

with open("uart.log", "w") as f:
    f.write("\n".join(out) + "\n")

# ---- uart-image-304.log: pre-deploy control capture (no burst feature) ----
cb = []
cb.append("# AVV-3000 pump telemetry — uart-image-304.log (3 h CONTROL capture, EXCERPT)")
cb.append("# Same unit slot, site test bench, image 3.0.4 (2025-11-30 pre-deploy")
cb.append("# validation sweep — BEFORE the 2 Hz STATUS burst feature (3.1.0).")
cb.append("# Format: ts_ms,seq,t_c,ch0_mmhg,ch1_mmhg,flags  (one line per SDR window).")
cb.append("# This file keeps one sample line per 300 s plus the gap summary.")
cb.append("ts_ms,seq,t_c,ch0_mmhg,ch1_mmhg,flags")
for ts in range(120, T_END, SAMPLE_S * 1000):
    cb.append(line(ts))
cb.append("")
cb.append("# ---- gap summary (full capture) ----")
cb.append("# STATUS bursts emitted: 0 (feature present from image 3.1.0)")
cb.append("# window-seq gaps observed: 0 across all 3 h — seq is contiguous")
cb.append("# ring-overflow counters (svc self-report at session end): 0")
cb.append("# NOTE for the fleet trail: this slot was re-imaged 3.0.4 -> 3.1.2 ->")
cb.append("# 3.2.1 without hardware change; the gap signature appears exactly at")
cb.append("# the image step that introduced the STATUS burst cadence.")
with open("uart-image-304.log", "w") as f:
    f.write("\n".join(cb) + "\n")

print("wrote uart.log (%d lines) and uart-image-304.log (%d lines)"
      % (len(out), len(cb)))
