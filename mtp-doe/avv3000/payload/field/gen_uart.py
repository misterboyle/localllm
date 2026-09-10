#!/usr/bin/env python3
"""Generate payload/field/uart.log — deterministic 3 h telemetry capture excerpt.

Model: one telemetry line per SDR window (2 ms) = 5,400,000 lines over 3 h.
The telemetry service ALSO emits a 100 ms status burst at 2 kHz every 5 s.
The service ring (capacity 64) overflows under a burst when the window stream
has not drained, dropping 12..19 consecutive lines per burst (seeded B2).

The committed uart.log is an EXCERPT: sample lines + 5-line windows around
each gap + the full-capture histogram (the histogram is the B2 reveal).
Regenerate: python3 gen_uart.py > uart.log   (fixed seed, byte-stable)
"""
import random

rng = random.Random(0xA7A3007)  # fixed seed — byte-stable output

T_W = 2  # ms per window line
N_LINES = 3 * 3600 * 1000 // T_W  # 5,400,000
BURST_PERIOD_MS = 5000
BURST_LEN_LINES = 200          # 100 ms @ 2 kHz status burst
RING_CAP = 64
GAP_SPAN = (12, 19)            # dropped consecutive lines per overflow

def temp_at(ts_ms):
    base = 36.8 + (ts_ms / (3 * 3600 * 1000)) * 1.1  # 36.8 -> 37.9 C
    wave = 0.05 * math.sin(ts_ms / 900000.0)
    return base + wave

import math

# ---- build the gap manifest (same burst/gap model every cycle ----
# but make gap sizes deterministic per burst index, and the FASTEST
# sampling on the continuing window stream.
gaps = []   # (burst_idx, gap_start_seq, gap_size)
next_gap_seq = 0

def seq_at(ts_ms):
    return ts_ms // T_W + 1

# Precompute gap events: a burst starts every 5000 ms. The window stream is
# continuous; the ring overflows at burst start when the ring is full (it is
# full whenever the service has been writing >= RING_CAP lines since any
# drain — i.e. essentially always at burst onset). Drop span is per-burst.
burst_ts = []
t = BURST_PERIOD_MS
while t <= 3 * 3600 * 1000:
    burst_ts.append(t)
    t += BURST_PERIOD_MS

gap_events = []  # (gap_start_seq, gap_size)
prev = 0
for bi, bt in enumerate(burst_ts):
    if bi % 2 == 0:  # overflow only on alternate bursts (ring drains between)
        gap_size = rng.randint(*GAP_SPAN)
        gs = seq_at(bt)
        gap_events.append((gs, gap_size))

# ---- emit excerpt ----
out = []
out.append("# AVV-3000 pump telemetry — uart.log (3 h capture, affected unit, EXCERPT)")
out.append("# Format: ts_ms,seq,t_c,ch0_mmhg,ch1_mmhg,flags  (flags O=occl_det A=alarm H=htemp L=ltemp)")
out.append("# Full capture = 1 line per SDR window (2 ms): 5,400,000 lines / 3 h.")
out.append("# This file keeps: (a) one sample line per 60 s, (b) 5-line windows")
out.append("# around a representative subset of ring-gap events (seq holes),")
out.append("# (c) the aggregate gap summary at the end. Body lines are verbatim;")
out.append("# only the selection is editorial.")
out.append("ts_ms,seq,t_c,ch0_mmhg,ch1_mmhg,flags")

def line(ts_ms):
    s = seq_at(ts_ms)
    t = temp_at(ts_ms)
    ch0 = 41 if ts_ms < 20_000 else (42 + (ts_ms // 200_000) % 4)
    ch1 = ch0 + 1
    return f"{ts_ms},{s},{t:.1f},{ch0},{ch1},"

# sample lines: one per 60 s
for ts in range(120, 3 * 3600 * 1000, 60_000):
    out.append(line(ts))

# gap windows: 5 lines around every 25th gap (2 before, hole, 2 after)
for gi, (gs, gsz) in enumerate(gap_events):
    if gi % 25 != 0:
        continue
    ts0 = (gs - 1) * T_W
    out.append("# -- gap: seq %d..%d missing (ring overflow under 2 Hz burst) --" % (gs, gs + gsz - 1))
    for k in range(2):
        out.append(line(ts0 - (2 - k) * T_W))
    for k in range(2):
        out.append(line(ts0 + (gsz + k) * T_W))

# ---- aggregate gap summary over the FULL model ----
out.append("")
out.append("# ---- full-capture gap summary (3 h) ----")
out.append(f"# gap events: {len(gap_events)} total, all inside 2 Hz status-burst minutes")
out.append(f"# missing lines total: {sum(s for _, s in gap_events)}")
out.append("# size distribution (missing seq lines per gap event):")
dist = {}
for _, s in gap_events:
    dist[s] = dist.get(s, 0) + 1
for s in sorted(dist):
    out.append(f"#   {s} missing: {dist[s]} events")
out.append("# gaps in NON-burst minutes: 0  => ring overflow is burst-correlated")

print("\n".join(out))