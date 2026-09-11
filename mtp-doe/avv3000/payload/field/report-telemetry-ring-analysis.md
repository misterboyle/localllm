# Field Engineering Report — Telemetry Window Drops, Site NV-2

**Document:** FN-RPT-2026-0531-A
**Relates to:** field ticket AVV-FT-4471 (`TICKET.md`), unit AVV3-4471
**Author:** R. Okonjo, tier-2 field service, with bench support (K. Vestergaard)
**Status:** evidence complete; root cause NOT declared — engineering owns that

---

## 1. What this report adds to the ticket

The ticket (opened 2026-05-19) is about red-safelight false alarms at
clinical temperature. This report is about the second thing that fell out
of the evidence: the pump telemetry console on the affected unit has been
silently eating pressure records since its first boot after the 3.1.0
image step — 12 to 19 records every 10 seconds like clockwork, ~1.5
records/second averaged. That is 16,816 lost window records in the three
hours of `uart.log` alone (1,080 gap events, 360 in each hour-window,
sizes 12..19 lines). Nobody flagged it as a sensor problem.
It is very likely not one. It is a service problem with a clean,
temperature-independent signature, and until it is fixed, every field
trace we take of the alarm ticket is missing evidence by construction.

## 2. The capture and its arithmetic

The console line format is documented in the service banner (echoed at
boot, see `console-boot-3.2.1.log`):

    ts_ms,seq,t_c,ch0_mmhg,ch1_mmhg,flags

where `seq` is the 1-based SDR window sequence: the acquisition engine
free-runs a 2 ms window (datasheet §7.2 step 3 reference setup, CFG =
0x0000_0049 → `SDR_WINDOW=1`), so a healthy console emits exactly one
line every 2 ms, 500 lines per second, and `seq == ts_ms/2 + 1` for
every line. A ring drop therefore shows as a `seq` hole with the wall
clock still advancing — exactly what `uart.log` shows. The excerpt keeps
5-line windows around one gap in four; the two lines before a hole and
the two after it always bracket the missing block cleanly:

    ...
    4996,2499,36.8,41,42,
    4998,2500,36.8,41,42,
    (2501..2518 never emitted)
    5036,2519,36.8,41,42,
    5038,2520,36.8,41,42,

The first gap sits at seq 2501 — that is, 5.0 s after boot, which is
when the FIRST status burst of the session fires. There is no warm-up.
The drops are as old as the feature.

Two things were true in lockstep across all three hours, and they are the
entire basis of the diagnosis:

1. **Every gap begins at a burst onset.** The service emits its 100 ms
   status burst every 5000 ms at 2 kHz — two status records per
   millisecond, 200 per burst, interleaved into the same ring as the
   window stream. Take any gap onset ts_ms mod 5000: it is zero, every
   time, 1,080 of 1,080 — gaps begin at burst onsets and only ever at
   the every-OTHER burst that actually overflows (see §5).
2. **No gap has ever happened outside a burst.** Zero sequence holes
   in non-burst minutes across the entire capture, and zero across the
   full 96 h bench run log (not attached; the bench attachment simply
   never showed a hole big enough to notice — the gap per burst is so
   regular it only becomes visible in aggregate, which is why the
   summary histogram in `uart.log` is the artifact that cracked this).

## 3. What we ruled out, in order, with the evidence that killed each

- **UART bandwidth / console flow control.** A 2 kHz burst is 2,400
  additional chars/s on top of 500 lines/s ≈ 6 chars each: well inside
  the 115200 console budget, and flow control would show gradual lag
  (ts deltas widening) BEFORE any hole. Ts deltas before and after every
  examined gap are exactly 2 ms; the stream has no lag, it has holes.
- **Flash write wear / logfs hiccup** (first theory on the ticket).
  The console is a serial TTY to the station harness, not to storage;
  plus: fsync stalls are log-normal and cluster sporadically, these do
  not — 1,080 events in exactly three hours, mean 10 s spacing ± 0
  (every other burst), size distribution flat over 12..19
  (119/126/141/150/143/116/149/136 for 12..19). A filesystem does not
  fail that periodically. A counter with an off-by-something does.
- **Hardware (UART FIFO, DMA) failure on the affected unit.** Ruled out
  by `uart-image-304.log`: same slot, same cable, same station, image
  3.0.4, three hours — zero holes. The slot re-imaged 3.0.4 → 3.1.2 →
  3.2.1 with no hardware change; the signature appears exactly at the
  3.1.0 step that introduced the burst. Also ruled out by
  `fleet-gap-survey.csv`: 10 units of the same image family at two
  sites, all with the identical 12..19-hole signature; the two units
  still parked at 3.0.x images: spotless. Fleet behaviour is software.
- **"It's just the status burst being logged and it doesn't matter."**
  It matters twice. First, the dropped records are WINDOW records —
  pressure/flag records, the security stream (alarm evidence lives in
  those lines). Status records are cheap; if anything is to be shed
  under pressure it has to be status. Second, the stall that coincides
  with the ring-full state is the same stall that correlates with the
  watchdog-reset histogram (`watchdog-histogram.txt`); the reset stream
  and the gap stream have never been observed apart.

## 4. Working model of the broken admission path

No service source was available in the field; the model below is inferred
from the boot echo (`console-boot-3.2.1.log`), the burst geometry, and
the hole statistics. Engineering: treat as a hypothesis to confirm
against the source, not as a given.

The service keeps ONE bounded ring (boot echo: 64 records) feeding one
console writer. Two producers push into it: the window stream (500/s)
and the status burst (2,000/s for 100 ms every 5 s). If admission is
unguarded FIFO ("append while count < capacity, drop the NEWCOMER when
full"), the burst behaves like this:

- Writer drains R records/ms. Between bursts the ring hovers near full
  because window arrivals (500/s = 0.5/ms) roughly balance a slow drain;
  any slack closes.
- At burst onset, 200 status records arrive over 100 ms, at 2/ms. Every
  arrival that finds the ring full is dropped — silently — while the
  writer keeps eating window records. The ring never refills with
  status; it just reorders nobody.
- The observed hole in the SEQ numbers, though, is in the WINDOW stream,
  not the status stream. That means the pre-fix ring is NOT dropping the
  newcomer; it is EVICTING on full — overwriting the oldest record to
  preserve FIFO order — and at burst onset the oldest records are window
  records. A 100 ms burst at writer-limited throughput therefore
  overwrites 12..19 window records depending on how much the writer
  drained during the burst; the flat size distribution then follows from
  the drain rate dithering (the ring drains a hair under 2 records/ms at
  head temperature and the burst is exactly as long as the ring is
  deep). This all lines up with the ticket observation that at 22 °C the
  bench saw gaps in aggregate but almost no watchdog resets (faster
  drain, shorter stalls).

Under that model the FIX SHAPE is obvious but — deliberately — not
specified here: guard the ring so that status traffic can never consume
the capacity the window stream owns, and never evict a window record to
admit anything, with counters on both decisions so the field can verify
the fix by counting, not by histogram archaeology.

## 5. Why every other burst (and not all of them)

Gap onsets sit exactly 10,000 ms apart — every SECOND burst overflows,
the intervening ones survive. That alternation must fall out of the
arithmetic of whatever the real admission policy is (burst length ×
status rate versus drain rate, with the ring only recovering between
bursts every other cycle); the 12..19 size spread is then the fractional
part that drifts with console pacing and the 1 kHz tick. Nobody has
produced that arithmetic from the source yet. Whoever confirms §4 writes
it down, because the alternation is the single most diagnostic fact in
the capture — it says the ring is *full with a period*, and full-with-a-
period means the admission path, not the transport path, is where the
loss lives.

## 6. Relationship to the watchdog and to the alarm ticket

Three separate things are in motion and must not be merged:

1. **The ring drops window records at burst instants** (this report).
   Root cause is the shared-ring admission policy. Fixable statically.
2. **The watchdog resets** (`watchdog-histogram.txt`): the controller
   petting path lives on the same core that the console writer blocks;
   when the writer stalls long enough (rings full + flash-sync
   coincidence) the watchdog fires. The temperature bias in the
   histogram is drain-rate bias, not sensor bias. Fixing the admission
   policy should shrink the histograms at 37 °C without moving the
   watchdog timeout; if it does not, that is a second defect, and its
   own ticket.
3. **The red-safelight alarm itself** (the ticket): per `i2c.pcap.txt`
   capture-3 notes, the latch event is silicon-honest; what is missing
   when we look at the host record is missing because of (the
   driver's) double STAT reads eating evidence and (this report's)
   window drops losing the neighbours. This ring fix, alone, does not
   fix the alarm ticket's visibility by itself — read
   `i2c-annotation-4471.md` before declaring the ticket closed when the
   gaps stop.

## 7. Asks / next steps

- Confirm §4's admission model against the service source and the boot
  defaults (ring capacity and burst cadence appear in the boot echo;
  drain rate does not — instrument it).
- Acceptance must be replay-shaped, not histogram-shaped: take the
  affected unit's capture profile (500 windows/s + 2 kHz bursts, full
  ring before each burst) and require zero window losses by COUNTER, in
  a fixed-seed test, at whatever burst density we like — with the
  eviction and rejection path each explicitly counted and each pinned
  to the right class (status evictable, window sacred).
- Re-run the alarm-ticket bench protocol after the ring fix; false
  alarms should persist (they are the other two faults) — state this
  up front to the clinic so a partial improvement is not mis-sold.

Attachments referenced: `uart.log`, `uart-image-304.log`,
`fleet-gap-survey.csv`, `watchdog-histogram.txt`,
`console-boot-3.2.1.log`, `service-visit-log.csv`.
