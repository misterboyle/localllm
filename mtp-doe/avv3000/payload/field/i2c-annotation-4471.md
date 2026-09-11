# I2C Capture Annotations — Unit AVV3-4471, ticket AVV-FT-4471

**File pair:** `i2c.pcap.txt` (raw excerpts, three captures) + this
annotation set. Read the pcap header first; it defines the line format.
This document is the reading guide field service uses before asking
engineering questions, because two of the three "mysteries" in the trace
have documented answers a reader skips at their peril.

**Register shorthand** (datasheet §6): 0x00 CTRL, 0x04 CFG, 0x08 STAT,
0x0C ALARM_CFG, 0x10 ALARM_STATUS, 0x20 ERR_STAT, 0x24 IRQ_MASK,
0x28 TH_RATE, 0x2C SAFETY_GATE, 0x40.. data port (window results, one
32-bit result per channel per window, read via the I2C host bridge here;
the AXI side is the native path).

---

## 1. The transaction idioms, annotated

Capture 1 (site unit, production image) opens with a four-write init:

    0,W,0x0C:0x01      ALARM_CFG enable-style write (windowed low byte)
    1,W,0x04:0x49      CFG = 0x49   <- the Rev-B init literal, datasheet §7.2
    2,W,0x0C:0x00
    3,W,0x24:0x1F      IRQ_MASK = 0x1F (all alarm sources, DRDY IRQ off)

Then the steady loop, every 2010-ish ms — and THIS is the whole ballgame
for capture 1:

    2010,R,0x08:0x01   STAT read #1: DRDY set — window ready
    2011,R,0x08:0x00   STAT read #2: DRDY GONE
    2012,R,0x40:0x2A   data CH0
    2013,R,0x48:0x3C   data CH1

Read #2 returning 0x00 is not the part misbehaving. STAT reads are
destructive-by-design on Rev B: the FIRST read observing DRDY clears it
(ES-2026-04 ES-2.1.3; §6.9 note). The second read is the Rev-A
"confirm" pattern — on Rev A it was harmless; on Rev B it (a) proves the
flag was consumed, (b) sets STAT_VIOLATION in ERR_STAT, and (c) walks
ERR_CNT [7:4] in STAT, which then self-reports in every later read
until ERR_STAT is read-to-cleared. Watch capture 1's tail: 0x3B =
OCCL_DET + DRDY + ERR_CNT=3. The part counted three violations, on a
unit whose "evidence integrity" the entire ticket is about.

Two lines in the loop are NOT double reads and should not be annotated
as such (we did, briefly):

    26010,R,0x08:0x02   DRDY not posted (BUSY), window skipped
    44010,R,0x08:0x02   same

Those windows are skipped cleanly — the data port is not touched when
DRDY wasn't seen. Skipping is fine; *confirmation reading* is the bug.

Capture 2 is the bench controller on the same silicon: one STAT, then
straight to data. It reads like the Rev-B datasheet. It IS the Rev-B
datasheet. Capture 3 is the site unit again, later, during two actual
alarm events; its notes are inline, the summary is below.

## 2. What the site driver believes vs what the silicon does

Belief (Rev-A era code, still shipping in the site host stack):

1. Poll STAT until DRDY.
2. Read STAT AGAIN to "make sure the flag is stable before taking the
   data" — because on Rev A, noise on the ready path was real.
3. Read data.

Rev B reality: step 2 consumes the DRDY event a second time (no-ops as
"DRDY gone"), latches STAT_VIOLATION, and increments ERR_CNT. The
datasheet's §7.4.2 is titled "Two-read STAT pattern (Rev-A legacy —
superseded)" and the errata carries the teeth. The failure signature of
the pattern is *not* "data are wrong" — the data port is a separate
windowed register and reads fine. The failure signature is *event
visibility*: the record that a given window happened, with the flags it
had AT the ready edge, is gone from a second read's perspective and —
combined with the service's ring losses (separate fault, own report) —
gone from the console log too. When the field says "the alarm has no
corresponding pressure event", this is the mechanism that makes that
sentence unprovable: the corresponding event may have existed, been
real, and been eaten.

Do not "fix" this in the part. Errata compatibility appendix G exists so
that nobody proposes silencing the counters; the violation reporting is
the DESIGN working.

## 3. The alarm events in capture 3, one at a time

180010..180030 (event A): STAT 0x35 (DRDY + OCCL_DET + ERR_CNT 3),
second read 0x34 (DRDY eaten, exactly capturing the violation on the
worst possible window), ERR_STAT 0x02, ALARM_STATUS 0x21 (OCCL +
LATCHED), operator resets via CTRL.ALARM_CLR, ALARM_STATUS falls to
0x00, engine keeps sampling. The data-port pairs around it show no
step — the controller's recorded picture does not contain anything an
occlusion is visible in.

That sentence needs to be read three times before it is quoted: **it is
NOT evidence the alarm was false.** The debounce geometry (§8.2, and
the reset OCCL_DEB=0xE row of Table 8-2) requires a run of consecutive
over-threshold windows before a latch; windows in exactly that run are
what the double-read + ring-eviction pair deletes, as surely as if the
log rotation ate the file. Absence of the record is the predicted
CONSEQUENCE of these faults, not a refutation of the alarm. Capture 3
shows silicon doing its job around a latch and the host being structurally
unable to see why.

## 4. Open question we are NOT closing in the field — the 1-in-5000

If every ready window is double-read, the violation rate is 500/s and
the unit should red-flag constantly. It does not: ~1-in-5000 sessions.
The reconciling arithmetic is a DUTY-CYCLE statement and is done in the
datasheet's own Appendix J.2 ("The 1-in-N arithmetic"), not here; two
things follow from it that the field can verify: (a) ERR_CNT walking
without an alarm is the NORMAL state on this build — the visible defect
needs a coincidence, not a rate; (b) any proposed "fix" that reduces
violation counts but leaves even one STAT re-read path in the driver
chain keeps (a) true and the ticket open. The bench experiment (noted,
not run at this visit): with a single-read controller and the site's
service, does the 37 °C reset histogram move? with a single-read
controller AND a guarded ring? that pair of runs separates the read-
path fault from the ring fault with no new instrumentation. Do it
before proposing anything to the clinic.

## 5. Annotation discipline for whoever extends this file

- Cite the register from §6 with its field bits, not with a byte value
  and vibes. 0x35 is four facts, write all four.
- A capture line never lies; annotations do. Every inference in this
  file is labelled as one.
- If a later capture from a FIXED build exists, do not edit these
  notes; add capture 4 and date it. The sequence of the notes is part
  of the trail.
