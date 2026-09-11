# Triage Notes — AVV-FT-4471 working session (NV-2 site + bench link)

**Date:** 2026-05-26 (second week on the ticket)
**Attendees:** R. Okonjo (tier-2 FS), K. Vestergaard (bench), D. Marchetti
(device eng, on video), two more notebooks than coffee.
**Raw notes — chronological, conclusions at the end. Not polished on
purpose: several of us keep mis-quoting the same numbers in the same
wrong ways, so the mis-quotations are recorded here next to the
corrections.**

---

## Where we started

Ticket text says: ~1-in-5000 infusion sessions, RED occlusion alarm with
no pressure event, affected deployments run warm (head 37 °C class),
bench at 22 °C cannot reproduce in 96 h including deliberate threshold
crossings. Attached at session start: uart.log, original 2-minute i2c
excerpt, watchdog histogram, thermal derating csv.

## Thread 1 — "It's gain drift, obviously" (DEAD, keep this note)

The thermal csv is real and it is scary-looking at first glance: gain
error −0.36 % at 37 °C, −3.4 % at 75 °C, margin column going negative at
75 °C. Marchetti's first read was "the compare margin is collapsing at
body temperature, swap the ASIC lot, done."

It died in six minutes because it dies WITHOUT anyone opening the
silicon. The margin column is not gain — it is the compare margin under
the 6σ rule, and it is computed IN THE CSV (margin = 960 − 6·noise; the
sheet says so, and the arithmetic checks out row per row — Vestergaard
verified all rows by hand and by script). At 37 °C the margin is 864
LSB. The risk zone the sheet itself defines is "margin < ~100 LSB".
Two orders of magnitude to spare. Full write-up with the arithmetic
laid out for the review board: `thermal-analysis-appendix.md`; the
datasheet also does the same kill in one line of its own Appendix E.2,
which is where it should have been looked at first.

Mis-quoted number of the session (round one): "-3.4 % at 75 °C means
the alarm trips early at 37 °C." No. It means at 75 °C you have
ACCURACY problems — and at 75 °C you also have ALARM_HTEMP running
first (datasheet §8.3 thermal path), which the csv's own note says.
Thermal derating is a decoy on this ticket. It is not nothing: it is
just not a false-alarm budget at the ticket's temperature.

## Thread 2 — "Then noise floor says the unit is sick" (DEAD)

Unit 4471's noise row is 16 LSB at 37 °C vs the 4 LSB the csv shows at
22-25 °C — noisy-looking. Compared against what? The csv's rows are a
fleet summary INCLUDING this unit's operating point. And calibration:
`calibration-certificate-4471.txt` (attached later that evening) shows
this unit's 25 °C SNR floors in-line with lot survey
(`snr-lot-survey.csv`) — the 71/82 dB figures of the datasheet §9.2
spec hold on this unit with margin on both filter settings. The CH0
floor rising 4→16 LSB across 22→37 °C is the normal shape of that
column across temperature for the fleet — the "sick unit" reading was
a temperature bin confusion (bench numbers vs site numbers compared as
if same temperature).

## Thread 3 — the watchdog histogram (ALIVE, split into two claims)

Histogram: resets per temperature bin, 4 weeks, affected unit. Mass at
37–38 °C (11 resets vs ~1-2 baseline), sister units show the same shape
with the same image. Round two of mis-quotations, preserved verbatim
from the whiteboard:

- Marchetti: "resets correlate with temperature, so a thermal fault."
- Okonjo: "resets correlate with the 2 Hz burst, and temperature only
  changes HOW OFTEN the stall condition coincides with the burst."

Both statements are compatible and only ONE is testable on site: the
reset timestamps, when they exist, land inside 100 ms AFTER a burst
onset (histogram-file extension). Temperature-biased, burst-caused. The
drain-rate story (console writer slows when the app core thermally
throttles) explains the bias without any thermal fault in the signal
chain. The claim to test at next visit, not yet tested: after a
ring-admission fix, the 37 °C bin should empty WITHOUT touching the
watchdog timeout or the thermal policy.

## Thread 4 — the uart.log gaps (ALIVE — biggest thread, own report)

This is now `report-telemetry-ring-analysis.md`. 16,816 dropped window
records in 3 h; onsets mod 5000 all zero; alternate bursts; 3.0.4 image
clean (control capture attached); fleet-wide on 3.1.x/3.2.x images,
clean on held 3.0.x units. The room's consensus: the telemetry service
has no guard between its status burst and its security stream, and it
evicts under full-ring conditions at burst onsets. Nobody in the room
could name the exact policy from the artifact alone (evict-oldest vs
drop-newcomer); Okonjo's working model says evict-oldest because the
seq holes are in the WINDOW stream, not in the status lines, and the
status lines themselves are never missing from the file... which is
true but weak, because burst lines may simply not carry seq numbers.
Parking the policy question for the source walk. One thing everyone
agreed: as long as this is broken, "the alarm has no corresponding
pressure event" is not provable from telemetry, because whole windows
of the trace are vanishing on a schedule. Which is probably why the
ticket sat for 8 months. To be blunt about it, per Okonjo: the gaps
make the ALARM ticket un-debuggable from telemetry and the ALARM ticket
is the reason anyone looked at telemetry. They fed each other.

## Thread 5 — the double STAT read (ALIVE — from the pcap, quiet killer)

Vestergaard's bench rig recorded the new captures while the site capture
from 2026-05-19 was still running on loop. Side-by-side the difference
is trivial to see and infuriating: bench controller — one STAT read per
window; site controller — two, ~1 ms apart, first returns DRDY, second
returns the post-consumption picture. It is a Rev-A read pattern; the
Rev-B protocol changed when the reads became destructive (ES-2.1.3 in
the errata sheet; datasheet §7.4.2 documents the supersession). The
site host stack is old code that got carried at re-imaging; the bench
never ran it, which is precisely why 96 h of bench "proved" the part
innocent. The violation counters on the affected unit are the smoking
fingerprint — ERR_CNT walking, ERR_STAT sticky STAT_VIOLATION — the
part is LOGGING the driver at us, in-band.

Why 1-in-5000 still needs an answer, though: if every window is double-
read, every window is violated, which is 500/s, not 1-in-5000. Nobody
has that today. Two candidate reconciliations were argued and neither
survived to morning as a statement: (a) the alarm-visible violations
are only the ones that land adjacent to ring-blocked console writes
(then it's the service's again — see thread 4), (b) only some windows
feed BOTH the alarm evidence path and the console under load (timing
coincidences again). Marchetti's note: "1-in-N arithmetic is a DUTY
CYCLE statement; the errata compatibility appendix does exactly that
duty-cycle math — read Appendix J.2 before proposing anything." This
stays open with the number attached.

## Thread 6 — the ×1000 rate spike (SEGREGATED)

The "×1000 rate" panel anomaly had its own five minutes and then got
quarantined on purpose. It reproduces at 22 °C on the bench on demand,
which already proves it is not sharing a cause with an alarm that 96 h
cannot reproduce there. It smells like a display-layer scale applied
twice (or applied where the panel's internal units already ARE the per-
millisecond engine cadence times 1000). App note says once, datasheet
says what the panel shows, the QC-panel artifact `qc-panel-rate-spike.md`
collects the traces. Separate ticket class. If someone says "the spike
and the alarm co-occur" — of course they co-occur, both ride the same
console feed through the ring under thread 4's storm; co-occurrence is
contaminated by the shared path.

## Round-table conclusions (2026-05-26, 19:40)

1. Thermal accuracy: CLOSED as a decoy, with arithmetic (§4 of the
   appendix is the kill chain). Keep the csv on the ticket as
   DISPROVING evidence, not supporting.
2. Unit hardware / lot: CLOSED — calibration and lot survey in-line.
3. Three live threads, one ticket each, no merging: ring admission
   (telemetry losses), read-path contract (destructive reads /
   violation counters), display scale (×1000). Keep separate root
   causes, separate fixes, separate acceptance.
4. Everything that says "reproduce at 37 °C" from now on goes through
   `bench-repro-protocol.md` (loaded console or not, burst cadence on
   or off — the matrix is IN the protocol; a cold unloaded bench result
   means NOTHING on this ticket and has already cost the ticket 8
   months).
5. Before the next site visit: read the errata sheet BEFORE the
   datasheet section people remember. Twice this week the fix proposed
   was something the silicon already does differently by documented
   design (once §7.4.2, once the mid-cycle HOLD behaviour).
6. Do not let anyone close FT-4471 when the gaps stop. The gaps are one
   of three faults and the loudest, not the one that flashes red.
