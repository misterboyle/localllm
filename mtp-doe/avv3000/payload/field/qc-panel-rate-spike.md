# QC Panel Rate Anomaly — "×1000 for two seconds" (segregated sub-ticket)

**Part of:** AVV-FT-4471 evidence set, explicitly split from the alarm
root-cause hunt on 2026-05-26 (triage thread 6). Reproduces on the bench
at 22 °C on demand. The alarm does NOT. Different cause, different fix,
different acceptance; the ONLY reason these share a document folder is
that both were witnessed during one infusion.

## 1. What the operators see

During high-activity moments (the 2 Hz status-burst storms at pump-head
temperature, or a bench-replayed equivalent), the QC panel's infusion
rate readout snaps to ~1000× the true value for about two seconds, then
falls back to truth. No alarm state. No flag column set on the affected
display lines. The panel never shows a rate between "true" and "true ×
1000" — the spike is an all-or-nothing presentation event, and its
duration matches the panel's own redraw/age window, not any plant
process.

## 2. What the panel is supposed to show (this is specified twice)

The engine and panel internals run the per-ms cadence domain; the
display unit is counts/second. The app note's QC-panel section states
the whole rule, and it is one line: display value = (counts per ms) ×
1000, applied EXACTLY ONCE, on the way out of the panel. The unit-scale
rule exists because the panel's internal sample accounting already
counts in the ms domain; a display path that "converts again," or that
accepts an upstream value that has already been scaled, shows counts/s
× 1000 while every internal check on the same numbers stays green. That
is the entire symptom class, and it is exactly why this anomaly is a
presentation-layer bug and not a data bug: the wire records are correct
(see §4), the console stream is correct, and the number on the wall is
off by an order-cubed factor at moments when the panel refresh path
re-derives its scale factor from... which is the code walk to do.

## 3. Why "it co-occurs with the red-alarm event" is contaminated evidence

Both the alarm path and the panel render path drain the SAME telemetry
console feed, and the telemetry ring under status burst is dropping and
reordering presentation samples on a schedule (separate fault — see the
ring report, on this ticket's shelf). During a burst storm the panel's
input stream is visibly chunked: a two-second window of missing or
duplicated arrival timestamps is precisely the trigger shape for a scale
factor recomputed across a discontinuity (the classic: "compute rate
from count delta / timestamp delta" — timestamps stall, count keeps,
delta-t below the guard, scale multiplies; two seconds later the age
window re-seeds and truth returns). Whether the specific mechanism is
that one does not matter for the segregation call: a defect that
reproduces instantly at 22 °C with a loaded console feed CANNOT share a
root cause with an alarm the same cold, loaded bench could not produce
in 96 h.

## 4. What is NOT wrong (keep these people honest)

- **The wire is honest.** Every record on `uart.log` during captured
  spikes carries sane per-window counts; integrating the trace over the
  spike window gives the true volume. Nobody should use the panel
  screenshot as evidence of over-pressure, and no pressure trace in any
  captured spike comes anywhere near the 960 LSB occlusion trip. (The
  ticket mentions the spike with the alarm because the operators saw
  both. The operator report is honest; the inference is where it dies.)
- **The engine cadence is honest.** ×1000 is not a missing clock or a
  prescaler glitch — the WAVE_DATA staircase under the panel keeps
  app-note cadence through the event (its own checks hold; plateau math
  unaffected).
- **The ADC chain is honest** — same argument as §2 of the triage notes
  two orders of magnitude in the other direction.

## 5. Acceptance asks for whoever fixes this

1. Property-style unit check the app note already implies (its §5 is
   the normative text): feed a synthetic constant-rate count sequence
   in the ms domain, assert displayed value == true counts/second —
   AND feed a pre-scaled sequence, assert it is NOT scaled again. The
   ×1000-multiply guard has to run OFF-SCREEN (headless), because the
   bug class lives in the presentation conversion and the acceptance
   must not depend on a display existing.
2. One test pins "no intermediate values": a spike is scale doubling,
   not noise; partial-scale readings in a test trace mean someone
   "fixed" the symptom with a clamp and made the anomaly a stealth
   anomaly.
3. Re-run the spike repro with a guarded telemetry ring (§ the ring
   report) once that exists: the spike should PERSIST until its own fix
   lands. If it "magically" disappears with the ring fix, the repro is
   not pinned and nobody has actually fixed the scale path.
