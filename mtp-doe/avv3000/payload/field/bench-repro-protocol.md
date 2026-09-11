# Service Bulletin SB-012 — Bench Reproduction Protocol for the FT-4471 Family

**Supersedes:** the "96 h continuous run at 22 °C with deliberate
threshold crossings" protocol, which produced a false acquittal and is
retired with prejudice.
**Why this exists:** three lab reports in a row said "cannot reproduce"
with a clean conscience, and each one was executing a protocol that
structurally excludes one of the two necessary conditions. This document
is the minimum matrix; partial runs get partial conclusions, per the
verdict table.
**Applies to:** AVV-3000 pumps on images 3.1.0 through 3.2.1 presenting
intermittent red-safelight reports and/or telemetry sequence gaps.

## 0. Preconditions (all of them; note every "obvious" one bit us)

- Pump head instrumented; head temperature (not ambient) logged at the
  controller's own thermistor channel, ±0.5 °C.
- Console attached as AT THE SITE: TTY to the station harness with the
  production console consumer running. No "we'll capture to file from
  the debug port instead" — the ring that drops records sits behind the
  production consumer path; bench without it is a different system,
  full stop.
- Image identity recorded from the boot banner (below the fold in
  `console-boot-3.2.1.log`), not from the asset tag.
- Service status burst ENABLED (it is on by default in the images under
  test; if it is off, nothing here works — confirm in the boot echo).
- Telemetry self-report counters at zero: ring high-water, rejected,
  evicted, window-overwrite. If they don't exist on the build, this is
  the protocol where you ask for them (§4).

## 1. The matrix

Run all four arms for 3 h each, no shorter. Temperature is the head
temperature held ±1 °C; the console load arm is: production consumer
ON (loaded) vs TTY open but consumer idle (unloaded — the console keeps
working, the ring still serves the window stream, but the burst path is
starved of the read-allocate race).

| Arm | Head T | Console | Watch for |
|---|---|---|---|
| A | 22 °C | loaded | sequence gaps at burst instants (expect PRESENT — 12..19-line holes, onsets on alternate 5 s bursts); watchdog resets (expect rare/none) |
| B | 22 °C | unloaded | gaps (expect absent — if present here, the model from the ring report is wrong, escalate) |
| C | 37 °C | loaded | gaps AS ARM A exactly; plus: watchdog resets begin clustering within 100 ms after burst onsets; IF the unit also carries the Rev-A read-path driver, this is the arm that can host a false-alarm-class latch event |
| D | 37 °C | unloaded | gaps absent; resets absent — the warm-but-unloaded arm isolates temperature from load at constant temperature |

Every arm: dump the console capture with the seq-hole summary appended
(per `uart.log`'s format), dump svc counters per arm-end, and archive
per-arm. Per-arm, not per-session: arm-to-arm leakage in lab notes is
where the earlier false acquittals were manufactured ("we ran warm and
saw nothing" — which arm was unloaded, though?).

## 2. Reading the gaps, if you are new to this

A capture is a line per 2 ms window with `seq == ts/2 + 1`. Gaps are
seq holes with the ts continuing. Quick pass:

    awk -F, 'NF>4 && p && $2>p+1 {print NR": missing "p+1".."$2-1} {p=$2}' capture.log

If your "gaps" are not 12..19 lines, do not paste them into the ticket
without checking they are not log rotation, TTY disconnects, or the
capture script's own buffering — four field "gap" reports since April
had three of those. Legit signatures begin at burst onsets (ts mod 5000
== 0) on alternate bursts; anything else is a different animal and its
own ticket.

## 3. What each verdict means

- **A gaps, B clean, C gaps + reset clustering, D clean:** the expected
  picture — shared-ring admission fault with drain-rate biasing its
  consequences. Do NOT also conclude anything about the alarm itself
  from these four arms; it is separately visible only via the
  read-path counters (§4).
- **A clean:** wrong build or arm not loaded as specified. Re-run; if
  again clean with counters proving burst emission off, the unit is
  outside the family and the ticket report that routed it here is wrong.
- **B or D with gaps:** escalate — load-independence falsifies the
  working model and opens the transport/UART possibility this protocol
  was partly written to exclude.
- **C with latch events:** snapshot ERR_STAT + ALARM_STATUS + the raw
  window record for the whole debounce run and file with engineering
  before powering anything down; that snapshot either contains the
  alarm or indicts the evidence path.

## 4. The counters (the ask that makes arms unnecessary at next image)

Instrument and echo at boot and on demand: burst count emitted; ring
high-water; records rejected (and by class); records evicted (and by
class — evicting STATUS is policy-correct, evicting a WINDOW record is
the violation class everyone is histogramming around); window records
dropped at admission. With those live, one 10-minute arm-C run replaces
a week of histogram archaeology, and the fix's acceptance test
(unguarded → guarded) is a delta of counters rather than a diff of
captures. The field does not get to specify the design; it gets to
insist that a bug that can only be found by three hours of arithmetic
on a log becomes a bug that announces itself in a register.

## 5. What NOT to conclude from a protocol pass

- A clean arm-D does not clear the part. Nothing here ever indicts the
  part; the errata-compatibility behavior (destructive STAT reads,
  mid-cycle HOLD deferral) is documented design and stays.
- "Alarms did not fire during 3 h × 4 arms" — base rate of the visible
  symptom is ~1-in-5000 sessions; twelve hours of negative result at
  ANY temperature says almost nothing about the alarm and, per the
  appendix arithmetic, thermal accuracy said nothing before that. Only
  the fault-level counters close loops.
