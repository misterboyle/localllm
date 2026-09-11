# Thermal Analysis Appendix — closing the derating decoy on AVV-FT-4471

**Purpose:** write down, in one place, the arithmetic that closes the
"thermal gain drift causes the false occlusion alarms" theory, so it
closes ONCE. The ticket keeps collecting this theory because the thermal
csv LOOKS like a motive: errors grow with temperature, the margin column
goes negative at 75 °C, and the symptom is temperature-biased. The
numbers themselves — the exact rows of `thermal-derating.csv`, same
table, same units — show the theory cannot reach the symptom at the
ticket's operating point. Datasheet Appendix E.2 performs this kill in
one table; this appendix is the slow version for reviewers.

## The rule the csv is built on

Quoting the csv header: margin = OCCL_TH (960 LSB) − 6·σ_window(CH0).
Three facts make this the right lens:

1. The occlusion compare fires on the *averaged* CH0 window crossing the
   threshold, so its false-trip budget is the noise tail at the
   threshold, not the mean accuracy error. Gain error shifts the MEAN.
   A −0.36 % gain error at 37 °C moves a ~1000-count operating point by
   three and a half counts — three and a half LSB, not hundreds. Mean
   shifts are calibration business (and the offset column stays inside
   the datasheet's ±8 LSB after-cal window's neighbourhood — see
   §9.2).
2. The compare debounces over consecutive windows (§8.2; the OCCL_DEB
   encoding, reset 0xE, in Table 8-2). Independent noise samples
   ANDing over a run is where the 6σ shorthand comes from; a
   false-trip-likely margin is defined on the NOISE column, not on gain
   or offset columns.
3. Every row in the csv satisfies margin == 960 − 6·noise exactly — the
   column is arithmetic, not measurement. The measurements are the noise
   column. So "margin collapsing at 75 °C" is a paraphrase of "the noise
   column got big enough that the threshold sits under the 6σ horizon" —
   legitimate engineering concern at 75 °C, and at 75 °C only, where the
   thermal-alarm path (§8.3) is running first anyway (the csv's own
   75 °C annotation says so).

## The table that kills the theory

Rows quoted from `thermal-derating.csv` at the ticket's operating
temperature and neighbours; the margin column verbatim:

| T (°C) | noise (LSB) | 6σ | margin (LSB) | reading |
|---|---|---|---|---|
| 22 | 4 | 24 | 936 | quiet |
| 31 | 6 | 36 | 924 | quiet |
| 36 | 13 | 78 | 882 | quiet |
| **37** | **16** | **96** | **864** | **the ticket's row — quiet** |
| 42 | 32 | 192 | 768 | quiet |
| 60 | 110 | 660 | 300 | low but silent |
| 70 | 165 | 990 | −30 | risk zone (nobody's pump runs here) |

The rule-of-thumb cut the csv header uses: margin under ~100 LSB earns
intermittent-latch worry. 864 is that threshold's eight-and-a-half-fold
overachievement. For 37 °C gain/offset drift to open a false-trip path
that the noise analysis says is 864 LSB wide, the drift would have to be
non-monotonic, temperature-specific, and roughly 900 LSB at *one
channel only* — and the gain error row at other temperatures would then
have to lie. The fleet agrees with the shape: `snr-lot-survey.csv` has
lot floors matching the §9.2 spec figures (71/82 dB at the two filter
settings) across lots including this unit's (certificate attached to
the ticket). If thermally-induced accuracy error were the false-alarm
mechanism, the margin column would have to cross ~800 LSB between 36
and 37 °C. It crosses 18.

## What survives from the thermal thread (do not throw these out)

- **The temperature BIAS of the watchdog histogram is real data** — the
  extension bins at 36.5/37.5 and the sister-unit histograms. It is
  bias-not-cause: temperature modulates how often the console-writer
  stall coincides with burst onsets (drain rate under thermal throttle).
  The distinction only becomes visible once you have the burst-instant
  alignment (histogram extension note 1), and it is why the 96 h cold
  bench "proved" innocence: at 22 °C the coincidence rate collapses
  while the ring keeps dropping windows at exactly the same per-burst
  rate (that non-coincidence — gaps at 22 °C unchanged, resets at 22 °C
  near zero — is the single most underrated line of evidence on the
  ticket, and it lives in plain view: uart.log, every gap comment, every
  burst-aligned ts).
- **The 70–75 °C rows are genuine system business** — different product
  question (should a safelight pump ever be there, and does ALARM_HTEMP
  get there first), owned by the reliability doc, not this ticket. Keep
  the rows, they are load-bearing for a different argument.
- **"Thermal chamber at 37 °C WITH loaded console" is the repro
  condition that eventually broke the case open** — `bench-repro-protocol.md`
  is exactly that matrix: loaded vs unloaded console × 22 vs 37 °C. The
  thermal axis of that matrix is where the derating theory went to die
  in person: the loaded-cold arm reproduces the drops, the unloaded-warm
  arm reproduces neither symptom. Temperature is a rate knob on a
  software race; it is not in the causal chain at all.

## Review posture

If a new theory enters the ticket as "but the ASIC accuracy derates
with temperature," the disposition is: cite this appendix (or E.2 in
the datasheet, same numbers), cite the 37 °C row, ask for the 900-LSB
mechanism, close if none arrives. Theories that arrive instead as "the
margin math is the wrong model for correlated noise" deserve a hearing
— but note the debounce-and-averaging structure already cuts
correlation twice, and nobody has brought a correlated-noise trace that
comes anywhere near the threshold either (every captured CH0 in the
ring-report's windows sits at 1000±40 raw LSB with ±16 LSB-class noise
— the trace simply never contains a thousand-count excursion, least of
all at the alarm instants, where the only thing near the threshold is
... nothing. That part of this ticket remains exactly as unproven as it
was, and structurally un-provable, until the read-path and ring faults
stop eating the evidence — which is the actual thermal-free punchline
of the whole thermal thread).
