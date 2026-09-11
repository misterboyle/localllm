# AVA-QC Stair-Step Waveform Patterns — App Note AN-8851-07

**Applies to:** ADS-8851 waveform engine (WAVE_CFG/WAVE_DATA/WAVE_STATUS),
AVV-3000 pump firmware (`fw/`), QC panel (`gui/`), plant simulator (`sim/`).
**Inherited feature:** AVA-QC is the AVV-2200→AVV-3000 carried-over infusion
quality-control pattern family (AVV-2200 register names and profile_a are
kept verbatim for cross-platform comparability).

---

## 1. Purpose

The AVA-QC (Aspiration Verification and Quality Control) staircase patterns
drive the AVV-3000 pump through a deterministic sequence of set-point steps so
that the QC panel and the closed-loop plant can verify: (a) the pump follows
its programmed profile, (b) pressure transients are clamped, and (c) occlusion
detection fires at the correct threshold. The pattern family is a **staircase
with plateau holds** — the pump set-point steps by a fixed increment every
hold interval, holds, and (for loop profiles) steps back.

## 2. Pattern parameterization

Every pattern is a sequence of steps defined by:

| Parameter | Meaning | Units | Where it lives |
|---|---|---|---|
| `step` | set-point increment per plateau | LSB (16-bit span) | WAVE_CFG.STEP_LSB |
| `hold` | plateau duration | ms | WAVE_CFG.HOLD_MS |
| `count` | number of plateaus in the ramp | — | profile table (§3) |
| `dir` | ramp direction (up = infusion, down = withdrawal) | — | WAVE_CFG.INVERT |
| `cycle` | ramp repetitions before DONE | — | profile table (§3) |

The engine output (WAVE_DATA) starts at 0 (or the current value on restart)
and increments by `step` every `hold` ms until the ramp completes, then
returns toward the start at the same cadence (for up/down profiles) or sets
DONE (for one-way profiles).

## 3. Profile tables (definitive)

### Profile A — Inspection (pump priming verification, default)

Ramp up 10 steps, hold, ramp down 10 steps; repeat indefinitely.

| Field | Value |
|---|---|
| steps_up | 10 |
| steps_dn | 10 |
| hold_ms (default) | 500 ms |
| step_lsb (default) | 12 LSB |
| loop | yes |
| end | returns to 0, DONE pulses once per full cycle |

Equivalent WAVE_CFG (rev-B golden driver): `PROFILE=1, HOLD_MS=500,
STEP_LSB=12, INVERT=0`.

### Profile B — Flush (high-rate rinse)

Ramp up 8 steps, hold at top, ramp down 8 steps; loop.

| Field | Value |
|---|---|
| steps_up | 8 |
| hold_top_ms | 1200 ms |
| hold_ms (default) | 800 ms |
| step_lsb (default) | 20 LSB |
| loop | yes |

Equivalent WAVE_CFG: `PROFILE=2, HOLD_MS=800, STEP_LSB=20, INVERT=0`.

### Profile C — Prime (micro-staircase for line charging)

Ramp up 14 small steps; one-way; stops at top and sets DONE.

| Field | Value |
|---|---|
| steps_up | 14 |
| hold_ms (default) | 200 ms |
| step_lsb (default) | 4 LSB |
| loop | no |

Caution (ES-2026-04 ES-2.1.2): do not change HOLD_MS mid-cycle on profile C;
stop the engine first.

Equivalent WAVE_CFG: `PROFILE=3, HOLD_MS=200, STEP_LSB=4, INVERT=0`.

## 4. Closed-loop expectation (plant trace test)

The plant simulator (`sim/plant.c`) models pump → line compliance →
occlusion. When the waveform engine runs profile X, the CH0 pressure trace is
expected to follow the staircase: each plateau reached within 3 × plateau
time constant, monotonic within tolerance, peak ≤ clamp threshold. The
`fv_staircase` test (r_sim, 25 pts) compares the traced CH0 against the
profile's golden staircase with fixed seed and tolerance:

- plateau accuracy: ± 6 % of step
- rise time: reach 95 % of plateau within 2 × hold
- no overshoot beyond 10 % of step above the plateau target
- clamp: peak across the whole run ≤ clamp threshold (ALARM_CFG-derived)

## 5. QC panel expectation (unit-scale rule)

The QC panel (`gui/`) displays infusion rate as counts/sec. The panel's
internal units are counts/ms (engine cadence). Display conversion: rate_s =
rate_ms × 1000. The `t_gui_unit` test feeds synthetic count sequences and
asserts the displayed value equals the true counts/sec — a panel that
multiplies by 1000 again (or fails to convert) shows rates ×1000.

## 6. AVV-2200 comparability note

Profile A ("profile_a") register names and the inspection-pattern constants
are byte-identical to the AVV-2200 platform's implementation. Runs measured on
AVV-3000 profile A remain comparable to AVV-2200 baseline data for the
waveform rung.
## 7. Programming the engine — WAVE_CFG encodings, step by step

### 7.1 The register the patterns actually live in

The profile tables above (§3) carry semantic values (steps, ms, LSB).
The engine itself only ever sees one word: datasheet §6.13, WAVE_CFG:

| Bits | Field | Range | Notes |
|---|---|---|---|
| [1:0] | PROFILE | 0..3 | 0 = off; 1/2/3 = A/B/C |
| [15:2] | HOLD_MS | 1..16383 | plateau hold, milliseconds |
| [21:16] | STEP_LSB | 1..63 | step per hold tick, counts of the 16-bit span |
| [22] | INVERT | 0/1 | ramp direction (not a sign; see 7.5) |
| [31:23] | rsvd | 0 | write 0 |

Two structural rules:

- HOLD_MS is stored where an LSB-aligned field sits — the shift is 2, so
  writing an ms value UNshifted leaves ms/4 there and writing an engine
  tick count (ms × 10 000) runs straight through the 14-bit ceiling.
  The reference driver range-checks both (1..16383) and its argument
  check fires before the bus ever moves — a recipe that "configures
  fine" and then never forms plateaus is almost always rule one: an
  unshifted ms write leaving 125 ms in a 500 ms profile.
- The hold domain goes to 16 383 ms deliberately: the app-note defaults
  (500/800/200 ms) and the profile-B top dwell (1 200 ms) must all be
  writable with the payload and the register agreeing on ms units —
  the register was widened to follow this note, not the other way
  around (§16 revision history). The engine's own budget: one hold tick
  = HOLD_MS × 1 ms; at 10 MHz ACLK that is HOLD_MS × 10 000 clock ticks
  and the engine's tick counter is 28-bit wide — 16 383 × 10 000 fits
  with ~60 % to spare, and no legal hold can wrap it.

### 7.2 The three default recipes, as words

Per datasheet Table A-2 (same shift, same field positions; the
derivation column is the one to trust over any decimal-to-hex in prose):

| Profile | WAVE_CFG word | Derivation |
|---|---|---|
| A (500 ms, 12 LSB) | 0x000C_07D1 | 1 \| (500 << 2) \| (12 << 16) |
| B (800 ms, 20 LSB) | 0x0014_0C82 | 2 \| (800 << 2) \| (20 << 16) |
| C (200 ms, 4 LSB)  | 0x0004_0323 | 3 \| (200 << 2) \| (4 << 16)  |

A recipe transport (the QC-panel-to-controller WAVE_RECIPE frame) carries
hold as an UNSIGNED 16-bit little-endian field for exactly these ms
values, and the receiving driver re-encodes into WAVE_CFG — never pass
a wire-format hold through to the register untouched.

### 7.3 Arming, running, disarming

Order (datasheet A.5 is the register-level list; the rules that matter
for this note):

1. ADC_EN must already be set — the waveform engine is armed by
   CTRL.WF_EN (RUN = WF_EN & PROFILE ≠ 0). Arming order between
   ADC_EN/WF_EN has a documented history (§7.2 step 2 rationale in the
   datasheet); follow the reference init first, wave second.
2. Write WAVE_CFG (profile + hold + step + direction) — then set
   WF_EN. Writing PROFILE ≠ 0 while WF_EN is already set starts the
   engine from base immediately.
3. Poll WAVE_STATUS.RUN for confirmation (RUN=1); a DONE read is not a
   substitute for RUN — DONE says a cycle closed, not that the engine
   is armed.
4. To stop: PROFILE→0 (driver `wave_stop`). The output FREEZES AND
   RETAINS at its current value (datasheet C.1) — a stopped engine is
   the pump holding its last set-point, NOT a pump returning to zero.
   Any recipe that wants "back to base" after a profile must step the
   waveform there or command the plant explicitly; the host owns the
   actuator path.
5. Any write to WAVE_CFG clears DONE (any of them: hold-only rewrite,
   re-arm, disarm-via-profile-0). A test that reads DONE and then
   rewrites the recipe and reads DONE again expecting history is
   testing the wrong register for history (there is none).

### 7.4 Changing a running pattern

The configuration-change matrix is datasheet Table C-3; the three lines
a recipe author must know by heart:

- HOLD_MS written mid-cycle does NOT retune the running cycle. Loop
  profiles pick it up at the NEXT cycle start; one-way profiles (C)
  running to their end never apply it. The driver's
  `wave_set_hold` workaround (write with PROFILE=0, rewrite with the
  profile) exists precisely to make "apply now" explicit — that
  stop-rewrite-restart drops the current cycle and is the ES-2.1.2
  documented behaviour, not an apology for it.
- PROFILE→other restarts the engine from base with the new profile's
  constants — it is a stop and an arm in one word, not a smooth
  transition. Any QC flow "switching B to C mid-run" should be written
  as an explicit stop and start with the plant quiesced, even though
  one write can do it.
- A re-arm (PROFILE 0→x, or WF_EN off→on with PROFILE ≠ 0) restarts
  the cycle from tick 0, output from base. There is no resume-at-
  plateau mode.

### 7.5 INVERT is a direction, not a sign

STAIRCASE direction up = infusion-going steps, down = withdrawal-going
steps (INVERT=1 runs Table C-4's sequences toward the base first). The
pressure channel itself never goes negative by inversion: the CH0/CH1
domain is the gauge domain of datasheet Table 5-1 — unsigned, 0..32767
counts for 0..3000 mmHg; the parser-side contracts enforce it (negative
samples are spec violations, never clamped). A driver that models
"withdraw" by negating sample values has invented a domain the payload
does not contain.

## 8. What a closed-loop trace should look like — worked

Section 4 states the tolerances. This section computes them for the
profile defaults from the reference plant (datasheet Appendix E.1, also
`sim/plant.c`): first order, DC gain G = 6 counts-count, τ = 40 ms,
relief ceiling 32 000 counts, seeded sensor noise ±1 count.

### 8.1 Plateau values

At profile A's defaults (STEP 12, HOLD 500): engine targets 12·n, plant
steady states 6 × that — CH0 settles toward 72·n counts:

| tick n | engine LSB | CH0 expected | margin to 960 trip |
|---|---|---|---|
| 1 | 12 | 72 | 888 |
| 4 | 48 | 288 | 672 |
| 8 | 96 | 576 | 384 |
| 10 (top) | 120 | 720 | 240 (25 %) |

The top-plateau number is the load-bearing constant of the whole
profile family: it is the ONLY legitimate way the closed-loop run gets
within shouting distance of the occlusion trip, and it stays 25 % short
of it BY DESIGN (the "peak ≤ clamp threshold" tolerance of §4 IS this
margin, and it only holds because 720 < 960 was chosen deliberately).

Down-ramp runs the same geometry in reverse with the plant following
down with the same τ (first-order systems track down as well as up).
Profile B reaches 8 × 20 = 160 engine counts; the dwell at top lasts
1 200 ms (30τ — deeply settled; a run that hasn't settled at the end of
the dwell is a broken plant or a broken dwell, not a slow profile).
Profile C tops at 14 × 4 = 56 counts — its entire excursion is tiny on
purpose (line charging), which is why its plateau checks can run at the
tolerances of §4 without any noise concession: at 4-count steps, the §4
accuracy band (±6 % of step) is wider than the noise term, and the sim
suppresses plant noise on C for exactly this reason (datasheet E.4).

### 8.2 Where each tolerance comes from

- **Accurate within ± 6 % of step at plateau end.** With HOLD = 12.5τ
  at profile defaults, the first-order residue at end of hold is
  e^−12.5 ≈ 4 ppm of the final value — the 6 % band is not tight on the
  physics; it is there to catch the recipe-scale errors of §12 (a
  125 ms "profile A" sits at 3.1τ and lands its plateaus at ~5 % low —
  right at the band's rim on purpose: the tolerance is a trap for the
  classic misconfiguration).
- **95 % of a plateau within 2 × HOLD.** 3τ ≈ 120 ms versus
  2 × 500 = 1 000 ms of budget: holds the (pun deliberate) across every
  profile default in §3.
- **Overshoot ≤ 10 % of step.** A first-order plant has none; the band
  exists for the plant-noise spikes and for catching a test that
  actually wired a second-order model into `sim/`.
- **Peak ≤ clamp threshold over the whole run** (ALARM_CFG-derived).
  §8.1: legal profile tops never approach the 960-LSB trip; the relief
  ceiling (32 000, below the channel's 16-bit full scale) is never the
  clamp that saves a run — the ALARM saves the run long before the
  relief valve defines the trajectory (E.5's full-occlusion trajectory
  trips the comparator well before the ceiling is interesting).

### 8.3 What "wrong shape" means

A trace is compared plateau-by-plateau: the check points are the HOLD
tick instants, plus the inter-tick shape (a plateau that is reached
late but correct is a rise-time failure; reached on time but undershot
is accuracy; monotonic within tolerance means no plateau regresses more
than the noise term between neighbors — for profile C, whose plateaus
are 4 counts, "regress" is measured against the same noise bound as
everything else). DONE timing closes the run: for A/B, DONE pulses at
down-ramp's end and the cycle restarts (a trace that stops at top and
stays is a C-shaped engine running an A recipe); for C, DONE arrives
once, with the output parked at 56, and stays until the next WAVE_CFG
write.

## 9. Cadence — what the engine is NOT slaved to

The pattern engine's only clock is its 1 ms timebase (§6.13 HOLD_MS,
CLKS_PER_MS = 10 000 at 10 MHz ACLK). It is NOT slaved to, and does not
interact with:

- the acquisition window period (2 ms reference config — the window
  stream samples the PLANT, it does not advance the waveform);
- the oversampling/filter configuration (CFG fields — noise/latency of
  the CH0 picture, never the staircase);
- any host service cadence — including the telemetry console, the QC
  panel refresh, and the status-report bursts of the station software.

Two consequences that have actually cost people days:

1. A console/log gap (missing window lines around the station's 2 Hz
   status burst) is NOT waveform evidence. A log hole with wall-clock
   ts continuity and no seq continuity is a service-side loss; the
   plant and engine both kept running through it. Before writing
   "engine stalled here" on any trace, check WAVE_STATUS.RUN in-band
   (or the retained-output signature below) — a stopped engine leaves
   WAVE_DATA RETAINED at the stop value (§7.3 rule 4), so "flat line at
   a plateau top with RUN=0" is a stopped engine, and "missing
   evidence" with retained output and RUN=1 is somebody else's buffer.
2. "Fast" profile variants for tests are engine-bench material and must
   be labelled as such (see §13). A conformance run of THIS note is at
   §3 default values, because the tolerances, the plant margins, and the
   comparability claim of §6 are all stated about those values.

## 10. Occlusion monitor interaction (and the latch arithmetic)

The waveform engine drives pump set-points; the occlusion comparator
reads the PRESSURE channel (CH0) — they share a plant, not logic
(datasheet §8.1/§8.2, C.5). Everything below is the §8.2 geometry stated
for the people who write the closed loops:

- Trip basis: the averaged window value `CH0_avg ≥ OCCL_TH`
  (OCCL_TH reset 0xF → 960 counts = ~88 mmHg; unit conversion Table
  8-3; values above the 960 maximum are not representable — wanting a
  higher trip point means gating the alarm path too, not asking Table
  8-3 for a fifth hex digit).
- Debounce: the latch closes at the close of the FIRST WINDOW AFTER
  `OCCL_DEB + 1` consecutive OCCL_DET windows. The debounce counter is
  evaluated at window close BEFORE that window's increment — which is
  why the count you OBSERVE is one more than the count the name
  suggests: at reset OCCL_DEB = 0xE, latching happens at the close of
  the 16th consecutive window; at OCCL_DEB = 2 (as used by the
  occlusion-focused verif suite), at the close of the 4th (§6.11/§8.2/
  Table 8-2 agree; the observed pre-increment geometry is the
  normative wording).
- Release: ONLY via CTRL.ALARM_CLR — an over-threshold condition that
  drops back re-arms nothing by itself; and to even let CLR stick, the
  averages must fall below `OCCL_TH − OCCL_HYST` (hysteresis; with the
  reset OCCL_HYST = 0 there is NO hysteresis — ES-2.1.1's soft-reset
  reset-value loss of OCCL_HYST is the same trap from the other side:
  re-write ALARM_CFG after any SWRST).
- After a release, debounce restarts at zero: a re-latch needs a fresh
  full run of consecutive trip windows.

What this means for an AVA-QC run: at §8.1's geometry a LEGAL run never
presents a trip window at all (top plateau 720, margin 240, and the
noise term is two orders short of it — the 6σ discipline of E.2). Every
alarm during a healthy profile trace is therefore either (a) the plant
being actually occluded (a real event — report it), or (b) a
threshold/evidence path that has been tampered with outside the
waveform contract. The suite for the latch geometry (occlusion traces
with DEB=2 latching at the 4th close, CLR-only release, hysteresis
re-arm, relief clamp under full occlusion) lives at the sim rung and
derives its expectations from exactly this wording.

## 11. Errata you will meet mid-pattern (ES-2026-04)

Profile programming is where the revision-B errata sheet stops being
background reading:

- **ES-2.1.3 (read destructiveness)**: during a QC session the host is
  reading STAT constantly (DRDY per window). One STAT read per window,
  ever. A "confirmatory" second read clears the flag and banks
  STAT_VIOLATION + ERR_CNT (§6.9, §7.4.2 supersession) — and a QC
  session with ERR_CNT walking is a session whose window evidence is
  no longer trustworthy, which includes any assertion you are about to
  make about an alarm inside it.
- **ES-2.1.2 (mid-cycle HOLD deferral)**: covered in §7.4; author
  recipes so HOLD only changes between sessions, and use the explicit
  stop-rewrite-restart when a human does change it.
- **ES-2.1.1 (OCCL_HYST lost on SWRST)**: a warm-restarted QC session
  that keeps using its cached ALARM_CFG runs hysteresis-zero — chatter
  at exactly-threshold plateaus, which looks like flaky occlusion
  detection and is flaky ALARM_CFG. Re-write ALARM_CFG after any soft
  reset (the §7.2 init walk of the driver does this; hand-rolled
  session restarts in test code forget it).

## 12. Misconfiguration gallery — what goes wrong and how it shows

Each row: what was written, what register state resulted, what the
trace does. All observed in the field or in test code at least once.

1. **Unshifted ms in HOLD.** Recipe "500" written raw → field holds
   125 in the ms decode → plateaus 4× too fast; plant rides ~5 % under
   each target, rise/fall never quiesces; §4 accuracy band starts
   failing at exactly the plateau ends. Fix: encode per §7.2 (or use
   the driver).
2. **Engine ticks in HOLD.** ms × 10 000 → range violation; the
   reference driver bounces it before the bus sees it. A system without
   the range check truncates into a different legal value and writes a
   silently different recipe — which is why the check belongs where the
   recipe is formed, not only in the bus layer.
3. **STEP 0 write.** No legal 0 (1..63); on a driver without a range
   check the engine advances by nothing — output flat at 0 with RUN=1
   and a happily looping DONE. RUN ≠ motion; WAVE_DATA trend is motion.
4. **"Withdrawal by negative values."** INVERT reinterpreted as a sign
   → negative samples written or parsed → domain violation per §7.5;
   parsers correctly reject, and the pipeline "mysteriously" drops
   every withdrawal-direction sample.
5. **Mid-run profile swap as a feature.** Switching recipes mid-cycle
   (one WAVE_CFG write) is legal, restarts from base, and has zero
   relation to the previous trajectory (§7.4); "glitchy staircase"
   tickets resolve to whoever did it.
6. **DONE-as-RUN.** Testing "is it running" by polling DONE: a loop
   profile only pulses DONE once per 10 000/14 000 ms cycle, a poller
   that misses the window concludes "not running" — RUN is the state
   bit, DONE is an event flag.
7. **Expecting return-to-zero on stop.** §7.3 rule 4: retained output
   is the design; "pump stayed at 720 counts after the abort" is the
   engine behaving; the host's abort routine owns going home.
8. **Panel showing counts/s a thousand times large.** The §5 rule
   (scale once, at presentation) violated by a second conversion in
   the display chain — the field reports this as "rate spike ~×1000 for
   ~2 s"; see the field evidence set's panel artifact; the engine and
   the wire are innocent.
9. **Reading the log instead of the register.** The console tells a
   STORY; WAVE_STATUS and WAVE_DATA are the STATE. Both matter — but a
   trace with holes and an in-band RUN=1 is a service bug wearing a
   waveform costume (§9 consequence 1).

10. **Cached ALARM_CFG across a warm restart.** SWRST re-kills the
   hysteresis field (ES-2.1.1) — a session that restarts softly and
   keeps its cached config runs hysteresis-0: chatter at exactly-trip
   plateaus that never survives a cold restart. Re-write ALARM_CFG in
   every init, not only the first one.
11. **Re-arm expecting resume.** Any WAVE_CFG write clears DONE
   (rule 5) AND re-arming restarts from base (§7.4): "it skipped the
   top plateau on the second lap" resolves to someone rewriting the
   recipe between laps and calling the engine stateful.

## 13. Verification strategy for this note (three layers, one rule)

The rule first: **where this note states a value, the conformance test
runs THAT value.** A test that quietly substitutes "a legal faster
configuration" has stopped being conformance and become a second source
of truth about what the pattern means — and it has quietly made itself
unable to fail in exactly the places §8 computed margins at the default
cadence.

Layer 1 — engine geometry (RTL). The waveform engine's state machine,
cadence, DONE/CYCLE behaviour and profile geometry are verified against
this note's tables at the cycle-accurate level. Cadence choices shorter
than §3 defaults are legitimate THERE (geometry is hold-relative — see
§8.2's arithmetic applying at any hold) provided the harness says out
loud what cadence it runs and why; "HOLD=1 ms in an engine bench tests
geometry, not the profile" is the kind of comment that stays in the
header of such a bench.

Layer 2 — closed-loop conformance (sim). `fv_staircase`-class runs:
plant physics of §8, profile A at app-note defaults AS WRITTEN (a
500 ms hold, not a stand-in), §4 tolerances every plateau, seeded
noise, one full cycle minimum plus entry/exit margins in the trace
window; the occlusion companion (latch geometry at DEB=2, CLR-only
release, re-arm) per §10. Fixed seed, hermetic, no wall-clocked
anything.

Layer 3 — ladder acceptance. The score ladder binds layers 1/2 with the
driver, service and panel rungs; the conformance identity of this note
(the literal table of §7.2, the values of §3, the geometry of §10) is
re-verified by it on every run. A change to this note is a payload
change: it has to keep §7.2's words encodable against the driver's
range checks, and every cross-cited section of the datasheet in
agreement before it's true.

## 14. AVV-2200 comparability, extended (§6 continued)

Byte-identical across the platforms: profile A's constants (10/10/12;
"500 ms" semantics — AVV-2200's native ms units, AVV-3000's as
§7.2-encoded ms), the register NAME space this note uses (WAVE_CFG/
WAVE_DATA/WAVE_STATUS), and the display-scale discipline of §5.
Different, deliberately: AVV-3000's HOLD domain width (§7.1) and the
destructive-STAT read contract of §11 — the latter inherited a
byte-compatible register map onto a NON-compatible read semantics
(es-2.1.3 supersession), which is exactly the class of "identical
register, different meaning" that §6's comparability claim carefully
excludes: waveform RUNS remain comparable; a Rev-A read sequence is not
comparable, it is superseded.

## 15. Worked sessions (what good looks like, line by line)

Session A — inspection, default (the clinical default):

    w = 0x000C_07D1                    ; profile A, 500 ms, 12 LSB
    CTRL.WF_EN = 1
    ; t+500     WAVE_DATA 12   (plant heads for 72)
    ; t+5000    WAVE_DATA 120  top plateau; plant ~720, margin 240
    ; t+10000   WAVE_DATA 0; DONE pulse; CYCLE=0->1 from here onward
    ; repeats: same staircase every 10 s
    CTRL.WF_EN = 0 and PROFILE=0 at session end (or simply PROFILE=0)
    ; WAVE_DATA retains whatever PROFILE=0 caught it at; if that
    ; matters to the plant, bring it to base FIRST (down-ramp does).

Session B — flush, including the errata lesson:

    w = 0x0014_0C82                    ; profile B, 800 ms, 20 LSB
    ; 8 up steps to 160 (8 x 800 = 6400 ms), HOLD_TOP 1200 ms dwell
    ; (plant at 6x160 = 960-ish? NO: 6 x 160 engine counts is a
    ; plant-side 960 counts at G=6 — which is EXACTLY AT the occlusion
    ; TRIP value; see the note below on profile B and alarm margin),
    ; 8 down steps: cycle 14 000 ms, DONE pulse per cycle, loop.
    Mid-run, a recipe push changes hold 800->600:
    ; RUNNING cycle keeps 800 (ES-2.1.2/Table C-3); it lands NEXT cycle.
    Same push via wave_set_hold (stop-rewrite-restart): cycle restarts
    from base at 600 immediately; plant sees the staircase over again.

    Scope note — why conformance (§13 layer 2) exercises A and C and
    not full-B: profile B's engine top (160 counts) against the pump
    plant's nominal closed-loop DC gain (Appendix E.1's G = 6 — a
    number DERIVED from the profile-A excursion, 120 -> 720) computes
    to ~960 CH0 counts: exactly the reset-OCCL_TH trip value, at the
    centre of the noise band rather than 6 sigma clear of it. Full-B
    closed-loop at the §3 defaults against the stock threshold is
    therefore NOT a documented conformance configuration — it tests
    the alarm latch geometry (a different trace suite's job, §10), not
    the staircase. Profile B is verified at engine level (layer 1:
    geometry, dwell, DONE/loop) and, where a plant is attached at all,
    only at a STEP_LSB/OCCL_TH pairing that restores §8.2 margins —
    with the pairing stated in the test header. Anyone tempted to
    "add B to the closed-loop suite" without changing either the step
    or the threshold is scheduling a latch and calling it a finding.

Session C — prime, one-way:

    w = 0x0004_0323                    ; profile C, 200 ms, 4 LSB
    ; 14 ticks to 56 counts, 2 800 ms; DONE sticky at the end.
    ; ES-2.1.2 caution (the §3 one): hold changes during the 2.8 s run
    ; are never applied (one-way profile; the running cycle is the last
    ; one) — stop before re-arming with a different cadence.
    ; DONE stays after success: re-arm = ANY WAVE_CFG write then WF_EN;
    ; a poller that reads DONE at 2 810 ms and rewrites the recipe sees
    ; DONE fall (write-clears-DONE is by design, §7.3 rule 5).

## 15.1. Where the profiles sit in the clinical sequence

The three patterns are not a menu of flavors; they occupy fixed slots of
a session and the trace expectations differ slightly because their
plants do:

- **C (prime) runs first**, on an un-pressurized line — its whole
  contract is "14 quiet increments and a stuck DONE", and its margins
  are so unexciting by design (240 LSB of CH0 excursion at the top)
  that it doubles as the suite's "is the plumbing honest" canary: if a
  C trace has the wrong shape, believe nothing downstream of it,
  including the alarm history.
- **A (inspection) is the run the ticket lives and dies on** — the
  clinical verification that the pump tracks its program before a
  dosing interval, which is why §13 pins it to §3 values literally and
  why the 25 % margin of §8.1's top plateau is quoted whenever a
  "false occlusion" report arrives: a legal A run is a machine-
  checkable statement that the pressure pipeline stayed a quarter of
  the trip away from the trip at every instant it could see.
- **B (flush) separates the waveform question from the pressure
  question on purpose** (§15's scope note): its engine geometry is
  verifiable everywhere; its closed-loop pressure behaviour at stock
  thresholds is a calibration topic, not a conformance topic.

A session that presents out of order (B before C, A without a prior
C-done) is not disallowed by anything in silicon — nothing in §6.13
closes those state machines — but it is undefined for trace tooling,
which assumes the §15 ordering for its "expected current profile" state
machine.

## 15.2. Alarms, gates, and what actually stops the pump

Loud clarification for integrators, because the safelight chain makes it
look like hardware interlocks exist and they do not stop engines:

- A latched ALARM_OCCL with SAFETY_GATE.GATE_EN forces the SAFELIGHT
  MODE to red (datasheet Table 8-4) — a colour. The waveform engine
  keeps its state and its retained output through it (§7.3 rule 4);
  the comparator never writes CTRL, WAVE_CFG, or WAVE_DATA —
  occlusion monitoring is explicitly independent of the engine
  generation path (§8.1).
- That means **quiescing the plant on a latch is host policy**: the
  QC flow that wants "stop under red" must itself PROFILE→0 (output
  retained), and move the plant home if retained-in-position is not
  where the process wants to sit (down-drive, vent, operator — process
  document territory, not this note's).
- AUTO_GREEN returning the lamp to green after a clear is NOT a resume
  affordance: it is a lamp policy (§8.4's "colour is UI") and a
  session that re-arms a staircase on the green edge without the
  evidence review that should sit between latch and re-arm is using a
  LED as a protocol gate.

The verification that belongs to this interaction is small and belongs
in conformance: latch during an A run → assert RUN unchanged, WAVE_DATA
retained, ALARM_STATUS latched, effective colour red with the gate on;
CLR → un-latch subject to §10's hysteresis; nothing above it implied.
## 16. Quick tables (everything else, in one place)

### 16.1 Encodings and ranges

| Quantity | Legal domain | Where | Trap if wrong |
|---|---|---|---|
| PROFILE | 0..3 (A/B/C) | §6.13 [1:0] | writing 4+ truncates to a profile |
| HOLD_MS | 1..16383 ms | §6.13 [15:2], shift 2 | shift/scale traps: §12 rows 1-2 |
| STEP_LSB | 1..63 | §6.13 [21:16] | 0 = silent dead engine (§12 row 3) |
| INVERT | 0/1 | §6.13 [22] | direction, NOT sign (§7.5) |
| Cycle tick budget | hold × CLKS_PER_MS < 2^28 | engine | impossible to overflow at legal holds (16 383 x 10 000 = 1.6e8 < 2^28) |
| CH0/CH1 sample domain | 0..32767 (0..3000 mmHg) | Table 5-1 | negative "withdrawal" (§12 row 4) |
| OCCL_TH | 4 bits, 64-count steps | §6.10, Table 8-3, reset 0xF = 960 | no representations above 960 |
| OCCL_DEB | 4 bits | §6.10, Table 8-2, reset 0xE | observed latch count = DEB + 2 windows (§10) |

### 16.2 Pattern timing at §3 defaults

| Profile | Steps | Cycle | Cadence | Loop | DONE |
|---|---|---|---|---|---|
| A | 10 up + 10 down | 10 000 ms | 500 ms / 12 LSB | yes, indefinite | pulse at cycle close |
| B | 8 up + dwell + 8 down | 14 000 ms | 800 ms / 20 LSB, dwell 1 200 | yes, indefinite | pulse at cycle close |
| C | 14 up | 2 800 ms | 200 ms / 4 LSB | no (one-way) | sticky until any CFG write |

Conformance layer runs §13: A (full), C (full), engine-geometry for B;
each trace window = whole cycles + entry/exit settling, fixed seed.

### 16.3 Symptom index

| Symptom seen | First page to believe | Second |
|---|---|---|
| plateaus low by a nearly-constant % | §12 row 1 (125 ms hold) | §8.2 residue math |
| staircase absent (flat 0, RUN=1) | STEP=0 write-back | recipe truncation on the wire |
| alarm during legal profile run | §10 (illegal) / plant | ERR_STAT walk (§11) |
| "engine stopped" holes in log | §9 consequence 1 (service gap vs engine state) | WAVE_DATA retention |
| DONE never observed | reading it after a CFG rewrite (§7.3 rule 5) | confusing run with cycle loop |
| rate × 1000 on panel | §5 scale-once rule | field evidence set's panel artifact |

### 16.4 Glossary for this note only

- **plateau** — one HOLD interval of constant WAVE_DATA; a check point
  at its close (§8.3).
- **tick** — one HOLD expiry; NOT a clock edge and NOT an acquisition
  window closure.
- **cycle** — one full up/(dwell)/down run: A/B loop cycles, C's single
  pass; the DONE unit (§3).
- **trip / latch** — OCCL comparator reaching the threshold / the
  debounced ALARM_OCCL register state; a latch is not an event of the
  waveform engine and vice versa (§10).
- **conformance (here)** — running the §3 numbers under §4 tolerances,
  §13 rule; anything else is an engine bench and must say so.

- **retained output** — WAVE_DATA's stopped-engine state: the last
  value, indefinitely; a feature for the pump (no surprise motion on
  error) and a trap for the log-reader (flat ≠ idle).
- **single-read contract** — one STAT read per window, the Rev-B read
  rule of §11; violations are device-counted, not just frowned upon.

### 16.5 Change-impact (what breaks together)

| Touch this | Must re-verify |
|---|---|
| §3 profile numbers | §7.2 literals, §8 margins, E.1's G derivation, all of §13 |
| §6.13 field layout | §7.1/7.2 encodings, driver range checks, t_waveform encodings, hlink WAVE_RECIPE width |
| §10 wording | Table 8-2, §6.11, sim mirror comment, occlusion trace suite |
| §5 scale rule | panel core conversion, offscreen unit test, field "rate spike" artifact class |

### 16.6 Reading order for first-time reviewers

Spec authors: §3 (what the patterns are) → §4 (what conformance means)
→ §8 (why those tolerances are the physics and the traps) → §7 (what
"same thing" means at the wire). Implementers start at §7 and
§12, then §10 before touching the alarm path. Trace forensics: §8.3 →
§9 → §16.3. Nobody skips §11 regardless of role — half the "the part
misbehaves" reports in the family's history are §11 by the time the
trace walk finishes, and §11 problems masquerade as §10 problems
precisely when the evidence is hottest.

### 16.7 Quick FAQ (as asked, as answered)

**Q: Can profile A's DONE pulse be missed between two polls?**
A: Yes — a 20-step 500 ms loop pulses DONE once per 10 000 ms and ANY
WAVE_CFG write clears it; RUN is state, DONE is an event. Test the
cycle with WAVE_DATA trend crossings if you need more than the edge.

**Q: Does the engine wait for the acquisition window to be ready?**
A: Never (it has no idea windows exist, §9). A QC "recipe validated
end-to-end" has at least one acquisition path check separate from the
engine check for exactly this reason.

**Q: What is the shortest legal hold?**
A: 1 ms — legal in the register, and the plant cannot form plateaus
there at τ = 40 ms, so closed-loop conformance never runs it (§13's
rule); engine geometry checks at short holds are layer-1 work (and
label themselves).

**Q: The panel shows flat 0 rates but the engine is running. Ring or
engine?**
A: Engine runs and log flat at the same instant = ring/console; engine
running AND WAVE_DATA flat = STEP/encoding (§12 rows 1/3). Panel flat
0 with a live wire = the display family (§5/field panel artifact).

**Q: Does INVERT=1 make the down-ramp the "negative pressure" ramp?**
A: No — §7.5: direction of the staircase in set-point space; the
pressure domain stays unsigned; parsers enforce it.

**Q: B's DONE never arrives in the trace.**
A: B loops forever by §3 (DONE pulses per cycle, then restarts) — a
test waiting for DONE to STICK on B is waiting for a C behaviour; and
one not seeing B's pulse should be checking whether it polls slower
than the 14 s cycle or rewrote the cfg.

## 17. Revision history

- **2025-09 R0** — first issue: profiles A/B/C tables (§3 semantics),
  closed-loop tolerance set (§4), display scale rule (§5), AVV-2200
  carry-over declaration (§6).
- **2025-10 R1** — profile B top dwell documented explicitly (1 200
  ms) after a register-implementation mismatch in the golden tree
  (an 8-bit internal dwell register truncated 1 200 → 176; the note
  drives, the register followed, width documented where coded).
- **2025-11 R1.1** — HOLD_MS semantics aligned with the widened
  §6.13 field (1..16383) after the app-note defaults vs max-field
  conflict was arbitrated: the register widened to carry this note's
  stated defaults and dwell, not the defaults shrunk to fit the old
  field; WAVE_CFG literal table (§7.2) added with derivations (the
  one decimal-to-hex slip in the first literal table became the
  worked example for "trust the derivation column").
- **2026-01 R1.2** — §10 latch arithmetic restated to the observed
  pre-increment geometry (count "after DEB+1 windows" == latch at the
  close of window OCCL_DEB+2), matching the RTL wire, the sim mirror,
  and the datasheet §8.2 wording; Table 8-2 cross-reference added.
- **2026-02 R1.3** — §13 written as a rule after a conformance trace
  ran a stand-in cadence instead of the §3 default and the difference
  only mattered in the direction of hiding failures; §12 gallery
  grew rows 1/4/9 from post-mortems; profile-B alarm-margin note
  added to §15 (G = 6 top-dwell arithmetic at the stock trip was a
  documentation trap, now a calibration target).

*This app note and the datasheet Appendix C/E tables are the two
authoritative statements of the same engine; where they differ, this
note drives for profile MEANINGS (what the patterns mean clinically and
what their tolerances are), the datasheet drives for register/silicon
BEHAVIOUR. They agree; check §7.2's derivations when reading either.*

Addendum to the FAQ, on when to STOP trusting any of the above: if a
session's ERR_STAT reads nonzero STAT_VIOLATION or ERR_CNT walks while
the session runs, the read path is eating its own evidence and NOTHING
in this section's trace-reading rules applies — a corrupted-
instrumentation trace cannot distinguish engine stall from lost sample
(§11's first bullet says how; the field's entire eight-month stall
happened inside that clause). Re-arm the session with a single-read
driver FIRST, then debug the waveform.
