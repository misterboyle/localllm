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