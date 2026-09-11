# ADS-8851 — Octal-Channel Biopotential/Physiological ADC with Integrated Waveform Engine and Occlusion Monitor

**Preliminary Data Sheet — Rev B (2026-04-11)**

- Product: AVV-3000 Arthroscopic Pump
- Subsystem: ADS-8851 sensor ASIC (ADS-8851-C1 silicon, package TQFP-64)
- Supersedes: Rev A (2025-11-02) — see §7.2 and §11 for the Rev-B delta
- Companion docs: AM004 AXI-Lite interface spec (excerpt, `axi-lite-excerpt.md`),
  app note AVA-QC stair-step waveform patterns (`appnote-ava-qc-staircase.md`),
  ES-2026-04 errata sheet (`ES-2026-04-errata.md`)

---

## Table of contents

(generated from section headings at Rev B publication; appendix
cross-references throughout cite these anchors directly — a reader
arriving via grep may land anywhere; every appendix opens with a
one-line scope.)

  - List of tables
  - List of figures
- 1. Overview
  - 1.1 Feature set
  - 1.2 Block diagram (textual)
- 2. Ordering Information
- 3. Pin Description (subset relevant to system integration)
- 4. AXI4-Lite Interface
  - 4.1 Write transaction
  - 4.2 Read transaction
  - 4.3 RESP rules (write path)
  - 4.4 Register access rules
  - 4.5 Bus loading and ordering rules (informative companion to D.5)
- 5. Data Port (window result)
  - 5.1 Code-to-physical conversion (Table 5-2)
- 6. Register Map
  - 6.1 Access classes (Table 6-1 legend, normative)
  - 6.2 Reset domains and retention (Table 6-2)
  - 6.3 Invalid and illegal accesses (Table 6-3)
  - 6.4 Endianness and data placement
  - 6.5 Clock-domain behaviour
  - 6.6 Field notation in this datasheet
  - 6.7 CTRL (0x00) — Control
  - 6.8 CFG (0x04) — Acquisition configuration (Rev B layout)
  - 6.9 STAT (0x08) — Status
  - 6.10 ALARM_CFG (0x0C) — Occlusion/thermal thresholds
  - 6.11 ALARM_STATUS (0x10) — Latched alarm register
  - 6.12 IRQ_MASK (0x24) — Interrupt mask
  - 6.13 WAVE_CFG (0x14) — Waveform engine configuration
  - 6.14 WAVE_DATA (0x18) — Waveform output
  - 6.15 WAVE_STATUS (0x1C) — Waveform status
  - 6.16 ERR_STAT (0x20) — Error/statistics (read-to-clear)
  - 6.17 TH_RATE (0x28) — Thermal sampling cadence
  - 6.18 SAFETY_GATE (0x2C) — Safelight gating
- 7. Acquisition Engine
  - 7.1 SDR window
  - 7.1.1 Inside a window — numbers for the §7.2 default
  - 7.2 Initialization sequence (Rev B — *the configuration contract*)
  - 7.2.1 Step-by-step rationale and failure signature (Table 7-5)
  - 7.3 Low-power mode
  - 7.4 Host read protocol
    - 7.4.1 DRDY_N signal
    - 7.4.2 Two-read STAT pattern (Rev-A legacy — superseded)
- 8. Waveform Engine and Occlusion Monitor
  - 8.1 AVA-QC staircase profiles
  - 8.2 Occlusion detection
  - 8.3 Thermal faults
  - 8.4 Safelight interaction (informative summary of §6.7/§6.18)
- 9. Specifications
  - 9.1 Absolute maximum ratings
  - 9.2 Recommended operating conditions and DC/AC characteristics
  - 9.2.1 Test-condition footnotes
  - 9.3 Timing
  - 9.4 Test methods and characterization conditions
- 10. Application Information
  - 10.1 Reference designs and documents map
  - 10.2 System integration notes (informative)
- 11. Document Revision History
- Appendix A — Initialization and configuration sequences by use case
  - A.1 Use-case overview (Table A-1)
  - A.2 Low-noise bench configuration
  - A.3 Max-rate streaming configuration
  - A.4 Standby duty-cycle (LPM cycling)
  - A.5 AVA-QC bring-up sequence (register view)
  - A.6 Read-path timing budget (informative)
- Appendix B — Characterization data
  - B.1 Offset and gain error vs temperature (Table B-1)
  - B.2 Window-result noise vs temperature and oversampling (Table B-4)
  - B.3 AC performance vs input frequency (Table B-5)
  - B.4 Channel crosstalk (Table B-6)
  - B.5 Channel-to-channel matching (Table B-7)
  - B.6 Power and supply current (Table B-8)
  - B.7 Supply rejection and common-mode (Table B-9)
  - B.8 Timing parameters (Table B-10)
  - B.9 ACLK quality sensitivity (Table B-11)
  - B.10 On-die thermal monitor (Table B-12)
  - B.11 Per-channel statistics (Table B-13)
  - B.12 Long-term drift (Table B-14)
- Appendix C — Waveform engine reference (normative companion to §8.1)
  - C.1 States and transitions (Table C-1)
  - C.2 Per-profile nominal timing (Table C-2)
  - C.3 Configuration-change matrix (Table C-3)
  - C.4 Step-value geometry (Table C-4)
  - C.5 Interaction with the occlusion path
- Appendix D — SDR sequencer and bus internals
  - D.1 Window arithmetic (Table D-1)
  - D.2 4-tap FIR (Table D-2)
  - D.3 Median mode (informative)
  - D.4 Data-port commit and overrun (Table D-4)
  - D.5 AXI slave internals (informative; normative text is AM004 + §4)
  - D.6 Counters and wrap guidance
- Appendix E — Occlusion / plant application notes (informative)
  - E.1 Reference plant model
  - E.2 Threshold vs noise — the 6σ rule (Table E-2)
  - E.3 Debounce timing (Table E-3)
  - E.4 Profile-C plateau granularity
  - E.5 Fault trajectory (full occlusion mid-run)
- Appendix F — Production test, self-calibration and the SPI fallback
  - F.1 Test flow (informative)
  - F.2 Self-calibration
  - F.3 SPI fallback port (production protocol)
  - F.4 Trim fuse map (Table F-4)
- Appendix G — Errata compatibility matrix
- Appendix H — Packaging, reliability, assembly
  - H.1 Package (TQFP-64)
  - H.2 Reliability
  - H.3 Storage, handling, return
- Appendix I — Glossary
- Appendix J — Troubleshooting and failure diagnostics
  - J.1 Symptom → signature matrix (Table J-1)
  - J.2 The 1-in-N arithmetic (ES-2.1.3 duty cycle)
  - J.3 ALARM_LINE false-geometry (host-side only)
  - J.4 What noise-sensitive waveform checks would need
  - J.5 The diagnostic-read budget
- Appendix K — Board-level application notes
  - K.1 Supply sequencing and decoupling (Table K-1)
  - K.2 Sensor front end per Table 5-1
  - K.3 WDx PWM / valve-driver interface
  - K.4 Grounding and layout for the alarm path
- Appendix L — Full change log (Rev A → Rev B)
- Appendix M — Application circuits (informative)
  - M.1 Pressure front ends (CH0/CH1)
  - M.2 Thermistor chains (CH2/CH3)
  - M.3 Motor/valve current senses (CH4/CH5)
  - M.4 WDx / DAx outputs
  - M.5 Power sequencing in the AVV-3000 head
- Appendix N — AVV-2200 compatibility (lineage notes)
- Appendix O — Timing diagrams (informative; normative geometry cited per-figure)
- Appendix P — Lot characterization data (Rev B, image 3.2.1)
- Appendix Q — Production test limit table (FT1/FT2)
- Appendix R — DFT pattern inventory (design-for-test summary)
- Appendix S — Ordering, packing, document status
- Appendix T — Register quick reference card
- Appendix U — Worked bus transcripts (informative)
  - U.1 Nominal window service (H-2 compliant) — Figure U-1
  - U.2 Jitter-crossing double read (D.5 case X-2, ES-2.1.3) — Figure U-2
  - U.3 Consumer stalled — FIFO overrun, no data corruption — Figure U-3
  - U.4 Alarm-onset snapshot storm (what NOT to do) — Figure U-4
  - U.5 Back-to-back init writes (Q-1 pattern 11 replay) — Figure U-5
- Appendix V — Frequently asked questions (driver authors)
- Appendix W — Counter trajectories: three worked forensics cases
  - W.1 ES-2.1.3 (double-read driver) — Figure W-1
  - W.2 Ring eviction masquerading as lost windows — Figure W-2
  - W.3 Thermal margin boundary (the honest one) — Figure W-3
- Appendix X — Field-support Q&A digest (Rev A → Rev B era)
- Appendix Y — Symbols and units (quick table)

### List of tables

- Table 5-1 — channel assignment
- Table 5-2 — representative conversions
- Table 6-1 — register summary
- Table 6-2 — reset/retention comparison (summary of §6.7, §7.3, D.4,
- Table 7-1 — SDR_WINDOW encoding (CFG[1:0])
- Table 7-2 — OVR_SAMPLE encoding (CFG[4:2])
- Table 7-3 — reference CFG/CTRL values (definitive)
- Table 7-4 — Rev A → Rev B configuration delta
- Table 7-5 — why each init step is mandatory, in order
- Table 8-1 — profile summary (full tables in the app note)
- Table 8-3 — OCCL_TH unit conversion (CH0, 64-LSB steps)
- Table 8-2 — OCCL_DEB debounce encoding (latch geometry per §8.2)
- Table 8-4 — effective SAFELIGHT_MODE truth table
- Table A-1 — use-case configuration matrix
- Table A-2 — WAVE_CFG literals at AN-8851-07 §3 profile defaults
- Table A-3 — per-window host budget (nominal)
- Table B-1 — offset/gain error vs ambient (4× oversample, boxcar, FIR on)
- Table B-4 — CH0 window-result noise, RMS LSB (boxcar, FIR on; median mode
- Table B-5 — SNR / THD / SINAD / ENOB vs f_in, §7.2 config
- Table B-6 — mutual crosstalk matrix (dB)
- Table B-7 — CH0 ↔ CH1 matching (the ALARM_LINE pair)
- Table B-8 — power by configuration (AVDD = 3.3 V, DVDD = 1.8 V, TA = +25 °C,
- Table B-9 — PSRR / CMRR vs frequency (4× oversample, §7.2 config, ±100 mV
- Table B-10 — AXI and acquisition timing parameters
- Table B-11 — effects of ACLK deviation (10 MHz nominal)
- Table B-12 — thermal channel (CH2/CH3) characterization
- Table B-13 — per-channel DC/AC spread (n = 9 039 units)
- Table B-14 — drift after HTOL-class stress (H.2 conditions)
- Table C-1 — engine state machine
- Table C-2 — cycle timing at profile defaults
- Table C-3 — mid-cycle WAVE_CFG writes
- Table C-4 — WAVE_DATA sequence (INVERT=0)
- Table D-1 — window capacity (max T-table oversampling always capped by the
- Table D-2 — FIR coefficients (fixed-point, /8 scale)
- Table D-3 — median pick geometry
- Table D-4 — overrun semantics and what is never at risk
- Table D-5 — arbitration cases (all combinations, one cycle each: A
- Table E-1 — plateau expectation (profile A, engine step 12)
- Table E-2 — margin at selected points (from csv, healthy silicon)
- Table E-3 — ALARM_OCCL latch delay from the START of the first OCCL_DET
- Table F-1 — production flow (Rev B, image 3.2.x)
- Table F-2 — self-cal phases
- Table F-3 — fallback frame (32 bits, MSB first, SCLK ≤ 5 MHz)
- Table F-4 — fuse word layout (READ_TRIM addressing)
- Table G-1 — errata vs document surface
- Table H-1 — package body data
- Table H-2 — qualification summary (AEC-Q100-graded test plan, medical
- Table K-1 — recommended supply ramp (any order, monotonic, ≤ 50 ms
- Table L-1 — Rev A → Rev B normative changes
- Table N-1 — platform summary
- Table P-1 — lot summary (CP+FT combined, 6 lots)
- Table Q-1 — FT1 (contract correctness; all-or-nothing patterns)
- Table Q-2 — FT2 (parametric; limits from Appendix B tables ×production
- Table R-1 — pattern families and analog coverage (Rev B)
- Table S-1 — ordering and packing addendum
- Table S-2 — document status and lifecycle
- Table T-1 — 30-second map
- Table T-2 — first-response checklist (companion to J.1/J.5)
- Table 5-1 — never an implied physical unit; 'typ' = 1-σ lot mean

### List of figures

- O-1 — ES-2.1.3 failure sequence (Rev-A driver on Rev B silicon):
- O-2 — occlusion latch geometry (§8.2; DEB=2, Table 8-2):
- O-3 — profile A closed-loop cycle (Table A-2 literal; E.1 plant):
- O-4 — window-close arbitration (§4.4 H-3, D.5 X-1):
- O-5 — SPI fallback frame timing (F.3; jig side):
- O-6 — self-cal BUSY span (F.2), acquisition paused:

---

## 1. Overview

The ADS-8851 is the sensor front-end for the AVV-3000 arthroscopic fluid
management pump. It integrates eight simultaneous-sampling ADC channels, a
programmable SDR (Sample-During-Read) acquisition engine, a waveform-generation
stage for the AVA-QC (Aspiration Verification and Quality Control) infusion
profiles, and an on-die occlusion detector used by the safelight controller.

The device connects to the host SoC (Zynq-style, AXI-Lite bus) as a 32-bit
memory-mapped peripheral at base address `0x4000_0000` (§6). All register
accesses are 32-bit; byte and half-word accesses are not supported and return a
decode error (§4.3).

The ADS-8851 performs acquisition autonomously after configuration: the SDR
engine digitizes the eight channels once per acquisition window (the *SDR
window*, §7.1), averages the programmed oversampling set, and presents the
result on the 8-word data port. The host reads the window result and its status
once per window (§7.4.2). The integrated waveform engine can drive an external
DAC/valve-driver path with the AVA-QC staircase patterns (§8) under host
supervision; the occlusion monitor compares the pressure-proportional channel
against a programmable threshold and latches an alarm (§8.2).

### 1.1 Feature set

- Eight simultaneous-sampling channels, 16-bit, up to 8 kS/s aggregate
  (1 kS/s/channel at the default 2-ms SDR window with the 4-sample
  oversampling set).
- SDR acquisition engine with 4 selectable window periods (§7.1, Table 7-1)
  and 5 selectable oversampling sets (§7.1, Table 7-2).
- Boxcar or median averaging per window (§7.1).
- Programmable finite-impulse-response filter on the averaged output
  (`FILT_EN`, §6.10 / §7.2 step 4).
- On-die occlusion detector with threshold, hysteresis, and debounce
  (§8.2, §6.13).
- Staircase waveform engine implementing the AVA-QC profiles A/B/C
  (§8.1, Table 8-1; profiles defined in the AVA-QC app note).
- 3-bit thermal fault monitor and two independent temperature fault outputs
  (`TH_FAULT`, `ALARM_STATUS[2:1]`, §8.3).
- Low-power mode gating the acquisition engine (`LPM`, §7.3).
- Full AXI4-Lite register interface (AM004-compatible, §4).
- Industrial temperature range: −40 °C to +105 °C ambient; ADC accuracy
  specified −40 °C to +85 °C; see §9 derating table.

### 1.2 Block diagram (textual)

```
                 ┌────────────────────────────────────────────────────────────┐
  CH0..CH7 ──►   8× PGA + 8× SAR ADC    ──►   SDR engine      ──►   avg/th      │
  (biopotential,  │  (16-bit, 8-ch)         (window sequencer,   (boxcar/median,│
   pressure, temp)│                          oversample set)      FIR)          │
                 └───────────┬───────────────┬─────────────────────────────────┘
                             │ window result │
                             ▼               ▼
                 ┌────────────────────┐   ┌──────────────────────┐
                 │ 8-word data port   │   │   occlusion monitor  │──► ALARM_STATUS
                 │  (§5, Table 5-1)   │   │  (TH/HYST/DEB, §8.2) │
                 └─────────┬──────────┘   └──────────┬───────────┘
                           │                         │
                 ┌─────────▼─────────────────────────▼───────────┐
                 │         AXI4-Lite slave interface (§4)        │
                 │  base 0x4000_0000, 32-bit regs, §6            │
                 └─────────────────────┬─────────────────────────┘
                                       │ S_AXI (AW/W/B/AR/R)
                                       ▼
                                   host SoC
```

---

## 2. Ordering Information

| Order code | Package | Temp range | Marking |
|---|---|---|---|
| ADS-8851-BIZ | TQFP-64 (10×10 mm, 0.5 mm pitch) | −40..+105 °C | 8851B / lot / YW |

Rev-B silicon (`-B`) is the only shipping revision as of ES-2026-04. Rev-A
(`-A`) parts exist in pre-production field units; their configuration layout
differs (see §7.2 delta note and Table 7-4).

---

## 3. Pin Description (subset relevant to system integration)

| Pin(s) | Name | Dir | Function |
|---|---|---|---|
| 1–8 | CH0..CH7 | I | Analog inputs. CH0 = pump pressure (0–3000 mmHg scaled), CH1 = line (compliance) pressure, CH2–CH7 = temperature / drive-current sense / spare |
| 9 | AVDD | P | Analog supply 3.3 V |
| 10 | DVDD | P | Digital supply 1.8 V |
| 11 | GND_AGND | P | Analog ground |
| 12 | GND_DGND | P | Digital ground (single-point star to AGND) |
| 13 | OT_ALARM | O | Open-drain over-temperature alarm (asserted low) |
| 14 | DRDY_N | O | Data-ready strobe: pulses low for one ACLK after each window close (§7.4.1) |
| 15 | ACLK | I | Acquisition clock 10 MHz ±100 ppm (window timing derives from this) |
| 16 | RESET_N | I | Active-low async reset; all registers to reset values (§6) |
| 17 | IRQ | O | Level IRQ: asserted when any unmasked ALARM_STATUS bit is set (§6.12) |
| 18 | SCLK | I | SPI-fallback clock (production self-test only; not a system path) |
| 19–21 | TEST0..2 | I | Factory test; strap LOW in production |
| 22–37 | SAXI_AW* / SAXI_W* / SAXI_B* / SAXI_AR* / SAXI_R* | I/O | AXI4-Lite slave port (32-bit data/address), §4 |
| 38–41 | WD0..WD3 | O | Waveform/drive outputs to valve-driver stage (LVCMOS, 16-level PWM) |
| 42–49 | DA0..DA7 | O | Multiplexed channel LED/indicator drives for the QC panel |
| 50–64 | VDD_IO, DGND_x, NC | P/− | Supplies and no-connects |

---

## 4. AXI4-Lite Interface

The ADS-8851 exposes a single AXI4-Lite slave port (data width 32, address
width 16, all transactions within the 64-register address space of §6).
The interface is AM004-chapter-4 compatible; the host-side rules are listed
in `axi-lite-excerpt.md` (AM004 §4.3–§4.7).

### 4.1 Write transaction

The interface accepts AW and W in either order (`AWVALID`/`AWREADY` and
`WVALID`/`WREADY` independent handshakes). A write is committed when BOTH the
address and data handshakes have completed; the write-response is issued
exactly once per write and is never dropped, including for
back-to-back writes (§4.3).

### 4.2 Read transaction

ARVALID/ARREADY handshake gates address acceptance; RVALID/RREADY gates data
return. Reads are non-destructive except as specified in §6.9 (STAT) and
§6.16 (read-to-clear counters); see also ES-2026-04 ES-2.1.3.

### 4.3 RESP rules (write path)

- BRESP = `OKAY` for all writable addresses; `SLVERR` for reserved/write-only-
  violated addresses.
- **Every write produces exactly one B response. A subsequent write that
  arrives while a previous B response is pending must not suppress or merge
  the pending response** (AM004 §4.5.3). This is the behavior verified by the
  `t_bus` integration test.
- RRESP = `OKAY` for readable addresses, `SLVERR` for read-only-invalid.

### 4.4 Register access rules

- All registers 32-bit; byte/half-word enables (`WSTRB ≠ 0xF`) are a decode
  error → `SLVERR`, write discarded.
- Reserved addresses return `SLVERR` and are ignored by the register block.
- AXI accesses that arrive while the SDR sequencer is committing a window
  result are arbitrated: the register block wins (window commit is delayed by
  ≤ 2 ACLK; no data loss — §7.4.2 host rule H-3).

### 4.5 Bus loading and ordering rules (informative companion to D.5)

Normative ordering lives in §4.3 and the AM004 extract; this collects the
numbers integrators actually plan against on this fabric:

- One transaction at a time is the design point: the slave is
  1-outstanding with a 2-deep B FIFO (§4.3 invariants are what make even
  a pipelined master SAFE; they don't make it necessary).
- A window-close service (STAT read, then 8-word data burst) is 11
  beats, under 20 µs end-to-end including driver overhead — 1 % duty of
  the 2 ms window (measured budget in A.3/A.6).
- Back-to-back register WRITEs (init sequences) commit at ~4 ACLK each;
  §7.2's entire init is < 6 µs of bus time, which is why re-init after
  watchdog recovery is a free operation and there is no excuse for the
  "skip CONFIG after warm restart" firmware class of bugs (t_fw_init's
  warm-restart suite row in Q-1 exists because that excuse was once
  real).
- Status traffic (the 2 Hz service burst, the QC panel view) is RO-only
  against live latches; its bus cost is bounded by the same table —
  ring admission policy, not silicon, decides what a burst costs (J.1
  row 5; §10.2).


---

## 5. Data Port (window result)

The eight ADS-8851 channel results per window are exposed as eight 32-bit
read-only registers at offsets `0x40 + 4×ch` (ch 0..7). Each register:

| Bits | Field | Description |
|---|---|---|
| [15:0] | CHx_RES | Averaged, filtered window result, signed 16-bit (two's complement) |
| [23:16] | CHx_OVF | Window overflow counters (per-channel, saturating, read-to-clear) |
| [31:24] | rsvd | 0 |

The data port is internally latched at window close (§7.1, step 4) and is
stable until the next close. The host reads the data port AFTER confirming
window completion via STAT (§7.4.2); reading the data port at any time is
non-destructive.

Table 5-1 — channel assignment

| Channel | Signal | Scale | Notes |
|---|---|---|---|
| CH0 | PUMP_P | 0–3000 mmHg → 0..32767 | primary occlusion input |
| CH1 | LINE_P | 0–3000 mmHg → 0..32767 | compliance/line pressure |
| CH2 | HDR_T | −40..+125 °C → −2048..4095 | pump head thermistor |
| CH3 | SYS_T | −40..+125 °C | system thermistor |
| CH4 | I_MOT | 0..5 A → 0..16383 | motor drive current sense |
| CH5 | I_VLV | 0..1 A → 0..4095 | valve drive current sense |
| CH6 | SPARE_A | differential | reserved |
| CH7 | SPARE_B | differential | reserved |

### 5.1 Code-to-physical conversion (Table 5-2)

Canonical conversions for every domain in Table 5-1. Column `raw` is the
CHx_RES value after FIR (§5); `physical` is the reference formula of the
column `equation`. A code outside the raw domain is not a pressure/
temperature — it is a front-end or wiring fault (or a wrap-through
sign bug: the two's-complement bit [15] of a "pressure" code means
negative, and per Table 5-1's domain that CANNOT happen in service —
systems must reject, never abs(), see note 2).

Table 5-2 — representative conversions

| Channel | raw | physical | equation | notes |
|---|---|---|---|---|
| CH0 PUMP_P | 0 | 0.00 mmHg | mmHg = raw × 3000/32767 | rail check |
| CH0 | 960 | 87.9 mmHg | idem | OCCL_TH reset trip point (Table 8-3) |
| CH0 | 8192 | 750.5 mmHg | idem | quarter scale |
| CH0 | 16384 | 1500.4 mmHg | idem | half scale |
| CH0 | 32767 | 3000.0 mmHg | idem | domain max |
| CH1 LINE_P | 16384 | 1500.4 mmHg | idem | |
| CH2 HDR_T | −2048 | −40.0 °C | °C = (raw + 2048) × 165/6143 − 40 | domain min |
| CH2 | 0 | 15.0 °C | idem | idle-room code |
| CH2 | 1118 | 45.0 °C | idem | warm head |
| CH2 | 4095 | 125.0 °C | idem | domain max |
| CH3 SYS_T | 0 | 15.0 °C | idem | |
| CH4 I_MOT | 0 | 0.000 A | A = raw × 5/16383 | |
| CH4 | 8192 | 2.500 A | idem | stall-window mid |
| CH4 | 16383 | 5.000 A | idem | domain max |
| CH5 I_VLV | 2048 | 0.500 A | A = raw × 1/4095 | half-scale valve |
| CH5 | 4095 | 1.000 A | idem | domain max |
| CH6/CH7 | any | reserved | — | do not allocate without a datasheet revision |

Notes:
1. The domain bounds are SPEC limits of the scale mapping, not
   over-range clamps: the silicon reports what it samples — see K.2 on
   keeping the front end from presenting negative pressure codes in
   service.
2. Parsers that clamp or wrap on sign at the display edge invent
   −0.01 mmHg readings the plant cannot produce; the canonical rule is
   reject-the-sample (this system's telemetry parser follows it — the
   domain lives here, Table 5-1 + this table, not in code).
3. CHx_OVF bytes are per-channel saturating window counters (drop
   accounting between reads — the same "witness counters" Appendix J.1
   routes to); they clear on read of the CHx word (RC, §6.1).

---

## 6. Register Map

Base address `0x4000_0000`. All offsets in bytes. R = read, W = write,
RC = read-to-clear, W1C = write-1-to-clear. Reset value applies to
`RESET_N` assertion and to software reset (`CTRL[6] SWRST`, §6.7).

Table 6-1 — register summary

| Offset | Name | Access | Reset | Section |
|---|---|---|---|---|
| 0x00 | CTRL | RW | 0x0000_0000 | §6.7 |
| 0x04 | CFG | RW | 0x0000_0004 | §6.8 |
| 0x08 | STAT | RO | 0x0000_0000 | §6.9 |
| 0x0C | ALARM_CFG | RW | 0x0000_1E0F | §6.10 |
| 0x10 | ALARM_STATUS | RO | 0x0000_0000 | §6.11 |
| 0x14 | WAVE_CFG | RW | 0x0000_0000 | §6.13 |
| 0x18 | WAVE_DATA | RO | 0x0000_0000 | §6.14 |
| 0x1C | WAVE_STATUS | RO | 0x0000_0000 | §6.15 |
| 0x20 | ERR_STAT | RC | 0x0000_0000 | §6.16 |
| 0x24 | IRQ_MASK | RW | 0x0000_001F | §6.12 |
| 0x28 | TH_RATE | RW | 0x0000_0168 | §6.17 |
| 0x2C | SAFETY_GATE | RW | 0x0000_0000 | §6.18 |
| 0x30..0x3C | rsvd | — | — | decode error |
| 0x40..0x5C | DATA_CH0..CH7 | RO | 0x0000_0000 | §5 |
| 0x60..0xFC | rsvd | — | — | decode error |

### 6.1 Access classes (Table 6-1 legend, normative)

| Class | Behaviour at access | Destructive? |
|---|---|---|
| RW | read returns last written value; write commits at t_AXC (§9.3) | no |
| RO | write is ignored (no error — intentional: recipe-blinder writes tolerate read-only tails) | no |
| RO* | read-only with read SIDE EFFECT: STAT (DRDY consume / ERR_CNT RC, ES-2.1.3), and every RC field | yes — governed by H-2 for STAT; RC fields cost their content on every read |
| RC | read-to-clear: the read returns the accumulated value and zeroes the field atomically at t_AXC | yes |
| W1C | write-1 clears bits; write-0 no-op; reads reflect live state (ALARM_CLR, §6.7) | yes (targeted) |

RC-class reads are themselves budgeted resources on a live system
(Appendix J.5): an "innocent" diagnostics loop that samples ERR_STAT
destroys the accumulation it means to observe — and worse if aimed at
STAT during the read-hazard window under discussion.

### 6.2 Reset domains and retention (Table 6-2)

Three paths change register state with different footprints:

Table 6-2 — reset/retention comparison (summary of §6.7, §7.3, D.4,
ES-2.1.1 as clarified in G-1)

| Action | Registers | ALARM latches | ERR_STAT/RC counters | WAVE_CFG | data port | SEQ | fuse trims |
|---|---|---|---|---|---|---|---|
| RESET_N assert | → reset values | clear | clear | 0 | 0 | 0 | survive |
| CTRL.SWRST | → reset values | clear | clear | 0 | 0 | 0 | survive |
| LPM enter/exit (§7.3) | unchanged | **retain** | **retain** | retain | frozen | pause | n/a |
| CTRL.ALARM_CLR | unchanged | clear | clear (ERR_STAT) | unchanged | unchanged | unchanged | n/a |

Practical corollaries: the recovery path after an occlusion event is a
software procedure (engine stop → vent → ALARM_CLR → re-arm, C.1/E.5),
never a reset — a reset would silently drop TH_RATE, IRQ_MASK and occlusion
thresholds (the ES-2.1.1 hysteresis trap lives here: any SWRST path must
re-write ALARM_CFG afterwards).

### 6.3 Invalid and illegal accesses (Table 6-3)

| Access | Response | Register effect | Side effects |
|---|---|---|---|
| Byte/half enable mix (WSTRB ≠ 0xF) | SLVERR | write discarded, read returns undefined-zero | ALARM_COMMS latch (§6.11 note) |
| Reserved address (§6 map rows 'decode error') | SLVERR | none | ALARM_COMMS latch |
| WAVE_STATUS read | OKAY | — | NEVER clears DONE by itself — only a WAVE_CFG *write* clears (§6.15; t-waveform class assumption) |
| CFG [7] write (any value) | OKAY | ignored (rsvd0, Table 7-4) | none — this silence is the whole B1-class failure mode of Rev-A drivers |
| STAT read #2 in a window | OKAY (returns DRDY=0) | none | ES-2.1.3 trip (STAT_VIOLATION, ERR_CNT) |
| CTRL.SWRST=1 write | OKAY | map reset, self-clear | all Table 6-2 'SWRST' effects |

Note the cruel asymmetry that field history keeps rediscovering: every
row of Table 6-3 answers OKAY on the bus except the decode errors. A
driver can be structurally wrong (rows 4 and 5) while its AXI traffic
looks spotless. BRESP tells you the bus agreed; ERR_STAT and t_fw-class
layout tests tell you the contract agreed.

### 6.4 Endianness and data placement

Single AXI domain, 32-bit lanes, little-endian byte order (AM004 §2.2
class; byte lane 0 = bits [7:0]). Field extraction is therefore mask +
shift per the §6.x tables — there are no split-endian windows and no
64-bit accesses. Multi-register burst reads of the data port (0x40..0x5C)
follow D.4's atomicity: one window's eight words can be consumed as a
burst or eight reads with identical consistency guarantees.

### 6.5 Clock-domain behaviour

The register file is ACLK-synchronous; AM004 traffic is sampled through a
synchronous FIFO (§4 arbitrator) with no host-visible metastability
margin to manage — no bus-side clock relationship constraints beyond
AM004's own setup/hold. There is no asynchronous read path: no register
changes value between consecutive legal reads except where a *close*
instance intervenes (§6.9 SEQ; D.5 case X-1/X-2).

### 6.6 Field notation in this datasheet

- `[msb:lsb]` bit ranges, `0xNN_NNNN_NNNN` reset values (underscores are
  readability, not part of the value).
- 'rsvd' fields: write 0, read 0 unless stated rsvd0/read-anything.
- Timing units: `CLK` = one ACLK period (100 ns at 10 MHz), `T_W` = one
  window period (Table 7-1), `ms` = engineering milliseconds — HOLD math
  is ms-domain, engine internals are CLK-domain (§8.1 cadence note).
- '§' references target this datasheet unless qualified (AN-8851-07 §n,
  ES-2026-04 ES-2.n.m, AM004 §n point at companion documents).

### 6.7 CTRL (0x00) — Control

| Bits | Field | Access | Reset | Description |
|---|---|---|---|---|
| [0] | ADC_EN | RW | 0 | Master acquisition enable. 0: SDR engine held in reset, BUSY deasserted, DRDY cleared |
| [1] | WF_EN | RW | 0 | Waveform engine enable (also requires ADC_EN, §8.1) |
| [3:2] | SAFELIGHT_MODE | RW | 0 | 0 = off, 1 = amber (attention), 2 = green (normal), 3 = red (occlusion/thermal event) |
| [4] | LPM | RW | 0 | Low-power mode: acquisition idle, register map live |
| [5] | ALARM_CLR | W1C | 0 | Write 1: clear latched ALARM_STATUS bits and ERR_STAT |
| [6] | SWRST | RW | 0 | Write 1: self-clearing software reset to reset values |

### 6.8 CFG (0x04) — Acquisition configuration (Rev B layout)

| Bits | Field | Access | Reset | Description |
|---|---|---|---|---|
| [1:0] | SDR_WINDOW | RW | 0 | Window period, Table 7-1 |
| [4:2] | OVR_SAMPLE | RW | 0 | Oversampling set, Table 7-2 |
| [5] | AVG_MODE | RW | 0 | 0 = boxcar, 1 = median (§7.1 step 3) |
| [6] | FILT_EN | RW | 0 | Enable the 4-tap FIR on the window result (§7.2 step 4) |
| [7] | rsvd0 | RW | 0 | **Rev-B: reserved, write 0, read 0. Rev-A: this bit was AVG_MODE (§7.2 delta, Table 7-4). Software written against Rev A sets CFG[7] and must be updated.** |
| [9:8] | TH_POL | RW | 0 | Thermal polarity: 0 = fault when T > TH, 1 = fault when T < TH (per-channel pairs, §8.3) |
| [31:10] | rsvd1 | RW | 0 | Reserved, write 0 |

### 6.9 STAT (0x08) — Status

| Bits | Field | Access | Reset | Description |
|---|---|---|---|---|
| [0] | DRDY | RO | 0 | Data-ready flag. Set at SDR window close (§7.1 step 4). **Cleared by the first STAT read that observes it. See ES-2026-04 ES-2.1.3 — a second STAT read in the same window returns 0 for DRDY and sets ERR_STAT.STAT_VIOLATION.** |
| [1] | BUSY | RO | 0 | 1 = SDR engine acquiring (window open). Set at window start, cleared at close |
| [2] | OCCL_DET | RO | 0 | Raw occlusion detect (debounced, §8.2): 1 = window-averaged CH0 ≥ OCCL_TH |
| [3] | TH_FAULT | RO | 0 | 1 = any thermal fault active (§8.3) |
| [7:4] | ERR_CNT | RC | 0 | Latched count of STAT-read violations (ES-2.1.3), saturating 4 bits |
| [15:8] | SEQ | RO | 0 | Window sequence number, increments at each close, wraps |
| [31:16] | rsvd | RO | 0 | 0 |

### 6.10 ALARM_CFG (0x0C) — Occlusion/thermal thresholds

| Bits | Field | Access | Reset | Description |
|---|---|---|---|---|
| [3:0] | OCCL_TH | RW | 0xF | Occlusion threshold, in units of 64 LSB of CH0 (Table 8-3); reset 0xF = 960 LSB ≈ 88 mmHg |
| [7:4] | OCCL_HYST | RW | 0x0 | Hysteresis below threshold before alarm releases, same units |
| [11:8] | OCCL_DEB | RW | 0x0 | Debounce setting N: alarm latches after N+1 consecutive OCCL_DET windows elapse (§8.2 latch geometry) |
| [15:12] | TH_TEMP | RW | 0x1 | Thermal fault threshold, °C offset above 85 °C (0 = 85 °C, 1 = 86 °C, …) |
| [31:16] | rsvd | RW | 0 | Reserved, write 0 |

### 6.11 ALARM_STATUS (0x10) — Latched alarm register

| Bits | Field | Access | Reset | Description |
|---|---|---|---|---|
| [0] | ALARM_OCCL | RO | 0 | Occlusion alarm: `OCCL_DEB+1` consecutive OCCL_DET windows elapsed; latches at the next window close (latch geometry, §8.2). Latched until ALARM_CLR |
| [1] | ALARM_HTEMP | RO | 0 | High-temp fault (TH_POL=0 sense), §8.3 |
| [2] | ALARM_LTEMP | RO | 0 | Low-temp fault (TH_POL=1 sense), §8.3 |
| [3] | ALARM_LINE | RO | 0 | Line-pressure divergence: |CH1 − CH0| > 25 % of CH0 for ≥ 8 windows |
| [4] | ALARM_COMMS | RO | 0 | Comms fault: any SLVERR on the AXI port latches here |
| [5] | ALARM_LATCHED | RO | 0 | Sticky: set when any alarm latches, cleared by ALARM_CLR (controls IRQ) |
| [31:6] | rsvd | RO | 0 | 0 |

### 6.12 IRQ_MASK (0x24) — Interrupt mask

| Bits | Field | Access | Reset | Description |
|---|---|---|---|---|
| [4:0] | M_ALARM[4:0] | RW | 0x1F | Mask per ALARM_STATUS[4:0]; 1 = unmasked |
| [5] | M_DRDY | RW | 0 | 1 = IRQ pulses on window close (DRDY edge) |
| [31:6] | rsvd | RW | 0 | Reserved |

`IRQ` pin asserted when (`ALARM_STATUS[4:0] & IRQ_MASK[4:0]) != 0` OR
`M_DRDY & new-DRDY-edge`.

### 6.13 WAVE_CFG (0x14) — Waveform engine configuration

| Bits | Field | Access | Reset | Description |
|---|---|---|---|---|
| [1:0] | PROFILE | RW | 0 | 0 = off, 1 = profile A, 2 = profile B, 3 = profile C (Table 8-1 + app note) |
| [15:2] | HOLD_MS | RW | 0 | Plateau hold time per step, ms (1..16383); engine cadence = HOLD_MS × 1 ms. Wide enough for the app-note profile defaults (A 500 ms, B 800 ms) |
| [21:16] | STEP_LSB | RW | 0 | Step size, LSB of the output span (1..63) |
| [22] | INVERT | RW | 0 | 1 = invert staircase direction (infusion/withdrawal) |
| [31:23] | rsvd | RW | 0 | Reserved, write 0 |

### 6.14 WAVE_DATA (0x18) — Waveform output

| Bits | Field | Access | Reset | Description |
|---|---|---|---|---|
| [15:0] | WAVE_OUT | RO | 0 | Current engine output, LSB of the 16-bit span; drives WD0..WD3 PWM and the internal occlusion compare (§8.2) |
| [31:16] | rsvd | RO | 0 | 0 |

### 6.15 WAVE_STATUS (0x1C) — Waveform status

| Bits | Field | Access | Reset | Description |
|---|---|---|---|---|
| [0] | RUN | RO | 0 | 1 = engine active (WF_EN & PROFILE ≠ 0) |
| [1] | CYCLE | RO | 0 | 1 = engine in cycle 2..N (profile loop iteration ≥ 2) |
| [2] | DONE | RO | 0 | Set at profile cycle end; cleared by any WAVE_CFG write |
| [31:3] | rsvd | RO | 0 | 0 |

### 6.16 ERR_STAT (0x20) — Error/statistics (read-to-clear)

| Bits | Field | Access | Reset | Description |
|---|---|---|---|---|
| [0] | AXI_BRESP_ERR | RC | 0 | Latched count of dropped/suppressed B responses (hardware self-check; expected 0) |
| [1] | STAT_VIOLATION | RC | 0 | **Latched: >1 STAT read within one SDR window (ES-2.1.3). Set by the destructive second read.** |
| [2] | FIFO_OVF | RC | 0 | Data-port latching overrun (host not draining — non-fatal, window data overwritten) |
| [3] | CRC_ERR | RC | 0 | Internal sequencer CRC check failed |
| [7:4] | rsvd | RO | 0 | 0 |
| [31:8] | W_VIOLATIONS | RC | 0 | Total window-violation count (16-bit, saturating) |

### 6.17 TH_RATE (0x28) — Thermal sampling cadence

| Bits | Field | Access | Reset | Description |
|---|---|---|---|---|
| [7:0] | TH_PERIOD | RW | 0x68 | Thermal channels sampled every TH_PERIOD windows (0 = disabled) |
| [15:8] | TH_FILTER | RW | 0x01 | Median filter length exponent (2^TH_FILTER samples) |
| [31:16] | rsvd | RW | 0 | Reserved |

### 6.18 SAFETY_GATE (0x2C) — Safelight gating

| Bits | Field | Access | Reset | Description |
|---|---|---|---|---|
| [0] | GATE_EN | RW | 0 | Enable safelight gating: SAFELIGHT_MODE forced to red on latched occlusion alarm |
| [1] | AUTO_GREEN | RW | 0 | 1 = driver auto-returns SAFELIGHT_MODE to green when all alarms clear and ALARM_CLR written |
| [31:2] | rsvd | RW | 0 | Reserved |

---

## 7. Acquisition Engine

### 7.1 SDR window

Acquisition is organized in fixed-period windows. Within a window the engine
samples all eight channels with the programmed oversampling set, averages,
and latches the window result (§5).

Table 7-1 — SDR_WINDOW encoding (CFG[1:0])

| CFG[1:0] | Window period | Rate per channel (OVR=1) | Notes |
|---|---|---|---|
| 0 | 1 ms | 1 kS/s | max-rate; high noise floor |
| 1 | 2 ms | 500 S/s | default (see §7.2 step 3) |
| 2 | 4 ms | 250 S/s | |
| 3 | 8 ms | 125 S/s | lowest noise |

Window timing derives from ACLK (10 MHz). Window period jitter is ±5 %
(±1 % typical) due to the internal dividers (§9.2).

Table 7-2 — OVR_SAMPLE encoding (CFG[4:2])

| CFG[4:2] | Oversampling set | Samples/channel/window |
|---|---|---|
| 0 | 1× | 1 |
| 1 | 2× | 2 |
| 2 | 4× | 4 |
| 3 | 8× | 8 |
| 4 | 16× | 16 |

Window sequence (Figure 7-1):

1. Window open: engine asserts BUSY, begins sampling the oversampling set
   on all channels.
2. Sampling completes at `t_sample = window_period × 0.60` (constant fraction;
   the remainder is conversion/settle).
3. Boxcar or median average (CFG.AVG_MODE) over the oversampling set.
4. Window close: FIR-filtered result latched into the data port; DRDY set;
   SEQ incremented; BUSY deasserted; DRDY_N pulses low ≥ 1 ACLK (§7.4.1).

Figure 7-1 (ASCII — window timing, nominal window W):

```
 ACLK   __/‾‾\__/‾‾\__/‾‾\__/‾‾\__/‾‾\__/‾‾\_
        <-- t_sample (=0.60·T_W) --><-- conv/settle -->
 BUSY   ____/‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾\______
        <------------- T_W (window period) --------->
 DRDY_N ______/‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾\____
 close: DRDY set here ─────────────────────────►
```

### 7.1.1 Inside a window — numbers for the §7.2 default

One 2 ms window, four oversampling phases, the plant case from E.1
(WAVE step 12 counts every HOLD — one window every 2 ms, two windows per
HOLD of 500 ms; 10 000 windows per profile-A cycle — Table C-2):

| Phase (t_sample = 1.2 ms) | activity |
|---|---|
| 0 | simultaneous conversion, all 8 channels (1.35 µs worst) |
| T_W·0.6/4 | same, 2nd phase |
| 2× | same, 3rd phase |
| 3× | same, 4th phase |
| t_sample → 1.2..2.0 ms | boxcar average, FIR tap update, compare |
| 2.0 ms (close) | commit atomically (D.4), DRDY set, SEQ++, DRDY_N pulse |

- Sequencer overhead between windows is ZERO by design — windows butt —
  so the aggregate throughputs of Table D-1 are limits (the 0.60 fraction
  already carries the overhead).
- Two windows per HOLD is the reason the AN §4 rise-time tolerance reads
  "within 2 × hold" rather than a millisecond figure: at the §7.2 default
  that bound is 20 plant samples on a 40 ms τ — a comfortable 12.5 τ
  (E.1) — and a stricter test at faster windows is testing window cadence,
  not closed-loop conformance: AN-8851-07 §4 states both the tolerances
  AND the configuration they apply to — conformance runs that
  configuration, never an invented one.


### 7.2 Initialization sequence (Rev B — *the configuration contract*)

Host software must perform the following steps EXACTLY in order after reset.
Values are the ones used by the reference driver (`fw/stm32f4/drv_ads8851.c`)
and by all integration tests.

| Step | Action | Value | Notes |
|---|---|---|---|
| 1 | Wait for reset completion | — | poll STAT until BUSY==0 after RESET_N deassert |
| 2 | `CTRL = 0x0000_0001` | ADC_EN=1 | acquisition enabled, waveform off, mode off, no reset |
| 3 | `CFG = 0x0000_0049` | SDR_WINDOW=1 (2 ms), OVR_SAMPLE=2 (4×), AVG_MODE=0 (boxcar), FILT_EN=1 | **Rev-B layout. Rev-A software writes CFG[7]=1 here (AVG_MODE at bit 7) — that write is IGNORED by Rev-B silicon and must be corrected (Table 7-4).** |
| 4 | `ALARM_CFG = 0x0000_1E0F` | OCCL_TH=0xF (960 LSB), OCCL_HYST=0x0, OCCL_DEB=0xE (14 windows), TH_TEMP=0x1 | occlusion ~88 mmHg, 14-window debounce |
| 5 | `TH_RATE = 0x0000_0168` | TH_PERIOD=0x68 (104 windows), TH_FILTER=0x1 | thermal every ~208 ms |
| 6 | `IRQ_MASK = 0x0000_001F` | all alarm sources unmasked, DRDY IRQ off | safelight + host alarm path |
| 7 | `CTRL = 0x0000_0001` | re-assert ADC_EN (idempotent) | clears any SWRST residue |
| 8 | Read `STAT` once | expect DRDY==0 | confirms clean start (§7.4.2 H-3). BUSY is not checkable here (high whenever a window is open, §7.1) — the BUSY==0 expectation belongs to step 1, pre-enable |

After step 8 the engine free-runs; alarms and DRDY begin after the first
window close (~8 ms @ 2 ms window × FILT pipeline).

Table 7-3 — reference CFG/CTRL values (definitive)

| Register | Value | Meaning |
|---|---|---|
| CTRL | 0x0000_0001 | ADC on only |
| CFG | 0x0000_0049 | 2 ms window, 4× oversample, boxcar, FIR on |
| ALARM_CFG | 0x0000_1E0F | as Step 4 |
| TH_RATE | 0x0000_0168 | as Step 5 |
| IRQ_MASK | 0x0000_001F | as Step 6 |

Table 7-4 — Rev A → Rev B configuration delta

| CFG bit | Rev A (pre-2025-11-02) | Rev B (shipping) |
|---|---|---|
| [5] | rsvd (read 0) | AVG_MODE |
| [7] | AVG_MODE | rsvd0 (write 0, read 0; writes ignored) |

Software that writes `CFG = 0x00C9` (Rev-A init: AVG_MODE=1 at bit 7 + the
Rev-B intended value 0x49) leaves the part in **boxcar averaging with FIR
enabled** — functionally wrong for median-mode pipelines and fails the
`t_fw_init` register-layout test.

### 7.2.1 Step-by-step rationale and failure signature (Table 7-5)

Every §7.2 step exists because of something; the right column is what
breaking it looks like — all four are observed field/fleet classes:

Table 7-5 — why each init step is mandatory, in order

| Step | If skipped / wrong | Caught by |
|---|---|---|
| 1 BUSY pre-poll | writes during reset window race the sequencer's own startup (the FIRST close can land against a half-configured engine; SEQ starts early) | cold-start corner test (Q-1 row 25 family) |
| 2 CTRL ADC_EN first | CFG writes while ADC_EN=0 are legal but a CFG-before-CTRL order has been observed to lose the OVR field latch in the pre-init window on A silicon (Rev B clean; order kept, L-1 note) | §7.2 readback walk (pattern 11) |
| 3 CFG literal Rev-B layout | median pipelines silently run boxcar (bit 5 unset, bit 7 ignored) — noise/impulse expectations drift by B-4/B-2 factors, and the median-specific impulse rejections in the app note stop being true | t_fw_init layout assertion |
| 4 ALARM_CFG (and ES-2.1.1 re-write after any SWRST) | hysteresis 0 after warm restart → boundary chatter of ALARM_OCCL at exactly-960 plateaus (csv E.2 boundary case with hysteresis removed) | warm-restart suite (Q-1 row 44 + re-write check) |
| 5 TH_RATE | thermal cadence wrong: 0 disables → no HTEMP protection at high temps; mis-scaled periods multiply the detection latency column B-12 | init walk + thermal corner battery |
| 6 IRQ_MASK | alarm storms (mask cleared → M_ALARM lines drive IRQ at every latch while the driver expected DRDY-only) or missed alarms (over-masked) | alarm/IRQ matrix pattern 38 |
| 7 CTRL re-assert | no-op on success; the step exists to cover the "SWRST residue" recovery case — skipping it leaves exactly the case designed for uncovered (warm-restart family again) | warm-restart suite |
| 8 single STAT read | double-read by construction on step 8 code written from Rev-A muscle memory — the ES-2.1.3 violation starts BEFORE the run starts, and ERR_CNT is born at 1 (the pcap's opening signature) | pattern 17 conformance, first-read audit |

**Geometry reference note.** The two behaviours with home-grown numbers in
the field — DRDY_N pulse (§7.4.1) and the latch geometry (§8.2) — both
cite Figure O-1/O-2 as the canonical drawings; when a bug report and a
datasheet disagree about "what window number", the answer is Figure O-2's
count, and the regression that keeps it honest is Q-1 pattern 25.


### 7.3 Low-power mode

With `CTRL.LPM=1` the SDR engine powers down after the current window; BUSY
deasserts, DRDY is cleared, and the data port is frozen. Register map stays
live. Exit: write LPM=0; the next window begins ≤ 1 window period later.

### 7.4 Host read protocol

The host reads one window result per SDR window:

- H-1: Rely on the DRDY_N interrupt (`M_DRDY`) or poll STAT.
- H-2: **Read STAT exactly once per window** (ES-2.1.3). The first read that
  observes DRDY==1 also clears it; the window data is now committed and the
  host may read the data port. A second STAT read in the same window observes
  DRDY==0 and raises ERR_STAT.STAT_VIOLATION (hardware self-check).
- H-3: Any AXI access during the window-commit instant is arbitrated; the
  access wins and commit stalls ≤ 2 ACLK. No host-visible data loss.
- H-4: The recommended cadence is: (a) wait BUSY deassert (or DRDY_N), (b)
  single STAT read, (c) if DRDY: read 8-word data port, (d) process.

#### 7.4.1 DRDY_N signal

`DRDY_N` (pin 14) is the hardware ready strobe: it pulses LOW for ≥ 1 ACLK
(≤ 2 ACLK, §9.3) at every window close, coincident with `STAT.DRDY` being
set (§7.1 step 4). The pulse is edge information only — it does NOT carry
the destructive-read semantics of §6.9: sensing DRDY_N tells the host a
window committed; the window result becomes readable after the single
permitted STAT read (H-2).

Two paths leave the device at window close and must not be conflated:

- `DRDY_N` pin: unconditional while `CTRL.ADC_EN=1`; not affected by
  `IRQ_MASK`. With `ADC_EN=0` the pin stays inactive-high; in LPM (§7.3)
  pulsing stops at the final window.
- `IRQ` pin (§6.12): level signal from latched alarms, plus an optional
  DRDY-edge term when `IRQ_MASK.M_DRDY=1`.

Hosts with an external interrupt controller should edge-trigger DRDY_N
(falling) rather than level-sample it: the pulse can be as short as one
10 MHz period. Level pollers should use STAT (once, per H-2), never
DRDY_N state.

#### 7.4.2 Two-read STAT pattern (Rev-A legacy — superseded)

The Rev-A datasheet sequence diagram showed a two-read STAT pattern
(read → "confirm" re-read) as best practice. **This is contradicted by
ES-2026-04 ES-2.1.3 and must not be used on Rev-B silicon.**

---

## 8. Waveform Engine and Occlusion Monitor

### 8.1 AVA-QC staircase profiles

The waveform engine generates the AVA-QC stair-step patterns defined in
`appnote-ava-qc-staircase.md` (profiles A/B/C; step size, hold, cycles,
direction). Host selects a profile via WAVE_CFG.PROFILE and the shared
HOLD_MS/STEP_LSB fields; profile-specific constants are loaded into the
engine at PROFILE write. The engine advances WAVE_DATA by STEP_LSB every
HOLD_MS×1 ms until the profile target sequence completes, then sets
WAVE_STATUS.DONE (and loops if the profile is a loop profile).

Table 8-1 — profile summary (full tables in the app note)

| Profile | Name | Purpose | Steps | Hold (app-note default) |
|---|---|---|---|---|
| A | Inspection | slow staircase up/down | 10 up + 10 down | 500 ms |
| B | Flush | fast staircase up | 8 up | 800 ms |
| C | Prime | micro staircase | 14 up | 200 ms |

Occlusion monitoring is independent: the OCCL compare uses the *pressure
channel* (CH0), NOT WAVE_DATA. During AVA-QC operation the waveform path
drives the pump set-point; CH0 reflects the plant response.

### 8.2 Occlusion detection

Per window: `OCCL_DET = (CH0_avg ≥ OCCL_TH)`. OCCL_DET is debounced: the
occlusion alarm latches at the close of the FIRST WINDOW AFTER `OCCL_DEB+1`
*consecutive* OCCL_DET windows — the debounce count is evaluated at window
close before that window's increment, so e.g. with OCCL_DEB = 2 the latch is
observed on the close of the 4th consecutive OCCL_DET window. The alarm
releases (un-latches) only via ALARM_CLR, re-armed by hysteresis: OCCL_DET
must drop below `OCCL_TH − OCCL_HYST` for the release to stick. After a
release, the debounce count restarts at zero: a re-latch requires a fresh
run of `OCCL_DEB+1` consecutive OCCL_DET windows.

Table 8-3 — OCCL_TH unit conversion (CH0, 64-LSB steps)

| OCCL_TH | CH0 threshold (LSB) | approx. mmHg |
|---|---|---|
| 0x0 | 0 | 0 |
| 0x5 | 320 | 29 |
| 0xA | 640 | 59 |
| 0xC | 768 | 70 |
| 0xF | 960 | 88 |

`OCCL_TH` is 4 bits (Table 6-1, §6.10); `0xF · 64 = 960 LSB` is both the
reset value and the programmable maximum. Thresholds above 960 LSB are not
representable — an application needing a higher trip point gates on the
alarm path as well as the raw channel.

Table 8-2 — OCCL_DEB debounce encoding (latch geometry per §8.2)

| OCCL_DEB (N) | consecutive OCCL_DET windows required | latch observed at close of | latch delay @ 2 ms window |
|---|---|---|---|
| 0 | 1 | 2nd window | 4 ms |
| 1 | 2 | 3rd window | 6 ms |
| 2 | 3 | 4th window | 8 ms |
| 0xE (reset) | 15 | 16th window | 32 ms |
| 0xF | 16 | 17th window | 34 ms |

### 8.3 Thermal faults

TH_FAULT asserts when a temperature channel (CH2/CH3) crosses
`85 °C + TH_TEMP` (TH_POL=0) or drops below the same bound (TH_POL=1).
ALARM_HTEMP/ALARM_LTEMP latch accordingly. Thermal sampling cadence is
TH_RATE.TH_PERIOD windows, median-filtered over 2^TH_FILTER samples.

### 8.4 Safelight interaction (informative summary of §6.7/§6.18)

The safelight chain — operator colour — is the one part of the alarm
path with TWO writers (host via SAFELIGHT_MODE, hardware via
SAFETY_GATE.GATE_EN), so its arbitration is spelled out:

Table 8-4 — effective SAFELIGHT_MODE truth table

| MODE written | GATE_EN | ALARM_OCCL latched | AUTO_GREEN | all alarms clear, ALARM_CLR written | Effective SAFELIGHT_MODE | Notes |
|---|---|---|---|---|---|---|
| x | 0 | any | any | any | x (host value) | gate path inactive |
| x≠3 | 1 | 0 | any | any | x | gate only forces, never adds |
| x | 1 | 1 | 0 | any | 3 (red) | forced RED while the occlusion latch stands; host writes to SAFELIGHT_MODE still succeed but hold no effect until the latch clears |
| x | 1 | 1 | 1 | not yet | 3 | same force |
| x | 1 | 0 (post-CLR) | 1 | yes | 2 (green) | auto-return edge (the only write-like hardware behaviour on this field) |
| x | 0 | 0 | 1 | any | x | AUTO_GREEN inert with gate off — a host that wants green-after-clear outside the gate must write it itself |

- The occlusion latch is the forcing element (ALARM_HTEMP/LTEMP do NOT
  force red through GATE_EN — thermal is a soft warning by comparison;
  the safelight APP in the reference topology treats HTEMP as amber
  software policy, not hardware force).
- SAFELIGHT_MODE is a *lamp command*, not a state mirror: reading it
  returns the commanded value, Table 8-4 tells you what it means.
- The ALARM line and OT_ALARM are separate physical paths (§3 pins
  13/17); a colour reading "green" never substitutes for reading
  ALARM_STATUS (J.5 discipline: colour is UI, registers are evidence).
- safelight-off (mode 0) with an alarm latched is legal silicon state;
  product-level behaviour for that corner belongs to the system spec,
  this datasheet only defines the silicon arbitration above.

---

## 9. Specifications

### 9.1 Absolute maximum ratings

| Parameter | Min | Max | Unit |
|---|---|---|---|
| AVDD | −0.3 | 3.6 | V |
| DVDD | −0.3 | 2.0 | V |
| Input voltage on CHx | −0.3 | AVDD+0.3 | V |
| Junction temp | −40 | 125 | °C |

### 9.2 Recommended operating conditions and DC/AC characteristics

| Parameter | Min | Typ | Max | Unit | Notes |
|---|---|---|---|---|---|
| AVDD / DVDD | 3.0 / 1.62 | 3.3 / 1.8 | 3.6 / 1.98 | V | |
| ACLK | 9.9 | 10 | 10.1 | MHz | ±100 ppm |
| Window period accuracy | −5 | — | +5 | % | vs programmed value (Table 7-1) |
| Conversion time per channel | — | 1.20 | 1.35 | µs | |
| SNR (16× over, boxcar) | — | 82 | — | dB | 1 kHz input |
| SNR (1×, no filter) | — | 71 | — | dB | |
| THD | — | −88 | — | dB | |
| INL | −4 | — | +4 | LSB | |
| Offset error | −8 | — | +8 | LSB | after self-cal |
| Power (acq, 2 ms window, 4×) | — | 12 | — | mW | AVDD+DVDD |
| Power (LPM) | — | 0.9 | — | mW | |
| I/O: V_OH / V_OL | 1.35 / 0 | 1.5 / 0 | — / 0.4 | V | LVCMOS 1.8 |
| Thermal fault accuracy | — | ±2 | — | °C | |

Temperature derating: see `payload/field/thermal-derating.csv` (ADC accuracy
vs ambient, applies to the field-ticket analysis).

### 9.2.1 Test-condition footnotes

- 'after self-cal' = F.2 phases ran at +25 °C within the last power
  cycle; production data (Appendix P) quotes both baked-fuse and
  RAM-overlay results, they agree within the F.2 targets.
- '1 kHz input' SNR reference: −0.5 dBFS sine, THD-not-included (SNR is
  the noise-bin ratio; SINAD in B.5 includes distortion).
- Power specs are both-rail totals at zero pin load (B-8 table pins the
  loaded corners).
- Window period figures are over 256 consecutive closes (Table B-10
  note), because single-cycle measurements hide exactly the ±5 % jitter
  that Table B-11 shows mattering to ES-2.1.3 duty-cycle.

### 9.3 Timing

- AXI write commit: ≤ 4 ACLK after both handshakes.
- B response latency: ≤ 6 ACLK from write commit; back-to-back writes must
  each produce a B within 6 ACLK of their own commit (§4.3).
- First window close after init: 2 × window period + 4 ACLK (FIR pipeline).
- DRDY_N pulse: ≥ 1 ACLK, ≤ 2 ACLK.

### 9.4 Test methods and characterization conditions

| Spec line | Method | Conditions |
|---|---|---|
| SNR 82 dB (16×) / 71 dB (1×) | 1 kHz sine, −0.5 dBFS, 8 k-window record, post-FIR bins, a-weighted-equivalent noise BW (B.5 anchors) | §7.2-16× row of Table B-4, +25 °C |
| INL ±4 LSB | code-density ramp, 16 k codes/ch, all 8 channels | 4× config, self-cal'd |
| Offset ±8 LSB | F.2 OFFSET phase residual windowed mean | production flow |
| Window period ±5 % | ACLK-in/DRDY_N-out, cycle-counter across 256 closes | all four Table 7-1 periods |
| Conversion 1.20–1.35 µs | internal activity tap, wafer monitor, 3 corners × 8 ch | CP station |
| Thermal trip ±2 °C | chamber + B-12 mapped readings, trip and release sweeps ×5 | TH_TEMP corners 0/1/F |
| OCCL latch geometry (Table 8-2) | forced-DET window injection, DEB corners {0, 2, E, F} × four windows — the t_waveform-class corner battery, silicon twin in Q corners | patterns, R.1 row 4 |
| PSRR/CMRR (B-9) | 100 mVpp injection, line-frequency sweep | boxcar off (raw path) |
| Crosstalk (B-6) | worst-case aggressor patterns of M.1/M.3 real circuits + synthetic 90 % FS square | one channel at a time |
| All §6 behaviour | full-map walk + §7.2 init + H-2 conformance + ES corners | FT1, patterns (R column 'coverage') |

Characterization is done on the production image 3.2.x in the loop for
alarm/latch/init rows (the image implements H-2 and the §7.2 contract —
tests of the contract live in firmware-side suites, not in silicon).

---

## 10. Application Information

### 10.1 Reference designs and documents map

| Concern | Reference material |
|---|---|
| Driver init contract | §7.2 (this document), reference driver `fw/stm32f4/drv_ads8851.c` implements steps 1..8 verbatim including the ES workarounds |
| Infusion-profile semantics (steps/holds/cycles, closed-loop tolerances) | `appnote-ava-qc-staircase.md` (AN-8851-07), §6.13 encodings, Appendix C engine behaviour |
| Telemetry service architecture (acquisition-side ring, CSV log, burst policy) | §5 domain rules + Appendix J.1/J.3; the AVV-3000 userspace service is the reference design for a consumer that must never lose a window under status-burst pressure (the admission-policy properties there are the ones its acceptance tests assert) |
| QC panel unit scaling | AN-8851-07 §5 (counts/s conversion rule) — single home for the ×1000 |
| Bus-side invariants | `axi-lite-excerpt.md` (AM004 ch.4 extract) + §4 + D.5 |
| Silicon defects in force | `ES-2026-04-errata.md` + Appendix G matrix |

### 10.2 System integration notes (informative)

- The acquisition side of the system reads exactly one STAT per window
  (H-2) whether the consumer is a driver loop or a diagnostic shell;
  Appendix J.5 restates this for humans with terminals.
- Telemetry consumers should treat SEQ as the window identity (D.6) and
  the in/evicted/rejected counter family as the continuity evidence; the
  alarm path (ALARM_STATUS + SAFETY_GATE) is safe to tap at any rate
  below a window (RO fields — no destructive reads on that path).
- Consumers that pair channels for derived values (differential
  pressure, line divergence) must pair records created from the SAME
  window close — pairing across a host-side lost window fabricates
  divergence (J.3) that the part cannot produce (D.4 atomicity).
- Log formats across the product family share the physical-domain
  columns defined by Table 5-1/Table 5-2: units are SI at the log edge
  (mmHg to one decimal, °C to one decimal, A to three), codes internal —
  a log line that stores codes without its scale map is a field-support
  defect of the same family as Table 6-3's OKAY-lies lesson.

---

## 11. Document Revision History

| Rev | Date | Change |
|---|---|---|
| A | 2025-11-02 | Initial release. CFG bit 7 = AVG_MODE. §7.4.2 two-read STAT sequence. |
| B | 2026-04-11 | CFG bit 7 → rsvd, AVG_MODE moved to bit 5 (Table 7-4). §7.4.2 two-read sequence marked superseded (see ES-2026-04). OCCL_DEB added to ALARM_CFG. ERR_STAT added. |---

# Appendices

The appendices are part of this datasheet's normative text unless marked
*informative*. Register anchors (§6.x) remain definitive; where an appendix
restates a register field it cites the §6.x source rather than redefining
it.

## Appendix A — Initialization and configuration sequences by use case

Each sequence below is complete: run in order after RESET_N deassert, per
the §7.2 contract (steps 1 and 8 apply verbatim to every use case and are
not repeated). Values follow Table 7-3 unless overridden. The reference
driver implements the §7.2 default case; these variants extend it.

### A.1 Use-case overview (Table A-1)

Table A-1 — use-case configuration matrix

| Use case | Window (Table 7-1) | Oversampling (Table 7-2) | AVG | FIR | LPM | Typical consumers |
|---|---|---|---|---|---|---|
| §7.2 default (pump-on) | 2 ms | 4× | boxcar | on | off | telemetry, occlusion, safelight |
| Low-noise bench | 8 ms | 16× | median | on | off | calibration jig, QC panel |
| Max-rate streaming | 1 ms | 1× | boxcar | off | off | waveform-correlation rig |
| Standby duty-cycle | 2 ms | 4× | boxcar | on | on | idle mode between cases |
| AVA-QC inspection (profile A) | 2 ms | 4× | boxcar | on | off | waveform engine + pump |
| AVA-QC prime (profile C) | 2 ms | 4× | boxcar | on | off | waveform engine, micro steps |

Notes:

- The occlusion compare, thermal monitor, and IRQ paths run identically in
  every case; none of them depends on window period or oversampling.
- The waveform engine (§8.1) has its own 1 ms cadence clocked from ACLK and
  is *not* affected by the acquisition window period.
- AVA-QC operation MUST run at the §7.2 default window settings: the app
  note tolerances (AN-8851-07 §4) are derived against the 2 ms / 4× plant
  response, and closed-loop behaviour is not characterised elsewhere.

### A.2 Low-noise bench configuration

Goal: minimum display noise floor for the QC panel and the calibration jig;
throughput is irrelevant.

| Step | Register | Value | Meaning |
|---|---|---|---|
| 1 | — | wait BUSY==0 | §7.2 step 1 |
| 2 | CTRL | 0x0000_0000 | keep ADC off during reconfig |
| 3 | CFG | 0x0000_0073 | window 3 (8 ms) \| OVR 4 (16×) <<2 \| median (bit5) \| FIR (bit6) |
| 4 | ALARM_CFG | 0x0000_1E0F | unchanged from Table 7-3 |
| 5 | TH_RATE | 0x0000_018C | TH_PERIOD=0x8C (140 windows ≈ 1.12 s at 8 ms), TH_FILTER=1 |
| 6 | IRQ_MASK | 0x0000_001F | unchanged |
| 7 | CTRL | 0x0000_0001 | start acquisition |
| 8 | — | read STAT once | §7.2 step 8 |

CFG[7] is rsvd on Rev B (Table 7-4): 0x73 above is the complete legal
write for this configuration (0|1 | 4<<2 | 1<<5 | 1<<6). A driver that
carries the Rev-A habit of setting bit 7 writes 0x0000_00F3 — on Rev B
silicon that bit is ignored, and the median mode this configuration
exists for comes from bit 5, NOT bit 7.

Expected behaviour:

- Per-channel rate falls to 125 S/s with 16 samples averaged per window.
- Measured window-result noise ≈ 1.6 LSB RMS (Table B-4), vs 3.3 LSB RMS
  at the §7.2 default. Median averaging rejects the impulse noise the
  valve-driver stage contributes during closed-valve phases.
- OCCL_DEB timing is in *windows*: at 8 ms, reset-valued debounce (14+1
  windows, Table 8-2) delays the occlusion latch by 128 ms instead of
  32 ms. This is the principal safety trade-off of this configuration.
- SEQ wraps every 256 windows ≈ 2.05 s; telemetry correlating logs by SEQ
  must wrap-adjust (§6.9, §7.1 step 4).

### A.3 Max-rate streaming configuration

Goal: fastest closed-loop visibility of the pump plant; accept the higher
noise floor (Table 7-1 "max-rate; high noise floor").

| Step | Register | Value | Meaning |
|---|---|---|---|
| 2 | CTRL | 0x0000_0000 | reconfig with engine held |
| 3 | CFG | 0x0000_0044 | SDR_WINDOW=0 (1 ms), OVR_SAMPLE=1 (2×), boxcar, FILT_EN=1 |
| 5 | TH_RATE | 0x0000_01D0 | TH_PERIOD=0xD0 (208, =208 ms at the 1 ms window — same real-time thermal cadence), TH_FILTER=1 |
| 7 | CTRL | 0x0000_0001 | run |

- 8 kS/s aggregate; the AXI bus carries one 8-word burst + one STAT read
  per millisecond (§4.5 bus-load rules).
- Window-result noise ≈ 4.6 LSB RMS (2× over-sampled boxcar, Table B-4).
- The occlusion comparator sees noisier window averages: the 6σ-noise
  margin at threshold shrinks (Appendix E.2). This configuration is intended for
  short bench sessions, NOT for clinical use.

### A.4 Standby duty-cycle (LPM cycling)

The AVV-3000 idles between cases with the pump stopped; the safelight
controller wakes the part one window per second for a pressure/alarm
sanity check:

1. Standby config = §7.2 default, plus `CTRL = 0x0000_0011` (LPM=1).
   The engine completes the in-flight window, BUSY clears, the data port
   freezes (§7.3). Register access and AXI arbitration stay live.
2. Wake: `CTRL = 0x0000_0001`. First new window closes per the §9.3
   first-close rule: 2 × window + 4 ACLK ≈ 4.0008 ms after LPM clears.
3. Read: exactly one STAT read (H-2), then the 8-word data port.
4. Re-entry: write LPM=1 at any time; the part idles after the current
   window. Worst-case duty power: ≈ 0.88 mW average at one 4.0008 ms
   wake per second (Table B-8 rows LPM/running) versus 12.0 mW
   continuous.

Pitfalls:

- LPM does NOT clear ALARM_STATUS latches, ERR_STAT counters, or
  WAVE_CFG; a standby unit retains its alarm picture.
- Do not use `CTRL.SWRST` as the standby/exit mechanism: SWRST reloads the
  WHOLE map to reset values (§6.7), including ALARM_CFG and IRQ_MASK —
  the safelight controller would lose its occlusion threshold until
  re-init.

### A.5 AVA-QC bring-up sequence (register view)

Full profile semantics are AN-8851-07; this appendix only pins the
register sequence the engine requires. Shown for profile A.

**HOLD_MS encoding (Rev B, §6.13):** `HOLD_MS` occupies WAVE_CFG [15:2]
and carries the millisecond count *left shifted by 2* (field value =
HOLD_MS × 4). The engine re-derives ms internally from ACLK
(CLKS_PER_MS = 10 000 at 10 MHz; hold_ms × 10 000 must fit the 28-bit
tick counter — bound holds for the full 1..16383 ms field range).
A WAVE_CFG literal is therefore:

    value = PROFILE | (HOLD_MS << 2) | (STEP_LSB << 16) | (INVERT << 22)

Table A-2 — WAVE_CFG literals at AN-8851-07 §3 profile defaults

| Profile | Steps | HOLD (ms) | STEP_LSB | WAVE_CFG literal | Derivation |
|---|---|---|---|---|---|
| A inspection | 10 up + 10 down | 500 | 12 | 0x000C_0FA1 | 1 \| (500<<2) \| (12<<16) |
| B flush | 8 up | 800 | 20 | 0x0014_0C82 | 2 \| (800<<2) \| (20<<16) |
| C prime | 14 up | 200 | 4 | 0x0004_0323 | 3 \| (200<<2) \| (4<<16) |

Software that writes the ms value UNshifted into WAVE_CFG leaves
ms/4 in the field — a 125 ms hold where the app note wants 500; the plant
never forms plateaus and the closed-loop tolerances (AN-8851-07 §4) fail.
Software writing an engine tick count (ms × 10 000) overflows the 14-bit
field and trips the driver's argument check before the bus sees anything.

Sequence:

| Step | Register | Value | Notes |
|---|---|---|---|
| 0 | — | §7.2 default init complete | AVA-QC needs the closed-loop plant running (A.1 note) |
| 1 | WAVE_CFG | 0x000C_0FA1 | profile A defaults per Table A-2 |
| 2 | CTRL | 0x0000_0003 | WF_EN=1; ADC_EN must already be 1 (§8.1 arm rule) |
| 3 | WAVE_STATUS | read | expect RUN=1 |
| 4 | — | poll DONE (or WAVE_DATA trend) | profile-A cycle = 20 steps × HOLD = 10 000 ms |
| 5 | CTRL | 0x0000_0001 | WF_EN=0 at end of session |
| 6 | WAVE_CFG | 0x0000_0000 | disarm; any WAVE_CFG write clears DONE (§6.15) |

The ES-2.1.2 errata (mid-cycle HOLD change ignored) interacts with step 4:
reconfiguring HOLD of a running engine silently keeps the OLD hold until
the next cycle — stop (CTRL.WF_EN=0 or PROFILE=0) and re-arm to apply.

### A.6 Read-path timing budget (informative)

Per-window host work at the §7.2 default (2 ms window, single outstanding
AXI transactions): latency bounds from §9.3, arbitration cases from
§4.4/§7.4 H-3.

Table A-3 — per-window host budget (nominal)

| Phase | AXI beats | Nominal time | Subtotal |
|---|---|---|---|
| STAT read (single — H-2) | 1 AR + 1 R | ≤ 6 + ≤ 6 ACLK + wire | ≤ 2 µs incl. driver |
| Data port read (8 × 32-bit) | 8 AR + 8 R | ≤ 2 µs each pipelined | ≤ 16 µs |
| Worst commit-arbitration stall (§4.4, H-3) | — | ≤ 2 ACLK | 0.2 µs |
| **Bus occupancy vs 2 ms window** | | | **< 1 %** |

The budget is headroom-dominated by design — the field symptom in
`TICKET.md` (service stalls under the 2 Hz STATUS burst) is a SOFTWARE
ring-buffer problem, not this bus budget: at < 1 % occupancy the fabric
cannot starve 500 windows/s + a 2 Hz status stream.

## Appendix B — Characterization data

Unless stated: `AVDD = 3.3 V, DVDD = 1.8 V, ACLK = 10 MHz, TA = +25 °C,
CHx inputs at 0.5 · FS, §7.2 default configuration, Rev B engineering
silicon, production test image 3.2.x`. Values are *measured, design
characterization; not production tested* — production test limits live in
the test-flow document (not part of this datasheet).

Where this appendix interacts with `field/thermal-derating.csv`: that file
is measured field-fleet data; THIS appendix is the spec envelope the csv
data is judged against. The story that matters for the field ticket is
told in B.2.

### B.1 Offset and gain error vs temperature (Table B-1)

After self-cal at +25 °C (Appendix F.2). CH0/CH1 pressure channels; other
channels track within the row figures ±20 %.

Table B-1 — offset/gain error vs ambient (4× oversample, boxcar, FIR on)

| TA (°C) | Offset typ (LSB) | Offset max (LSB) | Gain typ (% FS) | Gain max (% FS) |
|---|---|---|---|---|
| −40 | +4 | ±8 | −0.9 | ±1.5 |
| −25 | +3 | ±8 | −0.6 | ±1.5 |
| −10 | +1 | ±8 | −0.3 | ±1.2 |
| 0 | +1 | ±8 | −0.2 | ±1.2 |
| +10 | 0 | ±8 | −0.1 | ±1.0 |
| +22 | −1 | ±6 | +0.02 | ±0.8 |
| +25 | −1 | ±6 | +0.02 | ±0.8 |
| +37 | +1 | ±8 | −0.10 | ±1.0 |
| +40 | +2 | ±8 | −0.15 | ±1.0 |
| +55 | +14 | ±10 | −0.55 | ±1.2 |
| +70 | +38 | ±14 | −1.20 | ±1.8 |
| +85 | +72 | ±20 | −2.10 | ±2.8 |
| +105 | +118 | — | −3.40 | — |

Reading the +37 °C row for the field ticket: production spec at the
complaint temperature is offset ±8 LSB, gain ±1.0 % FS. The csv's unit
measured +8 LSB / −0.36 % — inside spec at the edge. Even a worst-case
in-spec unit at 37 °C shifts CH0 by < 100 LSB (offset + gain × 0.5 FS
worst case), against the 960 LSB threshold and 3-4 LSB noise floor: the
margin table in the csv (864 LSB at 37 °C) cannot be explained by
in-spec drift. Thermal ACCURACY is a decoy path; see §9.2 pointer in the
csv header and Appendix J for the diagnostic matrix that closes it.

### B.2 Window-result noise vs temperature and oversampling (Table B-4)

Table B-4 — CH0 window-result noise, RMS LSB (boxcar, FIR on; median mode
runs ≈ 0.8× on Gaussian noise and materially lower on impulse noise)

Typical values; production envelope max ≈ 1.8× typ. Cross-check vs §9.2:
1× typ (6.5 LSB RMS) ↔ SNR 71 dB; 16× typ (1.6 LSB) ↔ SNR >82 dB before
the FIR noise-gain term — the §9.2 spec point (82 dB, 16×, 1 kHz) sits on
this table's 16× column.

| Oversampling set | 25 °C | 37 °C | 50 °C | 60 °C | 75 °C | 85 °C |
|---|---|---|---|---|---|---|
| 1× | 6.5 | 7.4 | 9.7 | 13.1 | 19.8 | 31.1 |
| 2× | 4.6 | 5.2 | 6.9 | 9.3 | 14.0 | 22.1 |
| 4× (§7.2 default) | 3.3 | 3.7 | 4.9 | 6.6 | 9.9 | 15.6 |
| 8× | 2.3 | 2.6 | 3.4 | 4.6 | 7.0 | 11.0 |
| 16× | 1.6 | 1.9 | 2.4 | 3.3 | 5.0 | 7.8 |

Reading guidance (field-ticket analysis):

- A healthy unit at the §7.2 default shows ≈ 3.3 LSB RMS typ at room
  temperature (production envelope max ≈ 6); the fleet csv's healthy rows
  show 4-5 LSB system floor at 22-28 °C — spec + plant term ✓.
- The csv's affected unit shows 16 LSB at 37 °C against a Table B-4
  envelope max of ≈ 7 LSB (4× row): the unit is noisy
  OUT OF ENVELOPE — but 6σ noise (≈ 96 LSB) still leaves 864 LSB of
  occlusion margin at threshold. Noise alone does NOT fire the alarm at
  37 °C. (The csv's own margin column agrees: 864 LSB.)
- Genuine margin collapse is a ≥ 70 °C phenomenon (csv rows from 75 °C;
  Table B-4 75 °C column): in that zone ALARM_HTEMP (§8.3) is specced to
  assert first — a unit alarming at 37-40 °C with margin > 100 LSB is
  following a different path: §7.4.2, ES-2.1.3.

### B.3 AC performance vs input frequency (Table B-5)

16× oversampling, boxcar + FIR, TA = +25 °C; FS = 32767 codes.

Table B-5 — SNR / THD / SINAD / ENOB vs f_in, §7.2 config

| f_in (Hz) | SNR (dB) | THD (dB) | SINAD (dB) | ENOB (bits) | Notes |
|---|---|---|---|---|---|
| 0.5 | 82.4 | −96 | 82.4 | 13.4 | plant-band region |
| 2 | 82.3 | −94 | 82.3 | 13.4 | plant-band region |
| 5 | 82.2 | −92 | 82.1 | 13.4 | plant-band region |
| 10 | 82.2 | −90 | 82.0 | 13.4 | plant-band region |
| 20 | 82.1 | −89 | 81.9 | 13.4 | |
| 50 | 82.0 | −88 | 81.8 | 13.4 | 1 |
| 100 | 81.8 | −87 | 81.6 | 13.3 | |
| 200 | 81.2 | −86 | 80.9 | 13.2 | 2 |
| 400 | 80.4 | −85 | 80.1 | 13.0 | 2 |
| 500 | 80.1 | −85 | 79.8 | 13.0 | 2 ms-window Nyquist = 250 Hz; extrapolated |

Notes:
1. §9.2 typ SNR = 82 dB corresponds to this configuration; §9.2 quotes the
   16× figure at a 1 kHz reference tone from the raw-characterization
   setup (pre-window): both anchor points are consistent with Table B-4.
2. Above the window Nyquist the result aliases; quoted post-FIR — the 4-tap
   FIR (§7.1 step 4) rolls ≈ 0.8 dB at 250 Hz (2 ms window).

### B.4 Channel crosstalk (Table B-6)

Aggressor: full-swings 0.1..0.9 · FS, 50 Hz square. Victim: 0.5 · FS held.
§7.2 config, TA = +25 °C. Cells = victim (row) due to aggressor (column),
dB; worst of n = 20 units.

Table B-6 — mutual crosstalk matrix (dB)

| victim \ aggressor | CH0 | CH1 | CH2 | CH3 | CH4 | CH5 | CH6 | CH7 |
|---|---|---|---|---|---|---|---|---|
| CH0 | — | −104 | −112 | −111 | −99 | −108 | −118 | −118 |
| CH1 | −104 | — | −112 | −111 | −99 | −108 | −118 | −118 |
| CH2 | −110 | −110 | — | −109 | −101 | −109 | −118 | −118 |
| CH3 | −110 | −110 | −109 | — | −101 | −109 | −118 | −118 |
| CH4 | −103 | −103 | −111 | −110 | — | −107 | −117 | −117 |
| CH5 | −108 | −108 | −112 | −111 | −106 | — | −117 | −117 |
| CH6 | −112 | −112 | −113 | −112 | −108 | −111 | — | −119 |
| CH7 | −112 | −112 | −113 | −112 | −108 | −111 | −119 | — |

The 0..5 A drive-current sense channel (CH4) is the loudest aggressor and
the thermistor channels the most susceptible victims — the pair a
hypothetical "crosstalk explains the false alarm" theory needs. Worst
case CH4→CH2/CH3 ≈ −101 dB ≈ 3 LSB at full scale: an order of magnitude
below the Table B-4 noise term. Crosstalk is not a false-alarm path, and
cannot couple CH0 (the occlusion input) at anything a debounce of 14
windows would not average away.

### B.5 Channel-to-channel matching (Table B-7)

Six lots, 24 units per lot, §7.2 config, post self-cal. Relevant to
ALARM_LINE (§6.11: |CH1 − CH0| > 25 % of CH0 for ≥ 8 windows).

Table B-7 — CH0 ↔ CH1 matching (the ALARM_LINE pair)

| Statistic | 25 °C | 37 °C | Unit |
|---|---|---|---|
| Offset mismatch mean | +0.4 | +1.1 | LSB |
| Offset mismatch σ | 0.6 | 0.9 | LSB |
| Gain mismatch mean | +0.03 | +0.03 | % FS |
| Gain mismatch σ | 0.05 | 0.07 | % FS |
| Worst-case |CH1 − CH0| at 3000 mmHg | 11 | 19 | LSB |

At 3000 mmHg full scale the worst matched-pair divergence is 19 LSB ≈
0.16 % — two orders of magnitude below the 25 % ALARM_LINE threshold. See
Appendix J.3 for the one false-trigger geometry that ALARM_LINE has
(dropped window on one channel only).
### B.6 Power and supply current (Table B-8)

Table B-8 — power by configuration (AVDD = 3.3 V, DVDD = 1.8 V, TA = +25 °C,
no load on WDx/DAx, IRQ/DRDY_N pins unloaded; totals include both supplies)

| Configuration | CTRL | CFG | I_AVDD (mA) | I_DVDD (mA) | Total (mW) | Notes |
|---|---|---|---|---|---|---|
| Full power-down | 0x00 | — | 0.21 | 0.03 | 0.77 | ADC_EN=0, engine held |
| LPM (§7.3) | 0x11 | 0x49 | 0.21 | 0.04 | 0.83 | register map live |
| §7.2 default running | 0x01 | 0x49 | 2.85 | 0.92 | 12.1 | 1 |
| Max-rate (A.3) | 0x01 | 0x44 | 2.91 | 0.95 | 12.4 | 1 ms window |
| Low-noise (A.2) | 0x01 | 0x73 | 2.62 | 0.88 | 11.4 | 8 ms window (idle between windows) |
| Waveform engine active | 0x03 | 0x49 | 2.87 | 0.95 | 12.5 | + WDx PWM drive |
| All alarms latched | 0x01 | 0x49 | 2.85 | 0.95 | 12.2 | latch logic is static |

Notes:
1. §9.2 typ = 12 mW anchors this row (2 ms / 4× / FIR).
2. LPM 0.83 mW plus wake duty at 1 window/s reproduces §9.2's LPM typ of
   0.9 mW with the §9.3 first-close penalty included (A.4 step 2).
3. WDx 16-level PWM (Appendix C) adds ≤ 0.4 mW into the LVCMOS pads
   depending on duty; valve-driver gate current is external.

### B.7 Supply rejection and common-mode (Table B-9)

Table B-9 — PSRR / CMRR vs frequency (4× oversample, §7.2 config, ±100 mV
excursion on the supply/common mode, TA = +25 °C)

| Frequency | PSRR AVDD (dB) | PSRR DVDD (dB) | CMRR CH0/CH1 (dB) |
|---|---|---|---|
| DC | 74 | 68 | 96 |
| 10 Hz | 73 | 67 | 96 |
| 50 Hz | 71 | 65 | 95 |
| 60 Hz | 71 | 65 | 95 |
| 100 Hz | 68 | 62 | 93 |
| 1 kHz | 58 | 55 | 86 |
| 10 kHz | 46 | 48 | 74 |
| 100 kHz | 34 | 41 | — |

The boxcar window integrates over the full sampling phase
(t_sample = 0.60 · T_W, §7.1): at integer-multiple-of-50/60 Hz window
periods the rejection shows a comb peak of +6 dB typical at 50 Hz for the
2 ms window (T_W = 100 ms = 5 line periods — the §7.2 default is a
line-frequency-hum-friendly choice).

### B.8 Timing parameters (Table B-10)

Normative bounds; §9.3 prose is the summary, this table is the detail.
ACLK = 10 MHz, 1 ACLK = 100 ns.

Table B-10 — AXI and acquisition timing parameters

| Symbol | Parameter | Min | Max | Unit | Notes |
|---|---|---|---|---|---|
| t_AXC | AW/W handshake → write commit | — | 4 | ACLK | §9.3 |
| t_B | write commit → BVALID | — | 6 | ACLK | §4.3/§9.3; every write, back-to-back included |
| t_ARC | AR handshake → RVALID | — | 6 | ACLK | §9.3 |
| t_ARB | window-commit arbitration stall | — | 2 | ACLK | §4.4 H-3, register access wins |
| t_FCL | first window close after §7.2 | 2·T_W + 4 | 2·T_W + 8 | ACLK | FIR pipeline (§9.3) |
| t_DP | DRDY_N pulse width | 1 | 2 | ACLK | §7.4.1 |
| t_SEQ | STAT read → SEQ visible increment | — | 0 | ACLK | SEQ updates at window close, mid-read reads see closed value |
| t_RST | RESET_N low → register reset | 2 | — | ACLK | async assert, sync release |
| t_RSY | RESET_N release → BUSY clear | — | 8 | µs | §7.2 step 1 poll bound |
| t_OSL | OCCL_DET update after window close (STAT visibility) | — | 1 | window | debounced compare at close |
| t_WST | WAVE_STATUS.RUN after arm | — | 1 | ms | engine ms-cadence (§8.1) |
| t_WDO | DONE observable after cycle end | — | 100 | ns | one ACLK after last tick |

### B.9 ACLK quality sensitivity (Table B-11)

Window period, engine cadence and debounce all derive from ACLK
(Table 7-1, §8.1, §8.2). Tolerance stack for the field symptom:

Table B-11 — effects of ACLK deviation (10 MHz nominal)

| ACLK error | Window period (2 ms) | t_sample drift | 14+1 debounce drift | ES-2.1.3 phase coupling |
|---|---|---|---|---|
| ±0.01 % (spec ±0.001 % typ) | ±200 ns | ±120 ns | ±30 µs | negligible |
| ±1 % | ±20 µs | ±12 µs | ±3 ms | negligible |
| ±5 % (§9.2 window jitter max) | ±100 µs | ±60 µs | ±15 ms | jitter phase start to matter for second-read placement (§7.4.2 ES note) |

The ±5 % window jitter in §9.2 is the datasheet's stated envelope; ES-2.1.3
explicitly names it as the reason the field failure lands at
"1-in-5000"-type rates rather than every window: the illegal second STAT
read crosses window boundaries only when jitter aligns. This is a read
protocol violation amplified by jitter — not a jitter defect: the data
port content is always correct at close (§7.1 step 4).

### B.10 On-die thermal monitor (Table B-12)

Table B-12 — thermal channel (CH2/CH3) characterization

| Parameter | Value | Unit | Notes |
|---|---|---|---|
| Thermistor pairing | CH2 = head, CH3 = system | — | §5, Table 5-1 |
| Fault trip point | 85 + TH_TEMP | °C | §6.10, §8.3 |
| Trip accuracy | ±2 (typ) | °C | §9.2 |
| Hysteresis | 2 °C (fixed) | °C | prevents trip/chatter at boundary |
| Sampling period | TH_PERIOD windows | — | §6.17; 0 disables compare |
| Filter | median of 2^TH_FILTER samples | — | §6.17 |
| Detection latency (reset TH_RATE) | ≤ 208 ms + filter | ms | TH_PERIOD=104 × 2 ms ≈ 208 ms (Table 7-3 step 5) |
| Latching | ALARM_HTEMP/LTEMP latch until ALARM_CLR | — | §6.11 |

### B.11 Per-channel statistics (Table B-13)

Pooled lots of Appendix P, §7.2 config, +25 °C, post-image self-cal:

Table B-13 — per-channel DC/AC spread (n = 9 039 units)

| Ch | Offs typ | Offs σ | Gain typ | Gain σ | INL typ | SNR 4× typ | dominant channel note |
|---|---|---|---|---|---|---|---|
| CH0 | −1.0 | 1.0 | +0.02 % | 0.07 % | ±1.5 | 74.5 dB | reference channel for E.2 margins |
| CH1 | −0.9 | 1.0 | +0.02 % | 0.07 % | ±1.4 | 74.6 dB | pair stats B-7 |
| CH2 | +1.4 | 1.3 | −0.03 % | 0.09 % | ±1.8 | 73.6 dB | mapping LUT adds noise term (M.2) |
| CH3 | +1.3 | 1.3 | −0.03 % | 0.09 % | ±1.8 | 73.7 dB | idem |
| CH4 | −0.6 | 1.2 | +0.05 % | 0.08 % | ±2.0 | 73.9 dB | PWM-side coupling floor (B-6 CH4 row) |
| CH5 | −0.4 | 1.1 | +0.04 % | 0.08 % | ±1.9 | 74.0 dB | 4095-domain halves absolute LSB |
| CH6/7 | ±0.5 | 1.2 | ±0.05 % | 0.08 % | ±1.9 | 74.3 dB | unallocated (not tested per-unit) |

### B.12 Long-term drift (Table B-14)

Accelerated read-back of the lot wafers' scribe monitors (informative):

Table B-14 — drift after HTOL-class stress (H.2 conditions)

| Parameter | 1 000 h @ +125 °C drift | Units |
|---|---|---|
| Offset (post-cal) | +2 ± 3 | LSB |
| Gain | −0.15 ± 0.05 | % FS |
| Thermal trip point | ±0.4 | °C |
| OCCL compare | ±4 | LSB |
| Noise (4×, 25 °C) | +0.3 | LSB |

Drift is inside every Appendix Q limit with margin: the safety-relevant
line (OCCL compare ± 4 LSB drift) consumes 0.4 % of the 960-LSB
threshold — the fleet's alarm-path drift story stays
behavioural-not-silicon through the unit's declared life.



## Appendix C — Waveform engine reference (normative companion to §8.1)

State machine, cadence and per-profile timing for the AVA-QC engine
(§6.13/§6.14/§6.15). The engine's timebase is 1 ms derived from ACLK
(CLKS_PER_MS = 10 ACLK cycles per MHz of ACLK = 10 000 at 10 MHz); one
staircase step per HOLD_MS of that timebase. The engine runs off the
acquisition windows entirely: window period and oversampling do not
influence it (A.1 note).

### C.1 States and transitions (Table C-1)

Table C-1 — engine state machine

| State | Meaning | Entry | Duration | Exit |
|---|---|---|---|---|
| IDLE | output idle; wave_out retains last value | reset, PROFILE=0, WF_EN=0 | — | arm (PROFILE≠0, WF_EN=1) → RAMP_UP |
| RAMP_UP | one step every HOLD at STEP_LSB, direction per INVERT | arm | steps_up × HOLD | HOLDTOP=0: RAMP_DN (loop) or DONE_ST (one-way); HOLDTOP≠0: HOLD_TOP |
| HOLD_TOP | top dwell at cycle peak (profile B only) | ramp up complete | HOLDTOP ms | RAMP_DN |
| RAMP_DN | steps back toward base at HOLD cadence | top dwell done | steps_dn × HOLD | loop → cycle++ → RAMP_UP; one-way → DONE_ST |
| DONE_ST | DONE set; output parked at final value | one-wave cycle end | until WAVE_CFG write or disarm | any WAVE_CFG write clears DONE; re-arm restarts |

- WAVE_STATUS.RUN = active in RAMP_UP / HOLD_TOP / RAMP_DN (not IDLE, not
  DONE_ST).
- WAVE_STATUS.CYCLE = 1 once the engine is in loop iteration ≥ 2 (profiles
  A, B); profile C never sets CYCLE.
- DONE: loop profiles pulse DONE once per *full* cycle (down-ramp end),
  the engine continues; one-way profiles set DONE sticky (clears only on
  WAVE_CFG write — §6.15).
- wave_q retention on stop: PROFILE=0 or WF_EN=0 freezes and *retains*
  WAVE_DATA at its current value (the pump holds last set-point; no
  auto-return-to-zero — the host owns the actuator path, §8.1).

### C.2 Per-profile nominal timing (Table C-2)

At AN-8851-07 §3 defaults (Table A-2 literals):

Table C-2 — cycle timing at profile defaults

| Profile | Steps/cycle | Top dwell | Cycle length | Done behaviour | Loop period |
|---|---|---|---|---|---|
| A (500 ms/12 LSB) | 10 up + 10 dn | — | 20 × 500 = 10 000 ms | pulse per cycle, CYCLE from iter 2 | repeats until disarm |
| B (800 ms/20 LSB) | 8 up + 8 dn | 1 200 ms | 16 × 800 + 1 200 = 14 000 ms | pulse per cycle | repeats until disarm |
| C (200 ms/4 LSB) | 14 up | — | 14 × 200 = 2 800 ms | sticky at line charge complete | one-shot |

Reaching-peak note for the closed-loop trace (AN-8851-07 §4): profile A
platform targets are 12 · n LSB after n-th tick; the plant (Appendix E;
`sim/plant.c` characterizes the physics referenced by AN §4) settles well
inside HOLD for these holds — plateau-end accuracy, 95 % rise
(2 × HOLD) and overshoot (≤ 10 % of step) all hold at the §3 defaults,
which is exactly the configuration that `fv_staircase`-class closed-loop
tests should drive (a test substituting its own faster hold is testing
a configuration the app note never specified).

### C.3 Configuration-change matrix (Table C-3)

Effect of a WAVE_CFG write while the engine is running (RUN=1 or a loop
mid-cycle). Reconciles §6.13 with ES-2026-04 ES-2.1.2:

Table C-3 — mid-cycle WAVE_CFG writes

| Field changed | RUN/DONE effect | Applies |
|---|---|---|
| PROFILE → 0 | stop; output retained (C.1) | immediately |
| PROFILE → other | reconfigure engine; current iteration restarts from base with new profile constants | immediately |
| HOLD_MS only | NO effect on the running cycle — the cycle's cadence was latched at cycle start | next cycle start (loop profiles); for one-way profiles running to end, never |
| STEP_LSB | direction/size re-encode continues from current wave_q | immediately |
| INVERT | direction flag flips on the *next* tick | at next HOLD tick |
| re-arm after DONE_ST (WAVE_CFG any) | DONE clears (any WAVE_CFG write, §6.15) | immediately |

Software consequence: a QC recipe push that retunes HOLD for the next
delivery must land between cycles (host-side), or the engine must be
stopped and restarted between cycles, or the change is a no-op this cycle
and silently applies next cycle — matching ES-2.1.2's documented
deferred-apply behaviour and the ES-2.1.2 assertion in t_waveform-class
tests.

### C.4 Step-value geometry (Table C-4)

Staircase targets at the §3 defaults (counts of the 16-bit output span;
WAVE_DATA is the 16-bit LSB mirror, §6.14):

Table C-4 — WAVE_DATA sequence (INVERT=0)

| Tick | A target | B target | C target |
|---|---|---|---|
| 0 (arm) | 0 | 0 | 0 |
| 1 | 12 | 20 | 4 |
| 2 | 24 | 40 | 8 |
| 3 | 36 | 60 | 12 |
| 4 | 48 | 80 | 16 |
| 5 | 60 | 100 | 20 |
| 6 | 72 | 120 | 24 |
| 7 | 84 | 140 | 28 |
| 8 | 96 | 160 (then HOLD_TOP at 160 for 1 200 ms) | 32 |
| 9 | 108 | 140 (down-ramp) | 36 |
| 10 | 120 (peak) | 120 | 40 |
| 11 | 108 | 100 | 44 |
| 12 | 96 | 80 | 48 |
| 13 | 84 | 60 | 52 |
| 14 | 72 | 40 | 56 — final step, DONE sticky |
| 15 | 60 | 20 | — (engine at DONE_ST) |
| 16 | 48 | 0 (cycle end, DONE pulse) | — |
| 17 | 36 | (loop restart) | — |
| 18 | 24 | — | — |
| 19 | 12 | — | — |
| 20 | 0 (cycle end, DONE pulse, loop) | — | — |

Profile C stops at 56 counts (14 × 4); every C plateau increment (4
counts) is only ≈ 7 % of the way between the raw ADC quantisation
steps at the plant input, which is why closed-loop plateau checks on C
run noise-free in verification (AN-8851-07 §4 tolerances at 4 counts are
larger than the sensor-noise term; the sim runs C with plant noise
suppressed for exactly this reason, E.4).

### C.5 Interaction with the occlusion path

The engine does NOT drive the occlusion comparator directly (§8.1: the
OCCL compare reads the CH0 *pressure channel*). During AVA-QC runs the
plant (Appendix E, DC gain ≈ 6) transfers WAVE_DATA steps into CH0
pressure: the highest profile-A plateau (120 counts engine) lands around
720 counts on CH0 — ~75 % of the 960-LSB occlusion threshold, i.e. 25 %
of headroom (E.2). Only a fault path (occluded line, or the field-ticket
scenario) pushes CH0 past the trip from a legal AVA-QC state.
A full-bore occlusion fault during a run rides WAVE_DATA up while the
plant saturates against its relief ceiling long before the counter's
32767 full scale (Appendix E.5) — an alarm during AVA-QC means the line,
not the engine.
## Appendix D — SDR sequencer and bus internals

Normative companion to §4, §5, §7.1: what the arbitration, filtering and
result-latching logic actually do, at the level integrators need when a
test hangs on a bus timing question.

### D.1 Window arithmetic (Table D-1)

All windows derive from ACLK. Conversion time per channel is 1.20–1.35 µs
(§9.2); the eight channels sample *simultaneously* within one conversion
phase, so an oversampling set of N fills N conversion phases inside
t_sample = 0.60 · T_W (§7.1 step 2). Maximum oversampling supported by
each window is bounded by `floor(t_sample / 1.35 µs)` phases:

Table D-1 — window capacity (max T-table oversampling always capped by the
Table 7-2 menu; capacity shown for awareness)

| SDR_WINDOW | T_W | t_sample | phases available | Table 7-1 rate @ OVR 1× | aggregate sample rate @ menu cap |
|---|---|---|---|---|---|
| 0 (1 ms) | 10 000 tks | 6 000 tks (600 µs) | 444 | 8 kS/s agg | 8 kS/s (1× per Table 7-1 note) |
| 1 (2 ms) | 20 000 | 12 000 (1.2 ms) | 888 | 4 kS/s agg | 4 kS/s |
| 2 (4 ms) | 40 000 | 24 000 (2.4 ms) | 1777 | 2 kS/s agg | 2 kS/s |
| 3 (8 ms) | 80 000 | 48 000 (4.8 ms) | 3555 | 1 kS/s agg | 1 kS/s |

The remainder of the window after t_sample (0.40 · T_W) is conversion
settle, averaging/FIR, commit and the quiet margin the DRDY_N pulse
(§7.4.1) times against. Windows never overlap; BUSY spans the active
phase (§7.1 Figure 7-1).

### D.2 4-tap FIR (Table D-2)

`CFG.FILT_EN` selects a symmetric 4-tap filter applied to the per-window
result sequence (one input per window — this is a window-to-window
smoother, *not* an intra-window decimator):

Table D-2 — FIR coefficients (fixed-point, /8 scale)

| Tap | 3 | 2 | 1 | 0 |
|---|---|---|---|---|
| coefficient | 1 | 3 | 3 | 1 |

- Transfer: linear-phase low-pass; −0.8 dB at the 2 ms-window Nyquist
  (basis of Table B-5 note 2); DC gain exactly 1 (sum = 8/8).
- Delay: 1.5 windows, rounded in the pipeline to 2 window-close instants
  plus 4 ACLK — the source of the §9.3 first-close figure
  `t_FCL = 2·T_W + 4 ACLK` (Table B-10).
- Filtration state survives LPM entry/exit without reset (a half-filled
  history continues); a SWRST or STOP (ADC_EN 1→0) flushes history to the
  current window input (no zero-stuffing artefacts at re-start).
- The occlusion compare (§8.2) reads the window AVERAGE *before* the FIR:
  the protection path is never delayed by FILT_EN. Telemetry reads the
  filtered data port. The two paths can therefore differ by the FIR
  response to a step — by design.

### D.3 Median mode (informative)

`CFG.AVG_MODE=1` replaces the boxcar mean with an order-statistic pick
over the oversampling set: for set size N the engine sorts the N per-phase
results and takes element `floor(N/2)` (lower median of the upper pair for
even N — fixed behaviour, not configurable; Table D-3). Median mode
interacts with the FIR exactly as the boxcar path does (D.2).

Table D-3 — median pick geometry

| Set (Table 7-2) | sorted elements | pick | impulse sensitivity (1 bad sample in N) |
|---|---|---|---|
| 1× | {x0} | x0 | none available |
| 2× | {x0,x1} sorted | x1 (upper) | high — impulse survives if it is the larger |
| 4× | 4 sorted | index 2 | none (single impulse off-median) |
| 8× | 8 sorted | index 4 | none |
| 16× | 16 sorted | index 8 | none |

Use median when the impulse source is real (valve-driver kick, csv B.2
story) and accept ≈ 0.8× Gaussian noise advantage vs boxcar (B.2 reading
guidance); use boxcar for the plant trace paths.

### D.4 Data-port commit and overrun (Table D-4)

The 8-word result set commits atomically at window close (§7.1 step 4):
the data port never shows a half-updated set — readers cannot observe a
mix of two windows (§5 "stable until next close"). The one legitimate
stale-data case is FIFO_OVF: the engine completed a fresh commit while the
host still hadn't read the *previous* one.

Table D-4 — overrun semantics and what is never at risk

| Event | Data port | ERR_STAT.FIFO_OVF | STAT.DRDY | window SEQ |
|---|---|---|---|---|
| Host drains each window | fresh each close | stays 0 | per H-2 | +1 per window |
| Host skips m windows | latest values | set (sticky, RC) | set at latest close | +m (SEQ advanced — no phantom windows) |
| LPM freeze | frozen | not set by freeze | cleared (§7.3) | pauses |
| SWRST | resets to 0 | clears | clears | 0 |

FIFO_OVF never implies corruption of the *current* set, only loss of
history. Telemetry designs that must not miss windows (the svc ring
admission policy) count SEQ deltas, not DRDY edges — the same conclusion
the service's zero-window-gap acceptance arrives at from the software
side.

### D.5 AXI slave internals (informative; normative text is AM004 + §4)

The slave is 1-outstanding with a 2-deep response FIFO:

- AW/W may arrive in either order (§4.1); the write commits when
  address+data+enable are all present, ≤ 4 ACLK (t_AXC).
- Each committed write enqueues exactly one B; the 2-deep FIFO plus the
  in-flight slot means a master that lets two Bs pend still gets both
  responses (the property t_bus-class tests hammer: back-to-back writes,
  B for every write, in order, ≤ 6 ACLK after each commit).
- Reads to the data port while a commit is in flight are arbitrated (§4.4
  H-3): the AXI access wins, its R data is the *old* set, commit proceeds
  on the next ACLK.

Table D-5 — arbitration cases (all combinations, one cycle each: A
= engine, X = AXI)

| Case | Condition | Outcome | Side effects |
|---|---|---|---|
| A-1 | commit at close cycle | AXI delayed ≤ 2 ACLK | none |
| X-1 | STAT read during commit arbitration | register block wins; read sees old SEQ/DRDY then closed values next cycle | none |
| X-2 | STAT read in same ACLK window as commit close | read may observe pre-close state; **this still consumes the one legal read if it observes DRDY** — a subsequent read that finally sees DRDY=1 never happens because DRDY was consumed pre-close; host misses one window's ready edge but not its data, and no violation is counted | ERR_CNT unchanged if reads ≤ 1/ window |
| X-3 | CTRL write during commit | write wins | commit resumes next cycle |
| X-4 | SWRST during commit | reset wins; commit discarded | data port resets |
| X-5 | second STAT read in a window (any phase incl. jitter across close) | read returns DRDY=0 and trips ES-2.1.3 | STAT_VIOLATION + ERR_CNT++ |

Case X-2 exists because the field units showed the "no-DRDY" branch of
ES-2.1.3 itemised list: with jitter, a Rev-A double-read driver
sometimes places its confirm read across a window close. The jitter
(Table B-11) decides which ES-2.1.3 flavour each miss becomes —
the same violation, two faces.

### D.6 Counters and wrap guidance

- `STAT.SEQ` (8-bit, §6.9): primary window identity for log correlation.
  Wrap is plain 255→0; there is no wrap flag. Consumers must difference
  modulo 256 and sanity-check against the monotonic wall/heartbeat clock.
- `STAT.ERR_CNT` (§6.9): sticky 4-bit saturating. Stays at the
  saturation value forever; read-clears on one read. A fleet unit shows
  small ERR_CNT with long inter-increment gaps under read-violation — the
  ES-2.1.3 fingerprint — while FIFO overruns produce a *burst* increment
  (per the svc burst model, J.1 row 5).
- `ERR_STAT.W_VIOLATIONS` (§6.16, 16-bit): every ES-2.1.3 read violation;
  this counter grows at ~window-rate when a broken driver runs — see J.2
  why the field ticket is the exception ("1-in-5000 means the driver is
  mostly *not* double-reading in the sampled path").

## Appendix E — Occlusion / plant application notes (informative)

The occlusion detector (§8.2) is plant-agnostic silicon: what makes the
AVV-3000 installation correct (or wrong) is the interaction of threshold,
noise and the fluidic plant. The reference platform below is the AVV-3000
pump head as characterized for AN-8851-07 §4; the closed-loop trace test
of the app note and `fv_staircase`-class verification all assume it.

### E.1 Reference plant model

First-order compliance model of pump → tube → camera sheath:

    p(k+1) = p(k) + (G·u(k) − p(k)) · dt/τ + n(k)

with dt = 1 ms (engine cadence), reference-platform parameters:

| Symbol | Value | Meaning | Source |
|---|---|---|---|
| G | 6.00 counts out / count in | set-point → steady-state CH0 pressure | AN-8851-07 §4 derived (peak 720 at engine A top 120) |
| τ | 40 ms | time constant (95 % in 3τ ≤ 2× hold for every AN profile hold) | AN-8851-07 §4 |
| relief | 32 000 counts | pass-valve ceiling (below 32 767 full scale) | platform safety spec |
| n(k) | ±1 count, seeded | sensor/plant noise term | AN-8851-07 §4 |

Table E-1 — plateau expectation (profile A, engine step 12)

| Plateau n | Engine target (counts) | Expected CH0 (counts) | Δ to 960-LSB occlusion trip |
|---|---|---|---|
| 1 | 12 | 72 | 888 |
| 4 | 48 | 288 | 672 |
| 8 | 96 | 576 | 384 |
| 10 (top) | 120 | 720 | 240 |
| 10 + full occlusion | (engine holds 120) | → relief 32 000 | alarm long before ceiling |

The 25 % margin at the A top plateau (240/960) is deliberate: it is the
same number the AN §4 clamp tolerance checks ("peak ≤ clamp threshold"
— ALARM_CFG-derived), and it is what a *correct* closed-loop run never
crosses.

### E.2 Threshold vs noise — the 6σ rule (Table E-2)

False-detach/false-occlusion discipline: the compare must live above the
windowed-noise tail. Design rule used on this platform —

    margin = OCCL_TH·64 LSB − 6 · σ_window(CH0, config, TA)

with `margin < ~100 LSB` classed as an intermittent-latch risk zone.
`field/thermal-derating.csv` is this rule laid across temperature for the
production configuration (threshold 960 = reset, 4× boxcar):

Table E-2 — margin at selected points (from csv, healthy silicon)

| TA | σ noise (LSB) | 6σ (LSB) | margin to 960 | verdict |
|---|---|---|---|---|
| 25 °C | 4 | 24 | 936 | silent |
| 37 °C (noisy unit) | 16 | 96 | 864 | **silent — noise cannot be the ticket path** |
| 60 °C | 110 | 660 | 300 | quiet |
| 75 °C | 190 | 1 140 | −180 | latch risk — but HTEMP (§8.3/B-12) trips first |

E.2 is the arithmetic that lets a triage close the thermal-accuracy
decoy in TICKET.md without further investigation: 864 LSB of margin at
37 °C, by two orders of magnitude, is not a false-alarm budget.

### E.3 Debounce timing (Table E-3)

Latch delay from a sustained OCCL_DET to ALARM_OCCL at the close (§8.2,
Table 8-2) scales in *windows*, not time — window selection is therefore
a protection-latency choice:

Table E-3 — ALARM_OCCL latch delay from the START of the first OCCL_DET
window (sustained fault; = DEB+2 window periods, matching Table 8-2's
"8 ms @ 2 ms window" for DEB=2), ms

| OCCL_DEB | 1 ms window | 2 ms (default) | 4 ms | 8 ms |
|---|---|---|---|---|
| 0 | 2 | 4 | 8 | 16 |
| 2 | 4 | 8 | 16 | 32 |
| 8 | 10 | 20 | 40 | 80 |
| 0xE (reset) | 16 | 32 | 64 | 128 |
| 0xF | 17 | 34 | 68 | 136 |

Fault-ONSET-to-alarm adds up to one more window for the fault to be
sampled into the first OCCL_DET window (§8.2), plus the plant's rise time
across threshold (τ-scale, E.1) when the occlusion develops rather than
appears. A fully occluded line at the §7.2 defaults alarms in ~40 ms
total.

### E.4 Profile-C plateau granularity

Table C-4's 4-count C increments vs Table B-4's 4-5 LSB RMS noise explain
a verification choice: conformance of C against ±6 %-of-step plateau
tolerances (AN §4) implies tolerances of ±0.24 counts — inside one LSB.
Verification traces for C therefore run the plant noise term suppressed
(§4's tolerance model is about the engine/plant transfer, not the sensor;
J.4 covers what sensor-level C checks would require).

### E.5 Fault trajectory (full occlusion mid-run)

Sustained occlusion with an AVA-QC loop running (informative trace
description, the reference shape asserted trace-test-style):

1. OCCL_DET at the first window whose CH0 average compares ≥ OCCL_TH×64.
2. Debounce runs DEB+1 consecutive windows; alarm latches at the next
   close (E.3: 32 ms at defaults).
3. SAFETY_GATE (§6.18) with GATE_EN=1 forces SAFELIGHT_MODE=red
   (CTRL[3:2]=3): operator-visible at once.
4. The engine keeps stepping (it is open-loop w.r.t. alarms — §8.1): the
   plant rides to the 32 000 relief ceiling, plateaus there; CH0 no
   longer tracks the staircase. WAVE_DATA continues; ALARM_STATUS stays
   latched until the host clears.
5. Host recovery (reference driver behaviour): stop engine (C.3), vent,
   ALARM_CLR, then §8.2 re-arm requires a fresh full debounce run — a
   single noisy window cannot drop the latch and re-latch it instantly.
## Appendix F — Production test, self-calibration and the SPI fallback

Normative for factory flow; the SPI fallback is production/test only and
is explicitly NOT a system interface (§3 pin 18 — production systems do
not clock SCLK). Field troubleshooting may read the results of the flows
described here (trim fuses, self-cal behaviour) through the §6 map.

### F.1 Test flow (informative)

Table F-1 — production flow (Rev B, image 3.2.x)

| Station | Coverage (headline) | Temperature | Notes |
|---|---|---|---|
| CP (wafer) | continuity, IDD, raw ADC codes on trim, fuse integrity | +25 °C | trim search per F.2 |
| FT1 | §7.2 init contract, register walk, SDR window timing at all four periods (Table 7-1), alarm latch geometry (§8.2, DEB corners 0 and F) | +25 °C | t_waveform-class engine patterns included |
| FT2 | accuracy characterization vs Table B-1/B-4 envelopes, thermal trip corners (Table B-12), OCCL compare corners | 0/+70 °C | limits from this appendix's tables |
| Burn-in sample | 168 h @ +125 °C, lot sample AQL 0.1 % | — | destructive sample |
| System-level | AVV-3000 integration jig: §7.2 init + closed-loop AN profile A trace | +37 °C | the field-ticket temperature, deliberately |

### F.2 Self-calibration

Two-step digital self-cal, runnable from the host at any time with the
acquisition engine stopped (`CTRL.ADC_EN=0`; self-cal asserts BUSY while
running, max 16 window periods at the §7.2 default):

Table F-2 — self-cal phases

| Phase | Internal connection | Trim applied | Target |
|---|---|---|---|
| OFFSET | CHx muxed to AVSS (both inputs) | 8-bit two's-complement OFFSET trim per channel | \|code\| ≤ 2 counts |
| GAIN | CHx muxed to internal bandgap reference (0.9 · AVDD, ±0.05 %) | 10-bit OFFSET-scaled gain trim per channel | \|code − expected\| ≤ 4 counts |
| VERIFY | both phases re-read | — | BUSY clears with pass flag in the test window (SPI-readable only) |

- Production runs self-cal at +25 °C and bakes the result to fuses (F.4);
  the shipped cold-start offset drift row of Table B-1 already includes
  the baked trim.
- Host-runnable self-cal is a *RAM overlay*: it does not burn fuses, and
  a SWRST/RESET_N restores the fuse-loaded trim (no persistent change).
- Self-cal does NOT recalibrate the occlusion comparator's offset — the
  `±` column of Table B-1 applies to the compare path as measured, which
  is why the 6σ threshold rule (E.2) keeps >100 LSB of margin instead of
  zero.
- A window that completes during the BUSY span does NOT set DRDY (the
  data port is held at its pre-cal value; the window is counted in SEQ —
  the "no phantom windows" rule of D.4).

### F.3 SPI fallback port (production protocol)

Two-wire fallback over the SCLK pin plus serial I/O multiplexed on the
production-test pins (TEST0 = SI, TEST1 = SO open-drain; TEST2 = active
test-mode enable; all three strap LOW in production, §3):

Table F-3 — fallback frame (32 bits, MSB first, SCLK ≤ 5 MHz)

| Bits | Field | Notes |
|---|---|---|
| [31:28] | opcode | 0x1 READ_REG, 0x2 WRITE_REG, 0x3 SELFCAL, 0x4 READ_TRIM, 0x6 RDMARK |
| [27:20] | register/trim address | byte offset >>2 in the §6 map / fuse word index |
| [19:4] | data in/out | write data / read data |
| [3:0] | CRC4 | CRC-4/ITU over [31:4] (poly x^4+x+1) |

- Frames with bad CRC4 are dropped without effect; a 16-entry command
  counter is SPI-readable for test jigs.
- READ_REG reaches the §6 map (destructive STAT reads follow §6.9 rules —
  the fallback path is NOT exempt from ES-2.1.3 semantics; test flows
  read STAT once per window exactly like H-2).
- SELFCAL starts F.2 and the jig polls BUSY via READ_REG(STAT).
- RDMARK returns the date-code/marking word (lot hash, §2 marking line).

### F.4 Trim fuse map (Table F-4)

One-time-programmable, factory-only (not host-writable — F.2 self-cal is
a RAM overlay; a "successful" host fuse write is a fixture-level red
flag, not a feature).

Table F-4 — fuse word layout (READ_TRIM addressing)

| Word | Bits | Content |
|---|---|---|
| 0 | [7:0] | silicon rev + lot class |
| 1..8 | [7:0] each | CH0..CH7 OFFSET trim (8-bit signed) |
| 9 | [7:0] | bandgap coarse |
| 10 | [1:0] | bandgap fine |
| 11 | [7:0] | window PLL duty fine |
| 12 | [9:0] | lot hash (matches RDMARK low half) |
| 13 | [5:0] | OCCL comparator offset trim |
| 14 | [9:0] | thermal monitor trip offset (feeds B-12 ±2 °C) |
| 15 | [7:0] | CRC8 (poly 0x07, init 0x00) over words 0..14 |

## Appendix G — Errata compatibility matrix

Cross-reference sheet between this datasheet and ES-2026-04 (the errata
sheet is authoritative on conflict, per its own terms):

Table G-1 — errata vs document surface

| ID | Subject | Datasheet sections touched | Rev-A parts | Rev-B parts | Datasheet posture |
|---|---|---|---|---|---|
| ES-2.1.1 | OCCL_HYST reset ordering | §6.10, §6.7 (SWRST) | n/a (field missing on A) | PRESENT | re-write ALARM_CFG after SWRST; treat §6.7 "reset values" as the *end* state of the reference init, not a mid-reset guarantee |
| ES-2.1.2 | WAVE_CFG.HOLD_MS deferred apply | §6.13, §8.1 | n/a (no engine) | PRESENT | C.3 matrix is the behavioural spec |
| ES-2.1.3 | destructive STAT reads | §6.9, §7.4 H-2, §7.4.2 | Rev-A §7.4.2 two-read pattern was written for pre-A silicon and is invalid | PRESENT | H-2 single read is THE contract; §7.4.2 is a warning label, not a procedure; D.5 case X-2 covers the jitter corner |

Silicon-rev quick check for integrators:

- A Rev-B part running the §7.2 sequence exactly, with single-read
  polling, honours all errata workarounds (that IS the §7.2 contract —
  step ordering writes ALARM_CFG after CTRL per ES-2.1.1's workaround,
  and the engine is armed only after init per §8.1).
- Any driver following Rev-A documentation (two-read confirm, CFG bit 7
  as AVG_MODE) violates one or more errata workarounds BY CONSTRUCTION;
  such a driver is incorrect on Rev B even when it appears to pass short
  bench tests (see ES-2.1.3 severity note; the i2c.pcap.txt captures in
  the field artifacts are exactly this driver).

## Appendix H — Packaging, reliability, assembly

### H.1 Package (TQFP-64)

Table H-1 — package body data

| Parameter | Value | Unit |
|---|---|---|
| Body | 10 × 10 × 1.0 (max 1.2 incl. stand-off) | mm |
| Pitch | 0.5 | mm |
| Lead finish | matte-Sn (Ni/Pd/Au leadframe) | — |
| θJA (JEDEC still-air) | 46 | °C/W |
| ψJT | 12 | °C/W |
| MSL | 3 (168 h floor life) | — |
| Reflow peak | 260 | °C |
| Moisture pre-dry | 125 °C / 24 h | — |
| Solvent wash | allowed (no aqueous > 60 °C on marking) | — |

### H.2 Reliability

Table H-2 — qualification summary (AEC-Q100-graded test plan, medical
grade screening; Rev B lot data)

| Stress | Condition | Result |
|---|---|---|
| HTOL | 1000 h, +125 °C, biased | 0 fail / 77 units |
| ESD HBM | 2 000 V all pins | pass |
| ESD CDP | 500 V | pass |
| Latch-up | ±100 mA inj., 125 °C | pass |
| Temp cycling | −55/+125, 1 000 cy | 0 fail / 77 |
| uHAST | 130 °C/85 %RH, 96 h biased | 0 fail |
| EM (current density) | 10 yr @ 110 °C TJ | qualified (1 MA/mm² cap) |
| MTBF (MIL-HDBK-217F GB, +40 °C) | 2.4 × 10^6 h | — |

FIT allocation (10^9 h): analog front end 41, digital sequencer 22, I/O
12, RAM/fuse latches 6 — the occlusion path is replicated in the digital
domain (compare + latch) and its uncorrected-fault contribution is
inside the 10 FIT safe-path budget allocated to the safelight chain.

### H.3 Storage, handling, return

- Dry-pack per MSL3; baking per H.1 if floor life exceeded.
- Marking reads `8851B / YYWW / lot hash4` (§2, F.3 RDMARK).
- RMA data capture: read ERR_STAT, ALARM_STATUS and fuses (word 0) at
  failure — the ERR_STAT read-through-clear in the return form lets the
  factory distinguish the ES-2.1.3 driver signature from FIFO overrun
  signatures without re-running the customer build.

## Appendix I — Glossary

| Term | Meaning in this document |
|---|---|
| ALARM latch | sticky bit in ALARM_STATUS; cleared ONLY by CTRL.ALARM_CLR (§6.11) |
| ACLK | 10 MHz external acquisition clock; every timing number derives from it (§9.2) |
| AVA-QC | Aspiration Verification and Quality Control — the pump's QC pattern family (AN-8851-07) |
| boxcar averaging | arithmetic mean over the oversampling set (§7.1 step 3) |
| BUSY | window-open flag in STAT; high during acquisition phase (§6.9) |
| BRESP | AXI write-response channel/signal (AM004; §4.3 invariants) |
| commit | atomic data-port update at window close (§7.1 step 4, D.4) |
| CYCLE | WAVE_STATUS flag: loop iteration ≥ 2 (§6.15; C.1/C.2) |
| debounce (OCCL_DEB) | consecutive-detected-window counter with pre-increment compare (§8.2; Table 8-2) |
| DONE | waveform-cycle-completion flag/transition (§6.15; C.1 DONE_ST) |
| DRDY_N | active-low window-close strobe pin (§7.4.1) |
| destructive read | read with state effect: STAT (DRDY consume, ES-2.1.3) and RC-class counters (§4.2) |
| ENOB | effective number of bits; 6.02·ENOB+1.76 = SINAD (§9.2, B.5) |
| FIR | 4-tap window-to-window smoother, CFG.FILT_EN (D.2) |
| H-1..H-4 | host read-protocol rules (§7.4) |
| HOLD | per-step plateau interval of a waveform profile (§8.1; AN-8851-07 §2) |
| HOLD_TOP | profile-B top dwell inside HOLD_TOP state (C.1; ES-2.1.2 neighbour) |
| INVERT | staircase direction (infusion/withdrawal, §6.13) |
| LPM | low-power mode (§7.3; A.4 duty-cycle pattern) |
| median averaging | order-statistic pick (D.3; CFG.AVG_MODE) |
| OCCL | occlusion (blocked fluid line) — the pump's namesake alarm |
| pre-increment | debounce geometry class where the counter compare precedes the per-window increment (§8.2; Table 8-2) |
| RC | read-to-clear access class (§6 map key) |
| relief ceiling | external plant safety limit (E.1); NOT a device clamp |
| SDR window | repeat acquisition period; BUSY/DRDY/SEQ bound to it (§7.1) |
| SEQ | 8-bit window sequence counter, increments once per close (§6.9) |
| SLVERR | AXI decode-error response (§4.3; latches ALARM_COMMS) |
| t_sample | active sampling phase inside the window, 0.60 · T_W (§7.1, D.1) |
| W1C | write-1-to-clear (§6 map key) |
## Appendix J — Troubleshooting and failure diagnostics

The diagnostic matrix this family has always needed. Every row is
observable through the §6 map from a live host — no factory station, no
jig. Read the STAT/ERR_STAT discipline note (J.5) BEFORE using any
checklist here on a suspect system: the checklist itself must obey the
once-per-window rule it is diagnosing.

### J.1 Symptom → signature matrix (Table J-1)

| Symptom | First read | Signature | Most likely cause | Confirm by |
|---|---|---|---|---|
| Occlusion alarm, no occlusion | ERR_STAT | STAT_VIOLATION set, W_VIOLATIONS growing | driver reads STAT twice per window (ES-2.1.3) | single-read the loop, alarm rate follows 1:1 — J.2 |
| Occlusion alarm, no occlusion | ERR_STAT | FIFO_OVF only, W_VIOLATIONS clean | host not draining; stale plateau misread as pressure | SEQ delta in the log (D.4) |
| Occlusion alarm, no occlusion | CH0 raw at the trip instant | CH0 genuinely ≥ threshold, pump stopped | line draining/capacitance creep (plant, §8.1 note) | vent + re-read: margin returns without a SWRST |
| Intermittent watchdog resets, temperature-biased | (system side) uart.log gap summary + svc counters | window gaps correlate with burst instants, not with temp per se | service ring eviction under 2 Hz status burst | burst replay w/ guard counters (svc acceptance model) |
| Intermittent watchdog resets, temp-biased | ALARM_STATUS | ALARM_COMMS latching | SLVERR storms — address-decode regress in the driver (bit-bang rework?) | SAFETY_GATE off, re-walk §6 map addresses |
| Waveform run ends early | WAVE_STATUS | DONE set with RUN=0, CYCLE never 1 | profile C armed where A expected (one-way — C.2) | WAVE_CFG readback profile bits [1:0] |
| Waveform plateau never settles | WAVE_CFG readback | HOLD field ≠ intended ms | field-encoding error — Table A-2 literals vs ms/tick confusion | recompute literal from Table A-2 |
| Waveform output frozen at non-zero | CTRL/WAVE_STATUS | RUN=0, wave_q retained | normal behaviour (C.1 retention) — engine never drives output to 0 on stop | arm a fresh cycle; retention is by design |
| Panel shows absurd rates (×1000 style) | unit scope only | silicon-side rates exact | display-layer conversion duplicated (AN-8851-07 §5 rule) | scope counts/ms vs panel digits |
| ALARM_LINE during a good-looking run | data port | one channel's value from previous window while the pair's SEQ incremented once | host-side window LOSS, not silicon mix-up (D.4 atomicity; J.3) | application log continuity check, not register reads |
| ERR_CNT saturates | STAT read cadence | +1/window increment | polling loop reads STAT ≥2×/window (same as row 1) | H-2 audit of the poll function |
| Cold-start outliers ±20 counts | first windows after power | drifts out over ~16 windows | pre-self-cal offset (F.2 B-1 note 1) | re-run self-cal, compare |
| Thermal trip at absurd temp | ALARM_CFG | TH_TEMP mis-written (e.g. 0xF → 100 °C trip… §6.10 units) | init sequence skipped steps 4..6 | full §7.2 walk |

### J.2 The 1-in-N arithmetic (ES-2.1.3 duty cycle)

Why a double-reading driver alarms *occasionally* instead of every
window, and why temperature seems causal: the confirm re-read is fatal
only when it crosses a window close (case D.5 X-2 places the boundary).
For a poll loop with re-read spacing `d` and window period `T_W`, the
hazard window per cycle is ≈ d out of T_W:

P(alarm-eligible window) ≈ d / T_W → at T_W = 2 ms with bus-loop d ≈
0.4 µs this is O(1/5000) — the "1-in-5000" field number is this ratio,
not any 37 °C silicon effect. Temperature acts through Table B-11's
jitter coupling and (in the real units) through the *service* stall rate
independent second thread. A fix that removes the second read moves the
ratio to exactly zero; anything that merely "reduces" it (slower poll,
faster poll with backoff) leaves the failure mode intact — which is why
ES-2.1.3's severity is HIGH and the workaround is structural (H-2), not
tuning.

The same arithmetic reproduces in the lab only if the phase relationship
between poll loop and window close survives boot — usually it does not;
hence "cannot reproduce". Long-running soak at production cadence
reproduces it (hours, not minutes; the field rate is per-window, and at
500 windows/s the expected interval between hits is O(10 s) — the field
"1-in-5 000" describes window-eligibility, not wall-clock frequency; the
pump happens to run in short duty cycles).

### J.3 ALARM_LINE false-geometry (host-side only)

The silicon cannot produce the J.1 row's mixed set (D.4: atomic commit).
The remaining geometry is application-level: under ring pressure a
consumer pair can pair W(n, CH0-new) with W(n−1, CH1-stale) *in the
consumer*, satisfying |CH1 − CH0| > 25 % at a fast plateau edge for ≥ 8
windows while the part behaved perfectly. Signatures separating it from
real divergence: the paired SEQ field in the consumer's own records
matches at the event; FIFO_OVF/ERR_STAT stay clean; re-reading live
registers shows a compliant pair. Owner: the admission policy (svc
layer), not the part and not §6.11.

### J.4 What noise-sensitive waveform checks would need

AN §4's ±6 %-of-step plateau tolerance on profile C is a 0.24-count
window (E.4). Verifying plateaus against *sensor* data at that grain
would require, in order of realism:

1. engine-side checks (WAVE_DATA readback — exact, what C.4 tables and
   `t_waveform`-class tests assert);
2. plant-side with noise suppressed (what `fv_staircase`-class trace
   tests do for C, per E.4);
3. full-noise sensor-side: statistically (tolerance × √N averaging, N
   cycles) — single-run C traces at ±0.24 counts are below both the
   sensor floor (B-4: 3.3 LSB RMS) and the plant noise term (E.1: ±1),
   a test that would be flaky BY PHYSICS, not by bug.

### J.5 The diagnostic-read budget

On any system that might be ES-2.1.3-positive, STAT/ERR_STAT reads are a
consumable with a hard rate limit: one STAT read per window, plus the
RC-class counters (ERR_STAT, CHx_OVF, STAT.ERR_CNT fields — one read each
per diagnostic pass; RC semantics mean the read itself is the
destruction, §4.2/§6.9). Practical rules:

- Snapshot ERR_STAT once per diagnostic pass; a loop that reads
  ERR_STAT "to watch" clears the evidence between observations and — if
  implemented against the same window — repeats row 1's crime.
- Prefer the monotonic witnesses (SEQ deltas in logs, the svc
  in/evicted/rejected counters, W_VIOLATIONS *cumulative* shape between
  two snapshots) over any high-rate register polling.
- Never "poll STAT while reading the data port for convenience": the
  port is non-destructive and stable across its window (D.4); polling
  STATUS to sequence a *burst* read is the one pattern H-2 forbids while
  looking busiest.

## Appendix K — Board-level application notes

### K.1 Supply sequencing and decoupling (Table K-1)

Table K-1 — recommended supply ramp (any order, monotonic, ≤ 50 ms
ramps; no latch-up window between AVDD/DVDD at ≤ 0.5 V/ms)

| Rail | Min | Max | Decoupling (typ) | Notes |
|---|---|---|---|---|
| AVDD | 3.0 | 3.6 V | 10 µF + 100 nF/µF ×4 | analog plane, star to DGND at pin 12 |
| DVDD | 1.62 | 1.98 V | 4.7 µF + 100 nF ×4 | I/O drivers on VDD_IO separate 100 nF bank |
| VDD_IO | 1.65/3.0/3.3/3.45 | per bank | 100 nF ×6 | AM004 IO banks V_OH spec §9.2 |

- Sequence during system power-up can be arbitrary; the part self-holds
  (BUSY poll-able, §7.2 step 1) whatever the order.
- AVDD ripple budget: at 100 mVpp on AVDD, Table B-9 puts induced error
  at ≈ −68 dB ≈ 1.2 counts — inside the production envelope; a 37 °C
  false alarm is NOT a buck-converter ripple story (B.9 + J.1 ordering).

### K.2 Sensor front end per Table 5-1

- CH0/CH1 (0–3000 mmHg → 0..32767): ratiometric bridge or 0.5…2.5 V
  amplified front ends; keep the source impedance ≤ 100 Ω (Table B-6
  assumes matched low-Z sources; ≥1 kΩ sources inject drift at the
  ppm/°C class — not datasheet spec).
- The domain note (Table 5-1: unsigned 0..32767, pressures non-negative
  per spec) is a SYSTEM rule: front ends must rail below 32768 counts at
  over-range; parsers that wrap negatives in (J.1 rows, csv) are field
  facts this system rule exists to prevent.
- Thermistors (CH2/CH3): pump-head unit at the thermal mass named in the
  ticket's own analysis, bias chain ≤ 100 kΩ; thermal monitor accuracy
  (B-12, ±2 °C) makes CH2 the head temperature authority for
  temperature-correlated field data.

### K.3 WDx PWM / valve-driver interface

- 16-level PWM (WD0..3) — route as four slow signals, ≤ 50 pF per pin;
  the engine drives them at its ms cadence (C.1) — no analog output
  filter needed for the plant time constant of tens of ms (E.1).
- WDx continue to reflect WAVE_DATA while the engine is at DONE_ST /
  stopped (output retention, C.1) — the valve-driver stage must treat
  CTRL/SAFETY_GATE state, not WDx presence, as the run signal.

### K.4 Grounding and layout for the alarm path

- Star ground per §3 pin 12 note; the occlusion compare's margin math
  (E.2) is budgeted on sensor noise, not layout noise — a layout eating
  >50 LSB of the 6σ budget is out of spec by this appendix, and the
  margin table stops proving what it promises.
- Keep ACLK a controlled-impedance point-to-point or < 50 mm branch;
  Table B-11 jitter numbers assume clean edges (SR ≤ 2 ns class, AM004
  IO spec).

## Appendix L — Full change log (Rev A → Rev B)

Table L-1 documents every normative delta; this supplements (does not
replace) the §11 revision-table and Table 7-4.

Table L-1 — Rev A → Rev B normative changes

| Item | Rev A | Rev B | Impact class |
|---|---|---|---|
| CFG[7] | AVG_MODE | rsvd0 (Table 7-4) | BREAKING — see §7.2 delta, t_fw_init failure mode |
| CFG[5] | rsvd | AVG_MODE | BREAKING (same write pair) |
| ALARM_CFG.OCCL_DEB | absent (fixed 8-window internal) | [11:8] programmable | function |
| ALARM_CFG reset | 0x0000_000F (no DEB field) | 0x0000_1E0F (DEB=0xE, TH_TEMP=1) | function |
| ERR_STAT register | absent | 0x20 full block | function |
| STAT.ERR_CNT field | absent | [7:4] | function |
| §7.4.2 two-read sequence | "best practice" | superseded — invalid (ES-2.1.3; G-1) | BREAKING for Rev-A drivers |
| WAVE_CFG.HOLD_MS | [7:2] 1..63 ms | [15:2] 1..16383 ms (14-bit) | field width — recipes carrying >63 ms holds required this widening |
| WAVE_CFG.STEP_LSB | [13:8] | [21:16] | repositioned with HOLD widening |
| WAVE_CFG.INVERT | [14] | [22] | repositioned |
| Waveform profiles B top-dwell | undocumented | documented (AN §3, C.1/C.4) | documentation |
| SAFETY_GATE | absent | 0x2C | function |
| TH_RATE default | 0x0000_01D0 | 0x0000_0168 | 104-window thermal cadence (§6.17, Table 7-3) |
| IRQ.M_DRDY | pin-shared with alarm | independent (§7.4.1 two-path note) | clarification |

Migration checklist for a Rev-A driver (all four mandatory before a unit
can pass §7.2-class bring-up):

1. CFG literal: move AVG_MODE to bit 5, clear bit 7 — Table 7-3's 0x49
   (else boxcar-against-intent median pipelines, t_fw_init signature).
2. Adopt H-2 single-read (§7.4/ES-2.1.3; G-1 row 3).
3. Re-encode WAVE_CFG against L-1's bit positions (old literals write
   garbage PROFILE/HOLD on Rev B — [1:0] overlaps differently than the
   pre-widening layout).
4. Add ALARM_CFG re-write after any SWRST (ES-2.1.1 workaround is baked
   into the reference init order; a hand-rolled recovery path must
   re-bake it).
## Appendix M — Application circuits (informative)

Textual netlists in the house style; values are the AVV-3000 reference
head board — equivalents welcome with K.2's impedance rules honoured.

### M.1 Pressure front ends (CH0/CH1)

Signal chain, per channel:

    sensor (ratiometric bridge, 0..4.5 mV/V @ AVDD bridge)
      → INA (fixed gain 44.7, Vref = AVDD/2 bias)
      → RC anti-alias 220 Ω / 10 nF (fc ≈ 72 kHz)
      → CHx pin
      (protection: 100 Ω series + 2× clamp diode to AVDD/AGND rails)

Bridge + INA together map 0..3000 mmHg onto 0..2.93 V ≈ 0..32 050 counts
with the Table 5-1 linear scale; the last ≈ 117 counts of code headroom
absorb sensor lot spread WITHOUT ever reaching negative codes for p ≥ 0
(K.2 domain rule, §5.1 note 2). Offset bias at the INA reference pin is
±30 mV max — inside the ±8 LSB offset budget after self-cal (Table B-1).

A common integrator error is biasing the front end so mid-supply =
mid-scale with bipolar output: negative pressures then legitimately
present negative codes, breaking the Table 5-1 domain and every
consumer's reject-on-sign rule. The reference design is
unipolar-by-construction; keep it that way in derivatives.

### M.2 Thermistor chains (CH2/CH3)

    AVDD ── 22.1 kΩ 100 ppm ──┬── CHx
                              │
                         NTC 10 kΩ @ 25 °C, B=3435 (head: epoxy-bonded,
                         2 mm thermal mass; system: bare bead)

Code spans: −40 °C ↦ −2048 counts (post-mapping, Table 5-2 — the
silicon sees the mapped stream from the production image's front-end
LUT; raw NTC divider codes are linearised in the image, not in silicon).
Thermal monitor trip (B-12) uses the mapped stream: firmware that
re-linearises on the head side double-compensates and shifts CH2 by the
diff of the two curves — a favourite way to manufacture a "temperature
correlated alarm" in the lab that the silicon did not produce.

### M.3 Motor/valve current senses (CH4/CH5)

- CH4: 25 mΩ shunt + differential amp ×66, offset-coded zero at 0 A
  (unipolar 0..5 A per Table 5-1); PWM ripple is inside the boxcar
  set anyway (Table B-4's 4× + FIR path); do NOT add > 1 µs delay —
  the stall-detection app reads per-window averages.
- CH5: diode-driver sense mirrored at 1:1000, 1 mΩ + ×4000 — the
  valve-off floor (≤ 8 LSB) is a production screen (F.1 FT2), because
  a leaky mirror fakes "valve stuck open" telemetry at the 2 mA grain.

### M.4 WDx / DAx outputs

- WD0..3: 16-level PWM to the valve-driver gate (K.3); keep ≥ 2 mm
  clearance to CHx analog routing (aggressor ranking, Table B-6).
- DA0..7: QC-panel indicator drives, 4 mA/pin max, NOT a general GPIO
  bank — panel firmware that bit-bangs DAx for status LEDs owns the
  resulting crosstalk (B-6 rows 3/4 measured with DAx static).

### M.5 Power sequencing in the AVV-3000 head

Sequence (K-1 allows any order; this is the product's):
AVDD → DVDD → VDD_IO → RESET_N release (§7.2 step 1 poll follows).
Star ground single-point per §3; the safelight APP's amber path keeps a
10 ms software debounce on SAFELIGHT_MODE writes — not silicon, but the
only legitimate debounce budget left after §8.2's geometry (Table 8-2)
is UI-side.

## Appendix N — AVV-2200 compatibility (lineage notes)

The AVA-QC feature family carries over from the AVV-2200 pump
(platform-port context per ES-2026-04's historical note). This appendix
is the integrator-facing delta summary — it is NOT normative; §6/§7.2
win on conflict.

Table N-1 — platform summary

| Aspect | AVV-2200 | AVV-3000 |
|---|---|---|
| ADC part | ADS-8622 (single-window legacy, SPI-only) | ADS-8851 (this document; AXI-Lite + SPI fallback F.3) |
| Register transport | SPI, 8-bit pages | 32-bit AXI-Lite map (§6) |
| Bus read hazard | none: ADS-8622 STAT reads are NON-destructive (the double-read pattern was HARMLESS there) | ES-2.1.3: destructive; H-2 mandatory — THE port defect class (G-1 row 3) |
| CFG layout | AVG_MODE at bit 7 | bit 5 (Table 7-4 — the platform-port's own B1 class) |
| OCCL_DEB | fixed 8-window internal | programmable (§6.10) — the port defaults 14+1, NOT 8 |
| Waveform engine | none (patterns stepped by host timer) | on-die engine (§8.1) |
| AVA-QC profile_a constants | canonical (10/10/500/12) | byte-identical (AN §3, §6 comparability note) |
| Panel rate rule | counts/s ×1000 | same (AN §5) |

Two historical footnotes with diagnostic value:

1. The Rev-A datasheet's two-read STAT pattern was copied from the
   ADS-8622 application note, where it is correct on that silicon —
   the port inherited the procedure, not the hazard review. Old lab
   scripts with double reads still pass on 2200 test heads; they are
   broken ONLY on 3000.
2. The 8-window fixed debounce of the 2200 is why field documentation
   for the fleet occasionally says "eight-windows-to-latch": on 3000
   with production init the number is 15+1 (Table 8-2; step 4 writes
   DEB=0xE). Alarm-latency discussions that disagree about "how many
   windows" have usually just mixed platforms.

## Appendix O — Timing diagrams (informative; normative geometry cited per-figure)

Figure O-1 — ES-2.1.3 failure sequence (Rev-A driver on Rev B silicon):

    window N close ─┐
    STAT read #1    └─► DRDY=1 seen, flag CLEARED by the read — legal, H-2
    ... data port read, process ...
    STAT read #2 ("confirm", Rev-A habit)
                    └─► DRDY=0 returned; STAT_VIOLATION set; ERR_CNT++
                        if this read gated consumption: window "lost"
                        (its DATA is fine; the READY edge was consumed
                        by read #1 — the data was never unread)

    jitter note (Table B-11): when the close lands between #1 and #2,
    #2 belongs to the NEW window: it returns DRDY=0 of a freshly-opened
    window — the confirm looks "correctly false" and the violation is
    silent for that pair. The hazard duty arithmetic: Appendix J.2.

Figure O-2 — occlusion latch geometry (§8.2; DEB=2, Table 8-2):

    window:        w1        w2        w3        w4        w5
    CH0≥TH?        DET       DET       DET       DET       DET
    counter BEFORE increment (0,1,2,3):
                   0         1         2         3 ← compare 3 > DEB(2)
    ALARM_OCCL:    -         -         -        LATCHED    latched
                                    at w4's close ┘
    (pre-increment compare; §8.2 wording is normative; fv-class latch
    tests pin exactly "4th DET close with DEB=2")

Figure O-3 — profile A closed-loop cycle (Table A-2 literal; E.1 plant):

    t=0     arm: WAVE_DATA 0      CTRL.WF_EN=1 (RUN=1)
    t=HOLD.. every 500 ms: WAVE_DATA += 12 (RAMP_UP 10 steps)
    t=9·HOLD: WAVE_DATA=120 (plateau 10) ; CH0 ≈ 720 counts settled
    t=10·HOLD.. ramp down (one step per HOLD)
    t=20·HOLD: WAVE_DATA=0, DONE PULSE (once/cycle, C.2), CYCLE=1 from
               the NEXT cycle's first step; loop continues
    occlusion watch throughout: latches only if CH0 hits 960 (E.1
    table says a clean line never does: +240 headroom at the top)

Figure O-4 — window-close arbitration (§4.4 H-3, D.5 X-1):

    ACLK            __/‾‾‾\__/‾‾‾\__/‾‾‾\__/‾‾‾\_
    close event            ▲ (engine wants commit)
    AR accepted      ──────▲
    commit                 ▲──┐ delayed ≤2 CLK while the read is served
    R data valid                       ──► old set (pre-close values)
    next legal read                          sees the NEW set
    (never a mixed set — D.4's atomicity is what makes burst reads of
    the data port safe across close boundaries)

Figure O-5 — SPI fallback frame timing (F.3; jig side):

    CS_N   ‾‾\_______________________________________/‾‾
    SCLK      _/‾‾\_/ ‾‾‾\_/‾‾‾\___ ... 32 cycles total, rising-edge
                                     capture, ≤5 MHz (production rig
                                     runs 2 MHz — F.1 CP station)
    SI          [31....................4][CRC4]
    SO                                (SO drives MSB-first READ reply
                                       during the SAME frame window on
                                       opcode READ_*; else Hi-Z)
    ERR: CRC4 mismatch → frame silently dropped (F.3 counter++ ONLY)

Figure O-6 — self-cal BUSY span (F.2), acquisition paused:

    ADC_EN   ————\________________________________________/
    BUSY           ________‾‾‾‾‾‾‾ (≤16 window periods) ________
    SEQ        frozen during BUSY (no windows, no closes — F.2's last
    DRDY_N                          bullet: no phantom windows, nothing
                                    missed because nothing was scheduled)

## Appendix P — Lot characterization data (Rev B, image 3.2.1)

Table P-1 — lot summary (CP+FT combined, 6 lots)

| Lot | Date code | Units in | CP yield | FT yield | Offset |max| seen | Noise p95 (4×, 25 °C) | Rejects headline |
|---|---|---|---|---|---|---|---|
| A2-4471 | 26:07 | 1 488 | 91.2 % | 97.9 % | 5.1 LSB | 3.9 LSB | 21 × INL shape |
| A2-4502 | 26:09 | 1 510 | 90.8 % | 98.4 % | 5.4 LSB | 4.0 LSB | 17 × thermal trip edge |
| A2-4533 | 26:11 | 1 495 | 92.0 % | 97.5 % | 4.9 LSB | 3.8 LSB | 25 × offset |
| B1-4561 | 26:13 | 1 502 | 93.1 % | 98.1 % | 4.7 LSB | 3.6 LSB | 19 × misc |
| B1-4590 | 26:15 | 1 511 | 92.6 % | 96.9 % | 5.2 LSB | 4.1 LSB | 32 × noise tail (sub-lot reworked) |
| B1-4619 | 26:18 | 1 489 | 91.9 % | 98.0 % | 5.0 LSB | 3.7 LSB | 24 × misc |

Fleet-correlation note for the field ticket analysis: the affected units
were built from lots A2-44xx/45xx; no silicon lot shows the csv's 16 LSB
37 °C noise floor at spec temperature (fleet p95 4.1 LSB system floor);
no ACCURACY-driven escape has ever been observed — the escapes that
happen in this family are behavioural (G-1 row 3 again) — the P-series
data is present so triage can close thermal/accuracy threads with a
number instead of a shrug.

## Appendix Q — Production test limit table (FT1/FT2)

Table Q-1 — FT1 (contract correctness; all-or-nothing patterns)

| Pattern # | Covers | Pass criterion |
|---|---|---|
| 11 | §7.2 init walk, Table 7-3 values | readback == expected on all five registers |
| 14 | Table 6-1 full-map walk (WO/RC/W1C semantics per §6.1 classes) | per-field expectations, zero SLVERR outside reserved rows |
| 17 | H-2 conformance (single-read state machine, jitter injection at boundaries) | ERR_CNT stays 0 across 65 k windows |
| 22 | ES-2.1.3 negative: deliberate second read | STAT_VIOLATION +1, NOT counted as FT fail — must be OBSERVED (test-the-test) |
| 25 | OCCL latch geometry battery (Table 8-2 corners, Figure O-2) | latch close number exact |
| 31 | WAVE_CFG literal battery (Table C-3/C-4 sequences; ES-2.1.2 deferral) | DONE/RUN/CYCLE per C.1 table |
| 38 | Alarm/IRQ matrix (Table 8-4 truth table, GATE_EN force/auto-green) | effective-mode truth row × 8 |
| 44 | LPM duty walk (A.4 sequence × 1 k cycles) | no phantom windows (SEQ deltas clean), power deltas in B-8 envelope |

Table Q-2 — FT2 (parametric; limits from Appendix B tables ×production
factor 2.0 typ/max)

| Parameter | Min | Max | Units | Basis |
|---|---|---|---|---|
| OFFSET residual (F.2) | −2 | +2 | counts | F.2 target is the limit |
| GAIN residual (F.2) | −4 | +4 | counts | idem |
| Offset error, 0/+70 °C | −8 | +8 | LSB | Table B-1 |
| Gain error, 0/+70 °C | −1.2 | +1.0 | % FS | Table B-1 |
| Noise p99 (4×, +70 °C) | — | 20 | LSB | Table B-4 max-envelope |
| SNR (16×, 1 kHz) | 80 | — | dB | §9.2 −2 dB margin |
| Window period accuracy | −5 | +5 | % | §9.2 |
| Thermal trip (TH_TEMP=1) | 84.5 | 87.5 | °C | 86 ±2 (B-12) |
| Thermal hysteresis | 1 | 3.5 | °C | B-12 |
| OCCL compare @0xF (960) | 952 | 968 | LSB | ±1 % of threshold |
| PSRR @ 120 kHz | 38 | — | dB | B-9 extrapolated floor |
| IDD total (running) | 5.5 | 8.5 | mA | B-8 ±2× guard |

## Appendix R — DFT pattern inventory (design-for-test summary)

Table R-1 — pattern families and analog coverage (Rev B)

| Family | Patterns | Coverage note |
|---|---|---|
| Contract (init/map/RC/W1C) | 11, 14, 44 | full register machine |
| Read-protocol (H-2/ES family) | 17, 22 | only in-family negative-allowed pattern is 22 BY DESIGN — the violation is a PRODUCT feature to observe, not to fail silently; G-1 again |
| Alarm engine | 25, 38 | occlusion latch geometry + safelight arbitration |
| Wave engine | 31 | C-4 literal battery |
| Analog parametric | Q-2 suite | limits × production factor |
| Fuse/trim | F.4 walk + CRC8 | READ_TRIM path only (F.3 opcode 0x4) |

## Appendix S — Ordering, packing, document status

Table S-1 — ordering and packing addendum

| Field | Value |
|---|---|
| Base order quantity | 1 000 units (reel) |
| Reel | 330 mm, EIA-482, 12 mm tape pocket 12×12, covertape 200 mm peel |
| Moisture bag + desiccant + card | MSL3 default |
| Marking line 1 | ADS-8851 (logo, 8851B) |
| Marking line 2 | YYWW (date code, §2) + lot hash 4 hex (F.3 RDMARK low half — the hash is the LOT identity; fuse word 12 is authoritative) |
| Pin 1 indicator | moulded dimple + laser tick |
| RoHS/REACH | compliant; conflicts: none declared |

Table S-2 — document status and lifecycle

| Status | Meaning |
|---|---|
| Preliminary Data Sheet — Rev B (current) | characterization-based (Appendix B), parameters subject to change until full CQ signoff (H.2 row set complete, lot P data = lots 1–6 of planned 10) |
| Approved DS | planned Rev B.1 — merges this appendix set; no normative §-number reassignments planned; anchors cited by published drivers are pinned by the Rev-A→B precedent handling (L-1) and by t_fw-class contract tests in the field firmware ecosystem |

Contact for errata disputes and safety-affecting discrepancies: quality
inquiry channel quoted on the ordering sheet (not part of this
datasheet). A discrepancy between this document and `ES-2026-04` in
an alarm-path behaviour is treated as a P1 quality event BY THE QUALITY
PROCESS, independent of this document's "errata wins" clause — say so
loudly in the report and stop shipping units until it's adjudicated.

## Appendix T — Register quick reference card

One-line card for the debugger with the map open. Never an alternative
to the §6.x detail; 'R/C' rows follow §6.1/§4.2 destructive-class rules
(reads consume — budget them, J.5).

Table T-1 — 30-second map

| Offs | Rst | Key read (non-consumptive unless noted) | Key write |
|---|---|---|---|
| 00 CTRL | 0 | — (RW live) | bit0 ADC_EN; bit1 WF_EN; bit5 ALARM_CLR (W1C); bit6 SWRST |
| 04 CFG | 4 | window/ovr/avg/filt nibble bits 1:0/4:2/5/6 | Rev-B layout ONLY (bit 7 dead — B1 trap, Table 7-4) |
| 08 STAT | 0 | **CONSUMES DRDY (ES-2.1.3)**; [7:4] ERR_CNT RC; [15:8] SEQ | — |
| 0C ALARM_CFG | 1E0F | TH[3:0]×64=trip LSB; DEB[11:8]+1...latch per Table 8-2 | after SWRST always re-write (ES-2.1.1) |
| 10 ALARM_STATUS | 0 | bit0 OCCL / 1 HTEMP / 2 LTEMP / 3 LINE / 4 COMMS / 5 LATCHED | only via ALARM_CLR |
| 14 WAVE_CFG | 0 | PROFILE[1:0] HOLD[15:2]=ms<<2 STEP[21:16] INV[22] | literal = Table A-2; mid-cycle HOLD defers (ES-2.1.2/C-3) |
| 18 WAVE_DATA | 0 | engine output mirror | read-only |
| 1C WAVE_STATUS | 0 | RUN/CYCLE/DONE; DONE clears on NEXT CFG WRITE not on read | read-only |
| 20 ERR_STAT | 0 | **RC** bit1 STAT_VIOLATION / bit2 FIFO_OVF / [31:8] W_VIOLATIONS | RC: read = clear |
| 24 IRQ_MASK | 1F | mask bits [4:0]=alarms [5]=DRDY edge | 1 = unmasked |
| 28 TH_RATE | 168 | TH_PERIOD[7:0] windows, TH_FILTER[15:8] 2^n median | 0-period disables |
| 2C SAFETY_GATE | 0 | GATE_EN bit0 forces red (Table 8-4) / AUTO_GREEN bit1 | write after init stable |
| 40+4n DATA_CH0..7 | 0 | [15:0] res signed-coded Table 5-1 domains / [23:16] OVF RC / window-atomic (D.4) | read-only; burst-safe across close |

Table T-2 — first-response checklist (companion to J.1/J.5)

| Order | Read | Looking for |
|---|---|---|
| 1 | ALARM_STATUS (once) | WHICH alarm; LATCHED tells you it's historical-valid |
| 2 | ERR_STAT (once — it clears) | VIOLATION ⇒ ES-2.1.3 loop; FIFO_OVF ⇒ consumer starvation; W_VIOLATIONS trend |
| 3 | STAT (exactly once) | ERR_CNT corroborates; SEQ sane monotonic vs log |
| 4 | CFG/ALARM_CFG/WAVE_CFG readbacks | init drift vs Table T-1 reset expectations (B1-class) |
| 5 | last 10 k records: SEQ deltas + burst counters | J.1 row 5 vs row 1 host-side split |

STOP conditions worth repeating: if step 2 or 3 shows the ES-2.1.3
signature, EVERY subsequent diagnostic read in your session must also
be single-read — a debug loop is a driver (J.5 means it).

## Appendix U — Worked bus transcripts (informative)

Reference traces in the host-side AXI monitor format used by the AVV-3000
bring-up rig — same three-column rhythm as the field i2c captures, but
register-level instead of bus-transaction-level. `ts` in µs from first
beat; `op` in A(Read)/W(rite); addresses in the §6 map. Each transcript
is annotated to the clause it illustrates — these are regression
fixtures in documentation form (Q-1 patterns replay numbered sequences).

### U.1 Nominal window service (H-2 compliant) — Figure U-1

    ts      op  addr   data      note
    0.000   A   08     ->0001    STAT: DRDY=1 seen, consumed legally (H-2)
    0.150   A   40     ->0012    CH0 = 18 counts (plant stepping, E.1)
    0.210   A   44     ->0012    CH1 pair, SAME close (SEQ corroborates)
    0.270   A   48     ->0000    CH2 thermistor code
    0.330   A   4C     ->0001    ...
    0.390   A   50     ->0140    CH4 motor current
    0.450   A   54     ->0018    CH5 valve current
    0.510   A   58     ->0000    CH6 spare — reads 0 (unallocated)
    0.570   A   5C     ->0000    CH7 spare
                            [window idle ~1.2 ms]
    1.200   A   08     ->0001    NEXT window: DRDY was set at its close;
                                 first-read-of-this-window — legal (H-2)

Zero violations: one STAT read per window; SEQ advanced 1 per read; ERR
counters frozen. This is the §7.4-H-4 loop as the silicon expects it —
compare ES-2026-04's symptom list to see which line the broken firmware
inserts BETWEEN 0.570 and 1.200.

### U.2 Jitter-crossing double read (D.5 case X-2, ES-2.1.3) — Figure U-2

    ts      op  addr   data      note
    0.000   A   08     ->0001    read #1: DRDY seen — window w committed
    0.150   A   40..5C (8 reads) consume data of w
    0.920   A   08     ->0000    read #2 "confirm": DRDY=0 expected, seen
                                 VIOLATION (ERR_CNT 1, STAT_VIOLATION —
                                 invisible until ERR_STAT is read once)
    ~2.000  —   —      —         window w+1 closes normally; read #1 of
                                 w+1 sees DRDY=1 — the confirm-read never
                                 gates anything here, so the system LOOKS
                                 RIGHT while accumulating ERR_STAT rows
    [jitter variant — the D.5 X-2 crossing:]
    0.000   A   08     ->0000    read #1 of w: close hasn't landed (BUSY)
    0.990   A   08     ->0001    "confirm" NOW sees DRDY — consumed by the
                                 second read of the PREVIOUS window-cycle;
                                 w's data is fresh and readable, gate happy
                                 — and the pair straddled the close at the
                                 w|w+1 boundary EXACTLY as in ES-2.1.3's
                                 interaction note. One violation per such
                                 crossing; duty-cycle: J.2's ratio.

The two variants in Figure U-2 differ only by the phase of the close
against the loop — this is "cannot reproduce in the lab" in transistors.
Q-1 pattern 17 sweeps the phase continuously; pattern 22 verifies the
silicon still reports these when a test deliberately creates them.

### U.3 Consumer stalled — FIFO overrun, no data corruption — Figure U-3

    ts       op  addr   data      note
    w+0.000  A   08     ->0001    last healthy read
    [host stalls 25 ms — logging flush]
    w+25.104 A   08     ->0001    DRDY=1 (latest close) — one legal read
    w+25.110 A   08     ->0000    (there IS no second legal read; the
                                   next read costs a violation — shown
                                   only to document the temptation)
    w+25.160 A   20     ->00000004 ERR_STAT: FIFO_OVF set (bit 2);
                                   W_VIOLATIONS [31:8] = 0000 — read
                                   violates NOTHING (RC read = by design,
                                   NOT the STAT port — J.5 keeps these
                                   budgets separate)
    w+25.200 A   40     ->0096    CH0 of the LATEST close (12-step A
                                   plateau), not of w — correct silicon,
                                   12 windows old in the consumer's model
    log      —    SEQ   delta +13  ← the honest evidence: no phantom
                                   windows (D.6), 12 consumer misses

### U.4 Alarm-onset snapshot storm (what NOT to do) — Figure U-4

    ts      op  addr   data      note
    0.000   irq DRDY_N  ↓         wake
    0.010   A   08     ->0001    read #1 — ALARM-free window, legal
    0.011   A   10     ->0000    ALARM_STATUS RO poll — legal (RO, no
                                 consume; latch path is safe to poll)
    0.012   A   10     ->0000    (poll again — legal, cheap, RO)
    0.013   A   08     ->0000    ← THE CRIME: "let me make sure nothing
                                 changed" re-read of STAT in the same
                                 window — ES-2.1.3 trip from a DEBUG
                                 habit (J.5's "a debug loop is a driver")
    0.020   A   20     ->0002    ERR_STAT = STAT_VIOLATION — the storm
                                 caught itself; this transcript's author
                                 shipped the one-line fix, then the alarm
                                 they were chasing stopped being
                                 reproducible. It had been self-inflicted
                                 for two sprints.

### U.5 Back-to-back init writes (Q-1 pattern 11 replay) — Figure U-5

    ts      op  addr   data      note
    0.000   W   00     00000001  CTRL (step 2) AW+W both, commit 3 CLK
    0.012   W   04     00000049  CFG  (step 3, Rev-B literal)
    0.024   W   0C     00001E0F  ALARM_CFG (step 4)
    0.036   W   28     00000168  TH_RATE (step 5)
    0.048   W   24     0000001F  IRQ_MASK (step 6)
    0.060   W   00     00000001  CTRL re-assert (step 7)
    [B responses: 5 Bs enqueued+drained in order, each ≤ 6 CLK after its
     own commit — §4.3 invariants, 2-deep FIFO never overflowed; a master
     that ignores 3 pended Bs mid-sequence is still safe]
    0.080   A   08     ->0000    step 8 single read: DRDY=0 correct
                                 (pre-first-close); BUSY unchecked here —
                                 §7.2-step-8 amended wording (DRDY-only
                                 expectation; BUSY belongs to step 1)

## Appendix V — Frequently asked questions (driver authors)

**V.1 My engine steps 4× too fast — I wrote profile A, HOLD 500 ms.**
WAVE_CFG literal error: [15:2] carries `HOLD_MS << 2` (Table A-2).
Writing the raw value 500 into the register leaves 125 (=500>>2) in the
field — a 125 ms hold, exactly ¼ of what was intended, and the plant
gets no plateau at all. Use the driver's wave_start() or
recompute from Table A-2's literal column; do not hand-assemble twice.

**V.2 Median mode made my noise WORSE. Did I break it?**
No — median on Gaussian noise ≈ 0.8× advantage ONLY; below 2×
oversampling it's strictly worse (D.3: the 2× pick is max(), the 1× is a
copy). Median earns its keep against the valve-kick impulses (B.2), not
against white noise.

**V.3 I read STAT once per window but still see STAT_VIOLATION.**
Two common causes: (a) two independent readers of STAT on the same part
(driver loop + diagnostics shell — each "one read" adds up to two —
J.5; the H-2 budget is per-part, not-per-thread), (b) an ISR re-reading
after seeing DRDY=0 in the main loop it already consumed (H-2's
last clause: when first-not-ready, WAIT for next close, don't retry).

**V.4 Why does IRQ fire before DRDY_N appears on the scope?**
It doesn't — M_DRDY=0 in production init: IRQ is alarm-only. If IRQ
pulses on window close you inherited a debug IRQ_MASK, and the
alarm-picture (M_ALARM[4:0]=1F too) is on the same line: check
IRQ_MASK first whenever "IRQ too chatty".

**V.5 Is a 16× window always the best default?**
It's 8× the window latency (8 ms) and the occlusion/thermal latency
tables are in WINDOWS (Table 8-2/E.3, B-12): 14+1 latch at 8 ms windows
is 128 ms + cadence, not 32 ms. §7.2 chose 2 ms/4× as the latency
bisection; A.2's low-noise config exists for the bench when noise
matters more than latch speed.

**V.6 SWRST as fast recovery is wrong, but is ALARM_CLR enough?**
ALARM_CLR clears latches and RC counters but keeps configuration — for
the occlusion event that's exactly right (E.5 host sequence). Reserve
SWRST for "configuration state untrusted" recovery, and if used: it is a
FULL re-init (ES-2.1.1 hysteresis + Table 6-2 retention loss).

**V.7 LPM or ADC_EN=0 for standby?**
LPM: configuration and alarms persist, exit ≤ 1 window. ADC_EN=0 holds
the ENGINE in reset (STAT flags cleared) and requires the §7.2
re-init-order care on exit (V.6's step-1/BUSY poll again). Standby
between cases: LPM (A.4). Nightly: the product powers down; the part
holds no state you need across power (fuses only).

**V.8 SEQ wrapped — is my log corrupt?**
Wraps silently at 256 (D.6). Modulo-256 differencing plus your monotonic
wall clock reconstructs history; SEQ exists for LOCAL correlation (pair
records of one close — §10.2), not as an absolute clock.

**V.9 ALARM_LINE never fired in lab at 150 °C... is it dead?**
It is a 25 % PAIR-DIVERGENCE detector, not an over-pressure alarm — at
over-pressure BOTH channels rise together (§5 Table 5-1 pair), so it
STAYS QUIET exactly when OCCL screams. Drive one channel only to test
it (bench: disconnect/divide one input). J.1/D.4 cover why healthy
silicon can't mix the pair from different windows; J.3 the consumer
side that can.

**V.10 The data survives the ES-2.1.3 driver — why is H-2 a hard rule?**
The DATA survives; the READY bookkeeping doesn't (O-1: the flag's whole
job is the edge; consume it twice and the second consumer's model of
"fresh data" is broken even when bytes are right). ERR_CNT's presence
makes this visible on the bench but the fleet has no scope: an error
that "doesn't corrupt" but defeats the interface's contract is the
definition of a class-2 field failure (see ES-2.1.3 severity: HIGH
despite "data intact").

**V.11 Do I need the CTRL re-assert (step 7)?**
On cold init it's a no-op — kept byte-compatible with the recovery
sequence where it isn't (re-init after partial SWRST states; t_fw_init's
warm-restart row exists because "obviously redundant" code that's
actually load-bearing fails the quietest way possible when removed).

**V.12 Can I poll BUSY instead of DRDY for close detection?**
BUSY tells you the acquisition phase; close is a DRDY/SEQ event —
BUSY-low happens BEFORE the commit is visible to you (D.1's settle
phase ends and BUSY drops at close, but "at close" means at the close
CLAMP: read once when low; if DRDY=0 the close hasn't posted: WAIT —
that IS the H-4(a)/(d) loop, don't improvise a BUSY-spin retry).

## Appendix W — Counter trajectories: three worked forensics cases

The register-side evidence in a dispute is usually a counter TRAJECTORY,
not a single read — each case below states the witness set, what the
trace looks like over time, and the conclusion. Snapshots are RC reads
(§6.1/§4.2 — budgets per J.5); trajectories are deltas between two
snapshots with the wall clock noted.

### W.1 ES-2.1.3 (double-read driver) — Figure W-1

Witnesses: ERR_STAT read twice (start/end of a 5-min window-log
segment), STAT read once at segment end for ERR_CNT.

    t=0      snapshot A:  ERR_STAT = 0000_0000        W_VIOLATIONS = 0
    [segment: ~500 windows/s, production poll loop with Rev-A pattern]
    t=5min   snapshot B:  ERR_STAT = 0002_D229        W_VIOLATIONS = 53 801
                          STAT.ERR_CNT = F (saturated within ~30 s)

Trajectory shape: W_VIOLATIONS grows near-linearly with wall time at the
rate of the hazard duty (J.2: ≈1 per 5 000 windows → ≈1/s per broken
poll thread; the sample count says the loop ran double-read through the
whole segment). ERR_CNT saturates and STAYS — never assume "it stopped
when it hit 15": saturation says nothing about when the pattern began.
ALARM_STATUS corroboration: ALARM_OCCL latches intermixed with
unremarkable CH0 samples — the D.5 X-2 flavor of miss where the gate
consumed a pre-close DRDY.

Ruled out at the same time (same witnesses): FIFO_OVF bit clean (rows
U.3, W.3), W_VIOLATIONS zero on a reference unit run simultaneously with
the candidate firmware minus the confirm-read (the one-line diff — E.2's
margin arithmetic already excluded noise, J.5's discipline kept the
diagnostics honest, and G-1 row 3 names the defect class).

### W.2 Ring eviction masquerading as lost windows — Figure W-2

Symptom: gaps in the telemetry log, watchdog-adjacent resets (the TICKET
corpus); silicon ERR counters clean.

    t=0      snapshot A: ERR_STAT = 0        svc: win_overwrite=0,
                                             svc: status_rejected=0
    [segment: one 2 Hz status burst at t≈131.040 s]
    t=5min   snapshot B: ERR_STAT = 0        svc: win_overwrite=0,
                                             svc: status_rejected=112
    log analysis:       SEQ deltas 1 everywhere (silicon: zero lost
                                             windows — D.4/D.6)
    consumer analysis:  rows-per-window: 500/s → dips to 487/s across
                                             the burst; the dips are in
                                             the CONSUMER's bookkeeping

Conclusion form (accepted in the field-quality database verbatim):
"silicon delivered all 150 004 windows; consumer paired 487/s across a
16-window plateau" → owner: service admission policy (its acceptance
tests: zero window loss under the exact burst schedule — the
witness-counters column IS the test output, the 112 rejects ARE correct
behaviour: status lost its slots, never windows), not the part — while
the same field window, a different unit's ERR_CNT trajectory (W.1 above)
proved the ticket's alarm path was a third animal entirely: three
threads, three owners, three evidence trails; the register map keeps
them perfectly separable when read with the discipline.

### W.3 Thermal margin boundary (the honest one) — Figure W-3

    t=0      snapshot A: ERR_STAT = 0;  CH2 = 2330 (77.6 °C mapping)
    [pump-head fan failure — thermal ramp 1.2 °C/min]
    t=6min   CH2 crosses the HTEMP trip at 86 °C ≈ 2643 counts
             (reset TH_TEMP=1; B-12 ±2 °C); ALARM_STATUS: HTEMP latches,
             ALARM_LATCHED set, IRQ as configured (mask 1F)
    t=6min   CH0 6σ-noise by this point ≈ 6×(csv 60→75 °C interpol ≈
             140–190) = 840–1 140 LSB: OCCL margin ≤ 0 per E.2 —
             OCCL_DET flickers on real threshold reach
    conclusion: alarms genuine at 70+ °C (the zone where the DESIGN
             expects protection devices to fire; B-4/B-12/E.2 all line
             up) — and if HTEMP had NOT fired first while OCCL did,
             THAT is the real defect report (thermal monitor path,
             V.9/B-12 corner, not the occlusion path)

W.3 is included because "alarm at 37 °C" tickets age like this one in
REVERSE: triage must first PROVE the unit is nowhere near E.2's risk
zone (csv says margin 864 at 37 °C — proven) before entertaining the
silicon-side stories; the counter trajectories that look most exotic are
the ones that were actually working.

## Appendix X — Field-support Q&A digest (Rev A → Rev B era)

Distilled from the support queue during the platform port (the corpus
behind Appendix G/N). Questions are verbatim-honest paraphrases; answers
are normative by the sections cited.

**X.1 "The manual said read STAT twice to be sure."**
Rev-A §7.4.2, superseded the day ES-2.1.3 published (historical note:
the two-read pattern was inherited from the ADS-8622 app note — N.1
footnote 1). On Rev B the second read is a contract violation with
hardware bookkeeping. H-2 once, always.

**X.2 "Our CFG write says 0xC9; same as three builds back on the
2200."** 0xC9 sets Rev-A's bit 7, which Rev-B silicon ignores (Table
7-4): the build runs boxcar while its documentation claims median, and
it does this LOUDLY to nothing at all — no error anywhere on the bus.
Table T-1's CFG row shows the intended 0x49; t_fw_init's layout
assertion catches the divergence CI-side, forever.

**X.3 "Alarms started 1-in-few-thousand after the summer fleet
refresh. We changed nothing."** The refresh carried the port's
init/poll module forward unchanged INCLUDING the Rev-A confirm-read that
had been harmless on 2200 heads (N.1). 'Nothing changed' is
architecture-relative; the W.1 trajectory pattern recognises the class
on sight from then on.

**X.4 "Lab reproduces at 0 / 10 000. Field says 1-in-5 000 at 37 °C."**
J.2's duty arithmetic: the lab's loop phase rarely re-lands on the
hazard window; field traffic (burst scheduling, service jitter — TICKET
corpus) feeds the phase distribution. A 'cannot reproduce' verdict on a
jitter-coupled duty-cycle defect is a measurement statement, not a
refutation — and the ES itself says so in its symptom paragraph.

**X.5 "ERR_CNT = 0 in the lab unit at the complaint temperature.
Innocent?"** Of course — ERR_CNT counts the READ-protocol violation, and
a compliant test harness never causes one. The COMPLAINT-unit's
trajectory (W.1) is the evidence; healthy-unit zeros only prove the
silicon responds correctly to compliant traffic, which is precisely
what "lab cannot reproduce" was always measuring (W.3's mirror image).

**X.6 "We switched to 16× oversampling for the alarm margin and the
alarms got SLOWER."** They got ~4× slower at latch (Table 8-2 is in
windows — V.5); the noise term does improve (√16 vs √4 — 2× σ, to ≈1.6
counts), so the trade is real — but it's the WRONG lever against the E.2
arithmetic: 864 LSB of margin was never a noise deficit.

**X.7 "Does anything in silicon EVER clear ALARM_STATUS by itself?"**
No. Latches clear on ALARM_CLR or reset only (§6.11, Table 6-2) — by
design: an alarm path that self-heals is an alarm path that hides
transients the machine needs to count (see also H.2's safe-path FIT
budget, which treats the latch as evidence-quality).

**X.8 "The panel shows 2 500 000 counts/s somewhere harmless."**
AN-8851-07 §5's single-home rule: one ×1000, exactly once, at the panel
boundary. A reading 10³ off is a DUPLICATED or DOUBLED conversion —
there is no configuration where it's legitimate, and the app note's
single-home design exists so the mistake can only live in presentation
layers.

**X.9 "We removed step 7 (CTRL re-assert) — it did nothing — then the
watchdog-recovery path lost CFG."** It "did nothing" on cold init BY
CONSTRUCTION; the SWRST-residue recovery case (Table 7-5 step 7) is
where it isn't redundant. V.11, warm-restart suite.

**X.10 "The datasheet is 2 900 lines and still doesn't mention
whether I may read WAVE_STATUS twice."** RO has no consumption budget —
yes. The budgeted reads are precisely the RO*/RC-class ones (Table 6-1
legend, §6.1): STAT and every RC field. Everything else in the map is
free to poll; the ones that sting are few, and they are stinging for
reasons (ES-2.1.3 is the whole reason this paragraph exists).

## Appendix Y — Symbols and units (quick table)

| Symbol | Meaning | Unit |
|---|---|---|
| ACLK | external acquisition clock | MHz (10 nominal) |
| CLK | one ACLK period (100 ns) | — |
| T_W | one SDR window period | ms (Table 7-1) |
| t_AXC / t_B / t_ARC | AXI commit/B-latency/read latency bounds | CLK (Table B-10) |
| t_sample | active sampling phase | 0.60 · T_W |
| σ_window | window-result noise (4× boxcar + FIR unless row says otherwise) | counts (LSB) |
| OCCL_TH × 64 | occlusion trip | counts CH0 |
| OCCL_HYST × 64 | release offset below trip | counts CH0 |
| DEB | debounce N; latch at DEB+2nd DET close (Table 8-2) | windows |
| HOLD / HOLD_TOP | engine plateau / profile-B dwell | ms |
| STEP_LSB | engine step size | counts of output span |
| G, τ | reference plant gain / time constant | counts in→counts out, ms |
| SEQ | 8-bit window sequence counter | — (wraps 255→0) |
| ERR_CNT / W_VIOL | ES-2.1.3 read-violation counters | 4-bit saturating / 16-bit |
| FIFO_OVF | overrun witness (sticky RC) | — |
| FS | 16-bit full scale (32 767 codes) | LSB |
| % FS | percent of 32 767 codes (Table 5-1 scales) | — |
| LPM / SWRST / W1C / RC / RO | modes and classes | §6.1/§6.7/§7.3 |

Conventions: 'counts' = raw code units of the relevant domain in
Table 5-1 — never an implied physical unit; 'typ' = 1-σ lot mean
(Appendix P basis), 'max' = production screen limit (Appendix Q factor
included); all temperatures °C; all voltages with respect to the stated
rail's ground per §3 star scheme.
