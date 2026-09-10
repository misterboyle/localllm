# ADS-8851 — Octal-Channel Biopotential/Physiological ADC with Integrated Waveform Engine and Occlusion Monitor

**Preliminary Data Sheet — Rev B (2026-04-11)**

- Product: AVV-3000 Arthroscopic Pump
- Subsystem: ADS-8851 sensor ASIC (ADS-8851-C1 silicon, package TQFP-64)
- Supersedes: Rev A (2025-11-02) — see §7.2 and §11 for the Rev-B delta
- Companion docs: AM004 AXI-Lite interface spec (excerpt, `axi-lite-excerpt.md`),
  app note AVA-QC stair-step waveform patterns (`appnote-ava-qc-staircase.md`),
  ES-2026-04 errata sheet (`ES-2026-04-errata.md`)

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
§6.21 (read-to-clear counters); see also ES-2026-04 ES-2.1.3.

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
| [11:8] | OCCL_DEB | RW | 0x0 | Debounce in consecutive windows (0 = 1 window) |
| [15:12] | TH_TEMP | RW | 0x1 | Thermal fault threshold, °C offset above 85 °C (0 = 85 °C, 1 = 86 °C, …) |
| [31:16] | rsvd | RW | 0 | Reserved, write 0 |

### 6.11 ALARM_STATUS (0x10) — Latched alarm register

| Bits | Field | Access | Reset | Description |
|---|---|---|---|---|
| [0] | ALARM_OCCL | RO | 0 | Occlusion alarm: OCCL_DET asserted for ≥ OCCL_DEB consecutive windows. Latched until ALARM_CLR |
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
| [9:2] | HOLD_MS | RW | 0 | Plateau hold time per step, ms (1..255); engine cadence = HOLD_MS × 1 ms |
| [15:10] | STEP_LSB | RW | 0 | Step size, LSB of the output span (1..63) |
| [16] | INVERT | RW | 0 | 1 = invert staircase direction (infusion/withdrawal) |
| [31:17] | rsvd | RW | 0 | Reserved, write 0 |

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

### 7.2 Initialization sequence (Rev B — *the configuration contract*)

Host software must perform the following steps EXACTLY in order after reset.
Values are the ones used by the reference driver (`fw/drv_ads8851.c`) and by
all integration tests.

| Step | Action | Value | Notes |
|---|---|---|---|
| 1 | Wait for reset completion | — | poll STAT until BUSY==0 after RESET_N deassert |
| 2 | `CTRL = 0x0000_0001` | ADC_EN=1 | acquisition enabled, waveform off, mode off, no reset |
| 3 | `CFG = 0x0000_0049` | SDR_WINDOW=1 (2 ms), OVR_SAMPLE=2 (4×), AVG_MODE=0 (boxcar), FILT_EN=1 | **Rev-B layout. Rev-A software writes CFG[7]=1 here (AVG_MODE at bit 7) — that write is IGNORED by Rev-B silicon and must be corrected (Table 7-4).** |
| 4 | `ALARM_CFG = 0x0000_1E0F` | OCCL_TH=0xF (960 LSB), OCCL_HYST=0x0, OCCL_DEB=0xE (14 windows), TH_TEMP=0x1 | occlusion ~88 mmHg, 14-window debounce |
| 5 | `TH_RATE = 0x0000_0168` | TH_PERIOD=0x68 (104 windows), TH_FILTER=0x1 | thermal every ~208 ms |
| 6 | `IRQ_MASK = 0x0000_001F` | all alarm sources unmasked, DRDY IRQ off | safelight + host alarm path |
| 7 | `CTRL = 0x0000_0001` | re-assert ADC_EN (idempotent) | clears any SWRST residue |
| 8 | Read `STAT` once | expect DRDY==0, BUSY==0 | confirms clean start (§7.4.2 H-3) |

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

§7.4.2 (legacy, superseded): the Rev-A datasheet sequence diagram showed a
two-read STAT pattern (read → "confirm" re-read) as best practice. **This is
contradicted by ES-2026-04 ES-2.1.3 and must not be used on Rev-B silicon.**

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
occlusion alarm latches when OCCL_DET is true for `OCCL_DEB+1` *
consecutive* windows. The alarm releases (un-latches) only via ALARM_CLR,
re-armed by hysteresis: OCCL_DET must drop below `OCCL_TH − OCCL_HYST` for
the release to stick.

Table 8-3 — OCCL_TH unit conversion (CH0, 64-LSB steps)

| OCCL_TH | CH0 threshold (LSB) | approx. mmHg |
|---|---|---|
| 0x0 | 0 | 0 |
| 0x5 | 320 | 29 |
| 0xA | 640 | 59 |
| 0xF | 960 | 88 |
| 0x1E | 1920 | 176 |

### 8.3 Thermal faults

TH_FAULT asserts when a temperature channel (CH2/CH3) crosses
`85 °C + TH_TEMP` (TH_POL=0) or drops below the same bound (TH_POL=1).
ALARM_HTEMP/ALARM_LTEMP latch accordingly. Thermal sampling cadence is
TH_RATE.TH_PERIOD windows, median-filtered over 2^TH_FILTER samples.

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

### 9.3 Timing

- AXI write commit: ≤ 4 ACLK after both handshakes.
- B response latency: ≤ 6 ACLK from write commit; back-to-back writes must
  each produce a B within 6 ACLK of their own commit (§4.3).
- First window close after init: 2 × window period + 4 ACLK (FIR pipeline).
- DRDY_N pulse: ≥ 1 ACLK, ≤ 2 ACLK.

---

## 10. Application Information

See `appnote-ava-qc-staircase.md` for infusion-profile configuration and the
AVV-3000 userspace service (telemetry, ring buffer, CSV log) reference
design; see `fw/drv_ads8851.c` for the reference driver implementing §7.2.

---

## 11. Document Revision History

| Rev | Date | Change |
|---|---|---|
| A | 2025-11-02 | Initial release. CFG bit 7 = AVG_MODE. §7.4.2 two-read STAT sequence. |
| B | 2026-04-11 | CFG bit 7 → rsvd, AVG_MODE moved to bit 5 (Table 7-4). §7.4.2 two-read sequence marked superseded (see ES-2026-04). OCCL_DEB added to ALARM_CFG. ERR_STAT added. |