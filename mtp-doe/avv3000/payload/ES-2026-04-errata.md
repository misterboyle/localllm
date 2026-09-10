# ES-2026-04 — ADS-8851 Errata Sheet (Rev B silicon)

**Status: OFFICIAL — applies to ALL ADS-8851-BIZ parts**
**Date: 2026-04-18 · Silicon: ADS-8851-C1 · Supersedes: none**

This sheet documents the three known silicon errata for Rev-B ADS-8851 parts.
Each entry lists the affected document reference, the silicon behavior, and
the approved workaround. Where a workaround changes a documented datasheet
procedure, the errata entry is authoritative over the datasheet.

---

## ES-2.1.1 — ALARM_CFG.OCCL_HYST reset value lost on soft reset

**Affects:** ADS-8851 datasheet Rev B §6.10 (ALARM_CFG), §6.7 (CTRL.SWRST)

**Silicon behavior:** A software reset (`CTRL[6] SWRST=1` → self-clearing)
restores ALARM_CFG to its reset value, but the `OCCL_HYST` field [7:4]
internally restores to `0x0` **before** the external reset value `0x0` is
applied — indistinguishable at the register level. The practical effect: after
any SWRST, hysteresis is 0 regardless of what the software wrote before the
reset. No other ALARM_CFG field is affected.

**Workaround:** Re-write ALARM_CFG after any software reset. The reference
driver performs this as part of its re-init path (see `fw/drv_ads8851.c`,
`ads8851_init` — ALARM_CFG is written after CTRL in the §7.2 sequence).

**Severity:** Low — no safety impact; hysteresis re-establishment is a
configuration-ordering matter.

---

## ES-2.1.2 — WAVE_CFG.HOLD_MS change ignored mid-cycle

**Affects:** ADS-8851 datasheet Rev B §6.13 (WAVE_CFG), §8.1

**Silicon behavior:** While the waveform engine is running
(`WAVE_STATUS.RUN=1`), a write to `WAVE_CFG` that changes only the `HOLD_MS`
field [9:2] is accepted (W returns OKAY) but does **not** take effect until
the current profile cycle completes. All other WAVE_CFG fields (PROFILE,
STEP_LSB, INVERT) apply immediately.

**Workaround:** Hosts that need a mid-cycle hold-time change must stop the
engine (`PROFILE=0`), rewrite WAVE_CFG, and restart. The AVA-QC app note
(profile C "Prime") explicitly recommends against mid-cycle hold changes.

**Severity:** Low — documented behavior, timing-only.

---

## ES-2.1.3 — STAT reads are destructive: second read in a window clears DRDY and flags a violation

**Affects:** ADS-8851 datasheet Rev B §6.9 (STAT), §7.4.2 (host read
protocol), **and Rev A §7.4.2's two-read STAT sequence**

**Silicon behavior:** `STAT.DRDY` [0] is a one-shot edge flag: it is **set at
SDR window close** and **cleared by the first STAT read that observes it**.
Any additional STAT read within the same SDR window:

- returns DRDY = 0 (the flag is already cleared — the second read observes
  nothing), **and**
- sets `ERR_STAT.STAT_VIOLATION` [1] and increments `STAT.ERR_CNT` [7:4]
  (read-to-clear) as a hardware self-check.

The Rev-A datasheet §7.4.2 showed a two-read STAT pattern — read, then a
"confirm" re-read — as best practice. **That procedure is invalid on Rev-B
silicon.** A driver written to Rev-A behavior (or to the Rev-B §7.4.2 text
before this errata) that reads STAT twice per window will:

1. observe DRDY on the first read and clear it,
2. observe DRDY = 0 on the "confirm" read,
3. treat the window as *not ready* if the confirm read gates consumption —
   permanently missing every window's data (the flags are consumed before
   the gate), or
4. at best, burn a redundant AXI transaction that trips STAT_VIOLATION.

The symptom in the field is **intermittent** because it interacts with window
timing jitter (±5 %, §9.2): the second read lands before or after the close
depending on jitter phase, so the failure rate is position-dependent — exactly
the "occlusion alarms fire 1-in-5000 at 37 °C, lab can't reproduce" pattern.

**Approved workaround (§7.4.2 H-2, authoritative):** read STAT exactly ONCE
per SDR window. First read observes DRDY=1 → data committed → read the data
port. If the first read observes DRDY=0 the window was not yet closed; do NOT
re-read STAT in the same window — wait for the next DRDY_N/BUSY edge and read
once then. The reference driver implements this (`ads8851_poll_window`,
`fw/drv_ads8851.c`).

**Verification:** the `t_drdy` (host) and `t_reg_arb` (RTL) integration tests
enforce the once-per-window rule; a second read is detected by
ERR_STAT.STAT_VIOLATION.

**Severity:** HIGH for any code written against Rev-A §7.4.2. Requires a
driver correction (one-line read-path change). No silicon fix planned.

---

## Historical note

ES-2.1.3 was written 2026-04-18 after the AVV-2200→AVV-3000 platform port
surfaced field units exhibiting the intermittent-alarm symptom. The datasheet
Rev B (2026-04-11) still carried the §7.4.2 text that Rev-A software (and the
initial port) followed; this sheet is the controlling document.