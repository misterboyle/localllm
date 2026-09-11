/* drv_ads8851.h — AVV-3000 ADS-8851 reference driver (public API)
 *
 * Contract documents: ADS-8851 datasheet Rev B (all §), ES-2026-04
 * errata sheet (referenced per call), AN-8851-07 AVA-QC app note.
 */
#ifndef DRV_ADS8851_H
#define DRV_ADS8851_H

#include <stdint.h>
#include "ads8851_defs.h"

/* §7.2 initialization sequence, exactly as written (steps 1-8).
 * Returns ADS8851_OK or negative error code (ads8851_defs.h).
 * ES-2026-04 ES-2.1.1: ALARM_CFG is re-written after CTRL in the
 * sequence — do not reorder. */
int ads8851_init(void);

/* §7.3 low-power mode: 1 = enter (acquisition idle, data port frozen),
 * 0 = exit. Register map stays live either way. */
void ads8851_lpm(int en);

/* One H-4 read cycle (§7.4): wait DRDY_N edge, ONE STAT read (H-2,
 * ES-2.1.3), read the 8-word data port on DRDY.
 * Returns 1 and fills ch[0..7] (raw register values, §5) when a window
 * was consumed; 0 when no window was ready (no edge / stale edge).
 * NEVER "confirm"-re-read here — ES-2.1.3 supersedes Rev-A §7.4.2. */
int ads8851_poll_window(uint16_t ch[ADS8851_NCH]);

/* §6.11 alarm snapshot / CTRL.ALARM_CLR (§6.7 W1C) — clears latched
 * ALARM_STATUS bits and ERR_STAT. */
uint32_t ads8851_alarm_read(void);
void ads8851_alarm_clear(void);

/* §6.7 SAFELIGHT_MODE: 0 off, 1 amber, 2 green, 3 red. */
void ads8851_safelight(uint32_t mode);

/* §6.18 SAFETY_GATE raw write (bit0 GATE_EN, bit1 AUTO_GREEN).
 * Auto-green semantics (driver returns SAFELIGHT_MODE to green when all
 * alarms clear after an ALARM_CLR) is implemented host-side by the
 * safelight app logic, not here. */
void ads8851_safety_gate(uint32_t gate);

/* ---- AVA-QC staircase waveform API (§8.1, AN-8851-07 §3) ---- */

/* profile: AN-8851-07 §3 (1=A Inspection, 2=B Flush, 3=C Prime);
 * hold_ms 1..16383 (WAVE_CFG.HOLD_MS — app-note defaults 500/800 fit),
 * step_lsb 1..63 (WAVE_CFG.STEP_LSB),
 * invert: 0 = up/infusion, 1 = down/withdrawal (WAVE_CFG.INVERT).
 * Writes WAVE_CFG then sets CTRL.WF_EN (§8.1 requires ADC_EN too —
 * ads8851_init() must have run). Returns ADS8851_OK / ADS8851_ERR_ARG. */
int ads8851_wave_start(uint32_t profile, uint32_t hold_ms,
                       uint32_t step_lsb, int invert);

/* Stop the engine (PROFILE=0, WF_EN cleared). §6.15 RUN drops. */
void ads8851_wave_stop(void);

/* Change HOLD_MS on a possibly-running engine.
 * ES-2026-04 ES-2.1.2: a mid-cycle HOLD_MS write is ignored until the
 * cycle completes — the approved workaround is stop/rewrite/restart,
 * implemented here (stop only if WAVE_STATUS.RUN). */
int ads8851_wave_set_hold(uint32_t hold_ms);

/* WAVE_STATUS (§6.15) and current WAVE_DATA (§6.14) raw reads. */
uint32_t ads8851_wave_status(void);
uint32_t ads8851_wave_out(void);

#endif /* DRV_ADS8851_H */
