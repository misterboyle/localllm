/* model_waveform.h — C mirror of rtl/waveform.sv (AVA-QC engine, §8.1)
 *
 * The fw behavioral model deliberately drops the engine (D3 in
 * model_ads8851.c: cadence "exercised where it is golden — rtl/tb/
 * t_waveform and sim/ (S5)"). sim/ is that second home: this mirror is
 * the device side of the closed loop, so fv_staircase drives the SAME
 * golden engine semantics that rtl/ is verified against.
 *
 * The docs are the arbiter for BOTH implementations (app note profile
 * tables §3 + waveform.sv as the cycle-accurate sibling). One documented
 * deviation: tick granularity is 1 ms here vs 10 MHz ACLK in RTL — all
 * engine time scales (HOLD_MS, hold-top 1200 ms) are whole ms, so
 * behavior is identical modulo that scale.
 */
#ifndef MODEL_WAVEFORM_H
#define MODEL_WAVEFORM_H

#include <stdint.h>

void wf_reset(void);

/* Advance the engine by exactly 1 ms. en = WF_EN && ADC_EN (§6.15/§8.1:
 * the engine state machine samples EN exactly where waveform.sv does —
 * arm/start and the cfg-restart branch only). */
void wf_clock(int en);

/* WAVE_CFG write edge, carrying write data (§6.13 + ES-2.1.2:
 * PROFILE/STEP_LSB/INVERT apply immediately, a HOLD_MS change on a
 * running engine applies at the next cycle start; PROFILE=0 stops).
 * en = WF_EN && ADC_EN at the write time. */
void wf_cfg_wr(uint32_t val, int en);

/* Engine views for WAVE_DATA (§6.14) and WAVE_STATUS (§6.15, RUN|CYCLE|
 * DONE bit image; DONE is one engine-clock wide on the loop profiles and
 * sticky on the one-way profile, as in waveform.sv). */
uint32_t wf_out(void);
uint32_t wf_status(void);

#endif /* MODEL_WAVEFORM_H */
