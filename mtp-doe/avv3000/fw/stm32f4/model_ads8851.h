/* model_ads8851.h — behavioral model of the ADS-8851 digital core
 *
 * Test-only. Backs port_model.c for the fw host tests AND the qemu
 * (cortex-m3) run of the SAME suite. Deliberately mirrors the golden
 * semantics of rtl/ads8851_ctrl.sv (which rtl/tb/t_bus + t_waveform
 * verify cycle-accurately); deviations that do not affect tested behavior
 * are listed at the top of model_ads8851.c.
 *
 * The docs are the arbiter for both implementations:
 *   §6 register map, §7.1 window, §7.4 read protocol, ES-2026-04.
 */
#ifndef MODEL_ADS8851_H
#define MODEL_ADS8851_H

#include <stdint.h>

/* ---- device control ---- */
void     mdl_reset(void);               /* RESET_N assert + release (§6) */
void     mdl_tick(uint32_t n_aclk);     /* advance the ACLK domain */

/* ---- register access (counts toward ES-2.1.3 bookkeeping) ---- */
uint32_t mdl_rd(uint32_t off);
void     mdl_wr(uint32_t off, uint32_t val);

/* ---- model-internal views (NOT bus reads — they bypass ES-2.1.3) ---- */
int mdl_drdy_line(void);           /* STAT.DRDY flag state (DRDY_N) — peek only:
                                    * consuming happens ONLY via a STAT read */
uint32_t mdl_win_len(void);             /* current window length, ACLK */
uint32_t mdl_ticks_until_close(void);
uint32_t mdl_seq(void);
uint32_t mdl_ctrl_state(void);       /* CTRL peek (§6.7): no bus read —
                                       * costs no ES-2.1.3 budget, logs no
                                       * event; same discipline as the peek
                                       * above. Used by sim/ to sample
                                       * WF_EN/ADC_EN each world ms. */

/* ---- analog input drive (sim/ S5; fw unit tests leave it alone) ----
 * The RTL digital core latches a constant CH0 (ch0_res = 42) because the
 * plant is modeled in sim/, not in rtl/. mdl_set_ch0_in overrides the
 * ADC-input side only: it feeds BOTH the window-close CH0 latch
 * (§5 / Table 5-1) and the occlusion compare (§8.2 / Table 8-3).
 * Default (and after mdl_reset) is the RTL constant, so every existing
 * host/qemu test keeps its exact expectations. */
void mdl_set_ch0_in(uint32_t raw);

/* ---- test observations ---- */
typedef struct {
    uint8_t  wr;      /* 1 = write event, 0 = read event */
    uint16_t off;     /* register offset */
    uint32_t val;     /* value written, or value returned by the read */
} model_evt;

uint32_t    mdl_log_count(void);        /* events retained (keep-first log) */
uint32_t    mdl_log_dropped(void);      /* events dropped at capacity (>0 = overflow) */
model_evt   mdl_log_get(uint32_t i);    /* i < mdl_log_count(), 0 = oldest */
uint32_t    mdl_stat_reads(void);       /* STAT reads inside active SDR windows */
uint32_t    mdl_stat_reads_inwin_max(void); /* max STAT reads seen in one window */
uint32_t    mdl_stat_violations(void);  /* ES-2.1.3 second-read events */
uint32_t    mdl_slverrs(void);          /* decode errors on either direction */
void        mdl_obs_reset(void);        /* clear log + counters, keep registers */

#endif /* MODEL_ADS8851_H */
