/* ads8851_defs.h — ADS-8851 register contract (datasheet Rev B §6, Table 6-1)
 *
 * Single source of truth on the fw side; mirrors rtl/ads8851_pkg.sv.
 * Every number here traces to a datasheet §/Table or ES-2026-04 id.
 * If this file and the datasheet disagree, the datasheet is the contract.
 */
#ifndef ADS8851_DEFS_H
#define ADS8851_DEFS_H

/* ---- register offsets (Table 6-1; base depends on the host bridge) ---- */
#define ADS8851_CTRL         0x00u  /* §6.7  RW  reset 0x00000000 */
#define ADS8851_CFG          0x04u  /* §6.8  RW  reset 0x00000004 (Rev-B layout) */
#define ADS8851_STAT         0x08u  /* §6.9  RO  destructive reads — see ES-2.1.3 */
#define ADS8851_ALARM_CFG    0x0Cu  /* §6.10 RW  reset 0x00001E0F */
#define ADS8851_ALARM_STATUS 0x10u  /* §6.11 RO  latched, clear via CTRL.ALARM_CLR */
#define ADS8851_WAVE_CFG     0x14u  /* §6.13 RW  */
#define ADS8851_WAVE_DATA    0x18u  /* §6.14 RO  */
#define ADS8851_WAVE_STATUS  0x1Cu  /* §6.15 RO  */
#define ADS8851_ERR_STAT     0x20u  /* §6.16 RC  read-to-clear */
#define ADS8851_IRQ_MASK     0x24u  /* §6.12 RW  reset 0x0000001F */
#define ADS8851_TH_RATE      0x28u  /* §6.17 RW  reset 0x00000168 */
#define ADS8851_SAFETY_GATE  0x2Cu  /* §6.18 RW  */
#define ADS8851_DATA_BASE    0x40u  /* §5    RO  CH0..CH7 at +4*ch */
#define ADS8851_NCH          8u

/* ---- CTRL (§6.7) ---- */
#define ADS8851_CTRL_ADC_EN      0x00000001u
#define ADS8851_CTRL_WF_EN       0x00000002u
#define ADS8851_CTRL_SAFELIGHT_M 0x0000000Cu /* [3:2] SAFELIGHT_MODE */
#define ADS8851_CTRL_LPM         0x00000010u
#define ADS8851_CTRL_ALARM_CLR   0x00000020u /* W1C */
#define ADS8851_CTRL_SWRST       0x00000040u /* self-clearing */

/* ---- CFG (§6.8, Rev-B layout; Rev-A had AVG_MODE at bit 7 — Table 7-4) ---- */
/* Reference init value — Table 7-3 / §7.2 step 3. NEVER set bit [7]:
 * Rev-B ignores it and t_fw_init fails (Table 7-4 delta note). */
#define ADS8851_REF_CTRL  0x00000001u /* §7.2 steps 2/7 */
#define ADS8851_REF_CFG   0x00000049u /* 2 ms window, 4x, boxcar, FIR */
#define ADS8851_CFG_BIT7  0x00000080u /* Rev-B rsvd0 — must stay 0 */

/* ---- STAT (§6.9) ---- */
#define ADS8851_STAT_DRDY     0x00000001u /* cleared by FIRST read observing it */
#define ADS8851_STAT_BUSY     0x00000002u
#define ADS8851_STAT_OCCL_DET 0x00000004u
#define ADS8851_STAT_TH_FAULT 0x00000008u
#define ADS8851_STAT_ERR_CNT  0x000000F0u /* [7:4] RC (mirrors ERR_STAT) */
#define ADS8851_STAT_SEQ      0x0000FF00u

/* ---- ALARM_STATUS (§6.11) ---- */
#define ADS8851_ALM_OCCL    0x00000001u
#define ADS8851_ALM_HTEMP   0x00000002u
#define ADS8851_ALM_LTEMP   0x00000004u
#define ADS8851_ALM_LINE    0x00000008u
#define ADS8851_ALM_COMMS   0x00000010u /* any SLVERR latches this */
#define ADS8851_ALM_LATCHED 0x00000020u

/* ---- ERR_STAT (§6.16, read-to-clear) ---- */
#define ADS8851_ERR_BRESP      0x00000001u
#define ADS8851_ERR_STAT_VIOL  0x00000002u /* ES-2.1.3: >1 STAT read per window */
#define ADS8851_ERR_FIFO_OVF   0x00000004u
#define ADS8851_ERR_CRC        0x00000008u
#define ADS8851_ERR_CNT_SHIFT  4u           /* [7:4] ERR_CNT                     */
#define ADS8851_ERR_WVIOL_SHIFT 8u          /* [31:8] W_VIOLATIONS               */

/* ---- WAVE_CFG fields (§6.13) ---- */
#define ADS8851_WAVE_PROFILE_M 0x00000003u  /* [1:0] 0=off 1=A 2=B 3=C */
#define ADS8851_WAVE_HOLD_S    2u           /* [9:2]  HOLD_MS, 1..255 ms   */
#define ADS8851_WAVE_HOLD_M    0x000003FCu
#define ADS8851_WAVE_STEP_S    10u          /* [15:10] STEP_LSB, 1..63      */
#define ADS8851_WAVE_STEP_M    0x0000FC00u
#define ADS8851_WAVE_INVERT    0x00010000u  /* [16] */

/* ---- WAVE_STATUS (§6.15) ---- */
#define ADS8851_WAVE_ST_RUN   0x00000001u
#define ADS8851_WAVE_ST_CYCLE 0x00000002u
#define ADS8851_WAVE_ST_DONE  0x00000004u

/* ---- window result port (§5): [15:0] signed result, [23:16] OVF (RC) ---- */
#define ADS8851_DATA_RES_M 0x0000FFFFu

/* driver error codes */
#define ADS8851_OK          0
#define ADS8851_ERR_RESET  (-1) /* §7.2 step 1: BUSY never cleared after reset */
#define ADS8851_ERR_DIRTY  (-2) /* §7.2 step 8: DRDY already set at clean start */
#define ADS8851_ERR_ARG    (-3) /* range violation (§6.13 fields) */

#endif /* ADS8851_DEFS_H */
