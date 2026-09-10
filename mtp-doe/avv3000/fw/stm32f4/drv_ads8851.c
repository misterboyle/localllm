/* drv_ads8851.c — AVV-3000 ADS-8851 reference driver (golden)
 *
 * Platform: STM32F4 safelight controller; the ADS-8851 register space is
 * reachable through the host bridge (transport in the hal port — I2C on
 * target, behavioral model in fixture tests, see hal_ads8851.h).
 *
 * Contract: ADS-8851 datasheet Rev B + ES-2026-04 errata. Every constant
 * and ordering decision traces to a §/Table/ES id; the tests
 * (fw/tests/t_fw_init, t_drdy, t_reg_arb) enforce the same references.
 */
#include "drv_ads8851.h"
#include "hal_ads8851.h"

/* §7.2 step-1 guard: BUSY must clear after RESET_N with no acquisition
 * running; a handful of reads is plenty. This guard never loops while
 * ADC_EN is set (BUSY is high during every open window, §7.1). */
#define INIT_RESET_GUARD 8u

/* ------------------------------------------------------------------ */
/* §7.2 initialization sequence (Rev B — the configuration contract)   */
/* ------------------------------------------------------------------ */
int ads8851_init(void)
{
    /* Step 1 — poll STAT until BUSY==0 after RESET_N deassert. */
    for (uint32_t g = 0; g < INIT_RESET_GUARD; g++) {
        if ((ads8851_hw_rd(ADS8851_STAT) & ADS8851_STAT_BUSY) == 0u)
            break;
        if (g + 1u == INIT_RESET_GUARD)
            return ADS8851_ERR_RESET;
    }

    /* Step 2 — CTRL = 0x0000_0001: ADC_EN, acquisition on (Table 7-3). */
    ads8851_hw_wr(ADS8851_CTRL, ADS8851_REF_CTRL);

    /* Step 3 — CFG = 0x0000_0049: 2 ms window, 4x oversample, boxcar,
     * FIR (Table 7-3). REV-B layout: AVG_MODE is bit [5]; bit [7] is
     * rsvd and reads 0 — Rev-A code setting CFG[7] must be corrected
     * (Table 7-4; that write is IGNORED by Rev-B silicon). */
    ads8851_hw_wr(ADS8851_CFG, ADS8851_REF_CFG);

    /* Step 4 — ALARM_CFG = 0x0000_1E0F (Table 7-3). Written AFTER CTRL
     * on purpose: ES-2026-04 ES-2.1.1 requires ALARM_CFG re-write after
     * any soft reset; this ordering keeps the re-init path compliant. */
    ads8851_hw_wr(ADS8851_ALARM_CFG, 0x00001E0Fu);

    /* Step 5 — TH_RATE = 0x0000_0168: thermal every 104 windows (§6.17). */
    ads8851_hw_wr(ADS8851_TH_RATE, 0x00000168u);

    /* Step 6 — IRQ_MASK = 0x0000_001F: alarm sources unmasked, DRDY IRQ
     * off (safelight watches DRDY_N on its own line, §7.4.1). */
    ads8851_hw_wr(ADS8851_IRQ_MASK, 0x0000001Fu);

    /* Step 7 — re-assert CTRL (idempotent; clears SWRST residue). */
    ads8851_hw_wr(ADS8851_CTRL, ADS8851_REF_CTRL);

    /* Step 8 — read STAT ONCE; DRDY must be 0 at a clean start
     * (§7.4.2 H-3). BUSY is intentionally not checked here: it is only
     * low pre-enable (step 1) and at the window-close instant (§7.1).
     * One read is also the whole ES-2.1.3 budget for window 0 — under
     * no circumstances add a confirming re-read.
     * ERR_DIRTY is defensive per §7.2: on golden-semantics silicon step-1
     * already rejects any start with a pending close (DRDY and
     * BUSY-cleared never co-observable, §7.1), so the suite exercises
     * the §7.2 failure path via ERR_RESET (t_fw_init warm-restart). */
    if ((ads8851_hw_rd(ADS8851_STAT) & ADS8851_STAT_DRDY) != 0u)
        return ADS8851_ERR_DIRTY;

    return ADS8851_OK;
}

/* ------------------------------------------------------------------ */
/* §7.3 low-power mode                                                 */
/* ------------------------------------------------------------------ */
void ads8851_lpm(int en)
{
    uint32_t c = ads8851_hw_rd(ADS8851_CTRL);

    if (en)
        c |= ADS8851_CTRL_LPM;
    else
        c &= ~ADS8851_CTRL_LPM;
    ads8851_hw_wr(ADS8851_CTRL, c);
}

/* ------------------------------------------------------------------ */
/* §7.4 host read protocol (H-1/H-2/H-4), ES-2026-04 ES-2.1.3          */
/* ------------------------------------------------------------------ */
int ads8851_poll_window(uint16_t ch[ADS8851_NCH])
{
    /* H-1: rely on the DRDY_N edge (§3 pin 14, §7.4.1). The hal port
     * must not touch the register bus while waiting. */
    if (!ads8851_hw_wait_drdy())
        return 0;

    /* H-2 (ES-2.1.3 H-2 override, authoritative): read STAT EXACTLY
     * ONCE per SDR window. This read observes DRDY and clears it in the
     * same breath. The Rev-A "confirm" re-read is a REVOCATION of the
     * window budget: it returns DRDY=0 and sets ERR_STAT.STAT_VIOLATION
     * + ERR_CNT — it never adds information. */
    uint32_t st = ads8851_hw_rd(ADS8851_STAT);
    if ((st & ADS8851_STAT_DRDY) == 0u) {
        /* Window not yet closed (edge lost/jitter, §9.2 ±5 %). H-2:
         * do NOT re-read this window; wait for the next edge. */
        return 0;
    }

    /* H-4(c): data port reads are non-destructive (§5). */
    for (uint32_t i = 0; i < ADS8851_NCH; i++)
        ch[i] = (uint16_t)(ads8851_hw_rd(ADS8851_DATA_BASE + 4u * i) &
                           ADS8851_DATA_RES_M);
    return 1;
}

/* ------------------------------------------------------------------ */
/* alarms (§6.7 CTRL.ALARM_CLR, §6.11)                                 */
/* ------------------------------------------------------------------ */
uint32_t ads8851_alarm_read(void)
{
    return ads8851_hw_rd(ADS8851_ALARM_STATUS);
}

void ads8851_alarm_clear(void)
{
    /* ALARM_CLR is W1C and self-cleared by hardware (§6.7); preserve the
     * remaining CTRL bits exactly as they are. */
    ads8851_hw_wr(ADS8851_CTRL,
                  ads8851_hw_rd(ADS8851_CTRL) | ADS8851_CTRL_ALARM_CLR);
}

/* ------------------------------------------------------------------ */
/* safelight (§6.7 SAFELIGHT_MODE [3:2])                               */
/* ------------------------------------------------------------------ */
void ads8851_safelight(uint32_t mode)
{
    uint32_t c = ads8851_hw_rd(ADS8851_CTRL);

    c = (c & ~ADS8851_CTRL_SAFELIGHT_M) | ((mode & 0x3u) << 2);
    ads8851_hw_wr(ADS8851_CTRL, c);
}

void ads8851_safety_gate(uint32_t gate)
{
    ads8851_hw_wr(ADS8851_SAFETY_GATE, gate & 0x3u); /* §6.18 fields only */
}

/* ------------------------------------------------------------------ */
/* AVA-QC waveform engine (§8.1, AN-8851-07 §3, ES-2.1.2)              */
/* ------------------------------------------------------------------ */
int ads8851_wave_start(uint32_t profile, uint32_t hold_ms,
                       uint32_t step_lsb, int invert)
{
    /* Field ranges (§6.13): PROFILE 1..A/B/C, HOLD_MS 1..255,
     * STEP_LSB 1..63. Profile constants themselves are AN-8851-07 §3:
     * A=500 ms/12 LSB, B=800 ms/20 LSB, C=200 ms/4 LSB. */
    if (profile < 1u || profile > 3u || hold_ms < 1u || hold_ms > 255u ||
        step_lsb < 1u || step_lsb > 63u)
        return ADS8851_ERR_ARG;

    uint32_t w = profile |
                 ((hold_ms << ADS8851_WAVE_HOLD_S) & ADS8851_WAVE_HOLD_M) |
                 ((step_lsb << ADS8851_WAVE_STEP_S) & ADS8851_WAVE_STEP_M) |
                 (invert ? ADS8851_WAVE_INVERT : 0u);
    ads8851_hw_wr(ADS8851_WAVE_CFG, w);

    /* §8.1: the engine runs with WF_EN (and requires ADC_EN from init).
     * RUNNING = WF_EN & PROFILE != 0 (§6.15). */
    ads8851_hw_wr(ADS8851_CTRL,
                  ads8851_hw_rd(ADS8851_CTRL) | ADS8851_CTRL_WF_EN);
    return ADS8851_OK;
}

void ads8851_wave_stop(void)
{
    uint32_t w = ads8851_hw_rd(ADS8851_WAVE_CFG) & ~ADS8851_WAVE_PROFILE_M;

    ads8851_hw_wr(ADS8851_WAVE_CFG, w);              /* PROFILE=0 (§6.13) */
    ads8851_hw_wr(ADS8851_CTRL,
                  ads8851_hw_rd(ADS8851_CTRL) & ~ADS8851_CTRL_WF_EN);
}

int ads8851_wave_set_hold(uint32_t hold_ms)
{
    if (hold_ms < 1u || hold_ms > 255u)
        return ADS8851_ERR_ARG;

    uint32_t cfg = ads8851_hw_rd(ADS8851_WAVE_CFG);
    uint32_t target = (cfg & ~ADS8851_WAVE_HOLD_M) |
                      ((hold_ms << ADS8851_WAVE_HOLD_S) & ADS8851_WAVE_HOLD_M);

    /* ES-2026-04 ES-2.1.2: while running, a HOLD_MS-only write is
     * accepted but IGNORED until the cycle ends. Approved workaround:
     * stop (PROFILE=0), rewrite, restart. */
    if (ads8851_hw_rd(ADS8851_WAVE_STATUS) & ADS8851_WAVE_ST_RUN) {
        ads8851_hw_wr(ADS8851_WAVE_CFG, target & ~ADS8851_WAVE_PROFILE_M);
        ads8851_hw_wr(ADS8851_WAVE_CFG, target);
    } else {
        ads8851_hw_wr(ADS8851_WAVE_CFG, target);
    }
    return ADS8851_OK;
}

uint32_t ads8851_wave_status(void)
{
    return ads8851_hw_rd(ADS8851_WAVE_STATUS);        /* §6.15 */
}

uint32_t ads8851_wave_out(void)
{
    return ads8851_hw_rd(ADS8851_WAVE_DATA) & 0xFFFFu; /* §6.14 */
}
