/* port_sim.c — hal port + closed-loop world (sim/, plan S5)
 *
 * Links drv_ads8851.c (the DUT of fv_staircase: the F1 waveform API home
 * for this rung — stub ads8851_wave_* / §8.1 and the trace goes flat, so
 * r_sim catches F1 at the fw layer exactly as waveform.sv stubs get
 * caught by r_rtl) against model_ads8851.c (digital core, ES-2.1.3
 * budget) + model_waveform.c (the engine, device side) + plant.c
 * (pump -> line -> occlusion), wiring them into the loop AN-8851-07 §4
 * describes:
 *
 *     WAVE_DATA --(bus intercept: §6.14 served by the engine mirror,
 *                 D3 in model_ads8851.c)--> pump set-point --> plant
 *     plant CH0 sample --mdl_set_ch0_in()--> window latch / OCCL compare
 *     CH0 read ----------drv ads8851_poll_window()---> test trace
 *
 * Engine reads are intercepted BEFORE the model (WAVE_STATUS/DATA are the
 * mirror's §6.14/§6.15 image); WAVE_CFG writes fan out to BOTH the model
 * register (the driver reads it back in wave_stop/set_hold) and the
 * mirror write edge (ES-2.1.2 semantics live in wf_cfg_wr/wf_clock).
 */
#include "ads8851_defs.h"
#include "hal_ads8851.h"
#include "model_ads8851.h"
#include "model_waveform.h"
#include "plant.h"
#include "sim.h"

#define ACLK_PER_MS 10000u            /* ACLK 10 MHz (§9.2), whole windows
                                       * are whole ms (Table 7-1) */

static int sim_en(void)               /* engine EN = WF_EN && ADC_EN (§8.1) */
{
    uint32_t c = mdl_ctrl_state();
    return ((c & ADS8851_CTRL_WF_EN) && (c & ADS8851_CTRL_ADC_EN)) ? 1 : 0;
}

/* Pump command register (§8.1 abstraction): once the engine has RUN,
 * the port keeps commanding WAVE_DATA — the same value firmware would
 * write, and waveform.sv deliberately RETAINS wave_q through stop/one-
 * way-DONE (no final-command race where RUN drops on the step edge).
 * Before the engine ever RUNs the host owns the pump — fv_occl commands
 * the fault case that way. */
static int engine_engaged;

static void world_ms(void)
{
    wf_clock(sim_en());               /* engine (device clock keeps running
                                       * while the CPU idles — RTL parity) */
    if (wf_status() & ADS8851_WAVE_ST_RUN)
        engine_engaged = 1;
    if (engine_engaged)
        plant_set_sp(wf_out());
    plant_step();                     /* physics: pump sp -> compliance */
    mdl_set_ch0_in(plant_ch0());      /* sensor drives the ADC input */
    mdl_tick(ACLK_PER_MS);
}

void sim_reset(void)
{
    engine_engaged = 0;
    mdl_reset();
    wf_reset();
    plant_reset();
}

void sim_run_ms(uint32_t n_ms)
{
    while (n_ms--)
        world_ms();
}

uint32_t ads8851_hw_rd(uint32_t off)
{
    if (off == ADS8851_WAVE_DATA)
        return wf_out();              /* §6.14 from the engine (D3 home) */
    if (off == ADS8851_WAVE_STATUS)
        return wf_status();           /* §6.15 RUN/CYCLE/DONE image */
    return mdl_rd(off);
}

void ads8851_hw_wr(uint32_t off, uint32_t val)
{
    mdl_wr(off, val);                 /* register map stays in the model:
                                       * the driver read-backs WAVE_CFG */
    if (off == ADS8851_WAVE_CFG)
        wf_cfg_wr(val, sim_en());     /* §6.13 field fan-out + ES-2.1.2 */
}

int ads8851_hw_wait_drdy(void)
{
    /* No STAT reads here (ES-2.1.3): DRDY comes from the model's line
     * peek, budget in world ms (§9.3 first-close margin + 9.2 jitter
     * class headroom, same shape as port_model.c). */
    uint32_t budget = 4u * (mdl_win_len() / ACLK_PER_MS) + 3u;

    if (mdl_drdy_line())
        return 1;
    while (budget--) {
        world_ms();
        if (mdl_drdy_line())
            return 1;
    }
    return 0;
}
