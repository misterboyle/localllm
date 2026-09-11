/* plant.c — deterministic fluidic plant (see plant.h for the contract)
 *
 * Single state: pressure in Q16 counts. One plant_step() = 1 ms.
 * No floats, no libc time, no static init dependencies: the same seed
 * gives the same trace on every platform (r_sim hermeticity rule, plan §5).
 */
#include "plant.h"

/* Pump flow / compliance (plant.h: G = FLOW/ORIF = 6.0012, tau = 65536/ORIF
 * = 40.0 ms). dP terms per ms, applied to the Q16 state. */
#define PLANT_FLOW_Q16 9830 /* pump: dP += sp * 9830/65536  (a/C)         */
#define PLANT_ORIF_Q16 1638 /* orifice: dP -= P * 1638/65536 (1/(R*C), ms)*/

/* Mechanical relief valve, CH0 counts (~2930 mmHg, plant.h). */
#define PLANT_CEILING 32000u

/* CH0 domain top (Table 5-1: 0..32767). */
#define PLANT_FS 32767u

/* Sensor noise: +/-1 count. Fixed seed (AN-8851-07 §4 "fixed seed"),
 * numerically reviewed in plant.h against the tightest plateau tolerance. */
#define PLANT_NOISE_SEED 0x5eed51a7u

static int64_t  p_q16;      /* pressure state */
static uint32_t sp_latched; /* pump set-point, WAVE_DATA domain */
static uint8_t  occluded;
static uint8_t  noise_on;
static uint32_t lcg;

void plant_reset(void)
{
    p_q16 = 0;
    sp_latched = 0u;
    occluded = 0u;
    noise_on = 1u;
    lcg = PLANT_NOISE_SEED;
}

void plant_set_sp(uint32_t sp)
{
    sp_latched = (sp > 0xFFFFu) ? 0xFFFFu : sp;
}

void plant_set_line(int occl)
{
    occluded = occl ? 1u : 0u;
}

void plant_set_noise(int on)
{
    noise_on = on ? 1u : 0u;
}

void plant_step(void)
{
    /* Explicit Euler, BOTH terms of the PRE-step state (applying decay to
     * p+flow would bias the DC gain by (1 - dt/tau) = -2.5 %):
     *   dP = q_pump(sp)/C  -  P/(R*C)
     * sp x FLOW_Q16 IS the Q16 flow delta (0.15 counts/ms per set-point
     * count = 0.15x65536 Q16) — no extra >>16: shifting per step would
     * truncate small set-points to zero flow. */
    int64_t d_q16 = (int64_t)sp_latched * (int64_t)PLANT_FLOW_Q16;

    if (!occluded)                    /* orifice flow to the cavity (zero
                                       * when the line is occluded) */
        d_q16 -= (p_q16 * (int64_t)PLANT_ORIF_Q16) >> 16;

    p_q16 += d_q16;

    /* relief valve / no vacuum */
    if (p_q16 > ((int64_t)PLANT_CEILING << 16))
        p_q16 = (int64_t)PLANT_CEILING << 16;
    if (p_q16 < 0)
        p_q16 = 0;
}

uint32_t plant_pressure(void)
{
    return (uint32_t)((p_q16 + 32768) >> 16);
}

uint32_t plant_ch0(void)
{
    int64_t v = (int64_t)plant_pressure();

    if (noise_on) {
        /* LCG (glibc constants), one draw per sample; +/-1 count */
        lcg = lcg * 1103515245u + 12345u;
        v += (int64_t)((lcg >> 16) % 3u) - 1;
    }
    if (v < 0)
        v = 0;
    if (v > (int64_t)PLANT_FS)
        v = (int64_t)PLANT_FS;
    return (uint32_t)v;
}
