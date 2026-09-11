/* plant.h — AVV-3000 fluidic plant: pump -> line compliance -> occlusion
 *
 * The physics rung of sim/ (plan S5): a single-file deterministic plant
 * (plant.c). The waveform engine's WAVE_DATA is the pump set-point (§8.1:
 * "the waveform path drives the pump set-point"), the plant turns it into
 * CH0 pressure (Table 5-1: CH0 PUMP_P, 0..3000 mmHg -> 0..32767), and the
 * port (port_sim.c) injects that sample into the ADC model, closing the
 * loop that AN-8851-07 §4 defines for fv_staircase.
 *
 * Physics (one state variable, Q16 fixed point, 1 ms integration — no
 * floats, no wall clock: bit-identical on every host):
 *
 *   dP/dt = q_pump(sp)/C - P/(R*C)        line open
 *   dP/dt = q_pump(sp)/C                  line occluded (no orifice flow)
 *
 * Parameter derivation (the §4 tolerances of AN-8851-07 pick the numbers):
 *
 *   PLANT_ORIF_Q16/PLANT_FLOW_Q16 set the open-line DC gain
 *   G = 9830/1638 = 6.0012 counts per set-point count. Profile A peaks at
 *   120 sp-LSB (app note §3: 10 x 12), so max plateau = 720 counts. The
 *   clamp that §4 measures against is ALARM_CFG's reset occlusion
 *   threshold OCCL_TH=0xF -> 960 counts (datasheet §6.10 / Table 8-3):
 *   G must keep plateau + 10 % overshoot budget (792) under 960 — 6.0
 *   leaves a 25 % margin.
 *
 *   PLANT_ORIF_Q16 sets tau = R*C = 65536/1638 = 40.0 ms:
 *     - rise to 95 % = 3*tau = 120 ms <= 2 x hold (tightest app-note
 *       default hold: profile C, 200 ms -> 400 ms)                       §4
 *     - residual plateau lag e^(-hold/tau): profile C 0.7 % of the step
 *       <= 6 % plateau-accuracy budget                                   §4
 *
 *   PLANT_CEILING = 32000 counts (~2930 mmHg): mechanical relief valve.
 *   Below the CH0 top of range (32767, Table 5-1) so the ADC never rails
 *   in the occlusion clamp test, and above every threshold reachable from
 *   OCCL_TH[3:0] (max 960, Table 8-3) so OCCL_DET always sees the crossing.
 *
 *   Noise: seeded LCG, +/-1 count — < half the tightest app-note plateau
 *   tolerance (6 % of the 24-LSB plant step of profile C = 1.44 counts,
 *   and 4.32 for profile A) when it is used; tests may run noiseless
 *   (fv_staircase profile A keeps it on, geometry probes turn it off).
 */
#ifndef PLANT_H
#define PLANT_H

#include <stdint.h>

/* Integration cadence: plant_step() is exactly one millisecond (holds are
 * ms-scale, WAVE_CFG.HOLD_MS §6.13; SDR windows are whole ms, Table 7-1). */
void plant_reset(void);

/* Pump set-point (same domain as WAVE_DATA counts, §6.14; 0..65535).
 * Only meaningful while the engine OR a test drives it — last value wins. */
void plant_set_sp(uint32_t sp);

/* Line state: 0 = open (orifice flow to cavity), 1 = occluded. */
void plant_set_line(int occluded);

/* Sensor-noise model on/off (reset default: on, PLANT_NOISE_SEED seeded). */
void plant_set_noise(int on);

/* Advance physics by exactly 1 ms. */
void plant_step(void);

/* Instantaneous pressure, CH0 counts domain (rounded, noise-injected when
 * on, clamped to the Table 5-1 domain 0..32767). This is what the port
 * feeds to the ADC analog input each ms. */
uint32_t plant_ch0(void);

/* Noiseless view of the same state (debug/peak bookkeeping). */
uint32_t plant_pressure(void);

#endif /* PLANT_H */
