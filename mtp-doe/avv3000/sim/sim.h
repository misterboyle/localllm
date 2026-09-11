/* sim.h — sim/ world owner contract (implemented by port_sim.c)
 *
 * The world is driven exclusively by sim_run_ms() and by the hal port's
 * DRDY wait (same tick function); there is no wall clock anywhere (the
 * r_sim hermeticity rule, plan §5). One ms = one wf_clock + one
 * plant_step + CH0 re-injection + 10000 ACLK of the ADC model (§9.2).
 */
#ifndef SIM_H
#define SIM_H

#include <stdint.h>

/* Reset the closed loop: ADC model (register defaults + ES budget),
 * waveform mirror, plant (state + noise seed). Call before every run. */
void sim_reset(void);

/* Advance the world deterministically by n milliseconds. */
void sim_run_ms(uint32_t n_ms);

#endif /* SIM_H */
