/* hal_ads8851.h — platform glue contract for drv_ads8851.c
 *
 * The driver never touches a bus directly; a port implements these three.
 * Fixture builds link port_model.c (behavioral model behind host + qemu
 * runs — the same suite executes on both). The shipping target links
 * port_stm32f4.c (I2C host-bridge access, DRDY_N on EXTI — see
 * payload/field/i2c.pcap.txt); MicroBlaze links mb/ads8851_mb_bsp.c.
 */
#ifndef HAL_ADS8851_H
#define HAL_ADS8851_H

#include <stdint.h>

/* 32-bit register read at a register-map offset (ads8851_defs.h). */
uint32_t ads8851_hw_rd(uint32_t off);

/* 32-bit register write at a register-map offset. */
void ads8851_hw_wr(uint32_t off, uint32_t val);

/* Wait for the next DRDY_N edge (§3 pin 14, §7.4.1).
 * Returns 1 = edge observed, 0 = timed out with no edge.
 * Must NOT read STAT (ES-2.1.3 once-per-window budget belongs to the
 * driver's single H-2 read). */
int ads8851_hw_wait_drdy(void);

#endif /* HAL_ADS8851_H */
