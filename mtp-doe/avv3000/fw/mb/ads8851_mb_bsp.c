/* ads8851_mb_bsp.c — MicroBlaze BSP layer for the ADS-8851 (optional rung)
 *
 * r_fw_mb (plan §5/§10.2): the fixture has no MicroBlaze GCC on macOS
 * (plan R4). The stable substitute accepted in §10 is a syntax/header/
 * semantics gate: this BSP + ../stm32f4/drv_ads8851.c must compile
 * -fsyntax-only under the host C11 compiler. Deviation recorded in the
 * plan progress log 2026-09-10: `clang -target microblaze` does not and
 * will not exist (upstream clang has no MicroBlaze backend), so the
 * target triple is replaced by the host compiler; what remains checked
 * is exactly what is portable-observable: header hygiene, prototypes,
 * const-correctness of the register contract, and the port API shape.
 * If a MicroBlaze GCC ever shows up, this file is the whole port.
 */
#include <stdint.h>
#include "ads8851_defs.h"

/* AXI-Lite peripheral base on the Zynq-style fabric (§6 — the real
 * address; MicroBlaze sees the register map directly, no I2C bridge). */
#define MB_ADS8851_BASE 0x40000000u

static inline uint32_t mb_rd32(uint32_t addr)
{
    return *(volatile uint32_t *)(uintptr_t)addr;
}

static inline void mb_wr32(uint32_t addr, uint32_t val)
{
    *(volatile uint32_t *)(uintptr_t)addr = val;
}

/* ---- hal_ads8851.h contract ---- */
uint32_t ads8851_hw_rd(uint32_t off);
void ads8851_hw_wr(uint32_t off, uint32_t val);
int ads8851_hw_wait_drdy(void);

uint32_t ads8851_hw_rd(uint32_t off)
{
    return mb_rd32(MB_ADS8851_BASE + off);
}

void ads8851_hw_wr(uint32_t off, uint32_t val)
{
    mb_wr32(MB_ADS8851_BASE + off, val);
}

/* DRDY_N arrives on AXI GPIO (ps_gpio irq): xgpio_drdy_wait() blocks on
 * the driver's ISR flag — never on a STAT read (ES-2026-04 ES-2.1.3).
 * The BSP glue itself is xparameters-generated; only the entry point is
 * referenced here so the syntax gate stays self-contained. */
extern int xgpio_drdy_wait(unsigned timeout_ms);

int ads8851_hw_wait_drdy(void)
{
    return xgpio_drdy_wait(50u) == 1 ? 1 : 0;
}
