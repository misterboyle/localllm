/* hlink_mb_port.c — MicroBlaze END of the HEAD-LINK (thin counterpart)
 *
 * The base controller runs the SAME hlink.c (STATUS consumer, CTRL/VER_REQ
 * producer); only the port differs. This is the axi_uartlite shape the
 * Zynq-style fabric exposes to the MicroBlaze.
 *
 * r_fw_mb (plan §10.2) syntax gate: compiled -fsyntax-only -Wconversion;
 * never linked by the test ladder (no MicroBlaze execution — plan R4).
 * The head side (STM32) keeps a drv port under the same gate via
 * port_stm32f4.c's hlink_hw_* half, so the port contract is checked from
 * BOTH ends here.
 */
#include <stdint.h>
#include "hlink_port.h"

/* axi_uartlite instance on the fabric (board fixed; distinct from the
 * TELEMETRY CONSOLE uartlite — three-serial naming trap, hlink.h). */
#define MB_HLINK_UART_BASE 0x40011000u

typedef struct {
    volatile uint32_t ctrl;   /* 0x0: loopback/reset-fifo bits   */
    volatile uint32_t status; /* 0x4: [2:0] tx fill, [3] rx valid */
    volatile uint32_t data;   /* 0x8: rx/tx fifo access          */
} uartlite_t;

static volatile uartlite_t *const huart =
    (volatile uartlite_t *)MB_HLINK_UART_BASE;

#define ULS_RX_VALID 0x8u
#define ULS_TX_FULL  0x7u  /* tx fifo holds 7 — full when fill reads 7 */

void hlink_hw_tx(const uint8_t *bytes, uint32_t n)
{
    for (uint32_t i = 0; i < n; i++) {
        /* Bounded spin (link-up watchdog is the heartbeat above, not this) */
        for (uint32_t t = 0; t < 1000000u; t++) {
            if ((huart->status & ULS_TX_FULL) != ULS_TX_FULL)
                break;
        }
        if ((huart->status & ULS_TX_FULL) == ULS_TX_FULL)
            return; /* stuck: drop the rest, the HB period retries */
        huart->data = bytes[i];
    }
}

int hlink_hw_rx(uint8_t *byte)
{
    if ((huart->status & ULS_RX_VALID) == 0u)
        return 0;
    *byte = (uint8_t)huart->data;
    return 1;
}
