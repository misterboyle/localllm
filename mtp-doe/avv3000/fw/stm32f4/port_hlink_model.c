/* port_hlink_model.c — hlink port backed by the fake-peer wire shim
 *
 * Fixture builds only (fw/tests + fw/qemu rungs). hlink_hw_tx lands in the
 * test-visible capture, hlink_hw_rx pulls injected peer bytes — the whole
 * link is a deterministic loop with no UART and no clock (hlink.h).
 */
#include "hlink_port.h"
#include "model_hlink.h"

void hlink_hw_tx(const uint8_t *bytes, uint32_t n)
{
    mhl_tx_give(bytes, n);
}

int hlink_hw_rx(uint8_t *byte)
{
    return mhl_rx_take(byte);
}
