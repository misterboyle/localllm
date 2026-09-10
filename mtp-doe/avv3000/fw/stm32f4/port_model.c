/* port_model.c — hal port backed by model_ads8851.c
 *
 * Links into every fixture build of the test suite: host tests (fw/tests)
 * and the qemu cortex-m3 run of the SAME tests (fw/qemu). Deterministic:
 * the model only advances when it is ticked here — no wall clock anywhere.
 *
 * DRDY_N edge (§7.4.1): the model's DRDY flag is the line state; waiting
 * means ticking ACLK until it asserts. Budget = 4 window lengths, which
 * covers the first window after (re)enable plus the §9.2 ±5 % jitter.
 */
#include "hal_ads8851.h"
#include "model_ads8851.h"

uint32_t ads8851_hw_rd(uint32_t off)
{
    return mdl_rd(off);
}

void ads8851_hw_wr(uint32_t off, uint32_t val)
{
    mdl_wr(off, val);
}

int ads8851_hw_wait_drdy(void)
{
    uint32_t budget = 4u * mdl_win_len() + 1000u;

    /* Pure pin observation (§7.4.1): peeking the model's DRDY costs no
     * ES-2.1.3 budget and does NOT consume the flag — only the driver's
     * ONE STAT read may consume it. A pending unread close returns
     * immediately; the data port holds the latest close either way (§5). */
    if (mdl_drdy_line())
        return 1;
    while (budget--) {
        mdl_tick(1u);
        if (mdl_drdy_line())
            return 1;
    }
    return 0; /* no edge inside the budget — windows not running */
}
