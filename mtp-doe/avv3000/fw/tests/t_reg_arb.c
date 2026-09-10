/* t_reg_arb.c — register-access arbitration & the ES-2.1.3 window rule.
 * Rung r_fw_host (plan §4 B3 counterpart to t_drdy).
 *
 * Role split (ES-2026-04 ES-2.1.3): t_drdy guards captured DATA; this test
 * guards the ACCESS PATTERN — at most ONE STAT read per SDR window — plus
 * §4.4/H-3 arbitration at the window-commit instant. The Rev-A "double-
 * read hack" passes t_drdy; it fails here via the ES-2.1.3 hardware
 * self-check (ERR_STAT.STAT_VIOLATION / ERR_CNT), whose liveness is
 * pinned by the negative-control test below (09-10 lesson: a check that
 * can never fire is a fixture bug).
 */
#include "unity.h"
#include "ads8851_defs.h"
#include "drv_ads8851.h"
#include "model_ads8851.h"

void setUp(void)
{
    mdl_reset();
    TEST_ASSERT_EQUAL_INT_MESSAGE(ADS8851_OK, ads8851_init(), "init");
}

void tearDown(void) {}

static void warm(uint32_t windows)
{
    uint16_t ch[ADS8851_NCH];
    while (windows--)
        TEST_ASSERT_EQUAL_INT_MESSAGE(1, ads8851_poll_window(ch), "warm capture");
}

/* Golden driver pattern: EXACTLY one STAT read per window — no violations,
 * no ERR_STAT residue, no SLVERRs (§7.4.2 H-2). */
static void test_driver_uses_one_stat_read_per_window(void)
{
    warm(10);

    TEST_ASSERT_EQUAL_UINT32_MESSAGE(0u, mdl_stat_violations(),
                                     "ES-2.1.3: >1 STAT read in a window");
    TEST_ASSERT_EQUAL_UINT32_MESSAGE(1u, mdl_stat_reads_inwin_max(),
                                     "driver STAT reads per window != 1 (§7.4.2 H-2)");
    TEST_ASSERT_EQUAL_HEX32_MESSAGE(0u, mdl_rd(ADS8851_ERR_STAT),
                                    "ERR_STAT not clean (§6.16)");
    TEST_ASSERT_EQUAL_UINT32_MESSAGE(0u, mdl_slverrs(), "SLVERR seen (§4.3)");
}

/* Negative control (fixture self-check): simulate the seed-era Rev-A
 * pattern — consume the window, then append two "confirm" STAT reads in
 * the SAME window. The hardware self-check must fire on both, ERR_CNT and
 * W_VIOLATIONS must count them, and ERR_STAT must be read-to-clear (§6.16).
 * If this test ever fails, the t_reg_arb hook against the hack is dead. */
static void test_double_read_hack_trips_es213(void)
{
    warm(1);

    (void)mdl_rd(ADS8851_STAT); /* the hack's "confirm" re-read  */
    (void)mdl_rd(ADS8851_STAT); /* and once more for ERR_CNT      */

    TEST_ASSERT_EQUAL_UINT32_MESSAGE(2u, mdl_stat_violations(),
                                     "ES-2.1.3 self-check not firing");
    TEST_ASSERT_EQUAL_UINT32_MESSAGE(3u, mdl_stat_reads_inwin_max(),
                                     "violation detector lost the read count");

    uint32_t e1 = mdl_rd(ADS8851_ERR_STAT); /* read-to-clear on first read */
    TEST_ASSERT_EQUAL_UINT32_MESSAGE(ADS8851_ERR_STAT_VIOL,
                                     e1 & ADS8851_ERR_STAT_VIOL,
                                     "ERR_STAT.STAT_VIOLATION (§6.16)");
    TEST_ASSERT_EQUAL_UINT32_MESSAGE(2u,
                                     (e1 & ADS8851_STAT_ERR_CNT) >> ADS8851_ERR_CNT_SHIFT,
                                     "STAT.ERR_CNT == 2 violations (§6.9)");
    TEST_ASSERT_EQUAL_UINT32_MESSAGE(
        2u, (e1 >> ADS8851_ERR_WVIOL_SHIFT) & 0xFFu,
        "W_VIOLATIONS counter (§6.16)");
    TEST_ASSERT_EQUAL_HEX32_MESSAGE(0u, mdl_rd(ADS8851_ERR_STAT),
                                    "ERR_STAT not cleared by read (§6.16 RC)");

    /* t_drdy-blindness, pinned: double-reads do NOT stop capture — so a
     * data-only test can never catch the hack (this is why the two tests
     * exist). Do not "fix" t_drdy to also fail it. */
    uint16_t ch[ADS8851_NCH];
    TEST_ASSERT_EQUAL_INT(1, ads8851_poll_window(ch));
    TEST_ASSERT_EQUAL_UINT16(42u, ch[0]);
}

/* §4.4 / §7.4.2 H-3: accesses at the window-commit instant win and lose
 * nothing — commit delay is hardware-internal (≤ 2 ACLK), never a bus
 * error, never lost window data, never clobbered registers. */
static void test_access_at_commit_instant_loses_nothing(void)
{
    warm(2);

    uint32_t n = mdl_ticks_until_close();
    TEST_ASSERT_GREATER_THAN_UINT32(1u, n);
    mdl_tick(n - 1u); /* park at the committing tick */

    /* (No STAT probe here: this window already spent its single read on
     * warm()'s capture — probing now would be an ES-2.1.3 violation and
     * would test the wrong rule. CFG/gate traffic exercises H-3.) */
    mdl_wr(ADS8851_SAFETY_GATE, 0x3u);
    TEST_ASSERT_EQUAL_HEX32_MESSAGE(0x3u, mdl_rd(ADS8851_SAFETY_GATE),
                                    "write lost at commit instant (H-3)");
    TEST_ASSERT_EQUAL_HEX32(0x00000049u, mdl_rd(ADS8851_CFG));

    uint16_t ch[ADS8851_NCH];
    TEST_ASSERT_EQUAL_INT_MESSAGE(1, ads8851_poll_window(ch),
                                  "window lost through arbitration (H-3)");
    TEST_ASSERT_EQUAL_UINT16(42u, ch[0]);

    TEST_ASSERT_EQUAL_UINT32(0u, mdl_slverrs());
    TEST_ASSERT_EQUAL_UINT32(0u, mdl_stat_violations());
}

int main(void)
{
    UNITY_BEGIN();
    RUN_TEST(test_driver_uses_one_stat_read_per_window);
    RUN_TEST(test_double_read_hack_trips_es213);
    RUN_TEST(test_access_at_commit_instant_loses_nothing);
    return UNITY_END();
}
