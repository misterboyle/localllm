/* t_drdy.c — window capture contract (datasheet §7.1 windows, §7.4 read
 * protocol H-1..H-4, §7.3 LPM freeze). Rung r_fw_host.
 *
 * Deliberate scope (plan §4, B3): this test asserts CAPTURED DATA ONLY.
 * The Rev-A "double-read hack" must PASS t_drdy (§4: it trips t_reg_arb
 * instead); a Rev-A two-read-CONFIRM consume fails here because the
 * confirm read sees DRDY=0 (ES-2.1.3) and the window never consumes.
 * Do not add ERR_STAT / read-count assertions here — that is t_reg_arb's
 * job, and mixing them would collapse the two-test errata trap.
 */
#include "unity.h"
#include "ads8851_defs.h"
#include "drv_ads8851.h"
#include "model_ads8851.h"

/* Digital-core window constants (model_ads8851.c header; RTL lines
 * "latch deterministic window results"). Window-data contract, §5. */
#define CH0_GOLDEN 42u
#define CH1_GOLDEN 43u

void setUp(void)
{
    mdl_reset();
    TEST_ASSERT_EQUAL_INT_MESSAGE(ADS8851_OK, ads8851_init(), "init");
}

void tearDown(void) {}

static void check_window(const uint16_t ch[ADS8851_NCH])
{
    TEST_ASSERT_EQUAL_UINT16_MESSAGE(CH0_GOLDEN, ch[0], "CH0 pump pressure");
    TEST_ASSERT_EQUAL_UINT16_MESSAGE(CH1_GOLDEN, ch[1], "CH1 line pressure");
    for (uint32_t i = 2u; i < ADS8851_NCH; i++)
        TEST_ASSERT_EQUAL_UINT16_MESSAGE(0u, ch[i], "CH2..7 idle");
}

/* One capture cycle per SDR window, sustained (§7.1 free-run). */
static void test_capture_sustained_eight_windows(void)
{
    uint16_t ch[ADS8851_NCH];

    for (uint32_t i = 0; i < 8u; i++) {
        TEST_ASSERT_EQUAL_INT_MESSAGE(1, ads8851_poll_window(ch),
                                      "window closed but nothing captured");
        check_window(ch);
    }
    TEST_ASSERT_GREATER_OR_EQUAL_UINT32_MESSAGE(8u, mdl_seq(),
                                                "SEQ must advance per close (§6.9)");
}

/* §7.3 LPM: DRDY clears and the data port freezes — captures go away;
 * exit re-arms the engine (next window within a period). */
static void test_capture_freezes_under_lpm(void)
{
    uint16_t ch[ADS8851_NCH];

    TEST_ASSERT_EQUAL_INT(1, ads8851_poll_window(ch));
    TEST_ASSERT_EQUAL_INT(1, ads8851_poll_window(ch));

    ads8851_lpm(1);
    TEST_ASSERT_EQUAL_INT_MESSAGE(0, ads8851_poll_window(ch),
                                  "capture must stop in LPM (§7.3)");
    TEST_ASSERT_EQUAL_INT(0, ads8851_poll_window(ch));

    ads8851_lpm(0);
    TEST_ASSERT_EQUAL_INT_MESSAGE(1, ads8851_poll_window(ch),
                                  "capture must resume after LPM exit (§7.3)");
    check_window(ch);
}

/* Window-close jitter (§9.2 ±5 %): the close may sit pending on DRDY_N
 * for a while before the driver looks. The pin wait must neither read
 * STAT nor consume the budget — every window after a late look must
 * still capture cleanly, with zero ES-2.1.3 violations. */
static void test_capture_recovers_from_jittered_edge(void)
{
    uint16_t ch[ADS8851_NCH];

    /* stall: close a window, then keep spinning with reads pending */
    uint32_t n = mdl_ticks_until_close();
    mdl_tick(n + 1234u);

    TEST_ASSERT_EQUAL_INT(1, ads8851_poll_window(ch)); /* late look */
    check_window(ch);
    TEST_ASSERT_EQUAL_INT(1, ads8851_poll_window(ch)); /* next window */
    check_window(ch);
    TEST_ASSERT_EQUAL_UINT32_MESSAGE(0u, mdl_stat_violations(),
                                     "edge waits must not read STAT (ES-2.1.3)");
}

/* Window period must come from CFG, not a hardcoded cadence (Table 7-1).
 * Drop the default 2 ms window to 1 ms mid-run and keep capturing. */
static void test_capture_follows_window_period_change(void)
{
    uint16_t ch[ADS8851_NCH];

    TEST_ASSERT_EQUAL_INT(1, ads8851_poll_window(ch));

    uint32_t cfg = (mdl_rd(ADS8851_CFG) & ~0x3u) | 0u; /* SDR_WINDOW=0 -> 1ms */
    mdl_wr(ADS8851_CFG, cfg);
    TEST_ASSERT_EQUAL_UINT32(10000u, mdl_win_len());

    TEST_ASSERT_EQUAL_INT(1, ads8851_poll_window(ch));
    check_window(ch);
}

int main(void)
{
    UNITY_BEGIN();
    RUN_TEST(test_capture_sustained_eight_windows);
    RUN_TEST(test_capture_freezes_under_lpm);
    RUN_TEST(test_capture_recovers_from_jittered_edge);
    RUN_TEST(test_capture_follows_window_period_change);
    return UNITY_END();
}
