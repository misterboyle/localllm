/* t_fw_init.c — init-sequence contract (datasheet Rev B §7.2, Table 7-3/7-4;
 * ES-2026-04 ES-2.1.1). Rung r_fw_host.
 *
 * Fail-catches: Rev-A CFG layout (bit 7 set / wrong CFG value), reordered
 * init (ALARM_CFG before CTRL), missing step, stray extra bus traffic in
 * init. Golden passes; the B1 seeded driver (Rev-A CFG word) fails here.
 */
#include "unity.h"
#include "ads8851_defs.h"
#include "drv_ads8851.h"
#include "model_ads8851.h"

void setUp(void)
{
    mdl_reset();
}

void tearDown(void) {}

static model_evt evt_at(uint32_t i, const char *what)
{
    TEST_ASSERT_GREATER_THAN_UINT32_MESSAGE(i, mdl_log_count(), what);
    return mdl_log_get(i);
}

static void assert_wr(uint32_t i, uint32_t off, uint32_t val, const char *what)
{
    model_evt e = evt_at(i, what);
    TEST_ASSERT_EQUAL_UINT8_MESSAGE(1u, e.wr, what);
    TEST_ASSERT_EQUAL_HEX16_MESSAGE((uint16_t)off, e.off, what);
    TEST_ASSERT_EQUAL_HEX32_MESSAGE(val, e.val, what);
}

static void assert_rd(uint32_t i, uint32_t off, const char *what)
{
    model_evt e = evt_at(i, what);
    TEST_ASSERT_EQUAL_UINT8_MESSAGE(0u, e.wr, what);
    TEST_ASSERT_EQUAL_HEX16_MESSAGE((uint16_t)off, e.off, what);
}

/* §7.2: the EXACT transaction sequence (8 bus operations, in order). */
static void test_init_write_sequence_is_7_2(void)
{
    TEST_ASSERT_EQUAL_INT_MESSAGE(ADS8851_OK, ads8851_init(), "init rc");
    TEST_ASSERT_EQUAL_UINT32_MESSAGE(0u, mdl_log_dropped(), "log overflow");
    TEST_ASSERT_EQUAL_UINT32_MESSAGE(8u, mdl_log_count(),
                                     "init bus traffic != §7.2");
    assert_rd(0u, ADS8851_STAT, "step 1: poll BUSY==0");
    assert_wr(1u, ADS8851_CTRL, 0x00000001u, "step 2 (Table 7-3)");
    assert_wr(2u, ADS8851_CFG, 0x00000049u, "step 3 — Rev-B layout");
    assert_wr(3u, ADS8851_ALARM_CFG, 0x00001E0Fu, "step 4");
    assert_wr(4u, ADS8851_TH_RATE, 0x00000168u, "step 5");
    assert_wr(5u, ADS8851_IRQ_MASK, 0x0000001Fu, "step 6");
    assert_wr(6u, ADS8851_CTRL, 0x00000001u, "step 7 (idempotent re-assert)");
    assert_rd(7u, ADS8851_STAT, "step 8: single STAT read");
}

/* Table 7-4: CFG bit 7 was AVG_MODE on Rev-A; Rev-B wants it 0 and keeps
 * AVG_MODE at bit 5. 0x00C9 (Rev-A value) is the classic wrong answer. */
static void test_cfg_word_is_revb_layout(void)
{
    TEST_ASSERT_EQUAL_INT(ADS8851_OK, ads8851_init());

    uint32_t cfg = mdl_rd(ADS8851_CFG);
    TEST_ASSERT_EQUAL_HEX32_MESSAGE(0x00000049u, cfg,
                                    "CFG != Table 7-3 Rev-B value");
    TEST_ASSERT_EQUAL_UINT32_MESSAGE(0u, cfg & ADS8851_CFG_BIT7,
                                     "CFG[7] set — Rev-A layout (Table 7-4)");
}

/* ES-2026-04 ES-2.1.1: ALARM_CFG must be (re-)written AFTER CTRL. */
static void test_alarm_cfg_written_after_ctrl(void)
{
    TEST_ASSERT_EQUAL_INT(ADS8851_OK, ads8851_init());

    uint32_t ctrl_i = 0xFFFFFFFFu, acfg_i = 0xFFFFFFFFu;
    for (uint32_t i = 0; i < mdl_log_count(); i++) {
        model_evt e = mdl_log_get(i);
        if (!e.wr)
            continue;
        if (e.off == ADS8851_CTRL && ctrl_i == 0xFFFFFFFFu)
            ctrl_i = i;
        if (e.off == ADS8851_ALARM_CFG)
            acfg_i = i;
    }
    TEST_ASSERT_TRUE_MESSAGE(ctrl_i != 0xFFFFFFFFu, "CTRL never written");
    TEST_ASSERT_TRUE_MESSAGE(acfg_i != 0xFFFFFFFFu,
                             "ALARM_CFG never written (ES-2.1.1)");
    TEST_ASSERT_GREATER_THAN_UINT32_MESSAGE(ctrl_i, acfg_i,
                                            "ALARM_CFG before CTRL (ES-2.1.1)");
}

/* §7.2 step 8: the single start-of-day STAT read must see DRDY low. */
static void test_init_step8_sees_clean_drdy(void)
{
    TEST_ASSERT_EQUAL_INT(ADS8851_OK, ads8851_init());

    model_evt e = evt_at(7u, "step 8 read");
    TEST_ASSERT_EQUAL_UINT32_MESSAGE(0u, e.val & ADS8851_STAT_DRDY,
                                     "DRDY already set at clean start (§7.4.2 H-3)");
}

/* Init must leave a perfectly clean device: no decode errors, no alarm
 * comms latch, no ES-2.1.3 read violations, ERR_STAT empty. */
static void test_init_leaves_no_errors(void)
{
    TEST_ASSERT_EQUAL_INT(ADS8851_OK, ads8851_init());

    TEST_ASSERT_EQUAL_UINT32_MESSAGE(0u, mdl_slverrs(), "SLVERR during init");
    TEST_ASSERT_EQUAL_UINT32_MESSAGE(0u, mdl_stat_violations(),
                                     "ES-2.1.3 violation during init");
    TEST_ASSERT_EQUAL_HEX32_MESSAGE(0u, mdl_rd(ADS8851_ERR_STAT),
                                    "ERR_STAT not clean after init");
    TEST_ASSERT_EQUAL_UINT32_MESSAGE(
        0u, mdl_rd(ADS8851_ALARM_STATUS) & ADS8851_ALM_COMMS,
        "ALARM_COMMS latched (any SLVERR — §6.11)");
}

int main(void)
{
    UNITY_BEGIN();
    RUN_TEST(test_init_write_sequence_is_7_2);
    RUN_TEST(test_cfg_word_is_revb_layout);
    RUN_TEST(test_alarm_cfg_written_after_ctrl);
    RUN_TEST(test_init_step8_sees_clean_drdy);
    RUN_TEST(test_init_leaves_no_errors);
    return UNITY_END();
}
