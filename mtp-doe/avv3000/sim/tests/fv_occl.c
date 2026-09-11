/* fv_occl.c — occlusion monitor in closed loop (rung r_sim, plan S5:
 * "occlusion clamp test"). Rung pairs with fv_staircase.c: here the pump is
 * commanded directly (plant API) while the FULL datasheet alarm path runs
 * through the ADC model: CH0 injection -> OCCL_DET compare (Table 8-3) ->
 * window debounce (OCCL_DEB) -> ALARM_OCCL latch -> ALARM_CLR (§8.2/§6.11).
 *
 * Golden geometry (model_ads8851.c == rtl/ads8851_ctrl.sv occ_armed wire,
 * cnt > DEB evaluated pre-increment at window close): with OCCL_DEB=2 the
 * alarm latches at the close that FOLLOWS DEB+1 consecutive OCCL_DET
 * windows — i.e. on the 4th DET window — and §8.2 makes release ALARM_CLR-
 * only (the model implements no hysteresis branch, mirroring the RTL; the
 * §8.2 hysteresis sentence stays out of scope for both implementations).
 */
#include "unity.h"
#include "ads8851_defs.h"
#include "drv_ads8851.h"
#include "model_ads8851.h"
#include "plant.h"
#include "sim.h"

/* Table 8-3 units: CH0 threshold = OCCL_TH x 64 */
#define TH_COUNTS(th)  ((uint32_t)(th) * 64u)
#define TH_OCF         TH_COUNTS(0xFu)          /* init/reset default: 960 */

/* plant relief valve (plant.h): occluded pressure can never exceed it */
#define RELIEF_COUNTS  32000u

void setUp(void)
{
    sim_reset();          /* model registers + engine mirror + plant seed */
}
void tearDown(void) {}

static uint16_t loop_sample(void)
{
    uint16_t ch[ADS8851_NCH];

    sim_run_ms(mdl_win_len() / 10000u);
    TEST_ASSERT_EQUAL_INT_MESSAGE(1, ads8851_poll_window(ch),
                                  "window lost");
    return ch[0];
}

/* Latch geometry, latch persistence, ALARM_CLR, and re-arm from scratch. */
static void test_fv_occl_latch_geometry(void)
{
    const uint32_t deb = 2u;          /* ALARM_CFG.OCCL_DEB [11:8] */
    const uint32_t latch_at = deb + 2u; /* RTL occ_armed: 4th DET window */
    uint32_t det = 0, w;
    int latched = 0;

    TEST_ASSERT_EQUAL_INT_MESSAGE(ADS8851_OK, ads8851_init(), "init");
    mdl_wr(ADS8851_ALARM_CFG, (deb << 8) | 0xFu);  /* TH 0xF, DEB 2 */

    plant_set_noise(0);              /* clean edges for geometry probing */
    plant_set_sp(256u);              /* 38 counts/ms rise: window-observable */
    plant_set_line(1);

    /* ramp into occlusion: count DET windows from the crossing */
    for (w = 1; w <= 300; w++) {
        uint16_t ch = loop_sample();

        if ((uint32_t)ch >= TH_OCF) {
            det++;
            latched = (ads8851_alarm_read() & ADS8851_ALM_OCCL) != 0u;
            if (latched)
                break;
        }
    }
    TEST_ASSERT_TRUE_MESSAGE(latched, "occlusion never latched");
    TEST_ASSERT_GREATER_OR_EQUAL_UINT32(det, latch_at);
    TEST_ASSERT_EQUAL_UINT32_MESSAGE(
        latch_at, det,
        "ALARM_OCCL must latch on the close following DEB+1 DET windows "
        "(§8.2; RTL occ_armed geometry)");

    /* vent: pressure falls, the alarm DOES NOT follow it down — release
     * is ALARM_CLR only (§6.11/§8.2) */
    plant_set_line(0);
    plant_set_sp(0);
    for (w = 1; w <= 100; w++) {
        if (loop_sample() + 40u < TH_OCF)
            break;
    }
    TEST_ASSERT_TRUE_MESSAGE(w <= 100, "pressure never fell below TH");
    for (w = 0; w < 10; w++)
        TEST_ASSERT_TRUE_MESSAGE(
            ads8851_alarm_read() & ADS8851_ALM_OCCL,
            "latched alarm must survive the pressure decay (§6.11)");
    TEST_ASSERT_TRUE(ads8851_alarm_read() & ADS8851_ALM_LATCHED);

    /* ALARM_CLR releases it (W1C, §6.7) — and nothing re-latches while
     * the line is open and the pump is idle */
    ads8851_alarm_clear();
    TEST_ASSERT_EQUAL_UINT32(0u, ads8851_alarm_read());
    for (w = 0; w < 10; w++)
        loop_sample();
    TEST_ASSERT_EQUAL_UINT32(0u, ads8851_alarm_read());

    /* re-arm: fresh occlusion walks the debounce again from zero — same
     * geometry as the first ramp (counters reset when DET breaks) */
    plant_set_line(1);
    plant_set_sp(256u);
    det = 0;
    latched = 0;
    for (w = 1; w <= 300; w++) {
        uint16_t ch = loop_sample();

        if ((uint32_t)ch >= TH_OCF) {
            det++;
            if (det < latch_at) {
                TEST_ASSERT_EQUAL_UINT32(
                    0u, ads8851_alarm_read() & ADS8851_ALM_OCCL);
            }
            if (ads8851_alarm_read() & ADS8851_ALM_OCCL) {
                latched = 1;
                break;
            }
        }
    }
    TEST_ASSERT_TRUE_MESSAGE(latched, "re-armed occlusion never latched");
    TEST_ASSERT_EQUAL_UINT32(latch_at, det);
}

/* Full-bore occlusion: pump at a real command level, line blocked — the
 * plant's relief valve holds pressure below its ceiling while the datasheet
 * alarm fires with the reset ALARM_CFG (OCCL_TH 0xF, OCCL_DEB 0xE). This is
 * the "clamp" of AN-8851-07 §4 pushed to the fault case. */
static void test_fv_occl_relief_clamp(void)
{
    uint16_t ch;
    uint32_t peak = 0, w;
    int latched = 0, crossed = 0;

    TEST_ASSERT_EQUAL_INT_MESSAGE(ADS8851_OK, ads8851_init(), "init");
    /* ALARM_CFG stays at the §7.2 step-4 reference 0x1E0F (TH 0xF, DEB 0xE) */

    plant_set_noise(0);
    plant_set_sp(4096u);
    plant_set_line(1);

    for (w = 1; w <= 500; w++) {
        ch = loop_sample();
        if ((uint32_t)ch > peak)
            peak = (uint32_t)ch;
        if ((uint32_t)ch >= TH_OCF)
            crossed = 1;
        if (ads8851_alarm_read() & ADS8851_ALM_OCCL) {
            latched = 1;
            break;
        }
    }
    TEST_ASSERT_TRUE_MESSAGE(crossed, "occlusion never crossed TH_OCF");
    TEST_ASSERT_TRUE_MESSAGE(latched,
                             "DEB+1 window alarm must fire well before the "
                             "relief valve is needed (§8.2)");

    /* let it ride to the relief valve, then verify the clamp bound */
    for (w = 1; w <= 500; w++) {
        ch = loop_sample();
        if ((uint32_t)ch > peak)
            peak = (uint32_t)ch;
        if (peak >= RELIEF_COUNTS - 100u)
            break;
    }
    TEST_ASSERT_LESS_OR_EQUAL_UINT32_MESSAGE(
        RELIEF_COUNTS, peak,
        "occluded pressure exceeded the plant relief ceiling");
    TEST_ASSERT_TRUE_MESSAGE(peak >= RELIEF_COUNTS - 100u,
                             "relief valve never came into play");
}

int main(void)
{
    UNITY_BEGIN();
    RUN_TEST(test_fv_occl_latch_geometry);
    RUN_TEST(test_fv_occl_relief_clamp);
    return UNITY_END();
}
