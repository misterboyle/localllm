/* fv_staircase.c — closed-loop AVA-QC staircase trace vs AN-8851-07 §4
 * (rung r_sim, 25 pts; F1 payoff test, plan §4/§5).
 *
 * The loop the app note defines: the waveform engine runs a profile, WAVE_DATA
 * drives the pump set-point (sim/plant.c), CH0 carries the plant response
 * back through the real ADC window path — driver init + poll_window only, so
 * every ES-2.1.3-conformant capture in the trace is a real bus transaction.
 * DUT for this rung: the F1 waveform API of drv_ads8851.c (§8.1/§6.13) —
 * stubbed wave_* leaves CH0 flat and every section below fails.
 *
 * Tolerances are fixed by AN-8851-07 §4 (integer forms, plant domain):
 *   plateau accuracy  +/-6 % of step     A: 6 % x 72 = 4.32 -> |e| <= 4
 *                                        C: 6 % x 24 = 1.44 -> |e| <= 1
 *   rise to 95 %      within 2 x hold (per plateau, from its step)
 *   overshoot         <= 10 % of step    A: 7.2 -> bound +/-7
 *   clamp             whole-run peak <= ALARM_CFG OCCL_TH x 64
 *                     (reset 0x1E0F -> 960 counts, Table 8-3) + no alarm
 * "Plateau p" = window [p*hold+1 .. (p+1)*hold] ms after wave_start: the
 * engine's first step lands one full hold after arm (app note §2 "starts at
 * 0 ... increments by step every hold ms"), so plateau p rides WAVE_DATA
 * step p. The plant is seeded (sim/plant.c PLANT_NOISE_SEED): hermetic.
 *
 * Hold choice (S1 flag): the app note §3 default hold (A: 500 ms, B: 800)
 * exceeds the WAVE_CFG.HOLD_MS field range 1..255 (§6.13) — the driver
 * rejects it. Profile A runs the largest legal even hold (250 ms); §4
 * tolerances are hold-RELATIVE ("2 x hold"), so the checks transfer unchanged.
 * Profile C runs its app-note defaults (200 ms / 4 LSB) exactly.
 */
#include <stdlib.h>

#include "unity.h"
#include "ads8851_defs.h"
#include "drv_ads8851.h"
#include "model_ads8851.h"
#include "plant.h"
#include "sim.h"

#define MS_PER_ACLK_WIN 10000u

void setUp(void)
{
    sim_reset();          /* model registers + engine mirror + plant seed */
    plant_set_noise(1);   /* seeded default; C test opts out explicitly */
}
void tearDown(void) {}

/* Plant-domain target: round(sp * FLOW/ORIF), plant.h DC gain 6.0012 —
 * recomputed here from the constants plant.h cites, not from plant.c. */
static int tgt(int sp)
{
    return (int)(((long)sp * 9830L + 819L) / 1638L);
}

/* AN-8851-07 §3 profile geometry in engine counts */
static int a_sp(int plateau)          /* A: up 10 / down 10 / loop, 12 LSB */
{
    int phase, steps;

    if (plateau <= 0)
        return 0;                     /* starts at 0 (§2) */
    phase = plateau % 20;
    if (phase == 0)
        phase = 20;
    steps = (phase <= 10) ? phase : (20 - phase);
    return 12 * steps;
}

static int c_sp(int plateau)          /* C: up 14, one-way sticky, 4 LSB */
{
    if (plateau <= 0)
        return 0;
    return 4 * ((plateau < 14) ? plateau : 14);
}

static uint32_t win_ms(void)
{
    return mdl_win_len() / MS_PER_ACLK_WIN;   /* Table 7-1 @ 10 MHz (§9.2) */
}

/* One closed-loop sample: world advances one window, capture via the real
 * driver path. A lost window fails the test outright — the whole checker is
 * window-relative. */
static uint16_t loop_sample(void)
{
    uint16_t ch[ADS8851_NCH];

    sim_run_ms(win_ms());
    TEST_ASSERT_EQUAL_INT_MESSAGE(1, ads8851_poll_window(ch),
                                  "window lost — loop timebase broken");
    return ch[0];
}

typedef struct {
    int hold;
    int tol_plateau;                  /* +/-(6 % of step), integer floor */
    int over;                         /* 10 % of step bound: A set, C 0 */
    int peak_max;                     /* §4 clamp: ALARM_CFG-derived */
} spec_t;

/* Stream the closed loop for run_ms, checking every plateau window against
 * the profile table: accuracy at window end, per-segment overshoot and 95 %
 * rise, global clamp. done_seen counts §6.15 DONE pulses. */
static void run_staircase(const spec_t *spec, int (*sp_fn)(int),
                          int run_ms, uint32_t *done_out, int *plateaus_out)
{
    int cur = 0;                      /* plateau under observation */
    int mx = 0, mn = 0x7FFFFFFF, t95 = 0;
    uint16_t prev = 0;
    uint32_t done_seen = 0, t;
    int plateaus = 0;

    for (t = win_ms(); t <= (uint32_t)run_ms; t += win_ms()) {
        int p;
        int T, P;
        uint16_t ch = loop_sample();

        TEST_ASSERT_LESS_OR_EQUAL_UINT32_MESSAGE(
            (uint32_t)spec->peak_max, ch,
            "trace exceeded the ALARM_CFG clamp threshold (§4/Table 8-3)");
        if (ads8851_wave_status() & ADS8851_WAVE_ST_DONE)
            done_seen++;

        p = (int)((t - 1u) / (uint32_t)spec->hold);
        if (p != cur) {
            /* boundary: cur's window just ended on `prev` */
            T = tgt(sp_fn(cur));
            P = tgt(sp_fn(cur - 1));
            TEST_ASSERT_TRUE_MESSAGE(abs((int)prev - T) <= spec->tol_plateau,
                                     "plateau-end accuracy vs 6 % of step (§4)");
            if (T != P && spec->over) {
                if (T > P)
                    TEST_ASSERT_LESS_OR_EQUAL_INT_MESSAGE(
                        T + spec->over, mx, "overshoot above plateau (§4)");
                else
                    TEST_ASSERT_GREATER_OR_EQUAL_INT_MESSAGE(
                        T - spec->over, mn, "undershoot below plateau (§4)");
            }
            if (T != P) {
                TEST_ASSERT_NOT_EQUAL_MESSAGE(0, t95,
                                              "plateau never reached 95 % (§4)");
                TEST_ASSERT_LESS_OR_EQUAL_INT_MESSAGE(
                    2 * spec->hold, t95 - cur * spec->hold,
                    "rise to 95 % slower than 2 x hold (§4)");
                plateaus++;
            }
            mx = mn = (int)ch;
            t95 = 0;
            cur = p;
        } else {
            if ((int)ch > mx)
                mx = (int)ch;
            if ((int)ch < mn)
                mn = (int)ch;
        }

        T = tgt(sp_fn(cur));
        P = tgt(sp_fn(cur - 1));
        if (T != P && t95 == 0 &&
            100 * abs((int)ch - P) >= 95 * abs(T - P))
            t95 = (int)t;

        prev = ch;
    }
    *done_out = done_seen;
    *plateaus_out = plateaus;
}

/* Profile A (inspection) in closed loop: 1.5 cycles of the app-note
 * staircase with sensor noise ON (seeded), §4 tolerance set. */
static void test_fv_staircase_profile_a(void)
{
    spec_t spec;
    uint32_t done_seen = 0;
    int plateaus = 0;

    TEST_ASSERT_EQUAL_INT_MESSAGE(ADS8851_OK, ads8851_init(), "init");
    plant_set_noise(1);               /* seeded — hermetic (plan §5) */

    spec.hold = 250;                  /* largest legal hold, see header */
    spec.tol_plateau = 4;             /* 6 % of 12 LSB x gain 6 = 4.32 */
    spec.over = 7;                    /* 10 % of 72 = 7.2 */
    spec.peak_max = 960;              /* OCCL_TH 0xF x 64 (reset ALARM_CFG) */

    TEST_ASSERT_EQUAL_INT(ADS8851_OK, ads8851_wave_start(1u, 250u, 12u, 0));
    sim_run_ms(1);        /* engine arms on the first engine clock (§6.15;
                           * waveform.sv ST_IDLE arm samples EN at the edge) */
    TEST_ASSERT_TRUE(ads8851_wave_status() & ADS8851_WAVE_ST_RUN);

    run_staircase(&spec, a_sp, 7601, &done_seen, &plateaus);

    TEST_ASSERT_TRUE_MESSAGE(plateaus >= 20,
                             "fewer than one full cycle of plateaus checked");
    TEST_ASSERT_EQUAL_UINT32_MESSAGE(1u, done_seen,
                                     "DONE pulses once per cycle (§6.15) — "
                                     "exactly one cycle in the run window");
    TEST_ASSERT_TRUE(ads8851_wave_status() & ADS8851_WAVE_ST_CYCLE);
    TEST_ASSERT_EQUAL_UINT32_MESSAGE(
        0u, ads8851_alarm_read(),
        "AVA-QC run must never occlude: peak stays under the clamp (§4)");
    TEST_ASSERT_EQUAL_UINT32_MESSAGE(0u, mdl_stat_violations(),
                                     "loop is ES-2.1.3 clean");

    ads8851_wave_stop();
    TEST_ASSERT_EQUAL_UINT32(0u, ads8851_wave_status() & ADS8851_WAVE_ST_RUN);
}

/* Profile C (prime) at its exact app-note defaults: one-way — runs to the
 * 14th plateau, then WAVE_DATA holds and DONE goes sticky, RUN clears
 * (§6.15/waveform.sv ST_DONE_ST). Noiseless: the C plateau tolerance is
 * 1.44 counts and +-1 LSB sensor noise cannot live inside it (plant.h). */
static void test_fv_staircase_profile_c(void)
{
    spec_t spec;
    uint32_t done_seen = 0;
    int plateaus = 0;
    uint32_t t;

    TEST_ASSERT_EQUAL_INT_MESSAGE(ADS8851_OK, ads8851_init(), "init");
    plant_set_noise(0);

    spec.hold = 200;                  /* app-note default, legal (§6.13) */
    spec.tol_plateau = 1;             /* 6 % of 4 LSB x gain 6 = 1.44 */
    spec.over = 0;                    /* overshoot unbounded-checked (§4
                                       * bound 10 % x 24 = 2.4 < noise-free
                                       * first-order: monotone by physics) */
    spec.peak_max = 960;

    TEST_ASSERT_EQUAL_INT(ADS8851_OK, ads8851_wave_start(3u, 200u, 4u, 0));
    sim_run_ms(1);        /* arm clock (profile A test explains the parity) */

    run_staircase(&spec, c_sp, 3101, &done_seen, &plateaus);

    TEST_ASSERT_TRUE_MESSAGE(plateaus >= 14,
                             "one-way profile must walk all 14 plateaus");
    TEST_ASSERT_EQUAL_UINT32_MESSAGE(56u, ads8851_wave_out(),
                                     "one-way C stops at 14 x 4 LSB (§3)");
    TEST_ASSERT_EQUAL_UINT32(0u, ads8851_wave_status() & ADS8851_WAVE_ST_RUN);
    TEST_ASSERT_TRUE(ads8851_wave_status() & ADS8851_WAVE_ST_DONE);

    /* DONE sticky: readable at every sample after the stop, RUN stays 0 */
    for (t = 0; t < 5u; t++) {
        loop_sample();
        TEST_ASSERT_EQUAL_UINT32(56u, ads8851_wave_out());
        TEST_ASSERT_TRUE(ads8851_wave_status() & ADS8851_WAVE_ST_DONE);
        TEST_ASSERT_EQUAL_UINT32(0u,
                                 ads8851_wave_status() & ADS8851_WAVE_ST_RUN);
    }
    TEST_ASSERT_EQUAL_UINT32_MESSAGE(0u, ads8851_alarm_read(), "alarm quiet");
    TEST_ASSERT_EQUAL_UINT32_MESSAGE(0u, mdl_stat_violations(), "ES-clean");
}

int main(void)
{
    UNITY_BEGIN();
    RUN_TEST(test_fv_staircase_profile_a);
    RUN_TEST(test_fv_staircase_profile_c);
    return UNITY_END();
}
