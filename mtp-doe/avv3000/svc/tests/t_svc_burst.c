/* t_svc_burst.c — ring admission policy under the 2 Hz status burst.
 * Rung r_svc. THE B2 fail-catch: the field unit (payload/field/uart.log)
 * lost 12..19 window lines per burst because status bursts could fill the
 * shared ring; the corrected policy (telem_svc.h) can never lose window
 * records to status. Seeds B2a/B2b invert guard/drop policy and fail here.
 *
 * Determinism: no clocks/threads — the tests drive svc_ms(); the field
 * profile replays the gen_uart.py model (2 ms window stream, ~2 kHz line
 * writer, 100 ms/2 kHz/5 s status burst) in pure simulation.
 */
#include <string.h>

#include "unity.h"
#include "telem_svc.h"

#define RUN_MS 29998u          /* bursts fully inside: 5000..25000 (5) */
#define N_WIN  (RUN_MS / 2u + 1u)   /* ts 0,2,..29998 -> 15000 records */
#define N_GEN  (5u * (SVC_BURST_MS * SVC_BURST_PER_MS)) /* 1000 status */

static svc_t g_svc;

/* sink capture */
static char g_status_line[80];
static uint32_t g_st_lines;
static uint32_t g_win_seen;
static uint32_t g_expect_seq;     /* continuity: seq must be +1 */
static uint32_t g_first_seq;      /* first window line seen     */
static uint32_t g_sink_fail;

static void sink(void *ctx, const char *line)
{
    svc_rec_t r;

    (void)ctx;
    if (line[0] == '#') {
        if (strlen(line) < sizeof g_status_line) {
            strcpy(g_status_line, line);
        }
        g_st_lines++;
        return;
    }
    if (svc_parse_line(line, &r) != SVC_OK) {
        g_sink_fail++;            /* non-comment line must re-parse */
        return;
    }
    if (g_win_seen == 0u) {
        g_first_seq = r.seq;
    }
    if (r.seq != g_expect_seq) {
        g_sink_fail++;            /* gap or reorder */
    }
    g_expect_seq = r.seq + 1u;
    g_win_seen++;
}

void setUp(void)
{
    svc_init(&g_svc, sink, 0);
    g_status_line[0] = '\0';
    g_st_lines = 0;
    g_win_seen = 0;
    g_expect_seq = 1;
    g_first_seq = 0;
    g_sink_fail = 0;
}

void tearDown(void) {}

/* gen_uart.py footprint: windows every 2 ms, seq = ts/2 + 1 */
static svc_rec_t field_win(uint32_t ts)
{
    svc_rec_t r;
    r.ts_ms = ts;
    r.seq = ts / 2u + 1u;
    r.t_c10 = (int16_t)(368 + ts / 1000u);   /* 36.8 -> 39.8 ramp */
    r.ch0_mmhg = 42;
    r.ch1_mmhg = 43;
    r.flags = 0;
    r.cls = SVC_WIN;
    return r;
}

static void run_field_profile(void)
{
    uint32_t ts;
    for (ts = 0; ts <= RUN_MS; ts += 2u) {
        svc_rec_t r = field_win(ts);
        TEST_ASSERT_EQUAL_INT_MESSAGE(SVC_OK, svc_ingest(&g_svc, &r),
                                      "field window record rejected");
        svc_ms(&g_svc, ts);
    }
    svc_flush(&g_svc);
}

/* 1. THE contract: field profile loses zero window lines, end to end.
 *    Sink asserts continuity (seq 1..N_WIN, parse-clean); stats assert
 *    the guard actually bled status (pressure real; no false green) and
 *    that no window was ever overwritten. */
static void test_field_profile_zero_window_gaps(void)
{
    svc_stats_t st;

    run_field_profile();

    TEST_ASSERT_EQUAL_UINT32_MESSAGE(0, g_sink_fail,
                                     "window stream: gap/reorder/unparseable");
    TEST_ASSERT_EQUAL_UINT32(N_WIN, g_win_seen);
    TEST_ASSERT_EQUAL_UINT32(1u, g_first_seq);
    TEST_ASSERT_EQUAL_UINT32(N_WIN + 1u, g_expect_seq);

    svc_stats(&g_svc, &st);
    TEST_ASSERT_EQUAL_UINT32(0u, st.win_overwrite);
    TEST_ASSERT_EQUAL_UINT32(0u, st.st_evicted);
    TEST_ASSERT_EQUAL_UINT32(N_WIN, st.win_in);
    TEST_ASSERT_EQUAL_UINT32(N_WIN, st.logged);
    /* liveness: 1 kHz writer vs 2.5 kHz burst-window demand — the guard
     * must bleed the burst hard (golden ~60% rejected; floor keeps a
     * guard-free seed from sneaking through as "everything fits") */
    TEST_ASSERT_GREATER_THAN_UINT32(N_GEN / 4u, st.st_rejected);
    TEST_ASSERT_GREATER_THAN_UINT32(N_GEN / 8u, st.st_in);
}

/* 2. accounting: every generated status is in or rejected — none lost
 *    silently; burst-free time generates none. */
static void test_burst_accounting(void)
{
    svc_stats_t st;

    svc_ms(&g_svc, SVC_BURST_PERIOD - 1u);
    svc_stats(&g_svc, &st);
    TEST_ASSERT_EQUAL_UINT32(0u, st.st_in + st.st_rejected);

    run_field_profile();

    svc_stats(&g_svc, &st);
    TEST_ASSERT_EQUAL_UINT32(N_GEN, st.st_in + st.st_rejected);
    TEST_ASSERT_EQUAL_UINT32(st.st_in, g_st_lines); /* in == logged (flush) */
    TEST_ASSERT_EQUAL_UINT32(st.st_in, st.st_logged);
}

/* 3. B2 fail-catch microprobe: status saturates the ring to its guard,
 *    then more status is rejected — and a window still gets in with zero
 *    losses of any kind. (Seeds that drop-on-full or drop the guard fail
 *    here: no guard -> st_rejected==0 geometry breaks; naive reject when
 *    physically full -> the window assert breaks.) */
static void test_window_beats_saturated_status(void)
{
    svc_stats_t st;
    svc_rec_t w = field_win(4000);
    uint32_t i;

    w.seq = 1000;
    for (i = 0; i < 32u; i++) {
        svc_rec_t s;
        s.ts_ms = i;
        s.seq = 0;
        s.t_c10 = 0;
        s.ch0_mmhg = 0;
        s.ch1_mmhg = 0;
        s.flags = 0;
        s.cls = SVC_STATUS;
        TEST_ASSERT_EQUAL_INT_MESSAGE(SVC_OK, svc_ingest(&g_svc, &s),
                                      "status below guard must be accepted");
    }
    for (i = 0; i < 8u; i++) {
        svc_rec_t s;
        s.ts_ms = i;
        s.seq = 0;
        s.t_c10 = 0;
        s.ch0_mmhg = 0;
        s.ch1_mmhg = 0;
        s.flags = 0;
        s.cls = SVC_STATUS;
        TEST_ASSERT_EQUAL_INT_MESSAGE(SVC_ST_DROP, svc_ingest(&g_svc, &s),
                                      "status at guard must be rejected");
    }

    TEST_ASSERT_EQUAL_INT(SVC_OK, svc_ingest(&g_svc, &w));
    svc_stats(&g_svc, &st);
    TEST_ASSERT_EQUAL_UINT32(32u, st.st_in);
    TEST_ASSERT_EQUAL_UINT32(8u, st.st_rejected);
    TEST_ASSERT_EQUAL_UINT32(0u, st.win_overwrite + st.st_evicted);

    g_expect_seq = 1000u;    /* continuity of just this one window */
    svc_flush(&g_svc);
    TEST_ASSERT_EQUAL_UINT32(1u, g_win_seen);   /* the window survived */
    TEST_ASSERT_EQUAL_UINT32(32u, g_st_lines);
    TEST_ASSERT_EQUAL_UINT32_MESSAGE(0, g_sink_fail, "output misordered");
}

/* 4. eviction geometry: only a writer stall with an ALL-WINDOW ring can
 *    cost a window record — counted, ordered, status untouched. */
static void test_window_overwrite_only_on_stall(void)
{
    svc_stats_t st;
    uint32_t i;

    for (i = 0; i < SVC_RING_CAP; i++) {
        svc_rec_t w = field_win(i * 2u);
        w.seq = 100u + i;
        TEST_ASSERT_EQUAL_INT(SVC_OK, svc_ingest(&g_svc, &w));
    }
    {
        svc_rec_t w = field_win(SVC_RING_CAP * 2u);
        w.seq = 164u;
        TEST_ASSERT_EQUAL_INT(SVC_OK, svc_ingest(&g_svc, &w));
    }
    svc_stats(&g_svc, &st);
    TEST_ASSERT_EQUAL_UINT32(1u, st.win_overwrite);
    TEST_ASSERT_EQUAL_UINT32(0u, st.st_evicted);

    g_expect_seq = 101u;     /* survivors are contiguous 101..164 */
    svc_flush(&g_svc);
    TEST_ASSERT_EQUAL_UINT32(SVC_RING_CAP, g_win_seen);
    TEST_ASSERT_EQUAL_UINT32(101u, g_first_seq); /* oldest (100) evicted */
    TEST_ASSERT_EQUAL_UINT32_MESSAGE(0, g_sink_fail, "survivors must be contiguous");
}

/* 5. status console line is byte-pinned: "#st,<ts>,<last drained win seq>"
 *    (telem_svc.h; last_win_seq advances at EMIT, not ingest). */
static void test_status_line_format(void)
{
    svc_rec_t w = field_win(22);   /* seq 12 */
    svc_rec_t s;

    TEST_ASSERT_EQUAL_INT(SVC_OK, svc_ingest(&g_svc, &w));
    svc_ms(&g_svc, 22u);           /* drains (budget 44) -> last_win_seq=12 */
    TEST_ASSERT_EQUAL_UINT32(1u, g_win_seen);

    s.ts_ms = 7;
    s.seq = 999;                   /* must be ignored; seq field rewritten
                                    * at generation only — manual probe
                                    * passes through as-is */
    s.t_c10 = 0;
    s.ch0_mmhg = 0;
    s.ch1_mmhg = 0;
    s.flags = 0;
    s.cls = SVC_STATUS;
    TEST_ASSERT_EQUAL_INT(SVC_OK, svc_ingest(&g_svc, &s));
    svc_flush(&g_svc);

    TEST_ASSERT_EQUAL_UINT32(1u, g_st_lines);
    TEST_ASSERT_EQUAL_STRING("#st,7,999", g_status_line);
}

int main(void)
{
    UNITY_BEGIN();
    RUN_TEST(test_field_profile_zero_window_gaps);
    RUN_TEST(test_burst_accounting);
    RUN_TEST(test_window_beats_saturated_status);
    RUN_TEST(test_window_overwrite_only_on_stall);
    RUN_TEST(test_status_line_format);
    return UNITY_END();
}
