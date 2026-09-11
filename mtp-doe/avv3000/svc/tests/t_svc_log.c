/* t_svc_log.c — CSV log byte contract (rung r_svc, the "log golden" part).
 *
 * Pins the canonical line format against payload/field/uart.log samples
 * (formatter + parser + flag normalization), then compares the emitted
 * 1 s service excerpt BYTE-FOR-BYTE against t_svc_log_ref.csv.
 *
 * The reference file is generated from the golden (--regen) and ties to
 * the payload format contract through the spot-checks below (each golden
 * line is hand-derived from run_excerpt()'s formulas — see those comments).
 *
 * Regen after an INTENTIONAL format change (payload-owner call):
 *   ./build/t_svc_log --regen
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "unity.h"
#include "telem_svc.h"

#define REF_FILE "t_svc_log_ref.csv"
#define BUF_CAP  (1u << 18)

static char g_buf[BUF_CAP];
static uint32_t g_len;
static uint32_t g_overflow;

void setUp(void)
{
    g_len = 0;
    g_overflow = 0;
    g_buf[0] = '\0';
}

void tearDown(void) {}

static void sink(void *ctx, const char *line)
{
    uint32_t n = (uint32_t)strlen(line);
    (void)ctx;
    if (g_len + n + 2u >= BUF_CAP) {
        g_overflow++;
        return;
    }
    memcpy(g_buf + g_len, line, n);
    g_len += n;
    g_buf[g_len++] = '\n';
}

/* ---------- 1. formatter byte-pin ---------- */

static void assert_fmt(uint32_t ts, uint32_t seq, int16_t t10, uint16_t c0,
                       uint16_t c1, uint8_t flags, const char *want)
{
    svc_rec_t r;
    char buf[80];
    r.ts_ms = ts; r.seq = seq; r.t_c10 = t10;
    r.ch0_mmhg = c0; r.ch1_mmhg = c1; r.flags = flags; r.cls = SVC_WIN;
    TEST_ASSERT_EQUAL_INT_MESSAGE((int)strlen(want),
                                  svc_format_win(&r, buf, (int)sizeof buf), want);
    TEST_ASSERT_EQUAL_STRING_MESSAGE(want, buf, "");
}

static void test_format_exact(void)
{
    /* the uart.log sample line, verbatim (payload/field/uart.log, first sample line) */
    assert_fmt(120u, 61u, 368, 41u, 42u, 0u, "120,61,36.8,41,42,");
    /* flag order is always O,A,H,L regardless of value bit order */
    assert_fmt(1u, 1u, 200, 0u, 0u, SVC_F_L | SVC_F_H, "1,1,20.0,0,0,HL");
    assert_fmt(1u, 1u, 200, 0u, 0u, SVC_F_O | SVC_F_A | SVC_F_H | SVC_F_L,
               "1,1,20.0,0,0,OAHL");
    /* fixed-point temps, no FP rounding */
    assert_fmt(1u, 1u, -5, 0u, 0u, 0u, "1,1,-0.5,0,0,");
    assert_fmt(1u, 1u, -123, 0u, 0u, 0u, "1,1,-12.3,0,0,");
    assert_fmt(1u, 1u, 32767, 0u, 0u, 0u, "1,1,3276.7,0,0,");
    /* numeric extremes: ts/seq full u32, temp full int16 bounds, channels
     * at the Table 5-1 domain bounds (0 / 32767) */
    assert_fmt(4294967295u, 4294967295u, 0, 0u, 32767u, SVC_F_O,
               "4294967295,4294967295,0.0,0,32767,O");
    assert_fmt(1u, 1u, INT16_MIN, 0u, 32767u, 0u, "1,1,-3276.8,0,32767,");
    assert_fmt(1u, 1u, INT16_MAX, 32767u, 0u, 0u, "1,1,3276.7,32767,0,");
}

/* ---------- 2. parser: exact, normalization, rejects ---------- */

static void test_parse_exact(void)
{
    svc_rec_t r;
    TEST_ASSERT_EQUAL_INT(SVC_OK, svc_parse_line("120,61,36.8,41,42,", &r));
    TEST_ASSERT_EQUAL_UINT32(120u, r.ts_ms);
    TEST_ASSERT_EQUAL_UINT32(61u, r.seq);
    TEST_ASSERT_EQUAL_INT16(368, r.t_c10);
    TEST_ASSERT_EQUAL_INT16(41, r.ch0_mmhg);
    TEST_ASSERT_EQUAL_INT16(42, r.ch1_mmhg);
    TEST_ASSERT_EQUAL_UINT8(0u, r.flags);
    TEST_ASSERT_EQUAL_UINT8(SVC_WIN, r.cls);

    /* flags normalized regardless of input order */
    TEST_ASSERT_EQUAL_INT(SVC_OK, svc_parse_line("1,2,3.4,5,6,LHO", &r));
    TEST_ASSERT_EQUAL_UINT8(SVC_F_L | SVC_F_H | SVC_F_O, r.flags);

    /* CRLF-tolerant (wire), negative temp */
    TEST_ASSERT_EQUAL_INT(SVC_OK, svc_parse_line("9,9,-0.5,0,1,A\r\n", &r));
    TEST_ASSERT_EQUAL_INT16(-5, r.t_c10);
    TEST_ASSERT_EQUAL_UINT8(SVC_F_A, r.flags);

    /* ignores: blank/eol/service comment lines ("#st,..." self-input) */
    TEST_ASSERT_EQUAL_INT(SVC_IGNORE, svc_parse_line("", &r));
    TEST_ASSERT_EQUAL_INT(SVC_IGNORE, svc_parse_line("\n", &r));
    TEST_ASSERT_EQUAL_INT(SVC_IGNORE, svc_parse_line("# anything", &r));
    TEST_ASSERT_EQUAL_INT(SVC_IGNORE, svc_parse_line("#st,7,12", &r));

    /* rejects */
    TEST_ASSERT_EQUAL_INT(SVC_EBAD, svc_parse_line("a,b,c,d,e,", &r));
    TEST_ASSERT_EQUAL_INT(SVC_EBAD, svc_parse_line("1,2,3,4,5,X", &r));
    TEST_ASSERT_EQUAL_INT(SVC_EBAD, svc_parse_line("1,2,3.4,5,6", &r));    /* no tail comma */
    TEST_ASSERT_EQUAL_INT(SVC_EBAD, svc_parse_line("1,2,3,4,", &r));       /* short */
    TEST_ASSERT_EQUAL_INT(SVC_EBAD, svc_parse_line("1,,2,3,4,", &r));
    TEST_ASSERT_EQUAL_INT(SVC_EBAD, svc_parse_line("1,2,3.45,5,6,", &r));  /* 2 decimals */
    TEST_ASSERT_EQUAL_INT(SVC_EBAD, svc_parse_line("1,2,.3,5,6,", &r));    /* no int part */
    /* Table 5-1 channel domain: bounds accepted, minus sign and over-range
     * REJECTED (never clamped, never wrapped) */
    TEST_ASSERT_EQUAL_INT(SVC_OK, svc_parse_line("1,2,-3276.8,0,32767,O", &r));
    TEST_ASSERT_EQUAL_INT16(-32768, r.t_c10);     /* temp signed, full int16 */
    TEST_ASSERT_EQUAL_UINT16(0u, r.ch0_mmhg);
    TEST_ASSERT_EQUAL_UINT16(32767u, r.ch1_mmhg);
    TEST_ASSERT_EQUAL_INT(SVC_EBAD, svc_parse_line("1,2,3.4,-0,6,", &r));     /* minus illegal */
    TEST_ASSERT_EQUAL_INT(SVC_EBAD, svc_parse_line("1,2,3.4,-1,6,", &r));
    TEST_ASSERT_EQUAL_INT(SVC_EBAD, svc_parse_line("1,2,3.4,-33000,6,", &r));
    TEST_ASSERT_EQUAL_INT(SVC_EBAD, svc_parse_line("1,2,3.4,32768,6,", &r)); /* outside domain */
    TEST_ASSERT_EQUAL_INT(SVC_EBAD, svc_parse_line("1,2,3.4,70000,6,", &r));
    TEST_ASSERT_EQUAL_INT(SVC_EBAD, svc_parse_line("1,2,3276.8,4,5,", &r));   /* temp +bound */
    TEST_ASSERT_EQUAL_INT(SVC_EBAD, svc_parse_line("1,2,-3276.9,4,5,", &r));  /* temp -bound */
    TEST_ASSERT_EQUAL_INT(SVC_EBAD, svc_parse_line("1,2,4000.0,5,6,", &r));   /* temp range */
}

/* ---------- 3. golden excerpt ---------- */

/* Deterministic 1 s service excerpt: 512 window records, seq = ts/2 + 1,
 * temperature ramp 36.8..37.9 cycling every 12 windows, ch0 = 41 +
 * ts/512, flags pattern A on ts%60==0, O+L on ts%512==0. No status burst
 * (first burst is ms 5000) — burst accounting lives in t_svc_burst.
 * THE single source of truth for t_svc_log_ref.csv. */
static void run_excerpt(svc_t *svc, uint32_t *n_out)
{
    svc_rec_t r;
    uint32_t ts, n = 0;

    svc_init(svc, sink, 0);
    for (ts = 0; ts <= 1022u; ts += 2u) {
        r.ts_ms = ts;
        r.seq = ts / 2u + 1u;
        r.t_c10 = (int16_t)(368 + (ts / 2u) % 12u);
        r.ch0_mmhg = (uint16_t)(41u + ts / 512u);
        r.ch1_mmhg = (uint16_t)(r.ch0_mmhg + 1u);
        r.flags = 0;
        if (ts % 60u == 0u) {
            r.flags |= SVC_F_A;
        }
        if (ts % 512u == 0u) {
            r.flags |= (uint8_t)(SVC_F_O | SVC_F_L);
        }
        r.cls = SVC_WIN;
        TEST_ASSERT_EQUAL_INT(SVC_OK, svc_ingest(svc, &r));
        svc_ms(svc, ts);
        n++;
    }
    svc_ms(svc, 1023u);
    svc_flush(svc);
    *n_out = n;
}

static void test_excerpt_matches_golden_file(void)
{
    svc_t svc;
    uint32_t n;
    FILE *f;
    long sz;
    char *ref;

    run_excerpt(&svc, &n);
    TEST_ASSERT_EQUAL_UINT32(512u, n);
    TEST_ASSERT_EQUAL_UINT32(0u, g_overflow);

    f = fopen(REF_FILE, "rb");
    TEST_ASSERT_NOT_NULL_MESSAGE(f,
        "golden excerpt missing — regenerate from a VERIFIED golden: " REF_FILE);
    fseek(f, 0, SEEK_END);
    sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    TEST_ASSERT_EQUAL_INT_MESSAGE((int)g_len, (int)sz, "excerpt length drift");
    ref = (char *)malloc((size_t)sz + 1u);
    TEST_ASSERT_NOT_NULL(ref);
    TEST_ASSERT_EQUAL_INT((int)sz, (int)fread(ref, 1, (size_t)sz, f));
    ref[sz] = '\0';
    fclose(f);

    if (memcmp(ref, g_buf, g_len) != 0) {
        long i;
        for (i = 0; i < sz && ref[i] == g_buf[i]; i++) { }
        fprintf(stderr,
                "\nexcerpt byte-diff at %ld:\n  ref: %.60s\n  got: %.60s\n",
                i, ref + i, g_buf + i);
        free(ref);
        TEST_FAIL_MESSAGE("log bytes differ from golden (intentional? "
                          "regenerate ref via --regen after review)");
    }
    free(ref);
}

/* Tie the ref file to the payload format contract — every string below is
 * hand-derived from run_excerpt() formulas, NOT copied from the ref file:
 *   ts=0:    seq 1,   (0)%12=0 -> 36.8, ch 41, flags A(0%60) + O,L(0%512)
 *   ts=2:    seq 2,   1%12=1  -> 36.9, ch 41, none
 *   ts=60:   seq 31,  30%12=6 -> 37.4, ch 41, A
 *   ts=512:  seq 257, 256%12=4 -> 37.2, ch 42, O,L
 *   ts=1022: seq 512, 511%12=7 -> 37.5, ch 42, none (last line)
 */
static void test_excerpt_spot_checks(void)
{
    svc_t svc;
    uint32_t n;

    run_excerpt(&svc, &n);
    TEST_ASSERT_NOT_NULL(strstr(g_buf, "0,1,36.8,41,42,OAL\n"));
    TEST_ASSERT_NOT_NULL(strstr(g_buf, "\n2,2,36.9,41,42,\n"));
    TEST_ASSERT_NOT_NULL(strstr(g_buf, "\n60,31,37.4,41,42,A\n"));
    TEST_ASSERT_NOT_NULL(strstr(g_buf, "\n512,257,37.2,42,43,OL\n"));
    TEST_ASSERT_NOT_NULL(strstr(g_buf, "\n1022,512,37.5,42,43,\n"));
}

/* ---------- 4. exhaustive type-domain contract ----------
 * Channels: EVERY value of the Table 5-1 domain (uint16 0..32767).
 * Temperature: EVERY int16 value (all 65536 — even v -> +v, odd v ->
 * v-32768), so both signed halves are covered. Each through
 * format -> parse -> re-format: value must survive and the line must be
 * canonical-stable (byte-identical re-format). This is the type-change
 * guarantee — no sign asymmetry at the bounds, no silent wrap. */
static void test_channel_temp_exhaustive_roundtrip(void)
{
    uint32_t bad = 0;
    long v;

    for (v = 0; v <= 32767L; v++) {
        svc_rec_t r, back;
        char l1[80], l2[80];
        int n;

        r.ts_ms = 1u + (uint32_t)v;
        r.seq = 1u;
        r.t_c10 = (int16_t)((v & 1L) ? (v - 32768L) : v);
        r.ch0_mmhg = (uint16_t)v;
        r.ch1_mmhg = (uint16_t)(32767L - v);
        r.flags = (uint8_t)(v & 15L);
        r.cls = SVC_WIN;

        n = svc_format_win(&r, l1, (int)sizeof l1);
        if (n < 0) { bad++; continue; }
        if (svc_parse_line(l1, &back) != SVC_OK) { bad++; continue; }
        if (back.ts_ms != r.ts_ms || back.ch0_mmhg != r.ch0_mmhg ||
            back.ch1_mmhg != r.ch1_mmhg || back.t_c10 != r.t_c10 ||
            back.flags != r.flags) {
            bad++;
        }
        if (svc_format_win(&back, l2, (int)sizeof l2) != n ||
            strcmp(l1, l2) != 0) {
            bad++;                    /* non-canonical encoding */
        }
    }
    TEST_ASSERT_EQUAL_UINT32_MESSAGE(0u, bad,
        "domain values failed format<->parse round-trip (see test header)");
}

int main(int argc, char **argv)
{
    if (argc > 1 && strcmp(argv[1], "--regen") == 0) {
        svc_t svc;
        uint32_t n;
        FILE *f;
        run_excerpt(&svc, &n);
        f = fopen(REF_FILE, "wb");
        if (f == 0) {
            fprintf(stderr, "cannot write %s\n", REF_FILE);
            return 1;
        }
        fwrite(g_buf, 1, g_len, f);
        fclose(f);
        printf("regen %s: %u lines, %u bytes\n", REF_FILE, n, g_len);
        return (g_overflow != 0u);
    }

    UNITY_BEGIN();
    RUN_TEST(test_format_exact);
    RUN_TEST(test_parse_exact);
    RUN_TEST(test_excerpt_matches_golden_file);
    RUN_TEST(test_excerpt_spot_checks);
    RUN_TEST(test_channel_temp_exhaustive_roundtrip);
    return UNITY_END();
}
