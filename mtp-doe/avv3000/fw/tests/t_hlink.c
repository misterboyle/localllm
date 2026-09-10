/* t_hlink.c — HEAD-LINK codec + heartbeat SM contract (hlink.h).
 * Rung r_fw_host (also runs on qemu cortex-m3 — same binary logic).
 *
 * Fake-peer pattern (same as model_ads8851 backs the driver tests):
 * model_hlink is the WIRE — tests inject peer bytes and capture head tx.
 * TX capture is parsed here with a HAND-WRITTEN decoder + crc (independent
 * oracle: hlink's encoder is never validated by hlink's own decoder).
 *
 * Coverage per plan §S3b: bad CRC drop + recovery, SOF slip/resync,
 * heartbeat timeout -> LOST (the safelight app's amber trigger), periodic
 * HB train, seq bookkeeping, flood/queue containment, STATUS payload.
 */
#include "unity.h"
#include "hlink.h"
#include "model_hlink.h"

void setUp(void)
{
    mhl_reset();
    hlink_init(0);
}

void tearDown(void) {}

/* ---- independent CRC-8 (poly 0x07, init 0x00) ---- */
static uint8_t crc8_local(const uint8_t *p, uint32_t n)
{
    uint8_t crc = 0u;

    for (uint32_t i = 0; i < n; i++) {
        crc ^= p[i];
        for (int b = 0; b < 8; b++)
            crc = (crc & 0x80u) ? (uint8_t)((crc << 1) ^ 0x07u)
                                : (uint8_t)(crc << 1);
    }
    return crc;
}

/* Walk the tx capture one frame at a time; 1 = frame, 0 = end, -1 = corrupt.
 * pl[HL_MAX_PAYLOAD] filled from the wire. */
static int px_next(const uint8_t *b, uint32_t n, uint32_t *at,
                   uint8_t *type, uint8_t *seq, uint8_t *pl, uint32_t *plen)
{
    if (*at >= n)
        return 0;
    if (b[*at] != HL_SOF)
        return -1;
    uint32_t len = b[*at + 1];
    if (len > HL_MAX_PAYLOAD || *at + 5u + len > n)
        return -1;
    if (crc8_local(&b[*at], 4u + len) != b[*at + 4u + len])
        return -1;
    *type = b[*at + 2];
    *seq = b[*at + 3];
    *plen = len;
    for (uint32_t i = 0; i < len; i++)
        pl[i] = b[*at + 4u + i];
    *at += 5u + len;
    return 1;
}

/* poll until the peer queue is drained: 1 poll moves <= RX_DRAIN_MAX
 * (256) bytes, which covers every injection below in one call. */
static void peer_pump(uint32_t now)
{
    hlink_poll(now);
}

/* ---- wire format (head -> peer) ---- */

static void test_crc8_matches_reference_vectors(void)
{
    /* CRC-8/SMBUS check value ("123456789") + the empty case. */
    TEST_ASSERT_EQUAL_HEX8_MESSAGE(0xF4u,
        hlink_crc8((const uint8_t *)"123456789", 9u), "crc8 check digit");
    TEST_ASSERT_EQUAL_HEX8_MESSAGE(0xC0u,
        hlink_crc8((const uint8_t *)"123", 3u), "crc8 short vector");
    TEST_ASSERT_EQUAL_HEX8(0x00u, hlink_crc8((const uint8_t *)"", 0u));
}

static void test_send_wire_format_is_hlink_h(void)
{
    hlink_status_t s = { 0x11223344u, 0x00AABBCCu, 0x05u, 0x15u, 0x0Bu };
    uint8_t pl[HL_STATUS_LEN];
    uint8_t cap[64], type, seq, fpl[HL_MAX_PAYLOAD];
    uint32_t plen, at = 0u, n;

    static const uint8_t want[HL_STATUS_LEN] = { 0x44, 0x33, 0x22, 0x11, 0x05,
                                                 0xCC, 0xBB, 0xAA, 0x15, 0x0B };
    hlink_fill_status(pl, &s);
    /* little-endian payload contract (hlink.h STATUS table) */
    TEST_ASSERT_EQUAL_HEX8_ARRAY(pl, want, HL_STATUS_LEN);

    TEST_ASSERT_EQUAL_INT(HL_OK, hlink_send(HL_T_STATUS, pl, HL_STATUS_LEN));
    peer_pump(5u); /* first poll also emits the opening HB */

    n = mhl_tx_copy(cap, sizeof(cap));
    TEST_ASSERT_EQUAL_UINT32(0u, mhl_tx_dropped());
    TEST_ASSERT_EQUAL_UINT32(15u + 5u, n); /* STATUS(4+10+1) + HB(4+0+1)  */

    /* frame 1: STATUS, SEQ starts at 0 */
    TEST_ASSERT_EQUAL_INT(1, px_next(cap, n, &at, &type, &seq, fpl, &plen));
    TEST_ASSERT_EQUAL_HEX8(HL_T_STATUS, type);
    TEST_ASSERT_EQUAL_HEX8(0u, seq);
    TEST_ASSERT_EQUAL_UINT32(HL_STATUS_LEN, plen);
    TEST_ASSERT_EQUAL_HEX8_ARRAY(fpl, pl, HL_STATUS_LEN);
    /* frame 2: heartbeat, LEN=0, seq increments per sent frame */
    TEST_ASSERT_EQUAL_INT(1, px_next(cap, n, &at, &type, &seq, fpl, &plen));
    TEST_ASSERT_EQUAL_HEX8(HL_T_HB, type);
    TEST_ASSERT_EQUAL_UINT32(0u, plen);
    TEST_ASSERT_EQUAL_HEX8(1u, seq);
    TEST_ASSERT_EQUAL_UINT32(n, at);
    TEST_ASSERT_EQUAL_UINT32(2u, hlink_stats()->tx_frames);
}

static void test_send_arg_boundary(void)
{
    uint8_t pl[HL_MAX_PAYLOAD];

    for (uint32_t i = 0; i < HL_MAX_PAYLOAD; i++)
        pl[i] = (uint8_t)i;

    TEST_ASSERT_EQUAL_INT_MESSAGE(HL_ERR_ARG,
        hlink_send(HL_T_STATUS, pl, HL_MAX_PAYLOAD + 1u), "len>max accepted");
    TEST_ASSERT_EQUAL_INT_MESSAGE(HL_ERR_ARG,
        hlink_send(HL_T_STATUS, (const uint8_t *)0, 1u), "payload-less nonzero len");
    TEST_ASSERT_EQUAL_INT(HL_OK, hlink_send(HL_T_STATUS, pl, HL_MAX_PAYLOAD));
    TEST_ASSERT_EQUAL_INT(HL_OK, hlink_send(HL_T_VER_REQ, 0, 0u));
    /* no poll => no HB traffic; exactly the 2 accepted frames */
    TEST_ASSERT_EQUAL_UINT32(2u, hlink_stats()->tx_frames);
    /* no HB (no poll): 1 max-size frame (4+32+1) + VER_REQ len0 (4+1) */
    TEST_ASSERT_EQUAL_UINT32(37u + 5u, mhl_tx_len());
}

/* ---- peer -> head decode ---- */

static void test_rx_frame_delivered(void)
{
    uint8_t pl[HL_STATUS_LEN] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
    hlink_frame_t f;

    TEST_ASSERT_EQUAL_INT(15, mhl_inject_frame(HL_T_STATUS, 7u, pl,
                                               sizeof(pl), 0));
    peer_pump(0u);

    TEST_ASSERT_EQUAL_INT(1, hlink_have_frame(&f));
    TEST_ASSERT_EQUAL_HEX8(HL_T_STATUS, f.type);
    TEST_ASSERT_EQUAL_HEX8(7u, f.seq);
    TEST_ASSERT_EQUAL_UINT8(HL_STATUS_LEN, f.len);
    TEST_ASSERT_EQUAL_HEX8_ARRAY(f.payload, pl, HL_STATUS_LEN);
    TEST_ASSERT_EQUAL_INT(0, hlink_have_frame(&f));
    TEST_ASSERT_EQUAL_UINT32(1u, hlink_stats()->rx_frames);
    TEST_ASSERT_EQUAL_UINT32(0u, hlink_stats()->rx_crc_err);
}

static void test_bad_crc_dropped_then_recovers(void)
{
    hlink_frame_t f;

    TEST_ASSERT(mhl_inject_frame(HL_T_STATUS, 3u, 0, 0u, 1) > 0);
    peer_pump(0u);
    TEST_ASSERT_EQUAL_INT_MESSAGE(0, hlink_have_frame(&f),
                                  "CRC-corrupt frame must not deliver");
    TEST_ASSERT_EQUAL_UINT32(1u, hlink_stats()->rx_crc_err);
    TEST_ASSERT_EQUAL_UINT32(0u, hlink_stats()->rx_frames);

    TEST_ASSERT(mhl_inject_frame(HL_T_STATUS, 4u, 0, 0u, 0) > 0);
    peer_pump(1u);
    TEST_ASSERT_EQUAL_INT(1, hlink_have_frame(&f));
    TEST_ASSERT_EQUAL_HEX8(4u, f.seq);
    TEST_ASSERT_EQUAL_UINT32(1u, hlink_stats()->rx_frames);
}

static void test_sof_slip_resync_pure_junk(void)
{
    const uint8_t junk[] = { 0x00, 0x11, 0x22 }; /* SOF-free noise */
    hlink_frame_t f;

    TEST_ASSERT_EQUAL_UINT32(sizeof(junk), mhl_inject(junk, sizeof(junk)));
    TEST_ASSERT(mhl_inject_frame(HL_T_HB, 20u, 0, 0u, 0) > 0);
    peer_pump(0u);

    TEST_ASSERT_EQUAL_INT(1, hlink_have_frame(&f));
    TEST_ASSERT_EQUAL_HEX8(20u, f.seq);
    TEST_ASSERT_EQUAL_UINT32(1u, hlink_stats()->rx_frames);
    TEST_ASSERT_EQUAL_UINT32(0u, hlink_stats()->rx_crc_err);
}

/* Decoy SOF (0xA5) inside the junk swallows 3 bytes of frame A: the
 * decoder must eat the CRC mismatch and still deliver frame B — bounded
 * recovery of one lost frame (documented in hlink.h). The CRC bytes 0x34
 * (A) / 0x33 (B) are the reference values computed for these exact
 * frames, keeping the trace deterministic. */
static void test_sof_slip_resync_decoy_sof(void)
{
    const uint8_t junk[] = { HL_SOF, 0x00 }; /* half frame: SOF + LEN=0 */
    hlink_frame_t f;

    TEST_ASSERT_EQUAL_UINT32(2u, mhl_inject(junk, sizeof(junk)));
    TEST_ASSERT(mhl_inject_frame(HL_T_HB, 30u, 0, 0u, 0) > 0); /* frame A */
    TEST_ASSERT(mhl_inject_frame(HL_T_HB, 31u, 0, 0u, 0) > 0); /* frame B */
    peer_pump(0u);

    TEST_ASSERT_EQUAL_UINT32(1u, hlink_stats()->rx_crc_err);    /* A died  */
    TEST_ASSERT_EQUAL_UINT32(1u, hlink_stats()->rx_frames);     /* B lived */
    TEST_ASSERT_EQUAL_INT(1, hlink_have_frame(&f));
    TEST_ASSERT_EQUAL_HEX8(31u, f.seq);
    TEST_ASSERT_EQUAL_INT(0, hlink_have_frame(&f));
    /* baseline seq set on the ONE delivery: no phantom gap */
    TEST_ASSERT_EQUAL_UINT32(0u, hlink_stats()->rx_seq_gap);
}

static void test_seq_gap_counted(void)
{
    TEST_ASSERT(mhl_inject_frame(HL_T_HB, 3u, 0, 0u, 0) > 0);
    peer_pump(0u);
    TEST_ASSERT(mhl_inject_frame(HL_T_HB, 5u, 0, 0u, 0) > 0); /* 4 lost    */
    peer_pump(1u);
    TEST_ASSERT(mhl_inject_frame(HL_T_HB, 6u, 0, 0u, 0) > 0); /* in order  */
    peer_pump(2u);

    TEST_ASSERT_EQUAL_UINT32(3u, hlink_stats()->rx_frames);
    TEST_ASSERT_EQUAL_UINT32_MESSAGE(1u, hlink_stats()->rx_seq_gap,
                                     "one skipped inbound SEQ must count");
}

static void test_seq_wrap_is_not_a_gap(void)
{
    const uint8_t seqs[] = { 254u, 255u, 0u, 1u };

    for (uint32_t i = 0; i < 4u; i++) {
        TEST_ASSERT(mhl_inject_frame(HL_T_HB, seqs[i], 0, 0u, 0) > 0);
        peer_pump(i);
    }
    TEST_ASSERT_EQUAL_UINT32(4u, hlink_stats()->rx_frames);
    TEST_ASSERT_EQUAL_UINT32_MESSAGE(0u, hlink_stats()->rx_seq_gap,
                                     "255->0 wrap is not a gap");
}

static void test_rx_queue_overflow_contains_flood(void)
{
    hlink_frame_t f;
    uint32_t got = 0u;

    /* 6 valid inbound frames vs. delivery queue depth 4 */
    for (uint32_t i = 0; i < 6u; i++) {
        TEST_ASSERT(mhl_inject_frame(HL_T_HB, (uint8_t)i, 0, 0u, 0) > 0);
    }
    peer_pump(0u);
    while (hlink_have_frame(&f))
        got++;

    TEST_ASSERT_EQUAL_UINT32(4u, got);
    TEST_ASSERT_EQUAL_UINT32(6u, hlink_stats()->rx_frames);
    TEST_ASSERT_EQUAL_UINT32_MESSAGE(2u, hlink_stats()->rx_overflow,
                                     "undeliverable frames must be counted");
    /* the dropped flood still proved liveness */
    TEST_ASSERT_EQUAL_INT(1, hlink_link_up());
}

/* ---- heartbeat state machine ---- */

static void test_hb_train_periodic_tx(void)
{
    uint8_t cap[64], type, seq_at, pl[HL_MAX_PAYLOAD];
    uint32_t plen, at = 0u, n, frames = 0u, expect_seq = 0u;

    /* 21 polls, 10 ms apart: HBs due at t = 0, 50, 100, 150, 200 */
    for (uint32_t t = 0; t <= 200u; t += 10u)
        peer_pump(t);

    n = mhl_tx_copy(cap, sizeof(cap));
    TEST_ASSERT_EQUAL_UINT32(5u * 5u, n); /* 5 x LEN-0 frames of 5 bytes   */
    while (1) {
        int r = px_next(cap, n, &at, &type, &seq_at, pl, &plen);
        if (r == 0)
            break;
        TEST_ASSERT_EQUAL_INT_MESSAGE(1, r, "corrupt tx stream");
        TEST_ASSERT_EQUAL_HEX8(HL_T_HB, type);
        TEST_ASSERT_EQUAL_UINT32(0u, plen);
        TEST_ASSERT_EQUAL_HEX8((uint8_t)expect_seq, seq_at);
        expect_seq++;
        frames++;
    }
    TEST_ASSERT_EQUAL_UINT32(5u, frames);
    TEST_ASSERT_EQUAL_UINT32(5u, hlink_stats()->tx_frames);
}

static void test_hb_timeout_lost_then_recover_up(void)
{
    const uint8_t junk[] = { 0x00 };
    uint32_t tx_at_up;

    peer_pump(0u);
    TEST_ASSERT_EQUAL_INT(0, hlink_link_up()); /* never heard from peer    */

    TEST_ASSERT(mhl_inject_frame(HL_T_HB, 100u, 0, 0u, 0) > 0);
    peer_pump(10u);
    TEST_ASSERT_EQUAL_INT_MESSAGE(1, hlink_link_up(), "HB rx must set UP");
    tx_at_up = hlink_stats()->tx_frames; /* 1: the opening HB only */
    TEST_ASSERT_EQUAL_UINT32(1u, tx_at_up);

    peer_pump(159u); /* 149 ms quiet: still UP (< hb_timeout_ms)     */
    TEST_ASSERT_EQUAL_INT(1, hlink_link_up());

    peer_pump(210u); /* 200 ms quiet >= 150 -> LOST (app runs amber)  */
    TEST_ASSERT_EQUAL_INT(0, hlink_link_up());
    TEST_ASSERT_EQUAL_UINT32(1u, hlink_stats()->link_lost);

    /* LOST must not stop the knocking (this timeline's HBs: 0/159/210): */
    TEST_ASSERT_TRUE_MESSAGE(hlink_stats()->tx_frames > tx_at_up,
                             "HB train stalled while LOST");

    mhl_inject(junk, sizeof(junk)); /* noise does NOT refresh ...          */
    peer_pump(211u);
    TEST_ASSERT_EQUAL_INT(0, hlink_link_up());

    TEST_ASSERT(mhl_inject_frame(HL_T_STATUS, 101u, 0, 0u, 0) > 0);
    peer_pump(212u);
    TEST_ASSERT_EQUAL_INT(1, hlink_link_up()); /* any valid frame = alive  */
    TEST_ASSERT_EQUAL_UINT32(1u, hlink_stats()->link_lost); /* no re-count */
}

int main(void)
{
    UNITY_BEGIN();
    RUN_TEST(test_crc8_matches_reference_vectors);
    RUN_TEST(test_send_wire_format_is_hlink_h);
    RUN_TEST(test_send_arg_boundary);
    RUN_TEST(test_rx_frame_delivered);
    RUN_TEST(test_bad_crc_dropped_then_recovers);
    RUN_TEST(test_sof_slip_resync_pure_junk);
    RUN_TEST(test_sof_slip_resync_decoy_sof);
    RUN_TEST(test_seq_gap_counted);
    RUN_TEST(test_seq_wrap_is_not_a_gap);
    RUN_TEST(test_rx_queue_overflow_contains_flood);
    RUN_TEST(test_hb_train_periodic_tx);
    RUN_TEST(test_hb_timeout_lost_then_recover_up);
    return UNITY_END();
}
