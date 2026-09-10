/* hlink.c — AVV-3000 HEAD-LINK golden implementation (see hlink.h)
 *
 * Zero allocation, zero wall clock, single global instance (one UART per
 * head). Byte-driven decoder + small delivery queue: the RX drain in
 * hlink_poll() bounds time, so a flood cannot starve the app loop.
 *
 * Error containment model (exercised by tests/t_hlink.c):
 *   garbage bytes ......... dropped while hunting SOF
 *   LEN > MAX ............. frame attempt aborted, resync next SOF
 *   CRC mismatch .......... whole frame dropped (no partial delivery)
 *   truncated frame ....... dies at the NEXT real frame's CRC check;
 *                           recovery <= 1 further frame (no SOF escape —
 *                           a mid-payload 0xA5 can never be trusted)
 *   inbound queue full .... valid frame dropped, rx_overflow counted
 * None of these tear down the link: only heartbeat TIMEOUT moves the SM
 * out of UP (a peer that sends CRC garbage is still a peer that exists).
 */
#include "hlink.h"
#include "hlink_port.h"

#define RX_QUEUE_DEPTH 4u
#define RX_DRAIN_MAX   256u   /* bytes per poll: 2x headroom at 115k8/10 ms */
#define TX_CHUNK       8u

enum { RX_SOF, RX_LEN, RX_TYPE, RX_SEQ, RX_PAYLOAD, RX_CRC };

struct hlink_state {
    hlink_cfg_t cfg;
    uint32_t    now;            /* tick stashed by hlink_poll (rx refresh) */

    /* rx decoder */
    uint8_t  rxst;
    uint8_t  buf[HL_MAX_FRAME];
    uint32_t bidx;
    uint32_t wlen;              /* LEN of the frame being assembled        */

    /* delivery queue (FIFO ring) */
    hlink_frame_t q[RX_QUEUE_DEPTH];
    uint32_t qhead, qcnt;

    /* seq tracking */
    uint8_t tx_seq;
    uint8_t rx_seq;
    uint8_t rx_seq_valid;

    /* heartbeat SM */
    int      hb_started;
    uint32_t last_hb_ms;
    int      link_up;
    uint32_t last_rx_ms;

    hlink_stats_t stats;
};

static struct hlink_state st;

uint8_t hlink_crc8(const uint8_t *p, uint32_t n)
{
    uint8_t crc = 0u;

    for (uint32_t i = 0; i < n; i++) {
        crc ^= p[i];
        for (int b = 0; b < 8; b++) {
            uint8_t msb = (uint8_t)(crc & 0x80u);
            crc = (uint8_t)(crc << 1);
            if (msb != 0u)
                crc = (uint8_t)(crc ^ 0x07u);
        }
    }
    return crc;
}

/* Freestanding discipline (fw/qemu): no libc, so no struct = (T){0}
 * compounds either — the compiler turns them into memset calls that do
 * not exist in the qemu link. Explicit field code only. */
static void stats_zero(void)
{
    hlink_stats_t *s = &st.stats;

    s->rx_frames = 0u;
    s->rx_crc_err = 0u;
    s->rx_len_err = 0u;
    s->rx_seq_gap = 0u;
    s->rx_overflow = 0u;
    s->tx_frames = 0u;
    s->link_lost = 0u;
}

void hlink_init(const hlink_cfg_t *cfg)
{
    if (cfg != 0)
        st.cfg = *cfg;                        /* 2 words: copies inline     */
    else {
        st.cfg.hb_period_ms = HL_HB_PERIOD_MS;
        st.cfg.hb_timeout_ms = HL_HB_TIMEOUT_MS;
    }
    st.now = 0u;
    st.rxst = RX_SOF;
    st.bidx = 0u;
    st.wlen = 0u;
    st.qhead = st.qcnt = 0u;
    st.tx_seq = 0u;
    st.rx_seq = 0u;
    st.rx_seq_valid = 0u;
    st.hb_started = 0;
    st.last_hb_ms = 0u;
    st.link_up = 0;
    st.last_rx_ms = 0u;
    stats_zero();
}

int hlink_send(uint8_t type, const uint8_t *payload, uint32_t len)
{
    uint8_t f[HL_MAX_FRAME];
    uint32_t n = 0u;

    if (len > HL_MAX_PAYLOAD || (len > 0u && payload == 0))
        return HL_ERR_ARG;

    f[n++] = HL_SOF;
    f[n++] = (uint8_t)len;
    f[n++] = type;
    f[n++] = st.tx_seq++;                    /* wraps 255->0 naturally */
    for (uint32_t i = 0u; i < len; i++)
        f[n++] = payload[i];
    f[n] = hlink_crc8(f, n);                 /* SOF..payload inclusive   */
    n++;

    for (uint32_t i = 0u; i < n; i += TX_CHUNK) {
        uint32_t chunk = (n - i > TX_CHUNK) ? TX_CHUNK : (n - i);
        hlink_hw_tx(&f[i], chunk);
    }
    st.stats.tx_frames++;
    return HL_OK;
}

void hlink_fill_status(uint8_t out[HL_STATUS_LEN], const hlink_status_t *s)
{
    out[0] = (uint8_t)(s->win_seq & 0xFFu);
    out[1] = (uint8_t)((s->win_seq >> 8) & 0xFFu);
    out[2] = (uint8_t)((s->win_seq >> 16) & 0xFFu);
    out[3] = (uint8_t)((s->win_seq >> 24) & 0xFFu);
    out[4] = s->err_cnt;
    out[5] = (uint8_t)(s->w_violations & 0xFFu);
    out[6] = (uint8_t)((s->w_violations >> 8) & 0xFFu);
    out[7] = (uint8_t)((s->w_violations >> 16) & 0xFFu);
    out[8] = s->alarm;
    out[9] = s->wave_st;
}

/* ---- rx decoder ---- */

/* Frame fully received with good CRC: counts, liveness, seq bookkeeping,
 * delivery. Called with the whole frame in st.buf[0..bidx). */
static void rx_frame_ok(void)
{
    uint8_t type = st.buf[2];
    uint8_t seq = st.buf[3];
    uint8_t len = st.buf[1];

    st.stats.rx_frames++;

    /* ANY CRC-valid inbound frame proves the peer is alive (§topology:
     * bidirectional heartbeat — STATUS piggybacks liveness just like HB). */
    st.link_up = 1;
    st.last_rx_ms = st.now;

    if (st.rx_seq_valid && seq != (uint8_t)(st.rx_seq + 1u))
        st.stats.rx_seq_gap++;
    st.rx_seq = seq;
    st.rx_seq_valid = 1u;

    if (st.qcnt == RX_QUEUE_DEPTH) {         /* app not draining: shed, keep */
        st.stats.rx_overflow++;              /* liveness bookkeeping above   */
        return;
    }
    {
        hlink_frame_t *f = &st.q[(st.qhead + st.qcnt) % RX_QUEUE_DEPTH];
        f->type = type;
        f->seq = seq;
        f->len = len;
        for (uint32_t i = 0u; i < len; i++)
            f->payload[i] = st.buf[HL_HDR_LEN + i];
        st.qcnt++;
    }
}

static void rx_byte(uint8_t b)
{
    switch (st.rxst) {
    case RX_SOF:
        if (b == HL_SOF) {
            st.bidx = 0u;
            st.buf[st.bidx++] = b;
            st.rxst = RX_LEN;
        }
        break;
    case RX_LEN:
        if (b > (uint8_t)HL_MAX_PAYLOAD) {
            st.stats.rx_len_err++;           /* bogus frame; the corrupt     */
            st.rxst = RX_SOF;                /* byte is spent — scan next    */
        } else {
            st.wlen = b;
            st.buf[st.bidx++] = b;
            st.rxst = RX_TYPE;
        }
        break;
    case RX_TYPE:
        st.buf[st.bidx++] = b;
        st.rxst = RX_SEQ;
        break;
    case RX_SEQ:
        st.buf[st.bidx++] = b;
        st.rxst = (st.wlen != 0u) ? RX_PAYLOAD : RX_CRC;
        break;
    case RX_PAYLOAD:
        st.buf[st.bidx++] = b;
        if (st.bidx == HL_HDR_LEN + st.wlen)
            st.rxst = RX_CRC;
        break;
    case RX_CRC:
        st.buf[st.bidx++] = b;
        if (hlink_crc8(st.buf, st.bidx - 1u) == b)
            rx_frame_ok();
        else
            st.stats.rx_crc_err++;
        st.rxst = RX_SOF;                    /* resync: hunt the next SOF    */
        break;
    default:
        st.rxst = RX_SOF;
        break;
    }
}

/* ---- periodic service ---- */

void hlink_poll(uint32_t now_ms)
{
    st.now = now_ms;

    for (uint32_t i = 0u; i < RX_DRAIN_MAX; i++) {
        uint8_t b;
        if (!hlink_hw_rx(&b))
            break;
        rx_byte(b);
    }

    /* HB tx: due by period, and the very first poll starts the train.
     * Runs in LOST too — the head never stops knocking. */
    if (!st.hb_started ||
        (uint32_t)(now_ms - st.last_hb_ms) >= st.cfg.hb_period_ms) {
        st.hb_started = 1;
        st.last_hb_ms = now_ms;
        (void)hlink_send(HL_T_HB, 0, 0u);
    }

    /* liveness timeout: UP -> LOST (safelight app reads link_up==0 and
     * runs amber + head-lost telemetry flag; hlink stays out of that) */
    if (st.link_up &&
        (uint32_t)(now_ms - st.last_rx_ms) >= st.cfg.hb_timeout_ms) {
        st.link_up = 0;
        st.stats.link_lost++;
    }
}

/* ---- views ---- */

int hlink_link_up(void)
{
    return st.link_up;
}

int hlink_have_frame(hlink_frame_t *out)
{
    const hlink_frame_t *f;

    if (st.qcnt == 0u)
        return 0;
    /* field copy (no 35-byte struct assignment -> memcpy freestanding);
     * payload beyond f->len keeps stale bytes — readers bound by len      */
    f = &st.q[st.qhead];
    out->len = f->len;
    out->type = f->type;
    out->seq = f->seq;
    for (uint32_t i = 0u; i < f->len; i++)
        out->payload[i] = f->payload[i];
    st.qhead = (st.qhead + 1u) % RX_QUEUE_DEPTH;
    st.qcnt--;
    return 1;
}

const hlink_stats_t *hlink_stats(void)
{
    return &st.stats;
}

void hlink_stats_clear(void)
{
    stats_zero();
}
