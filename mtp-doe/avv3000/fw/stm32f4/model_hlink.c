/* model_hlink.c — fake-peer wire shim implementation (see model_hlink.h) */
#include "model_hlink.h"
#include "hlink.h"

#define MHL_RXQ   512u   /* peer -> head bytes outstanding  */
#define MHL_TXCAP 2048u  /* head -> peer capture, linear    */

static uint8_t  rxq[MHL_RXQ];
static uint32_t rxq_rd, rxq_cnt;

static uint8_t  txcap[MHL_TXCAP];
static uint32_t txcap_n, txcap_drop;

void mhl_reset(void)
{
    rxq_rd = rxq_cnt = 0u;
    txcap_n = 0u;
    txcap_drop = 0u;
}

uint32_t mhl_inject(const uint8_t *bytes, uint32_t n)
{
    uint32_t i = 0u;

    while (i < n && rxq_cnt < MHL_RXQ) {
        rxq[(rxq_rd + rxq_cnt) % MHL_RXQ] = bytes[i];
        rxq_cnt++;
        i++;
    }
    return i;
}

int mhl_inject_frame(uint8_t type, uint8_t seq, const uint8_t *pl,
                     uint32_t pl_len, int bad_crc)
{
    uint8_t f[HL_MAX_FRAME];
    uint32_t n = 0u;

    if (pl_len > HL_MAX_PAYLOAD || (pl_len > 0u && pl == 0))
        return -1;

    /* independent mini-encoder: if hlink's own encoder and this one ever
     * disagree, the tx wire-format test (parsed with a hand-rolled
     * decoder) is what catches it. */
    f[n++] = HL_SOF;
    f[n++] = (uint8_t)pl_len;
    f[n++] = type;
    f[n++] = seq;
    for (uint32_t i = 0u; i < pl_len; i++)
        f[n++] = pl[i];
    f[n] = hlink_crc8(f, n);
    if (bad_crc)
        f[n] ^= 0xFFu;
    n++;

    return (mhl_inject(f, n) == n) ? (int)n : -1;
}

uint32_t mhl_tx_len(void)
{
    return txcap_n;
}

uint32_t mhl_tx_copy(uint8_t *out, uint32_t cap)
{
    uint32_t n = (txcap_n < cap) ? txcap_n : cap;

    for (uint32_t i = 0u; i < n; i++)
        out[i] = txcap[i];
    return n;
}

void mhl_tx_clear(void)
{
    txcap_n = 0u;
    txcap_drop = 0u;
}

uint32_t mhl_tx_dropped(void)
{
    return txcap_drop;
}

/* ---- port-facing ---- */

int mhl_rx_take(uint8_t *byte)
{
    if (rxq_cnt == 0u)
        return 0;
    *byte = rxq[rxq_rd];
    rxq_rd = (rxq_rd + 1u) % MHL_RXQ;
    rxq_cnt--;
    return 1;
}

void mhl_tx_give(const uint8_t *bytes, uint32_t n)
{
    for (uint32_t i = 0u; i < n; i++) {
        if (txcap_n < MHL_TXCAP)
            txcap[txcap_n++] = bytes[i];
        else
            txcap_drop++;
    }
}
