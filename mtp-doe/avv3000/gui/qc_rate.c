/* qc_rate.c — golden unit-scale core. See qc_rate.h for the contract. */
#include "qc_rate.h"

void qc_rate_reset(qc_rate_t *st)
{
    st->prev_count = 0;
    st->prev_ts = 0;
    st->have_prev = 0;
    st->cps = -1;
}

int32_t qc_rate_update(qc_rate_t *st, uint32_t ts_ms, uint32_t count)
{
    uint32_t dt;
    uint32_t dc;
    uint64_t scaled;

    if (!st->have_prev) {
        st->prev_count = count;
        st->prev_ts = ts_ms;
        st->have_prev = 1;
        return st->cps;                 /* -1 until second sample */
    }
    dc = count - st->prev_count;        /* wrap-correct */
    if (ts_ms == st->prev_ts) {
        return st->cps;                 /* no interval: keep last */
    }
    /* monotonic clock assumed; a step backward just retimes */
    dt = ts_ms - st->prev_ts;
    scaled = ((uint64_t)dc * 1000ull) / (uint64_t)dt;   /* §5: once */
    st->prev_count = count;
    st->prev_ts = ts_ms;
    st->cps = (scaled > (uint64_t)INT32_MAX) ? INT32_MAX : (int32_t)scaled;
    return st->cps;
}
