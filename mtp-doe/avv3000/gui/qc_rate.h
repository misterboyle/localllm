/* qc_rate.h — QC panel rate conversion core (rung r_gui, unit-scale
 * contract: AN-8851-07 §5 "QC panel expectation (unit-scale rule)").
 *
 * The engine cadence domain is COUNTS/ms. The panel DISPLAYS counts/sec.
 * Display conversion: rate_s = rate_ms * 1000 — applied EXACTLY ONCE,
 * in this core. The field artifact "rate spike x1000 times nominal"
 * (payload/field/TICKET.md) is that conversion applied twice (or in the
 * wrong layer): that is seeded bug B5, and it is caught here because t_gui_unit
 * feeds synthetic count sequences and asserts the DISPLAYED value equals
 * the true counts/sec.
 *
 * Pure C, integer-only (no FP, no Qt) so the conversion has exactly one
 * home; the Qt widget is presentation-only.
 */
#ifndef QC_RATE_H
#define QC_RATE_H

#include <stdint.h>

typedef struct {
    uint32_t prev_count;   /* cumulative engine counts at prev sample    */
    uint32_t prev_ts;      /* sample clock, ms (monotonic input)         */
    uint8_t  have_prev;    /* 0 until one sample establishes a baseline  */
    int32_t  cps;          /* last computed rate; -1 = no data yet       */
} qc_rate_t;

void qc_rate_reset(qc_rate_t *st);

/* Feed one cumulative-count sample at monotonic ts_ms.
 * Returns the updated display value in counts/sec, or -1 until a second
 * sample exists. Contract:
 *   cps = (delta_counts * 1000) / delta_ms   — truncate once, clamp to
 *   INT32_MAX; delta_counts uses uint32 wrap arithmetic (count wrap OK).
 *   A non-increasing ts (delta_ms == 0) keeps the previous value.
 * The *1000 (counts/ms -> counts/s, AN-8851-07 §5) happens HERE, once. */
int32_t qc_rate_update(qc_rate_t *st, uint32_t ts_ms, uint32_t count);

#endif /* QC_RATE_H */
