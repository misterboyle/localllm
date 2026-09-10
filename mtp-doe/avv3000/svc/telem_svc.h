/* telem_svc.h — AVV-3000 pump telemetry service contract (rung r_svc).
 *
 * Role (topology, plan "System topology"): the base-controller ARM Linux
 * userspace service. Consumes telemetry records from the telemetry FIFO
 * (plumbed by the fabric/MicroBlaze side), maintains a bounded ring, and
 * appends the security-relevant window stream to a CSV log on the
 * telemetry console. This is the reference design named by datasheet §10;
 * the field artifact payload/field/uart.log is an excerpt of this console.
 *
 * WIRE/LOG LINE FORMAT (canonical, byte-identical to uart.log):
 *   ts_ms,seq,t_c,ch0_mmhg,ch1_mmhg,flags
 *   - ts_ms:  int ms; seq: 1-based SDR window sequence (seq = ts/2 + 1)
 *   - ch0/ch1: uint gauge pressure, domain 0..32767 (datasheet Table 5-1:
 *             0–3000 mmHg -> 0..32767). The parser enforces the contract:
 *             negative values and > 32767 are REJECTED as spec violations
 *             (never clamped, never wrapped). (Temperatures, by contrast,
 *             ARE signed — Table 5-1 thermistors run −40..+125 °C.)
 *   - t_c:    degrees C, one decimal, produced WITHOUT fp printf (t_c10
 *             fixed-point: 36.8 -> "36.8", -5 -> "-0.5")
 *   - flags:  optional letters "O" "A" "H" "L", emitted in that fixed
 *             order when set (bit0 O=occl, bit1 A=alarm, bit2 H=htemp,
 *             bit3 L=ltemp). Empty column is legal (trailing comma).
 *   Service-emitted comment lines on the same console start with "#".
 *
 * STATUS BURST (field model, payload/field/gen_uart.py docstring): the
 * service emits a status burst every 5000 ms — 100 ms train at 2 kHz
 * (2 records/ms, 200 per burst) — on the SAME ring and console writer as
 * the 2 ms window stream (500 lines/s).
 *
 * RING ADMISSION POLICY (THE B2 CONTRACT — the seeded bug inverts this):
 *   The ring is shared and bounded: SVC_RING_CAP = 64 records, FIFO order.
 *   The console writer drains at most SVC_DRAIN_PER_MS records per ms.
 *   1. WINDOW records are the security stream (pressure/alarms). They
 *      are admitted whenever count < cap. A status burst may NEVER cause
 *      a window record to be lost.
 *   2. STATUS records are admissible only while strictly more than half
 *      the ring (free > SVC_STATUS_GUARD = cap/2) is free: the back
 *      half of the ring is reserved for the window stream. Otherwise the
 *      arriving STATUS record is REJECTED (st_rejected++).
 *   3. Only a writer stall can fill the ring with window data. A WINDOW
 *      arriving at count == cap evicts the OLDEST record:
 *        oldest was STATUS -> st_evicted++  (acceptable: status is cheap)
 *        oldest was WINDOW -> win_overwrite++ (POLICY VIOLATION — real
 *                              security data lost; must be 0 for the
 *                              entire field profile).
 *   Field unit (uart.log): recurring 12..19-line window gaps, EXCLUSIVELY
 *   inside 2 Hz status-burst windows — the pre-fix service admitted
 *   status to the full ring, so burst onsets found it full and window
 *   records were dropped. The watchdog-histogram.txt correlation is the
 *   same stall. t_svc_burst pins the corrected policy; B2 seeds land on
 *   items 2/3 (S6 owns the instances).
 *
 * Determinism: no clocks, no threads, no floats, no malloc. The caller
 * drives time via svc_ms(now_ms) (non-decreasing) and feeds records with
 * svc_ingest(); output goes to a sink callback.
 */
#ifndef TELEM_SVC_H
#define TELEM_SVC_H

#include <stdint.h>

#define SVC_RING_CAP       64u
#define SVC_STATUS_GUARD   (SVC_RING_CAP / 2u)  /* free must EXCEED this for STATUS */
#define SVC_DRAIN_PER_MS   1u                   /* console writer, 1 line/ms (=1 kHz
 *   — the field footprint: the 500/s window stream alone drains fine, a
 *   burst window's 2.5 kHz total demand does NOT; unused budget NEVER
 *   accrues while the ring is empty (no catch-up super-drains) */
#define SVC_BURST_PERIOD   5000u                /* ms between bursts */
#define SVC_BURST_MS       100u                 /* burst duration */
#define SVC_BURST_PER_MS   2u                   /* status records/ms inside burst */

/* record class */
#define SVC_WIN     0u
#define SVC_STATUS  1u

/* flags bits */
#define SVC_F_O  0x01u
#define SVC_F_A  0x02u
#define SVC_F_H  0x04u
#define SVC_F_L  0x08u

/* svc_ingest / svc_parse_line return codes (negative = not enqueued) */
#define SVC_OK        0   /* enqueued                                   */
#define SVC_ST_DROP  (-1) /* STATUS rejected by admission guard         */
#define SVC_IGNORE  (-2)  /* comment/blank line (svc_parse_line only)   */
#define SVC_EBAD    (-3)  /* malformed line                             */

typedef struct {
    uint32_t ts_ms;
    uint32_t seq;        /* SVC_WIN: window seq; SVC_STATUS: last drained seq */
    int16_t  t_c10;      /* deg C * 10, SIGNED per Table 5-1 thermistors
                            (unused by STATUS)                          */
    uint16_t ch0_mmhg;   /* 0..32767, Table 5-1 domain (unused by STATUS) */
    uint16_t ch1_mmhg;
    uint8_t  flags;      /* SVC_F_* bits; STATUS: 0                    */
    uint8_t  cls;        /* SVC_WIN | SVC_STATUS                       */
} svc_rec_t;

typedef struct {
    uint32_t win_in;        /* records accepted (svc_ingest OK, SVC_WIN)     */
    uint32_t st_in;         /* status records accepted (incl. burst-generated) */
    uint32_t st_rejected;   /* STATUS rejected by the guard (policy working) */
    uint32_t st_evicted;    /* full-ring eviction hit STATUS (tolerable)     */
    uint32_t win_overwrite; /* full-ring eviction hit WINDOW — must stay 0   */
    uint32_t logged;        /* WINDOW lines written to the sink              */
    uint32_t st_logged;     /* STATUS comment lines written to the sink      */
} svc_stats_t;

/* console writer: called once per output line, line NUL-terminated,
 * no trailing newline (sink adds it). ctx opaque to svc. */
typedef void (*svc_sink_fn)(void *ctx, const char *line);

typedef struct {
    svc_rec_t   ring[SVC_RING_CAP];
    uint32_t    head;          /* oldest index                           */
    uint32_t    count;
    uint32_t    budget;        /* drain credits carried across ms        */
    uint32_t    now_ms;        /* last handled time (monotonicity check) */
    uint32_t    last_win_seq;  /* most recently LOGGED window seq (0 = none) */
    svc_sink_fn sink;
    void       *ctx;
    svc_stats_t s;
} svc_t;

void svc_init(svc_t *svc, svc_sink_fn sink, void *ctx);

/* Enqueue a record (svc_ms must have been called for the current time
 * first or the clock untouched: records only drain inside svc_ms).
 * Returns SVC_OK or SVC_ST_DROP per the admission policy above.
 * SVC_WIN at count == cap evicts the oldest and returns SVC_OK. */
int svc_ingest(svc_t *svc, const svc_rec_t *rec);

/* Advance the model clock (non-decreasing; steps > 1 ms handled).
 * Per ms: generate burst STATUS records (2/ms while inside a burst),
 * then drain budget records FIFO to the sink. */
void svc_ms(svc_t *svc, uint32_t now_ms);

/* Final flush: drain everything regardless of budget (EOF path). */
void svc_flush(svc_t *svc);

void svc_stats(const svc_t *svc, svc_stats_t *out);

/* Parse one wire line into a SVC_WIN record; '#' or blank -> SVC_IGNORE.
 * Flags letters accepted in any order, normalized by the bit map.
 * Returns SVC_OK | SVC_IGNORE | SVC_EBAD. */
int svc_parse_line(const char *line, svc_rec_t *out);

/* Format a SVC_WIN record to its canonical wire/log line (no newline).
 * Returns length. STATUS records are formatted by the svc internals as
 * "#st,<ts>,<last_win_seq>". */
int svc_format_win(const svc_rec_t *rec, char *buf, int cap);

#endif /* TELEM_SVC_H */
