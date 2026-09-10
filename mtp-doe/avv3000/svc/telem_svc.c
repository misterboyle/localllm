/* telem_svc.c — AVV-3000 telemetry service golden implementation.
 * Policy contract + B2 landing zones: see telem_svc.h header.
 *
 * Freestanding discipline (same as fw/, so the suite stays host+qemu-able
 * if we ever port the rung): no malloc, no FP, no printf-family %f, no
 * threads/clocks. Output only via the sink callback.
 */
#include "telem_svc.h"

#define SVC_LINE_MAX 80

void svc_init(svc_t *svc, svc_sink_fn sink, void *ctx)
{
    uint32_t i;
    for (i = 0; i < SVC_RING_CAP; i++) {
        svc->ring[i].ts_ms = 0;
        svc->ring[i].seq = 0;
        svc->ring[i].t_c10 = 0;
        svc->ring[i].ch0_mmhg = 0;
        svc->ring[i].ch1_mmhg = 0;
        svc->ring[i].flags = 0;
        svc->ring[i].cls = SVC_WIN;
    }
    svc->head = 0;
    svc->count = 0;
    svc->budget = 0;
    svc->now_ms = 0;
    svc->last_win_seq = 0;
    svc->sink = sink;
    svc->ctx = ctx;
    svc->s.win_in = 0;
    svc->s.st_in = 0;
    svc->s.st_rejected = 0;
    svc->s.st_evicted = 0;
    svc->s.win_overwrite = 0;
    svc->s.logged = 0;
    svc->s.st_logged = 0;
}

/* push at tail; caller guarantees room */
static void ring_push(svc_t *svc, const svc_rec_t *rec)
{
    uint32_t tail = (svc->head + svc->count) % SVC_RING_CAP;
    svc->ring[tail] = *rec;
    svc->count++;
}

/* drop oldest, returning its class */
static uint8_t ring_evict(svc_t *svc)
{
    uint8_t cls = svc->ring[svc->head].cls;
    svc->head = (svc->head + 1u) % SVC_RING_CAP;
    svc->count--;
    return cls;
}

int svc_ingest(svc_t *svc, const svc_rec_t *rec)
{
    if (rec->cls == SVC_WIN) {
        if (svc->count == SVC_RING_CAP) {
            /* policy item 3: evict oldest — only a writer stall gets here */
            if (ring_evict(svc) == SVC_WIN) {
                svc->s.win_overwrite++;   /* must stay 0 (field profile) */
            } else {
                svc->s.st_evicted++;
            }
        }
        ring_push(svc, rec);
        svc->s.win_in++;
        return SVC_OK;
    }

    /* STATUS: policy item 2 — window stream reserves the back half */
    if (SVC_RING_CAP - svc->count > SVC_STATUS_GUARD) {
        ring_push(svc, rec);
        svc->s.st_in++;
        return SVC_OK;
    }
    svc->s.st_rejected++;
    return SVC_ST_DROP;
}

static int fmt_u32(char *b, uint32_t v)
{
    char tmp[10];
    int n = 0, i;
    do {
        tmp[n++] = (char)('0' + (v % 10u));
        v /= 10u;
    } while (v);
    for (i = 0; i < n; i++) {
        b[i] = tmp[n - 1 - i];
    }
    return n;
}

static int fmt_i16(char *b, int16_t v)
{
    if (v < 0) {
        b[0] = '-';
        return 1 + fmt_u32(b + 1, (uint32_t)(-(int32_t)v));
    }
    return fmt_u32(b, (uint32_t)v);
}

/* fixed-point one-decimal temp, no FP: 36.8 / -0.5 / -12.3 */
static int fmt_temp(char *b, int16_t t_c10)
{
    uint32_t u = (t_c10 < 0) ? (uint32_t)(-(int32_t)t_c10) : (uint32_t)t_c10;
    int n = 0;
    if (t_c10 < 0) {
        b[n++] = '-';
    }
    n += fmt_u32(b + n, u / 10u);
    b[n++] = '.';
    b[n++] = (char)('0' + (u % 10u));
    return n;
}

int svc_format_win(const svc_rec_t *rec, char *buf, int cap)
{
    char tmp[SVC_LINE_MAX];
    char *p = tmp;
    int i;
    static const char fname[4] = { 'O', 'A', 'H', 'L' };

    p += fmt_u32(p, rec->ts_ms);
    *p++ = ',';
    p += fmt_u32(p, rec->seq);
    *p++ = ',';
    p += fmt_temp(p, rec->t_c10);
    *p++ = ',';
    p += fmt_i16(p, rec->ch0_mmhg);
    *p++ = ',';
    p += fmt_i16(p, rec->ch1_mmhg);
    *p++ = ',';
    for (i = 0; i < 4; i++) {
        if (rec->flags & (uint8_t)(1u << i)) {
            *p++ = fname[i];
        }
    }
    *p = '\0';

    i = (int)(p - tmp) + 1;
    if (i > cap) {
        return -1;
    }
    for (p = tmp; *p != '\0'; p++) {
        buf[p - tmp] = *p;
    }
    buf[i - 1] = '\0';
    return i - 1;
}

/* emit one drained record on the console */
static void emit(svc_t *svc, const svc_rec_t *rec)
{
    char line[SVC_LINE_MAX];
    int n;

    if (rec->cls == SVC_WIN) {
        n = svc_format_win(rec, line, (int)sizeof line);
        if (n >= 0 && svc->sink != 0) {
            svc->sink(svc->ctx, line);
        }
        svc->s.logged++;
        svc->last_win_seq = rec->seq;
    } else {
        /* status comment: #st,<ts>,<last_win_seq-at-generation>
         * worst case 4 + 10 + 1 + 10 + 1 = 26 < SVC_LINE_MAX */
        char *p = line;
        *p++ = '#';
        *p++ = 's';
        *p++ = 't';
        *p++ = ',';
        p += fmt_u32(p, rec->ts_ms);
        *p++ = ',';
        p += fmt_u32(p, rec->seq);
        *p = '\0';
        if (svc->sink != 0) {
            svc->sink(svc->ctx, line);
        }
        svc->s.st_logged++;
    }
}

static void burst_gen(svc_t *svc, uint32_t ms)
{
    svc_rec_t r;
    uint32_t k;
    for (k = 0; k < SVC_BURST_PER_MS; k++) {
        r.ts_ms = ms;
        r.seq = svc->last_win_seq;
        r.t_c10 = 0;
        r.ch0_mmhg = 0;
        r.ch1_mmhg = 0;
        r.flags = 0;
        r.cls = SVC_STATUS;
        (void)svc_ingest(svc, &r);
    }
}

static int in_burst(uint32_t ms)
{
    return ms >= SVC_BURST_PERIOD &&
           (ms % SVC_BURST_PERIOD) < SVC_BURST_MS;
}

void svc_ms(svc_t *svc, uint32_t now_ms)
{
    uint32_t ms;
    if (now_ms <= svc->now_ms) {
        return;                       /* non-decreasing clock; equal = no-op */
    }
    for (ms = svc->now_ms + 1u; ms <= now_ms; ms++) {
        if (in_burst(ms)) {
            burst_gen(svc, ms);
        }
        svc->budget += SVC_DRAIN_PER_MS;
        while (svc->count > 0u && svc->budget > 0u) {
            svc_rec_t r = svc->ring[svc->head];
            ring_evict(svc);
            emit(svc, &r);
            svc->budget--;
        }
        /* writer never banks capacity while idle (a real 1 kHz writer
         * cannot super-drain a burst with credits from quiet seconds) */
        if (svc->count == 0u) {
            svc->budget = 0u;
        }
    }
    svc->now_ms = now_ms;
}

void svc_flush(svc_t *svc)
{
    while (svc->count > 0u) {
        svc_rec_t r = svc->ring[svc->head];
        ring_evict(svc);
        emit(svc, &r);
    }
}

void svc_stats(const svc_t *svc, svc_stats_t *out)
{
    *out = svc->s;
}

/* ---------------- wire parser ---------------- */

static int digit(char c)
{
    return c >= '0' && c <= '9';
}

/* unsigned decimal field; on failure leaves 0 in *v */
static const char *parse_u32(const char *s, uint32_t *v, char term)
{
    uint32_t acc = 0;
    int any = 0;
    while (digit(*s)) {
        acc = acc * 10u + (uint32_t)(*s - '0');
        s++;
        any = 1;
    }
    if (!any || *s != term) {
        return 0;
    }
    *v = acc;
    return s + 1;
}

static const char *parse_i16(const char *s, int16_t *v, char term)
{
    int neg = 0;
    const char *t;
    uint32_t u;

    if (*s == '-') {
        neg = 1;
        s++;
    }
    t = parse_u32(s, &u, term);
    if (t == 0 || u > 32768u) {
        return 0;
    }
    *v = neg ? (int16_t)(-(int32_t)u) : (int16_t)u;
    return t;
}

/* t_c: signed, mandatory one decimal digit ("36.8", "-0.5") */
static const char *parse_temp(const char *s, int16_t *v)
{
    int neg = 0;
    uint32_t ip = 0, fp = 0;
    const char *t;

    if (*s == '-') {
        neg = 1;
        s++;
    }
    t = parse_u32(s, &ip, '.');
    if (t == 0 || !digit(t[0])) {
        return 0;
    }
    fp = (uint32_t)(*t - '0');
    t++;
    if (*t != ',') {
        return 0;
    }
    if (ip * 10u + fp > 32767u) {
        return 0;
    }
    *v = (int16_t)(ip * 10u + fp);
    if (neg) {
        *v = (int16_t)-*v;
    }
    return t + 1;
}

int svc_parse_line(const char *line, svc_rec_t *out)
{
    const char *s = line;
    uint32_t ts, seq;
    int16_t t_c10;
    int16_t ch0, ch1;
    uint8_t flags = 0;

    while (*s == ' ' || *s == '\t') {
        s++;
    }
    if (*s == '\0' || *s == '\n' || *s == '\r') {
        return SVC_IGNORE;
    }
    if (*s == '#') {
        return SVC_IGNORE;                    /* service comment lines */
    }

    s = parse_u32(s, &ts, ',');
    if (s == 0) return SVC_EBAD;
    s = parse_u32(s, &seq, ',');
    if (s == 0) return SVC_EBAD;
    s = parse_temp(s, &t_c10);
    if (s == 0) return SVC_EBAD;
    s = parse_i16(s, &ch0, ',');
    if (s == 0) return SVC_EBAD;
    s = parse_i16(s, &ch1, ',');
    if (s == 0) return SVC_EBAD;

    while (*s != '\0' && *s != '\n' && *s != '\r') {
        if (*s == 'O') flags |= SVC_F_O;
        else if (*s == 'A') flags |= SVC_F_A;
        else if (*s == 'H') flags |= SVC_F_H;
        else if (*s == 'L') flags |= SVC_F_L;
        else return SVC_EBAD;
        s++;
    }

    out->ts_ms = ts;
    out->seq = seq;
    out->t_c10 = t_c10;
    out->ch0_mmhg = ch0;
    out->ch1_mmhg = ch1;
    out->flags = flags;
    out->cls = SVC_WIN;
    return SVC_OK;
}
