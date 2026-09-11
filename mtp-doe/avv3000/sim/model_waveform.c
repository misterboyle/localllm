/* model_waveform.c — ms-cadence mirror of rtl/waveform.sv (see header)
 *
 * State machine structure and every branch condition mirror waveform.sv;
 * tick arithmetic is 1 ms per wf_clock() instead of CLKS_PER_MS ticks.
 * Profile geometry is app note AN-8851-07 §3 (the authoritative tables),
 * identical to the localparams in waveform.sv.
 */
#include "model_waveform.h"
#include "ads8851_defs.h"

/* profile tables (AN-8851-07 §3 == waveform.sv localparams) */
#define PROF_A 1u
#define PROF_B 2u
#define PROF_C 3u

/* Geometry arbiter resolved pre-S6 (S5 flag 2): app note §3 is the
 * arbiter; waveform.sv now stores holdtop_ms in reg [10:0] (B's 1200 ms
 * fit; t_waveform covers the B dwell), so this mirror and the RTL agree. */
typedef struct { uint32_t up, dn, holdtop_ms; } profile_geo;

static profile_geo geo_of(uint32_t pf)
{
    switch (pf) {
    case PROF_A: { profile_geo g = { 10u, 10u, 0u    }; return g; }
    case PROF_B: { profile_geo g = { 8u,  8u,  1200u }; return g; }
    case PROF_C: { profile_geo g = { 14u, 0u,  0u    }; return g; }
    default:     { profile_geo g = { 0u,  0u,  0u    }; return g; }
    }
}

enum { ST_IDLE, ST_RAMP_UP, ST_HOLD_TOP, ST_RAMP_DN, ST_DONE_ST };

/* engine state */
static int      st;
static uint16_t wave_q;
static uint32_t hold_lat, step_lat;   /* applied cadence/arm values */
static uint32_t tick_cnt, step_cnt;   /* ms, engine-clock counts */
static int      sense;                /* 0 asc, 1 desc */
static int      iter_r;               /* CYCLE: one full cycle done */
static int      run_r, done_r;
static uint32_t applied_prof;         /* profile the ramp is armed with */

/* WAVE_CFG register-file mirror (§6.13 fields; the register fans out to
 * the engine ports exactly like the RTL register block) */
static uint32_t reg_prof, reg_hold, reg_step;
static int      reg_inv;

void wf_reset(void)
{
    st = ST_IDLE;
    wave_q = 0u;
    hold_lat = step_lat = 0u;
    tick_cnt = step_cnt = 0u;
    sense = 0;
    iter_r = run_r = done_r = 0;
    applied_prof = 0u;
    reg_prof = reg_hold = reg_step = 0u;
    reg_inv = 0;
}

/* waveform.sv: tick = tick_cnt >= hold_lat * CLKS_PER_MS - 1; hold 0
 * (never writable through the driver) never fires in RTL either. */
static int tick_ms(uint32_t len_ms)
{
    if (len_ms == 0u)
        return 0;
    return tick_cnt + 1u >= len_ms;   /* fires on the len_ms-th ms clock */
}

void wf_clock(int en)
{
    profile_geo g = geo_of(reg_prof);

    done_r = 0;                       /* default: DONE is a pulse (§6.15) */

    switch (st) {
    case ST_IDLE:
        if (en && reg_prof != 0u) {   /* arm: no tick_cnt reset (RTL parity;
                                       * tick_cnt is 0 here after reset or
                                       * a previous tick boundary) */
            hold_lat = reg_hold;
            step_lat = reg_step;
            sense = reg_inv;
            st = ST_RAMP_UP;
            run_r = 1;
            applied_prof = reg_prof;
        }
        break;

    case ST_RAMP_UP:
        if (tick_ms(hold_lat)) {
            tick_cnt = 0u;
            step_cnt++;
            if (sense)
                wave_q = (wave_q >= step_lat)
                             ? (uint16_t)(wave_q - step_lat) : 0u;
            else if (wave_q < (uint32_t)0xFFFFu - step_lat)
                wave_q = (uint16_t)(wave_q + step_lat);
            else
                wave_q = 0xFFFFu;

            if (step_cnt >= g.up) {
                if (g.holdtop_ms != 0u) {
                    st = ST_HOLD_TOP;
                } else if (g.dn != 0u) {
                    sense = !sense;
                    step_cnt = 0u;
                    st = ST_RAMP_DN;
                } else {              /* one-way profile (C): DONE sticky */
                    done_r = 1;
                    st = ST_DONE_ST;
                    run_r = 0;
                }
            }
        } else {
            tick_cnt++;
        }
        break;

    case ST_HOLD_TOP:                /* profile B top dwell (app note §3) */
        if (tick_ms(g.holdtop_ms)) {
            tick_cnt = 0u;
            sense = !sense;
            step_cnt = 0u;
            st = ST_RAMP_DN;
        } else {
            tick_cnt++;
        }
        break;

    case ST_RAMP_DN:
        if (tick_ms(hold_lat)) {
            tick_cnt = 0u;
            step_cnt++;
            if (sense)
                wave_q = (wave_q >= step_lat)
                             ? (uint16_t)(wave_q - step_lat) : 0u;
            else
                wave_q = (uint16_t)(wave_q + step_lat);

            if (step_cnt >= g.dn) {   /* cycle end */
                done_r = 1;
                iter_r = 1;
                sense = reg_inv;      /* re-arm direction */
                step_cnt = 0u;
                hold_lat = reg_hold;  /* ES-2.1.2: deferred HOLD_MS lands
                                       * at the next cycle start */
                st = ST_RAMP_UP;
            }
        } else {
            tick_cnt++;
        }
        break;

    default:                          /* ST_DONE_ST: sticky DONE */
        done_r = 1;
        run_r = 0;
        break;
    }
}

void wf_cfg_wr(uint32_t val, int en)
{
    uint32_t pf   = val & ADS8851_WAVE_PROFILE_M;
    uint32_t hold = (val & ADS8851_WAVE_HOLD_M) >> ADS8851_WAVE_HOLD_S;
    uint32_t step = (val & ADS8851_WAVE_STEP_M) >> ADS8851_WAVE_STEP_S;
    int      inv  = (val & ADS8851_WAVE_INVERT) != 0u;

    /* register file updates on every write (RTL register block) */
    reg_prof = pf;
    reg_hold = hold;
    reg_step = step;
    reg_inv  = inv;

    done_r = 0;                       /* waveform.sv cfg branch clears */

    if (pf == 0u) {                   /* stop */
        st = ST_IDLE;
        run_r = 0;
        iter_r = 0;
        step_cnt = 0u;
        applied_prof = 0u;
    } else if ((st == ST_IDLE || st == ST_DONE_ST) && en) {
        /* (re)start; wave_q NOT cleared — a restart ramps from the
         * current output (§6.13 note in waveform.sv) */
        hold_lat = hold;
        step_lat = step;
        tick_cnt = 0u;
        sense = inv;
        st = ST_RAMP_UP;
        run_r = 1;
        applied_prof = pf;
    } else {
        /* running: PROFILE/STEP/INVERT apply immediately; a HOLD_MS
         * change waits for the next cycle start (ES-2.1.2) */
        step_lat = step;
        sense = inv;
        if (pf != applied_prof)
            step_cnt = 0u;            /* geometry switch re-arms the ramp */
        applied_prof = pf;
    }
}

uint32_t wf_out(void)
{
    return wave_q;
}

uint32_t wf_status(void)
{
    return ((run_r ? ADS8851_WAVE_ST_RUN : 0u) |
            (iter_r ? ADS8851_WAVE_ST_CYCLE : 0u) |
            (done_r ? ADS8851_WAVE_ST_DONE : 0u));
}
