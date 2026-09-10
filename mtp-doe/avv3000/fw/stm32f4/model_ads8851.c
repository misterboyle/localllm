/* model_ads8851.c — behavioral model of the ADS-8851 digital core
 *
 * Host/qemu test shim for drv_ads8851.c. Mirrors the golden behavior of
 * rtl/ads8851_ctrl.sv (that RTL is the cycle-accurate reference, verified
 * by rtl/tb/t_bus + t_waveform):
 *
 *   §7.1     SDR window: free-runs while ADC_EN && !LPM; at close latch
 *            the window results, set DRDY, advance SEQ, reset the
 *            per-window STAT-read budget.
 *   ES-2.1.3 STAT reads are destructive: the FIRST read of a window
 *            clears DRDY; any further read in the SAME window returns
 *            DRDY=0 and sets ERR_STAT.STAT_VIOLATION + ERR_CNT +
 *            W_VIOLATIONS.
 *   §6.16    ERR_STAT is read-to-clear.
 *   §8.2     occlusion: window debounce (OCCL_DEB+1 consecutive windows),
 *            latch until ALARM_CLR.
 *
 * Documented deviations from the RTL (none affect tested contracts):
 *   D1  STAT reads are counted per §6.16 literally — "within one SDR
 *       window" — i.e. only while a window is ACTIVE. The RTL also
 *       counts reads made before the first window ever opened.
 *   D2  The read budget resets on the ADC_EN 0->1 transition as well as
 *       at every window close (RTL: close/reset only; combined with D1
 *       the observable behavior matches the docs).
 *   D3  The waveform engine is not cadence-modeled here: WAVE_STATUS
 *       reports RUN = WF_EN & ADC_EN & PROFILE!=0 (§6.15), CYCLE/DONE
 *       read 0 and WAVE_DATA reads 0. Engine cadence is exercised where
 *       it is golden — rtl/tb/t_waveform and sim/ (S5).
 *   D4  No AXI timing (commits are instant, §9.3 latencies modeled nowhere);
 *       bus-level behavior is r_rtl territory. Arbitration H-3 is modeled
 *       as "accesses never fail because of a commit" (what fw tests check).
 *   D5  CTRL.SWRST keeps test observations (write log / counters) while
 *       resetting device state, so init logging survives soft reset.
 *
 * Digital-core window results are deterministic (RTL lines "latch
 * deterministic window results"): CH0=42, CH1=43, CH2..7=0.
 */
#include "model_ads8851.h"
#include "ads8851_defs.h"

/* freestanding-friendly (this file is also compiled for cortex-m3 under
 * qemu, where there is no libc): zeroing without string.h */
static void mzero(void *p, unsigned n)
{
    char *q = (char *)p;
    while (n--)
        *q++ = 0;
}

#define MDL_LOG_CAP 128u

/* CH constants == rtl/ads8851_ctrl.sv (ch0_res = 42; data_regs latch
 * 42/43/0...). OCCL compare uses 64-LSB units (Table 8-3). */
#define MDL_CH0_CONST 42u
#define MDL_CH1_CONST 43u

typedef struct {
    /* register file (Table 6-1 reset values) */
    uint32_t ctrl, cfg, alarm_cfg, alarm_status, wave_cfg;
    uint32_t err_stat, irq_mask, th_rate, safety_gate;
    uint32_t data[ADS8851_NCH];
    /* sequencer */
    uint32_t win_cnt;
    uint8_t  seq;
    uint8_t  drdy;        /* STAT.DRDY flag / DRDY_N line state */
    uint16_t win_reads;   /* STAT reads in current window (capped) */
    uint8_t  prev_adc_en;
    /* occlusion (§8.2) */
    uint8_t  occ_cnt;     /* consecutive closed windows with OCCL_DET */
    uint8_t  latched;     /* ALARM_STATUS[5] sticky */
} mdl_state;

static mdl_state s;

/* observations */
static model_evt s_log[MDL_LOG_CAP];
static uint32_t s_log_n, s_log_drop;
static uint32_t s_stat_reads, s_stat_viol, s_reads_max, s_slverr;

static uint8_t writable(uint32_t off)
{
    /* RTL reg_wr_resp addr_ok set (§4.3 decode) */
    return off == ADS8851_CTRL || off == ADS8851_CFG ||
           off == ADS8851_ALARM_CFG || off == ADS8851_WAVE_CFG ||
           off == ADS8851_IRQ_MASK || off == ADS8851_TH_RATE ||
           off == ADS8851_SAFETY_GATE;
}

static void reset_regs(void)
{
    mzero(&s, sizeof(s));
    s.ctrl       = 0x00000000u; /* Table 6-1 */
    s.cfg        = 0x00000004u;
    s.alarm_cfg  = 0x00001E0Fu;
    s.irq_mask   = 0x0000001Fu;
    s.th_rate    = 0x00000168u;
}

void mdl_reset(void)
{
    reset_regs();
    mdl_obs_reset();
}

void mdl_obs_reset(void)
{
    mzero(s_log, sizeof(s_log));
    s_log_n = s_log_drop = 0;
    s_stat_reads = s_stat_viol = s_reads_max = s_slverr = 0;
}

static void log_evt(uint8_t wr, uint32_t off, uint32_t val)
{
    if (s_log_n < MDL_LOG_CAP) {
        s_log[s_log_n].wr = wr;
        s_log[s_log_n].off = (uint16_t)off;
        s_log[s_log_n].val = val;
        s_log_n++;
    } else {
        s_log_drop++; /* keep-first: init-sequence asserts stay readable */
    }
}

/* ---- window geometry (Table 7-1) ---- */
static uint8_t win_active(void)
{
    return (uint8_t)((s.ctrl & ADS8851_CTRL_ADC_EN) &&
                     !(s.ctrl & ADS8851_CTRL_LPM)); /* §7.3 */
}

uint32_t mdl_win_len(void)
{
    switch (s.cfg & 0x3u) {     /* CFG.SDR_WINDOW, Table 7-1 @ 10 MHz */
    case 0u: return 10000u;     /* 1 ms */
    case 1u: return 20000u;     /* 2 ms (init default) */
    case 2u: return 40000u;     /* 4 ms */
    default: return 80000u;     /* 8 ms */
    }
}

uint32_t mdl_ticks_until_close(void)
{
    if (!win_active())
        return 0u;
    return mdl_win_len() - s.win_cnt;
}

uint32_t mdl_seq(void) { return s.seq; }
int mdl_drdy_line(void) { return s.drdy; }

/* RTL ch0_res constant drives the occlusion compare level (§8.2) */
static uint8_t occldet_level(void)
{
    uint32_t th_unit = (s.alarm_cfg & 0xFu) * 64u; /* OCCL_TH x 64 LSB */
    return (uint8_t)((s.ctrl & ADS8851_CTRL_ADC_EN) &&
                     (MDL_CH0_CONST >= th_unit));
}

/* ---- window close: §7.1 step 4 + §8.2 debounce ---- */
static void window_close(void)
{
    uint8_t armed;

    /* RTL evaluated occ_armed from the pre-increment count (§8.2:
     * latch after OCCL_DEB+1 *consecutive* windows) */
    armed = (uint8_t)(s.occ_cnt > ((s.alarm_cfg >> 8) & 0xFu));

    s.win_cnt = 0u;
    s.seq = (uint8_t)(s.seq + 1u);
    s.drdy = 1u;
    s.win_reads = 0u;
    s.data[0] = MDL_CH0_CONST;
    s.data[1] = MDL_CH1_CONST;
    for (uint32_t i = 2u; i < ADS8851_NCH; i++)
        s.data[i] = 0u;

    if (occldet_level()) {
        if (s.occ_cnt < 0xFu)
            s.occ_cnt++;
    } else {
        s.occ_cnt = 0u;
    }
    if (armed && !(s.alarm_status & ADS8851_ALM_OCCL)) {
        s.alarm_status |= ADS8851_ALM_OCCL;
        s.latched = 1u;
    }
}

void mdl_tick(uint32_t n_aclk)
{
    while (n_aclk--) {
        uint8_t active = win_active();

        /* D2: budget resets as the first window opens */
        if (active && !s.prev_adc_en)
            s.win_reads = 0u;
        s.prev_adc_en = active;

        if (!active) {
            s.win_cnt = 0u;
            s.drdy = 0u;          /* §6.7 CTRL.ADC_EN / §7.3 LPM */
            continue;
        }
        if (s.win_cnt == mdl_win_len() - 1u)
            window_close();
        else
            s.win_cnt++;
    }
}

/* ---- ES-2.1.3 STAT bookkeeping ---- */
static void stat_read_accounting(void)
{
    if (!win_active())
        return; /* D1: "within one SDR window" (§6.16) */

    s_stat_reads++;
    if (s.win_reads == 0u) {
        s.win_reads = 1u;         /* first read: also clears DRDY */
        s.drdy = 0u;
    } else {
        if (s.win_reads < 0xFFFFu)
            s.win_reads++;        /* second+ read: hardware self-check */
        s_stat_viol++;
        s.err_stat |= ADS8851_ERR_STAT_VIOL;
        if ((s.err_stat & ADS8851_STAT_ERR_CNT) < 0xF0u)
            s.err_stat += 1u << ADS8851_ERR_CNT_SHIFT; /* saturate [7:4] */
        s.err_stat = (s.err_stat & ~(0xFFFFFFu << ADS8851_ERR_WVIOL_SHIFT)) |
                     ((((s.err_stat >> ADS8851_ERR_WVIOL_SHIFT) + 1u) &
                       0xFFFFFFu) << ADS8851_ERR_WVIOL_SHIFT);
    }
    if (s.win_reads > s_reads_max)
        s_reads_max = s.win_reads;
}

uint32_t mdl_rd(uint32_t off)
{
    uint32_t val = 0u;
    uint8_t ok = 1u;

    switch (off) {
    case ADS8851_CTRL:   val = s.ctrl; break;
    case ADS8851_CFG:    val = s.cfg; break;
    case ADS8851_STAT: {
        uint32_t busy = (uint32_t)(win_active() &&
                                   (s.win_cnt != mdl_win_len() - 1u));
        uint32_t bit2 = (uint32_t)(((s.alarm_status & ADS8851_ALM_OCCL) ||
                                    occldet_level()) ? 1u : 0u);
        val = ((uint32_t)s.seq << 8) |
              (s.err_stat & ADS8851_STAT_ERR_CNT) |
              (bit2 << 2) | (busy << 1) | s.drdy; /* §6.9 bit image */
        break;                                        /* snapshot, then… */
    }
    case ADS8851_ALARM_CFG:    val = s.alarm_cfg; break;
    case ADS8851_ALARM_STATUS: val = s.alarm_status |
                                   (s.latched ? ADS8851_ALM_LATCHED : 0u); break;
    case ADS8851_WAVE_CFG:  val = s.wave_cfg; break;
    case ADS8851_WAVE_DATA: val = 0u; break;                 /* D3 */
    case ADS8851_WAVE_STATUS:                                  /* §6.15, D3 */
        val = ((s.ctrl & ADS8851_CTRL_WF_EN) && (s.wave_cfg & 0x3u)) ?
              ADS8851_WAVE_ST_RUN : 0u;
        break;
    case ADS8851_ERR_STAT:
        val = s.err_stat;
        s.err_stat = 0u;           /* §6.16 read-to-clear */
        break;
    case ADS8851_IRQ_MASK: val = s.irq_mask; break;
    case ADS8851_TH_RATE:  val = s.th_rate; break;
    case ADS8851_SAFETY_GATE: val = s.safety_gate; break;
    default:
        if (off >= ADS8851_DATA_BASE &&
            off < ADS8851_DATA_BASE + 4u * ADS8851_NCH &&
            (off & 3u) == 0u)
            val = s.data[(off - ADS8851_DATA_BASE) / 4u];      /* §5 */
        else
            ok = 0u;               /* decode error → SLVERR, RDATA 0 (§4.3) */
        break;
    }

    if (off == ADS8851_STAT)
        stat_read_accounting();
    if (!ok) {
        s_slverr++;
        s.alarm_status |= ADS8851_ALM_COMMS; /* §6.11 "any SLVERR latches" */
        s.latched = 1u;
        val = 0u;
    }
    log_evt(0u, off, val);
    return val;
}

void mdl_wr(uint32_t off, uint32_t val)
{
    log_evt(1u, off, val);

    if (!writable(off)) {
        s_slverr++;                        /* §4.3: SLVERR, write discarded */
        s.alarm_status |= ADS8851_ALM_COMMS;
        s.latched = 1u;
        return;
    }

    if (off == ADS8851_CTRL) {
        if (val & ADS8851_CTRL_ALARM_CLR) { /* W1C fields clear state (§6.7) */
            s.alarm_status = 0u;
            s.latched = 0u;
            s.err_stat = 0u;
        }
        /* CTRL.SWRST: self-clearing soft reset to reset values (§6.7);
         * ES-2.1.1: ALARM_CFG returns to its reset value (indistinguishable
         * at the register level) — reference init re-writes it anyway. */
        if (val & ADS8851_CTRL_SWRST) {
            reset_regs();
            return; /* D5: observations survive the reset */
        }
        s.ctrl = val & ~(ADS8851_CTRL_ALARM_CLR | ADS8851_CTRL_SWRST);
        return;
    }

    switch (off) {
    case ADS8851_CFG:        s.cfg = val; break;
    case ADS8851_ALARM_CFG:  s.alarm_cfg = val; break;
    case ADS8851_WAVE_CFG:   s.wave_cfg = val; break;
    case ADS8851_IRQ_MASK:   s.irq_mask = val; break;
    case ADS8851_TH_RATE:    s.th_rate = val; break;
    case ADS8851_SAFETY_GATE: s.safety_gate = val; break;
    default: break;
    }
}

/* ---- observations ---- */
uint32_t mdl_log_count(void) { return s_log_n; }
uint32_t mdl_log_dropped(void) { return s_log_drop; }
model_evt mdl_log_get(uint32_t i) { return s_log[i]; }
uint32_t mdl_stat_reads(void) { return s_stat_reads; }
uint32_t mdl_stat_reads_inwin_max(void) { return s_reads_max; }
uint32_t mdl_stat_violations(void) { return s_stat_viol; }
uint32_t mdl_slverrs(void) { return s_slverr; }
