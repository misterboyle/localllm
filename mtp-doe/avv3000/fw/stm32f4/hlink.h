/* hlink.h — AVV-3000 HEAD-LINK protocol (STM32 pump-head side)
 *
 * Framed UART link between the pump head (STM32F4 safelight) and the base
 * controller (Zynq-style SoC: ARM Linux svc + fabric + MicroBlaze). The
 * base's MicroBlaze is the peer; fw/mb/hlink_mb_port.c is the thin MB-side
 * port for the same module.
 *
 * Naming trap (three serials on this system — S7 README must never conflate
 * them): HEAD-LINK (this module, control) vs TELEMETRY CONSOLE (where
 * payload/field/uart.log comes from) vs the QC-panel link.
 *
 * Frame format (all multi-byte payloads little-endian, Cortex-M native):
 *
 *   off  size  field
 *   0    1     SOF   0xA5
 *   1    1     LEN   payload octet count, 0..HL_MAX_PAYLOAD
 *   2    1     TYPE  HL_T_*
 *   3    1     SEQ   sender counter, +1 per sent frame, wraps 255->0
 *   4    LEN   PAYLOAD
 *   4+LEN 1   CRC8  poly 0x07 init 0x00 (CRC-8/SMBUS) over bytes 0..4+LEN-1
 *
 * SOF is inside the CRC (a corrupted SOF that still decodes as a frame is
 * caught); there is no escape mechanism, so a corrupt byte equal to SOF
 * inside a frame costs that frame — the decoder resynchronises on the next
 * SOF (recovery: every frame boundary is self-locating once SOF is found;
 * after a truncation, recovery takes at most one further frame).
 *
 * Frame types (directions per the system topology):
 *   HL_T_HB        head<->base  heartbeat, LEN=0, bidirectional
 *   HL_T_STATUS    head->base   telemetry snapshot, LEN=HL_STATUS_LEN
 *   HL_T_VER_REQ   base->head   LEN=0
 *   HL_T_VER_RSP   head->base   payload = version string bytes
 *   HL_T_CTRL      base->head   payload[0] = HL_CMD_*
 *   HL_T_CTRL_ACK  head->base   payload = [opcode, status]
 *
 * Heartbeat state machine: a sent HB every hb_period_ms; ANY correctly
 * decoded inbound frame refreshes liveness. No refresh for hb_timeout_ms
 * (>= 3 missed peer heartbeats) -> link_up = 0. The safelight APP reads
 * hlink_link_up() and does the amber + head-lost telemetry flag
 * themselves; hlink never touches the ADS-8851 driver.
 *
 * Time model: no wall clock anywhere (fixture-deterministic). The caller
 * supplies a monotonic millisecond tick to hlink_poll() — systick-based on
 * the head, fake-peer-supplied in tests. hlink_poll() also drains RX and
 * services the tx path; call it at least every hb_period_ms.
 */
#ifndef HLINK_H
#define HLINK_H

#include <stdint.h>

/* ---- wire constants ---- */
#define HL_SOF          0xA5u
#define HL_HDR_LEN      4u                  /* SOF LEN TYPE SEQ            */
#define HL_MAX_PAYLOAD  32u
#define HL_MAX_FRAME    (HL_HDR_LEN + HL_MAX_PAYLOAD + 1u) /* 37           */

/* frame types */
#define HL_T_HB         0x01u
#define HL_T_STATUS     0x02u
#define HL_T_VER_REQ    0x03u
#define HL_T_VER_RSP    0x04u
#define HL_T_CTRL       0x05u
#define HL_T_CTRL_ACK   0x06u

/* HL_T_CTRL opcodes (AVA-QC wave recipe push = the pump-domain CCM analogue) */
#define HL_CMD_WAVE_RECIPE 0x10u /* [cmd, profile, hold_ms, step_lsb, invert] */
#define HL_CMD_WAVE_LOOP   0x11u /* [cmd, enable]  wave loop run/stop        */
#define HL_CMD_SELFTEST    0x12u /* [cmd, mode]                              */
#define HL_CMD_TWEAK       0x13u /* [cmd, sel, lo, hi]  OCCL_TH / I_MOT gain */

/* STATUS payload (HL_STATUS_LEN = 10, little-endian):
 *   [0..3]  win_seq        window sequence (STAT.SEQ-derived, §6.9)
 *   [4]     err_cnt        STAT [7:4] (RC mirror of ERR_STAT, §6.16)
 *   [5..7]  w_violations   ERR_STAT [31:8] W_VIOLATIONS, 24-bit (ES-2.1.3)
 *   [8]     alarm          ALARM_STATUS [7:0] bitmap (§6.11)
 *   [9]     wave_st        [2:0]=WAVE_STATUS RUN/CYCLE/DONE, [4:3]=profile
 */
#define HL_STATUS_LEN   10u

/* heartbeat defaults: period 50 ms, timeout 150 ms (3 missed) */
#define HL_HB_PERIOD_MS   50u
#define HL_HB_TIMEOUT_MS  150u

/* api results */
#define HL_OK           0
#define HL_ERR_ARG      (-1)

typedef struct {
    uint32_t hb_period_ms;    /* HB transmit period              */
    uint32_t hb_timeout_ms;   /* peer-liveness timeout           */
} hlink_cfg_t;

typedef struct {
    uint8_t len;                              /* payload octets     */
    uint8_t type;                             /* HL_T_*             */
    uint8_t seq;                              /* sender SEQ         */
    uint8_t payload[HL_MAX_PAYLOAD];
} hlink_frame_t;

typedef struct {
    uint32_t rx_frames;       /* CRC-valid frames decoded                     */
    uint32_t rx_crc_err;      /* frame attempts failing CRC (dropped)         */
    uint32_t rx_len_err;      /* LEN byte > HL_MAX_PAYLOAD (dropped, resync)  */
    uint32_t rx_seq_gap;      /* inbound SEQ != last+1 (wrap-aware)           */
    uint32_t rx_overflow;     /* decoded fine, delivery queue full (dropped)  */
    uint32_t tx_frames;       /* frames handed to the port                    */
    uint32_t link_lost;       /* UP -> LOST transitions                        */
} hlink_stats_t;

typedef struct {
    uint32_t win_seq;
    uint32_t w_violations;    /* low 24 bits used */
    uint8_t  err_cnt;
    uint8_t  alarm;
    uint8_t  wave_st;
} hlink_status_t;

/* ---- lifecycle ---- */
/* cfg NULL -> HL_HB_* defaults. Clears stats + both queues. */
void hlink_init(const hlink_cfg_t *cfg);

/* ---- app->peer interface ---- */
/* Encodes and transmits immediately via hlink_hw_tx() (chunked). LEN=0 is
 * legal with payload == 0. Returns HL_OK / HL_ERR_ARG. The port must not
 * backpressure (UART FIFO/DMA contract — hal-style, see hlink_port.h). */
int hlink_send(uint8_t type, const uint8_t *payload, uint32_t len);

/* Fills a HL_STATUS payload (payload layout above). Pure. */
void hlink_fill_status(uint8_t out[HL_STATUS_LEN], const hlink_status_t *st);

/* ---- periodic service (call from the app main loop) ---- */
/* Drains the RX port through the decoder (bounded), runs the heartbeat SM,
 * emits due HB frames. now_ms: caller's monotonic tick. */
void hlink_poll(uint32_t now_ms);

/* ---- views ---- */
int hlink_link_up(void);                      /* 1 = peer refreshed in time  */
int hlink_have_frame(hlink_frame_t *out);     /* 1 = frame dequeued          */
const hlink_stats_t *hlink_stats(void);
void hlink_stats_clear(void);

/* CRC-8/SMBUS (poly 0x07, init 0x00); exposed for the fake-peer shim and
 * the test's independent builder/oracle. */
uint8_t hlink_crc8(const uint8_t *p, uint32_t n);

#endif /* HLINK_H */
