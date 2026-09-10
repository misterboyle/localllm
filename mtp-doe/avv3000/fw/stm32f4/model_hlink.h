/* model_hlink.h — fake-peer wire shim for hlink.c
 *
 * Test-only. Same shim-as-peer pattern as model_ads8851: it is never a
 * protocol participant — it is the WIRE. Tests inject raw bytes/frames as
 * the peer (rx side of hlink) and capture what hlink transmits.
 * Deterministic: no threads, no clocks; everything moves on hlink_poll().
 *
 * Used by the fw host tests AND the qemu cortex-m3 run of the same suite
 * (via port_hlink_model.c). Backs the MB-side story only symbolically —
 * the MB counterpart port (mb/hlink_mb_port.c) is under the syntax gate.
 */
#ifndef MODEL_HLINK_H
#define MODEL_HLINK_H

#include <stdint.h>

/* ---- test control ---- */
void mhl_reset(void);                        /* flush both directions        */

/* Peer -> head: raw bytes onto the wire (queue full drops the excess). */
uint32_t mhl_inject(const uint8_t *bytes, uint32_t n);

/* Peer -> head: build a well-formed frame (hlink wire format, own encoder
 * path for oracle independence on the payload content — CRC via public
 * hlink_crc8) and inject it. bad_crc != 0 corrupts the CRC byte.
 * Lengths validated like the real send; returns bytes injected or -1. */
int mhl_inject_frame(uint8_t type, uint8_t seq, const uint8_t *pl,
                     uint32_t pl_len, int bad_crc);

/* Head -> peer capture (linear; mhl_tx_dropped() != 0 means it overflowed). */
uint32_t mhl_tx_len(void);
uint32_t mhl_tx_copy(uint8_t *out, uint32_t cap); /* copy, do not clear   */
void     mhl_tx_clear(void);
uint32_t mhl_tx_dropped(void);

/* ---- port-facing (called by port_hlink_model.c only) ---- */
int  mhl_rx_take(uint8_t *byte);             /* 1 = byte to the module      */
void mhl_tx_give(const uint8_t *bytes, uint32_t n);

#endif /* MODEL_HLINK_H */
