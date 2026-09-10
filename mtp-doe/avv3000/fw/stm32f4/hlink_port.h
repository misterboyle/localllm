/* hlink_port.h — 2-function platform contract for hlink.c
 *
 * Same shape as hal_ads8851.h: hlink.c never touches a UART directly.
 * Fixture builds link port_hlink_model.c (fake-peer wire shim); the STM32
 * head target implements it in port_stm32f4.c (USART2); the MicroBlaze
 * base end uses mb/hlink_mb_port.c.
 *
 * Contract:
 *  - hlink_hw_tx(): push n bytes at the peer (burst-chunked n <= 8 by
 *    hlink.c, but ports may accept more). Must not drop or backpressure:
 *    a stuck TX is a board fault, not a link condition — the heartbeat
 *    timeout above the port handles silence, not slow bytes.
 *  - hlink_hw_rx(): one byte from the peer; 1 = byte stored,
 *    0 = nothing waiting. Never blocks (poll-driven, deterministic tests).
 */
#ifndef HLINK_PORT_H
#define HLINK_PORT_H

#include <stdint.h>

void hlink_hw_tx(const uint8_t *bytes, uint32_t n);
int  hlink_hw_rx(uint8_t *byte);

#endif /* HLINK_PORT_H */
