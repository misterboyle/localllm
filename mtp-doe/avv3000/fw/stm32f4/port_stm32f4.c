/* port_stm32f4.c — STM32F4 target port for drv_ads8851.c (REFERENCE)
 *
 * The safelight controller reaches the ADS-8851 register map THROUGH THE
 * HOST BRIDGE over I2C1 (the same transaction shape the field sniff
 * payload/field/i2c.pcap.txt shows on the affected unit): one write
 * transfer {op, offset_hi, offset_lo, d0..d3} initiates a bridge
 * read/write against the AXI-Lite window at Zynq base 0x4000_0000.
 * DRDY_N (§3 pin 14) is wired to PB13 input; waiting is done on the pin,
 * never by re-reading STAT (ES-2026-04 ES-2.1.3 — hal_ads8851.h).
 *
 * NOT BUILT BY THE FIXTURE LADDER (no target silicon in CI): this file is
 * compile-checked by `make -C fw tests-syntax` and is the honest-shape
 * reference the ticket's reader expects next to the model port. Bases and
 * bit positions follow the STM32F4 reference manual family; bridge
 * semantics follow the field capture. */

#include <stdint.h>
#include "hal_ads8851.h"
#include "hlink_port.h"

/* ---- minimal CMSIS-free register view (subset used here) ---- */
#define PERIPH_BASE   0x40000000u
#define I2C1_BASE     (PERIPH_BASE + 0x5400u)   /* AF9 I2C1 on PB6/PB7   */
#define GPIOB_BASE    0x40020400u
#define RCC_BASE      0x40023800u /* AHB1ENR at +0x30, APB1ENR at +0x40  */

typedef struct {
    volatile uint32_t cr1, cr2, oar1, oar2, dr, sr1, sr2, ccr, trise;
} i2c_t;
typedef struct {
    volatile uint32_t moder, otyper, ospeedr, pupdrr, idr, odr, bsrr,
                      lckr, afrl, afrh;
} gpio_t;
typedef struct {
    volatile uint32_t dummy[11];
    volatile uint32_t ahb1enr;      /* +0x30 */
    volatile uint32_t apb1enr;      /* +0x40 */
    volatile uint32_t apb2enr;      /* +0x50 */
} rcc_t;

static volatile i2c_t *const i2c1 = (volatile i2c_t *)I2C1_BASE;
static volatile gpio_t *const gpiob = (volatile gpio_t *)GPIOB_BASE;
static volatile rcc_t *const rcc = (volatile rcc_t *)RCC_BASE;

/* I2C1 CR1/SR1 fields (F4 line) */
#define I2C_CR1_PE     (1u << 0)
#define I2C_CR1_SWI    (1u << 8)
#define I2C_CR1_START  (1u << 8)
#define I2C_CR1_STOP   (1u << 9)
#define I2C_CR1_ACK    (1u << 10)
#define I2C_SR1_SB     (1u << 0)
#define I2C_SR1_ADDR   (1u << 1)
#define I2C_SR1_BTF    (1u << 2)
#define I2C_SR1_RXNE   (1u << 6)
#define I2C_SR1_TXE    (1u << 7)
#define I2C_SR1_AF     (1u << 10)
#define I2C_SR2_BUSY   (1u << 1)

#define BRIDGE_I2C_ADDR 0x68u  /* host bridge slave address (board fixed) */
#define BRIDGE_OP_RD    0x00u
#define BRIDGE_OP_WR    0x40u

#define DRDY_PIN        (1u << 13) /* PB13 <- ADS8851 pin 14 DRDY_N, active low */

/* ---- bridge transport ---- */
static void i2c_kill_errors(void)
{
    if (i2c1->sr1 & I2C_SR1_AF) {
        i2c1->sr1 &= ~I2C_SR1_AF;
        i2c1->cr1 |= I2C_CR1_STOP;
    }
}

static int i2c_xfer(const uint8_t *tx, uint32_t tx_len,
                    uint8_t *rx, uint32_t rx_len)
{
    uint32_t t;

    i2c1->cr1 |= I2C_CR1_START;
    for (t = 0; !(i2c1->sr1 & I2C_SR1_SB) && t < 100000u; t++)
        ;
    if (t >= 100000u) { i2c_kill_errors(); return -1; }

    i2c1->dr = (BRIDGE_I2C_ADDR << 1); /* W: address phase only for split */
    for (t = 0; !(i2c1->sr1 & I2C_SR1_ADDR) && t < 100000u; t++)
        ;
    if (t >= 100000u) { i2c_kill_errors(); return -1; }
    (void)i2c1->sr2; /* clear ADDR sequence */
    (void)i2c1->sr1;

    for (uint32_t i = 0; i < tx_len; i++) {
        while (!(i2c1->sr1 & I2C_SR1_TXE))
            ;
        i2c1->dr = tx[i];
    }
    if (rx_len) {
        i2c1->cr1 |= I2C_CR1_STOP;
        for (uint32_t i = 0; i + 1u < rx_len; i++) {
            while (!(i2c1->sr1 & I2C_SR1_RXNE))
                ;
            rx[i] = (uint8_t)i2c1->dr;
        }
        /* NACK+STOP before last byte */
        i2c1->cr1 &= ~I2C_CR1_ACK;
        while (!(i2c1->sr1 & I2C_SR1_RXNE))
            ;
        rx[rx_len - 1u] = (uint8_t)i2c1->dr;
        i2c1->cr1 |= I2C_CR1_STOP;
        i2c1->cr1 |= I2C_CR1_ACK;
    } else {
        while (!(i2c1->sr1 & I2C_SR1_BTF))
            ;
        i2c1->cr1 |= I2C_CR1_STOP;
    }
    return 0;
}

static int bridge_op(uint32_t off, uint32_t *val)
{
    uint8_t msg[7];
    uint8_t rx[4];
    uint32_t rc;

    msg[0] = BRIDGE_OP_RD | ((val != 0) ? 0u : BRIDGE_OP_WR);
    msg[1] = (uint8_t)(off >> 8);
    msg[2] = (uint8_t)(off & 0xFFu);
    if (val != 0) {
        /* write direction: data follows */
        msg[0] = BRIDGE_OP_WR;
        msg[3] = (uint8_t)(*val >> 24);
        msg[4] = (uint8_t)(*val >> 16);
        msg[5] = (uint8_t)(*val >> 8);
        msg[6] = (uint8_t)(*val);
        rc = (uint32_t)i2c_xfer(msg, 7u, 0, 0u);
        return rc == 0u ? 0 : -1;
    }
    /* read: 3-byte setup, then restart handled by a second transfer */
    if (i2c_xfer(msg, 3u, 0, 0u) != 0)
        return -1;
    i2c1->cr1 |= I2C_CR1_START;
    while (!(i2c1->sr1 & I2C_SR1_SB))
        ;
    i2c1->dr = (BRIDGE_I2C_ADDR << 1) | 1u;
    if (i2c_xfer((const uint8_t *)msg, 0u, rx, 4u) != 0)
        return -1;
    *val = ((uint32_t)rx[0] << 24) | ((uint32_t)rx[1] << 16) |
           ((uint32_t)rx[2] << 8) | rx[3];
    return 0;
}

/* ---- hal contract ---- */
uint32_t ads8851_hw_rd(uint32_t off)
{
    uint32_t v = 0u;
    if (bridge_op(off, &v) != 0)
        return 0u; /* bridge NAK: ALARM_COMMS path on the ASIC side */
    return v;
}

void ads8851_hw_wr(uint32_t off, uint32_t val)
{
    (void)bridge_op(off, &val);
}

int ads8851_hw_wait_drdy(void)
{
    /* PB13 input, active-low DRDY_N; conservative busy-poll (EXTI13 is the
     * production path; same edge semantics). Bounded: ~4 x 8 ms max window. */
    for (uint32_t spin = 0; spin < 2000000u; spin++) {
        if ((gpiob->idr & DRDY_PIN) == 0u)
            return 1;
    }
    return 0;
}

/* ---- HEAD-LINK port (hlink_port.h): USART2, PA2 TX / PA3 RX AF7 -------
 * The framed control link to the base controller (hlink.h). NOT the
 * telemetry console — three-serial naming trap, hlink.h header note. */
#define USART2_BASE   (PERIPH_BASE + 0x4400u)  /* APB1                     */
#define GPIOA_BASE    0x40020000u

typedef struct {
    volatile uint32_t sr, dr, brr, cr1, cr2, cr3, gtpr;
} usart_t;

static volatile usart_t *const usart2 = (volatile usart_t *)USART2_BASE;
static volatile gpio_t *const gpioa = (volatile gpio_t *)GPIOA_BASE;

#define USART_SR_RXNE  (1u << 5)
#define USART_SR_TXE   (1u << 7)
#define USART_CR1_UE   (1u << 13)
#define USART_CR1_TE   (1u << 3)
#define USART_CR1_RE   (1u << 2)

void hlink_hw_tx(const uint8_t *bytes, uint32_t n)
{
    uint32_t t;

    for (uint32_t i = 0; i < n; i++) {
        for (t = 0; !(usart2->sr & USART_SR_TXE) && t < 400000u; t++)
            ;
        if (t >= 400000u)
            return; /* TX stuck = board fault, not a link condition;
                     * the heartbeat period retries from the next frame */
        usart2->dr = bytes[i];
    }
}

int hlink_hw_rx(uint8_t *byte)
{
    if (!(usart2->sr & USART_SR_RXNE))
        return 0;
    *byte = (uint8_t)usart2->dr; /* reading DR also clears ORE: a dropped
                                  * byte is a codec problem — SOF resync  */
    return 1;
}

/* Bring-up order matters for the bridge; called once from board_init().
 * (Real projects put this in the clock-tree init; kept adjacent so the
 * syntax gate covers it too.) */
void port_stm32f4_init(void)
{
    rcc->ahb1enr  |= (1u << 1);              /* GPIOB clock               */
    rcc->apb1enr  |= (1u << 21);             /* I2C1 clock                */
    gpiob->moder  &= ~(0xFu << 26);          /* PB13 input (DRDY_N)       */
    gpiob->moder  |=  (0x0u << 26);
    gpiob->pupdrr |=  (1u << 26);            /* pull-up: idle HIGH = not  */
    i2c1->cr1 &= ~I2C_CR1_PE;
    i2c1->ccr = 100000u / 100000u;           /* placeholder 100 kHz map   */
    i2c1->trise = 200u;
    i2c1->cr1 |= I2C_CR1_PE | I2C_CR1_ACK;
    (void)i2c1->sr1;
    (void)i2c1->sr2;
    (void)I2C_CR1_SWI;

    /* HEAD-LINK: USART2 115200 8N1 (placeholder BRR for 16 MHz APB1) */
    rcc->ahb1enr    |= (1u << 0);            /* GPIOA clock               */
    rcc->apb1enr    |= (1u << 17);           /* USART2 clock              */
    gpioa->moder     = (gpioa->moder & ~0xF0u) | 0xA0u;  /* PA2/PA3 = AF */
    gpioa->afrl      = (gpioa->afrl & ~0xFF00u) | (7u << 8) | (7u << 12);
    gpioa->pupdrr   |= (1u << 12);           /* PA3 RX pull-up            */
    usart2->brr      = 139u;
    usart2->cr1      = USART_CR1_UE | USART_CR1_TE | USART_CR1_RE;
}
