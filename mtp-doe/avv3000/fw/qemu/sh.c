/* sh.c — ARMv7-M semihosting (bkpt 0xAB) for qemu-system-arm
 *
 * r_fw_qemu (plan §5): the SAME fw test suite runs on cortex-m3 here.
 * Console + exit go through semihosting; no peripheral is touched.
 * The runner additionally parses the RESULT= line from qemu_main.c,
 * so exit-status mapping never has to be trusted blindly.
 */
#include "sh.h"

static long sh_call(long op, const void *arg)
{
    register long a0 asm("r0") = op;
    register long a1 asm("r1") = (long)arg;

    asm volatile ("bkpt 0xAB" : "+r"(a0) : "r"(a1) : "memory");
    return a0;
}

void sh_putc(int c)
{
    char b = (char)c;
    (void)sh_call(0x03, &b); /* SYS_WRITEC: writes one char */
}

void sh_puts(const char *s)
{
    while (*s)
        sh_putc((unsigned char)*s++);
}

void sh_putu(unsigned v)
{
    char buf[10];
    int i = 0;

    do {
        buf[i++] = (char)('0' + (v % 10u));
        v /= 10u;
    } while (v);
    while (i)
        sh_putc(buf[--i]);
}

void sh_exit(int rc)
{
    /* SYS_EXIT with the AAPCS reason block: {ADPN=0x20026 "call exit", code}. */
    unsigned blk[2] = { 0x20026u, (unsigned)rc };

    (void)sh_call(0x18, blk);
    for (;;) /* if qemu kept running: wedge for the runner watchdog */
        asm volatile ("bkpt 0x7F");
}
