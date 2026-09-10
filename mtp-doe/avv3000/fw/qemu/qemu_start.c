/* qemu_start.c — cortex-m3 vectors + C startup for the fw qemu runner.
 * Target: qemu -M lm3s6965evb (generic CM3; flash @0, 64K RAM @0x20000000).
 * Faults/IRQs semihost-exit with a distinct code so a broken tree fails
 * fast instead of hanging (the runner watchdog is the last resort). */
#include "sh.h"

extern unsigned _sidata, _sdata, _edata, _sbss, _ebss;
extern int main(void);

void Reset_Handler(void);
static void dead(const char *why)
{
    sh_puts("\nFAULT: ");
    sh_puts(why);
    sh_puts("\n");
    sh_exit(111);
}

void Reset_Handler(void)
{
    unsigned *src = &_sidata, *dst = &_sdata;
    while (dst < &_edata)
        *dst++ = *src++;
    for (dst = &_sbss; dst < &_ebss;)
        *dst++ = 0u;

    int rc = main();
    sh_exit(rc);
}

#define DEAD(name, why) \
    static void name(void) { dead(why); }
DEAD(nmi_h, "NMI")
DEAD(hard_h, "HardFault")
DEAD(mpc_h, "MemManage")
DEAD(bfs_h, "BusFault")
DEAD(ufs_h, "UsageFault")
DEAD(svc_h, "SVCall-unexpected")
DEAD(dflt_h, "unexpected IRQ")

extern unsigned _stack_top;

__attribute__((section(".isr_vector"), used))
void (* const g_vectors[])(void) = {
    (void (*)(void))&_stack_top,
    Reset_Handler,
    nmi_h, hard_h, mpc_h, bfs_h, ufs_h,
    dflt_h, dflt_h, dflt_h, dflt_h,   /* 7..10 */
    svc_h, dflt_h, dflt_h,             /* SVCall, DebugMon, PendSV (unused) */
    dflt_h,                            /* SysTick */
};
