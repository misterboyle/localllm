/* qemu_main.c — entry for the qemu build of each fw test.
 *
 * The test files keep standard Unity mains (identical sources to the host
 * build); fw/qemu/Makefile renames `main` -> `fw_test_main` with -Dmain=…
 * for them only, so this shim can add the machine-readable RESULT line
 * that run_qemu.sh parses before trusting the semihosted exit status. */
#include "unity.h"
#include "sh.h"

extern int fw_test_main(void);

int main(void)
{
    int rc = fw_test_main();

    sh_puts("\nRESULT rc=");
    sh_putu((unsigned)(rc < 0 ? 255 : rc));
    sh_puts(" run=");
    sh_putu((unsigned)Unity.NumberOfTests);
    sh_puts(" fail=");
    sh_putu((unsigned)Unity.TestFailures);
    sh_puts("\n");
    return rc;
}
