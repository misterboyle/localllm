# fw/mb Makefile — r_fw_mb: MicroBlaze port syntax gate (plan §10.2)
# Host-compiler substitute for the clang -target microblaze R4 imagined;
# see ads8851_mb_bsp.c header comment for the rationale.
CC ?= clang
CFLAGS := -std=c11 -Wall -Wextra -Werror -fsyntax-only -I../stm32f4

.PHONY: check
check:
	$(CC) $(CFLAGS) ads8851_mb_bsp.c
	$(CC) $(CFLAGS) ../stm32f4/drv_ads8851.c
	@echo "r_fw_mb: syntax OK (bsp + driver)"
