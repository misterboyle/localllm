# fw/mb Makefile — r_fw_mb: MicroBlaze port syntax gate (plan §10.2)
# Host-compiler substitute for the clang -target microblaze R4 imagined;
# see ads8851_mb_bsp.c header comment for the rationale.
CC ?= clang
# -Wconversion: the closest free stand-in for "lint against microblaze"
# (register/integer width discipline as a real mb32 build would check;
# caveat: host int widths apply — sizeof(long) is the known blind spot).
CFLAGS := -std=c11 -Wall -Wextra -Werror -Wconversion -fsyntax-only -I../stm32f4

.PHONY: check
check:
	$(CC) $(CFLAGS) ads8851_mb_bsp.c hlink_mb_port.c \
	               ../stm32f4/drv_ads8851.c ../stm32f4/hlink.c \
	               ../stm32f4/model_ads8851.c ../stm32f4/port_model.c
	@echo "r_fw_mb: syntax + conversion OK (bsp, hlink mb port, driver, hlink, model, port)"
