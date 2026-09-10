# fw Makefile — aggregate runner for the fw rungs (plan §5):
#   r_fw_host (20 pts)  tests/   native Unity
#   r_fw_qemu (10 pts)  qemu/    same suite, cortex-m3 under qemu
#   r_fw_mb   (10 pts)  mb/      MicroBlaze port syntax gate (§10.2)
.PHONY: all host qemu mb tests-syntax check clean

all: host qemu mb tests-syntax

host:
	@$(MAKE) -C tests run

qemu:
	@$(MAKE) -C qemu run

mb:
	@$(MAKE) -C mb check

# reference target port: compile-checked, never linked (no silicon in CI)
tests-syntax:
	clang -std=c11 -Wall -Wextra -Werror -fsyntax-only \
	      -Istm32f4 stm32f4/port_stm32f4.c
	@echo "stm32f4 reference port: syntax OK"

check: all

clean:
	$(MAKE) -C tests clean
	$(MAKE) -C qemu clean
