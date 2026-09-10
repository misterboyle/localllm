# svc Makefile — telemetry service rungs (plan §5 r_svc, 15 pts):
#   tests/  t_svc_burst (ring admission, B2 fail-catch) + t_svc_log (CSV
#           byte contract vs golden excerpt)
# The stdin->console skin (svc_main.c) is compile-gated, not ladder-tested.
.PHONY: all tests syntax check clean

all: tests syntax

tests:
	@$(MAKE) -C tests run

# service skin: compile-checked, never linked by the ladder (no FIFO here)
syntax: | build
	$(CC) -std=c11 -Wall -Wextra -Werror -I. -c svc_main.c -o build/svc_main.o
	@echo "svc_main: compile OK"

build:
	@mkdir -p build

check: all

clean:
	@$(MAKE) -C tests clean
	rm -rf build
