# sim Makefile — closed-loop plant tests (rung r_sim, 25 pts: `make -C sim
# run`, the score.sh contract). plant.c + engine mirror + port link the REAL
# drv_ads8851.c against the behavior model — same discipline as fw/tests:
# everything builds from source, nothing but seeded fixed-point runs.
CC ?= cc
CSTD := -std=c11
WARN := -Wall -Wextra -Werror
OPT  := -O2            # mdl_tick runs ~1e8 ACLK steps across the fv runs
FW   := ../fw/stm32f4
INC  := -I. -I$(FW) -I../fw/tests/unity
B    := build

SRC := plant.c model_waveform.c port_sim.c \
       $(FW)/drv_ads8851.c $(FW)/model_ads8851.c
TESTS := fv_staircase fv_occl

.PHONY: all run clean
all: $(addprefix $(B)/,$(TESTS))

$(B):
	@mkdir -p $(B)

# vendored Unity (MIT) — outside our -Werror regime, same as fw/tests
$(B)/unity.o: ../fw/tests/unity/unity.c | $(B)
	$(CC) $(CSTD) $(INC) -w $(OPT) -c $< -o $@

$(B)/%.o: %.c | $(B)
	$(CC) $(CSTD) $(WARN) $(OPT) $(INC) -c $< -o $@

# fv_* names avoid a pattern clash with the generic %.o rule above
$(B)/fv_%.o: tests/fv_%.c | $(B)
	$(CC) $(CSTD) $(WARN) $(OPT) $(INC) -c $< -o $@

$(B)/%: $(B)/%.o $(B)/unity.o $(SRC) | $(B)
	$(CC) $(CSTD) $(WARN) $(OPT) $(INC) $< $(B)/unity.o $(SRC) -o $@

run: all
	@rc=0; for t in $(TESTS); do \
	  echo "== $$t"; ./$(B)/$$t || rc=1; \
	done; exit $$rc

clean:
	rm -rf $(B)
