# fw/tests Makefile — host Unity tests (rung r_fw_host: t_fw_init, t_drdy,
# t_reg_arb). Driver + model + port build from source on every link; the
# golden driver and the model must both stay green here (plan §5).
CC ?= cc
CSTD := -std=c11
WARN := -Wall -Wextra -Werror
INC  := -I. -Iunity -I../stm32f4
B    := build

FW_SRC := ../stm32f4/drv_ads8851.c ../stm32f4/model_ads8851.c \
          ../stm32f4/port_model.c
TESTS  := t_fw_init t_drdy t_reg_arb

.PHONY: all run clean
all: $(addprefix $(B)/,$(TESTS))

$(B):
	@mkdir -p $(B)

# vendored Unity (MIT) — compiled without our -Werror regime
$(B)/unity.o: unity/unity.c | $(B)
	$(CC) $(CSTD) $(INC) -w -c $< -o $@

$(B)/%.o: %.c | $(B)
	$(CC) $(CSTD) $(WARN) $(INC) -c $< -o $@

$(B)/%: $(B)/%.o $(B)/unity.o $(FW_SRC) | $(B)
	$(CC) $(CSTD) $(WARN) $(INC) $< $(B)/unity.o $(FW_SRC) -o $@

run: all
	@rc=0; for t in $(TESTS); do \
	  echo "== $$t"; ./$(B)/$$t || rc=1; \
	done; exit $$rc

clean:
	rm -rf $(B)
