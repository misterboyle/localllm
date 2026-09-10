# svc/tests Makefile — host Unity tests for the telemetry service
# (rung r_svc, plan §5: t_svc_burst + log golden). Vendored Unity is
# shared with fw/tests (one copy in the tree).
CC ?= cc
CSTD := -std=c11
WARN := -Wall -Wextra -Werror
UNITY_DIR := ../../fw/tests/unity
INC  := -I. -I$(UNITY_DIR) -I..
B    := build

SVC_SRC := ../telem_svc.c
TESTS   := t_svc_burst t_svc_log

.PHONY: all run clean
all: $(addprefix $(B)/,$(TESTS))

$(B):
	@mkdir -p $(B)

# vendored Unity (MIT) — compiled without our -Werror regime
$(B)/unity.o: $(UNITY_DIR)/unity.c | $(B)
	$(CC) $(CSTD) -I$(UNITY_DIR) -w -c $< -o $@

$(B)/%.o: %.c | $(B)
	$(CC) $(CSTD) $(WARN) $(INC) -c $< -o $@

$(B)/%: $(B)/%.o $(B)/unity.o $(SVC_SRC) | $(B)
	$(CC) $(CSTD) $(WARN) $(INC) $< $(B)/unity.o $(SVC_SRC) -o $@

run: all
	@rc=0; for t in $(TESTS); do \
	  echo "== $$t"; ./$(B)/$$t || rc=1; \
	done; exit $$rc

clean:
	rm -rf $(B)
