# fw/qemu Makefile — r_fw_qemu (plan §5): the SAME host test suite,
# cross-built for cortex-m3 and executed under qemu-system-arm with ARM
# semihosting. Acceptance (S3): host + qemu green on golden.
CC      := arm-none-eabi-gcc
SIZE    := arm-none-eabi-size
CSTD    := -std=gnu11   # register asm / attributes in sh.c (host builds stay -std=c11)
WARN    := -Wall -Wextra
ARCH    := -mcpu=cortex-m3 -mthumb
OPT     := -Os -ffunction-sections -fdata-sections
OPT     += -fno-builtin   # freestanding: keep our explicit mzero loops as loops
# Unity without libc stdio/math/setjmp: output goes to semihosting.
UNITYCFG := -DUNITY_EXCLUDE_STDIO_H -DUNITY_EXCLUDE_MATH_H \
            -DUNITY_EXCLUDE_SETJMP_H \
            -D'UNITY_OUTPUT_CHAR(a)=sh_putc(a)' \
            -D'UNITY_OUTPUT_CHAR_HEADER_DECLARATION=sh_putc(int)'
# shim/ shadows gcc's include_next wrappers (this toolchain has no newlib)
INC     := -I. -I../stm32f4 -I../tests -I../tests/unity -isystem shim
# freestanding: no libc exists in this toolchain (see shim/) — libgcc only
LDFLAGS := -T qemu.ld -nostdlib -lgcc -Wl,--gc-sections -Wl,-Map=build/map
B       := build

FW_OBJS  := $(B)/drv_ads8851.o $(B)/model_ads8851.o $(B)/port_model.o \
            $(B)/hlink.o $(B)/model_hlink.o $(B)/port_hlink_model.o
QEMU_SRC := qemu_start.c qemu_main.c sh.c
TESTS    := t_fw_init t_drdy t_reg_arb t_hlink
ELFS     := $(addprefix $(B)/,$(addsuffix .elf,$(TESTS)))

.PHONY: all run clean
.SECONDARY:            # keep intermediate .o files (build/ is gitignored)
all: $(ELFS)

$(B):
	@mkdir -p $(B)

# test .c files live in ../tests and keep their standard Unity mains;
# the qemu link supplies qemu_main.c, so `main` is renamed for them only
$(B)/%.o: ../tests/%.c | $(B)
	$(CC) $(ARCH) $(CSTD) $(WARN) $(OPT) $(UNITYCFG) $(INC) \
	      -Dmain=fw_test_main -c $< -o $@

$(B)/%.o: ../tests/unity/%.c | $(B)
	$(CC) $(ARCH) $(CSTD) -w $(OPT) $(UNITYCFG) $(INC) -c $< -o $@

$(B)/%.o: ../stm32f4/%.c | $(B)
	$(CC) $(ARCH) $(CSTD) $(WARN) $(OPT) $(UNITYCFG) $(INC) -c $< -o $@

$(B)/%.o: %.c | $(B)
	$(CC) $(ARCH) $(CSTD) $(WARN) $(OPT) $(UNITYCFG) $(INC) -c $< -o $@

$(B)/%.elf: $(B)/%.o $(B)/qemu_main.o $(B)/qemu_start.o $(B)/sh.o \
            $(B)/unity.o $(FW_OBJS) | $(B)
	$(CC) $(ARCH) $(OPT) $(LDFLAGS) $^ -o $@
	@$(SIZE) $@

run: all
	@rc=0; for t in $(TESTS); do \
	  echo "== $$t (qemu cortex-m3)"; \
	  ./run_qemu.sh $(B)/$$t.elf || rc=1; \
	done; exit $$rc

clean:
	rm -rf $(B)
