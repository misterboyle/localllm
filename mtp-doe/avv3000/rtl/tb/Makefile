# rtl/tb Makefile — verilator testbenches (rung r_rtl)
VERILATOR ?= verilator
RTLDIR    := ..
WARN      := -Wno-fatal -Wno-WIDTH -Wno-UNUSEDSIGNAL -Wno-UNUSEDPARAM \
             -Wno-PINMISSING -Wno-SYNCASYNCNET -Wno-WIDTHCONCAT \
             -Wno-CASEX -Wno-CASEINCOMPLETE -Wno-UNOPTFLAT
SOURCES   := $(RTLDIR)/ads8851_pkg.sv $(RTLDIR)/axi_lite_bridge.sv \
             $(RTLDIR)/waveform.sv $(RTLDIR)/ads8851_ctrl.sv

.PHONY: all clean run

all: t_bus t_waveform

# --cc --exe --build (NOT --binary: --binary implies --main and generates a
# second main() that collides with the testbench's)
t_bus: t_bus.cc $(SOURCES)
	$(VERILATOR) --cc --exe --build $(WARN) --top-module ads8851_ctrl \
	  --Mdir obj_dir_tbus -o t_bus t_bus.cc $(SOURCES)

t_waveform: t_waveform.cc $(SOURCES)
	$(VERILATOR) --cc --exe --build --trace $(WARN) --top-module ads8851_ctrl \
	  --Mdir obj_dir_twave -o t_waveform t_waveform.cc $(SOURCES)

run: all
	./obj_dir_tbus/t_bus
	./obj_dir_twave/t_waveform

clean:
	rm -rf obj_dir_tbus obj_dir_twave