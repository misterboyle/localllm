# AVV-3000 fixture — top-level quality gate (plan R2: make -C avv3000 check).
# Rungs land one stage at a time; score.sh (S7) wires the same targets into
# the ladder.
.PHONY: check rtl fw clean

check: rtl fw

rtl:
	@$(MAKE) -C rtl/tb run

fw:
	@$(MAKE) -C fw all

clean:
	@$(MAKE) -C rtl/tb clean
	@$(MAKE) -C fw clean
