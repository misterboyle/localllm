# AVV-3000 fixture — top-level quality gate (plan R2: make -C avv3000 check).
# Rungs land one stage at a time; score.sh (S7) wires the same targets into
# the ladder.
.PHONY: check rtl fw svc gui clean

check: rtl fw svc

rtl:
	@$(MAKE) -C rtl/tb run

fw:
	@$(MAKE) -C fw all

svc:
	@$(MAKE) -C svc all

# gui arrives with the Qt tier (env.sh install-gui); score.sh gates it
gui:
	@$(MAKE) -C gui all

clean:
	@$(MAKE) -C rtl/tb clean
	@$(MAKE) -C fw clean
	@$(MAKE) -C svc clean
	-@$(MAKE) -C gui clean
