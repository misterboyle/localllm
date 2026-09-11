# AVV-3000 fixture — top-level quality gate (plan R2: make -C avv3000 check).
# Rungs land one stage at a time; score.sh (S7) wires the same targets into
# the ladder.
.PHONY: check payload-check rtl fw svc sim gui clean

check: rtl fw svc sim payload-check

# S1 payload gate (S1 acceptance tooling): §3 size gates + cross-ref +
# register-map pin. WIRED INTO `check` 2026-09-11 — all four §3 gates
# green after the S1 expansion sessions (datasheet/appnote/axi/field).
payload-check:
	@tests/test-payload-consistency.sh

rtl:
	@$(MAKE) -C rtl/tb run

fw:
	@$(MAKE) -C fw all

svc:
	@$(MAKE) -C svc all

sim:
	@$(MAKE) -C sim all

# gui arrives with the Qt tier (env.sh install-gui); score.sh gates it
gui:
	@$(MAKE) -C gui all

clean:
	@$(MAKE) -C rtl/tb clean
	@$(MAKE) -C fw clean
	@$(MAKE) -C svc clean
	@$(MAKE) -C sim clean
	-@$(MAKE) -C gui clean
