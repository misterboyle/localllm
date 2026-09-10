#!/usr/bin/env bash
# score.sh — AVV-3000 scoring ladder (plan §5): one scalar out of 100 from
# hermetic tests. exit code = score (0..100). Per-rung JSON object plus a
# summary line. No network (everything below is local make/test-execution);
# full ladder TEST EXECUTION must stay < 90 s (builds belong to `make check`).
#
# Rung statuses: PASS (scored), FAIL (0), SKIP (0, tool missing — r_gui
# without Qt), PENDING (0, stage not built yet — r_sim before S5).
# Golden tree expectation: 100 once all stages exist (r_sim lands S5).
#
# S7 extends this with head-to-head ingestion (source tag, mtp-harvest);
# the rung table below is the contract S7 consumes unchanged.
set -uo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

BREW_PREFIX="$(brew --prefix 2>/dev/null || echo /opt/homebrew)"
QMAKE="$BREW_PREFIX/opt/qt/bin/qmake"

# name|pts|make-target-command|gate ("" = always, else path that must exist)
RUNGS=(
  "r_rtl|15|make -C rtl/tb run|"
  "r_fw_host|20|make -C fw/tests run|"
  "r_fw_qemu|10|make -C fw/qemu run|"
  "r_fw_mb|10|make -C fw/mb check|"
  "r_svc|15|make -C svc/tests run|"
  "r_gui|5|make -C gui run|$QMAKE"
  "r_sim|25|make -C sim run|sim/Makefile"
)

total=0
json=""
for entry in "${RUNGS[@]}"; do
  IFS='|' read -r name pts cmd gate <<< "$entry"
  status=""
  if [ -n "$gate" ] && [ ! -x "$gate" ] && [ ! -f "$gate" ]; then
    if [ "$name" = "r_sim" ]; then status="PENDING"; else status="SKIP"; fi
    detail="gate missing"
  elif out="$($cmd 2>&1)"; then
    status="PASS"; detail=""
  else
    status="FAIL"; detail="rc=$?"
  fi
  if [ "$status" = "PASS" ]; then
    total=$((total + pts))
    pass_json=true
  else
    pass_json=false
    echo "--- $name ($pts pts) $status: $cmd"
    echo "$out" | tail -20
  fi
  [ -n "$json" ] && json="$json, "
  json="$json\"$name\": {\"pts\": $pts, \"pass\": $pass_json, \"status\": \"$status\"}"
done

echo "{\"score\": $total, \"rungs\": {$json}}"
echo "SCORE $total"
exit "$total"
