#!/usr/bin/env bash
# env.sh — toolchain check/install for the AVV-3000 fixture.
# Tiers: core (verilator qemu arm-none-eabi-gcc) and gui (qt).
# Usage:
#   ./env.sh check          print tool status per rung
#   ./env.sh install-core   brew install verilator qemu arm-none-eabi-gcc
#   ./env.sh install-gui    brew install qt (large; r_gui only)
set -euo pipefail
CONF_DIR="$(cd "$(dirname "$0")" && pwd)"
CORE_FORMULAS=(verilator qemu arm-none-eabi-gcc)
GUI_FORMULAS=(qt)

check() {
  local missing=0
  for t in verilator make cmake clang gcc python3 qemu-system-arm arm-none-eabi-gcc; do
    if command -v "$t" >/dev/null 2>&1; then
      echo "  ok    $t -> $(command -v "$t")"
    else
      echo "  MISS  $t"
      missing=1
    fi
  done
  qtpath="$(brew --prefix 2>/dev/null)/opt/qt/bin/qmake"
  if [ -x "$qtpath" ]; then
    echo "  ok    qt (r_gui/offscreen) -> $qtpath"
  else
    echo "  MISS  qt (r_gui deferred; install-gui)"
  fi
  if [ "$missing" -eq 1 ]; then
    echo "note:  run './env.sh install-core' (and optionally install-gui)"
  fi
}

install-core() {
  brew install "${CORE_FORMULAS[@]}"
}

install-gui() {
  brew install "${GUI_FORMULAS[@]}"
}

case "${1:-check}" in
  check)        check ;;
  install-core) install-core ;;
  install-gui)  install-gui ;;
  *) echo "usage: $0 {check|install-core|install-gui}"; exit 1 ;;
esac