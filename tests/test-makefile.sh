#!/usr/bin/env bash
# Validate Makefile has required targets
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"

MAKEFILE="$ROOT_DIR/Makefile"

if [ ! -f "$MAKEFILE" ]; then
  echo "ERROR: Makefile not found"
  exit 1
fi

fail=0
required_targets=(help setup deps config start stop check lint test clean)

for target in "${required_targets[@]}"; do
  if grep -q "^${target}:" "$MAKEFILE" || grep -q "^${target} " "$MAKEFILE"; then
    echo "  $target: present"
  else
    echo "  $target: MISSING"
    fail=1
  fi
done

exit "$fail"
