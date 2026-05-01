#!/usr/bin/env bash
# Validate Python scripts have valid syntax
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"

fail=0

for py in "$ROOT_DIR"/*.py; do
  [ -f "$py" ] || continue
  name="$(basename "$py")"

  if python3 -m py_compile "$py" 2>/dev/null; then
    echo "  $name: valid syntax"
  else
    echo "  $name: SYNTAX ERROR"
    fail=1
  fi
done

exit "$fail"
