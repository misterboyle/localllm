#!/usr/bin/env bash
# Validate shell scripts pass ShellCheck
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"

if ! command -v shellcheck > /dev/null 2>&1; then
  echo "SKIP: shellcheck not installed"
  exit 0
fi

fail=0
scripts=(
  "$ROOT_DIR/start-server.sh"
  "$ROOT_DIR/snapshot-memory.sh"
)

# Add test scripts
for t in "$SCRIPT_DIR"/test-*.sh; do
  [ -f "$t" ] && scripts+=("$t")
done

for script in "${scripts[@]}"; do
  name="$(basename "$script")"
  if shellcheck -e SC1090 -e SC1091 "$script" > /dev/null 2>&1; then
    echo "  $name: clean"
  else
    echo "  $name: ShellCheck warnings"
    shellcheck -e SC1090 -e SC1091 "$script" || true
    fail=1
  fi
done

exit "$fail"
