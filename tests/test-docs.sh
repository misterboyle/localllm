#!/usr/bin/env bash
# Validate AGENTS.md files exist and are non-empty
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"

fail=0

# Root AGENTS.md
if [ -s "$ROOT_DIR/AGENTS.md" ]; then
  echo "  AGENTS.md: present ($(wc -l < "$ROOT_DIR/AGENTS.md") lines)"
else
  echo "  AGENTS.md: MISSING or empty"
  fail=1
fi

# .opencode/AGENTS.md
if [ -s "$ROOT_DIR/.opencode/AGENTS.md" ]; then
  echo "  .opencode/AGENTS.md: present ($(wc -l < "$ROOT_DIR/.opencode/AGENTS.md") lines)"
else
  echo "  .opencode/AGENTS.md: MISSING or empty"
  fail=1
fi

exit "$fail"
