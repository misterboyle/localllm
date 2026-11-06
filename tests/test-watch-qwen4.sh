#!/usr/bin/env bash
# Legs for the Qwen 4 launch watcher: needle, controls, alert tiers, dedup,
# and one live read of the real Hugging Face org listing.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

python3 "$SCRIPT_DIR/test-watch-qwen4.py"
