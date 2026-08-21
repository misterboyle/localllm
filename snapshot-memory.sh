#!/usr/bin/env bash
# Takes a memory snapshot: server processes (llama.cpp + MLX), system memory,
# server health (ports from servers.jsonc), prompt cache usage.
set -euo pipefail

PHASE="${1:?Usage: $0 <phase-label>}"
LOG="$HOME/.localllm/memory-snapshots.log"
CONF_FILE="$(cd "$(dirname "$0")" && pwd)/servers.jsonc"

echo "" | tee -a "$LOG"
echo "=== MEMORY SNAPSHOT: $PHASE ($(date '+%H:%M:%S')) ===" | tee -a "$LOG"

# Server processes (RSS under-reports GPU/Metal memory — see wiki
# applesilicon-memory-monitoring; treat as a floor, not a peak)
# shellcheck disable=SC2009
ps aux | grep -E "llama-server|mlx_lm.server|optiq serve" | grep -v grep | \
  awk '{printf "PID: %s, MEM%%: %.1f, RSS: %.1f GB, CMD: %s\n", $2, $4, $6/1048576, $11}' | \
  tee -a "$LOG"

# System memory
vm_stat | head -1 | tee -a "$LOG"

# Server health — read ports from servers.jsonc
if [ -f "$CONF_FILE" ]; then
  ports=$(python3 -c "
import json, re
raw = open('$CONF_FILE').read()
raw = re.sub(r'//.*$', '', raw, flags=re.MULTILINE)
cfg = json.loads(raw)
for name, srv in cfg.get('servers', {}).items():
    print(name + ' ' + str(srv['port']))
" 2>/dev/null || echo "")
  if [ -n "$ports" ]; then
    while read -r name port; do
      if [ -n "$name" ]; then
        if curl -sf "http://localhost:$port/health" > /dev/null 2>&1; then
          echo "Port $port ($name): up" | tee -a "$LOG"
        else
          echo "Port $port ($name): down" | tee -a "$LOG"
        fi
      fi
    done <<< "$ports"
  else
    echo "No servers in $CONF_FILE" | tee -a "$LOG"
  fi
else
  echo "Config not found: $CONF_FILE" | tee -a "$LOG"
fi

# Prompt cache disk usage
echo "Prompt cache:" | tee -a "$LOG"
du -sh ~/.localllm/prompt_cache/ 2>/dev/null | tee -a "$LOG" || echo "  (not initialized)" | tee -a "$LOG"
