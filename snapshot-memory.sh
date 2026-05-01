#!/usr/bin/env bash
# Takes a memory snapshot for MLX servers
set -euo pipefail

PHASE="${1:?Usage: $0 <phase-label>}"
LOG="$HOME/.localllm/memory-snapshots.log"

echo "" | tee -a "$LOG"
echo "=== MEMORY SNAPSHOT: $PHASE ($(date '+%H:%M:%S')) ===" | tee -a "$LOG"

# MLX server processes
ps aux | grep mlx_lm.server | grep -v grep | awk '{printf "PID: %s, MEM%%: %.1f, RSS: %.1f GB\n", $2, $4, $6/1048576}' | tee -a "$LOG"

# System memory
vm_stat | head -1 | tee -a "$LOG"

# Server health — read ports from config
CONF_FILE="$HOME/.localllm/models.jsonc"
if [ -f "$CONF_FILE" ]; then
  ports=$(python3 -c "
import json, re
raw = open('$CONF_FILE').read()
raw = re.sub(r'//.*?$', '', raw, flags=re.MULTILINE)
cfg = json.loads(raw)
for name, srv in cfg.get('servers', {}).items():
    if srv.get('enabled', False):
        print(srv['port'])
" 2>/dev/null || echo "")
  if [ -n "$ports" ]; then
    for port in $ports; do
      status=$(curl -sf "http://localhost:$port/health" 2>/dev/null || echo "down")
      echo "Port $port: $status" | tee -a "$LOG"
    done
  else
    echo "No enabled servers in config" | tee -a "$LOG"
  fi
else
  echo "Config not found at $CONF_FILE" | tee -a "$LOG"
fi

# Prompt cache disk usage
echo "Prompt cache:" | tee -a "$LOG"
du -sh ~/.localllm/prompt_cache/ 2>/dev/null | tee -a "$LOG" || echo "  (not initialized)" | tee -a "$LOG"
