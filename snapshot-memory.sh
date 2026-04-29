#!/usr/bin/env bash
# Takes a memory snapshot and logs it with a phase label
set -euo pipefail

PHASE="${1:?Usage: $0 <phase-label>}"
TEST_LOG="$HOME/.localllm/test-2slot.log"

echo "" | tee -a "$TEST_LOG"
echo "=== MEMORY SNAPSHOT: $PHASE ($(date '+%H:%M:%S')) ===" | tee -a "$TEST_LOG"

# Process memory
ps aux | grep llama-server | grep -v grep | awk '{printf "Process: MEM%%: %.1f, RSS: %.1f GB, VSZ: %.1f GB\n", $4, $6/1048576, $5/1048576}' | tee -a "$TEST_LOG"

# System memory (from Activity Monitor via vm_stat)
vm_stat | head -1 | tee -a "$TEST_LOG"

# Slots status
echo "Slots:" | tee -a "$TEST_LOG"
curl -s http://localhost:30080/slots | python3 -c "
import sys, json
slots = json.load(sys.stdin)
for s in slots:
    status = 'ACTIVE' if s['is_processing'] else 'idle'
    print(f'  Slot {s[\"id\"]}: {status}')
" 2>/dev/null | tee -a "$TEST_LOG"

# Prompt cache state
grep "cache state:" ~/.localllm/dense.log | tail -1 | tee -a "$TEST_LOG"

# Checkpoint summary — current ring size per slot (from latest checkpoint creation)
echo "Checkpoints:" | tee -a "$TEST_LOG"
for slot_id in 0 1 2 3 4 5 6 7; do
  latest=$(grep "id  ${slot_id}.*created context checkpoint" ~/.localllm/dense.log | tail -1 | sed -E 's/.*checkpoint ([0-9]+) of ([0-9]+).*/\1\/\2/')
  if [[ -n "$latest" ]]; then
    echo "  Slot $slot_id: $latest" | tee -a "$TEST_LOG"
  fi
done
