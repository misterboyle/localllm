#!/usr/bin/env bash
# 2-slot AB test: restarts dense server with 2 slots, monitors memory
set -euo pipefail

CONF_FILE="$HOME/.localllm/models.conf"
TEST_LOG="$HOME/.localllm/test-2slot.log"

echo "[$(date '+%H:%M:%S')] Starting 2-slot test..." | tee "$TEST_LOG"

# Backup config
cp "$CONF_FILE" "${CONF_FILE}.bak"

# Modify config: 2 slots, 64GB cache-ram, keep everything else same
sed -i '' 's/^DENSE_SLOTS=.*/DENSE_SLOTS=2/' "$CONF_FILE"
sed -i '' 's/^DENSE_CACHE_RAM=.*/DENSE_CACHE_RAM=65536/' "$CONF_FILE"

echo "[$(date '+%H:%M:%S')] Config updated: DENSE_SLOTS=2, DENSE_CACHE_RAM=65536" | tee -a "$TEST_LOG"

# Stop existing servers
if [ -f "$HOME/.localllm/pids/dense.pid" ]; then
  kill "$(cat "$HOME/.localllm/pids/dense.pid")" 2>/dev/null || true
  sleep 2
fi

# Clear old log
: > "$HOME/.localllm/dense.log"

# Start server with 2 slots
nohup bash "$HOME/localllm/start-server.sh" > >(sed 's/^/[2SLOT-TEST] /' >> "$TEST_LOG") 2>&1 &
echo $! > "$HOME/.localllm/pids/test2slot.pid"

echo "[$(date '+%H:%M:%S')] Server starting, PID: $(cat $HOME/.localllm/pids/test2slot.pid)" | tee -a "$TEST_LOG"

# Wait for server ready
for i in $(seq 1 60); do
  if curl -sf http://localhost:30080/health > /dev/null 2>&1; then
    echo "[$(date '+%H:%M:%S')] Server ready" | tee -a "$TEST_LOG"
    break
  fi
  sleep 1
done

# Record baseline memory
echo "=== BASELINE MEMORY ===" | tee -a "$TEST_LOG"
ps aux | grep llama-server | grep -v grep | awk '{printf "MEM%%: %.1f, RSS: %.1f GB, VSZ: %.1f GB\n", $4, $6/1024, $5/1024}' | tee -a "$TEST_LOG"

# Show server config
echo "=== SERVER CONFIG ===" | tee -a "$TEST_LOG"
tail -5 "$HOME/.localllm/dense.log" | tee -a "$TEST_LOG"

echo "" | tee -a "$TEST_LOG"
echo "=== TEST SETUP COMPLETE ===" | tee -a "$TEST_LOG"
echo "Config backed up to ${CONF_FILE}.bak" | tee -a "$TEST_LOG"
echo "To restore: bash $HOME/localllm/restore-config.sh" | tee -a "$TEST_LOG"
echo "" | tee -a "$TEST_LOG"
echo "Next steps:" | tee -a "$TEST_LOG"
echo "  1. Start opencode session A (Slot 0 agent)" | tee -a "$TEST_LOG"
echo "  2. Grow conversation to 50K tokens" | tee -a "$TEST_LOG"
echo "  3. Run: bash $HOME/localllm/snapshot-memory.sh phase-A-50K" | tee -a "$TEST_LOG"
echo "  4. Grow to 100K tokens" | tee -a "$TEST_LOG"
echo "  5. Run: bash $HOME/localllm/snapshot-memory.sh phase-A-100K" | tee -a "$TEST_LOG"
echo "  6. Start opencode session B (Slot 1 agent)" | tee -a "$TEST_LOG"
echo "  7. Grow to 50K tokens" | tee -a "$TEST_LOG"
echo "  8. Run: bash $HOME/localllm/snapshot-memory.sh phase-B-50K" | tee -a "$TEST_LOG"
echo "  9. When done, restore config: bash $HOME/localllm/restore-config.sh" | tee -a "$TEST_LOG"
