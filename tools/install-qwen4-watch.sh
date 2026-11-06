#!/usr/bin/env bash
# Install the Qwen 4 launch watcher as a LaunchAgent (macwatch's shape).
# Idempotent: re-running re-reads the plist from this repo.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LABEL="com.michael.qwen4-watch"
PLIST_SRC="$SCRIPT_DIR/$LABEL.plist"
PLIST_DST="$HOME/Library/LaunchAgents/$LABEL.plist"
LOG_DIR="$HOME/.localllm/logs"
TOPIC_FILE="$HOME/.localllm/watch/ntfy_url"

log() { echo "[$(date '+%H:%M:%S')] $*"; }

[ -f "$PLIST_SRC" ] || { log "missing $PLIST_SRC"; exit 1; }
# Guard the thing launchd actually does — run it under /usr/bin/python3 — not the
# exec bit, which launchd never consults and which an editor reset broke once.
/usr/bin/python3 -m py_compile "$SCRIPT_DIR/watch-qwen4.py" \
  || { log "watcher does not compile under /usr/bin/python3"; exit 1; }

mkdir -p "$LOG_DIR" "$(dirname "$TOPIC_FILE")"
if [ ! -f "$TOPIC_FILE" ]; then
  log "no push channel: write the full ntfy URL to $TOPIC_FILE (alerts still log to $LOG_DIR)"
fi

# Bootout is expected to fail on a first install; it is not an error worth dying on.
launchctl bootout "gui/$(id -u)/$LABEL" 2>/dev/null || true
cp "$PLIST_SRC" "$PLIST_DST"
launchctl bootstrap "gui/$(id -u)" "$PLIST_DST"
launchctl kickstart -k "gui/$(id -u)/$LABEL" 2>/dev/null || true
sleep 6
LOG="$LOG_DIR/qwen4-watch.log"
# Read the cadence back from the installed artifact rather than restating it: a
# hardcoded "15 min" here was already wrong once by the time it printed.
INTERVAL="$(/usr/libexec/PlistBuddy -c "Print :StartInterval" "$PLIST_DST" 2>/dev/null || echo '?')"
log "installed: $(launchctl print "gui/$(id -u)/$LABEL" 2>/dev/null | grep -E 'state|last exit' | tr -s ' ' | paste -sd' ' -)"
log "first run logged below (exit 0 and no ALERT line = not shipped; alert tier every ${INTERVAL}s, advisory tier every 900s)"
tail -8 "$LOG" 2>/dev/null || log "no log yet at $LOG"
