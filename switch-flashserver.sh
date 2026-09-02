#!/usr/bin/env bash
# Switch the flash server on 30085 between profiles in servers.jsonc:
#   ./switch-flashserver.sh flash-next-offload --yes   # PLE on-direct test (build-direct)
#   ./switch-flashserver.sh flash-next --yes           # vanilla baseline / rollback (build/bin)
# Only ever ONE server on this box (zero RAM headroom; serve's collision guard
# is the backstop). SWITCHING KILLS THE ASSISTANT SESSION - 30085 serves it.
# Run from a real terminal.
set -u
cd ~/localllm

TARGET="${1:-}"
[ "$TARGET" = "flash-next" ] || [ "$TARGET" = "flash-next-offload" ] || {
  echo "usage: $0 <flash-next|flash-next-offload> --yes"; exit 1; }
[ "${2:-}" = "--yes" ] || {
  echo "This stops whatever flash server is running and starts $TARGET (~3 min load)."
  echo "The assistant session served by 30085 dies. Re-run with: $0 $TARGET --yes"
  exit 1; }

LOG="$TARGET"
echo "== resolve check =="
./serve cmd "$TARGET" | tr ' ' '\n' | grep -E 'llama-server$|lazy-mode|on-direct' | head -4

echo "== stopping any flash server =="
./serve stop flash-next flash-next-offload >/dev/null 2>&1
while pgrep -f 'llama-server.*30085' >/dev/null; do sleep 2; done
curl -sf -m 2 http://127.0.0.1:30085/health >/dev/null && { echo "port still answering, aborting"; exit 1; }
sleep 5   # Metal deferred-release grace, per 2026-09-01 incident
vm_stat | awk '/Pages wired down/{gsub(/[^0-9]/,"",$2); printf "wired before start: %.1f GB\n", $2*16384/2^30}'

echo "== starting $TARGET =="
./serve start "$TARGET"   # prints a premature health ✗ during load — ignore, we poll

for i in $(seq 1 120); do
  curl -sf -m 2 http://127.0.0.1:30085/health >/dev/null && { echo "HEALTHY: $TARGET (port 30085)"; HEALTH=1; break; }
  sleep 5
done
if [ "${HEALTH:-0}" != "1" ]; then
  echo "NOT HEALTHY after 10 min. tail $LOG.log; rollback:"
  echo "  ./serve stop $TARGET && ./serve start flash-next"
  exit 1
fi
[ "$TARGET" = "flash-next-offload" ] && grep -m1 'PLE direct read' "$LOG.log"

cat <<'EOF'
== NEXT ==
  1. sampler (fresh label -> its own file):
       ./ple-sample.sh offload | tee ~/localllm/samples-offload.csv     # or baseline
  2. restart opencode — the new session runs on the switched server.
== ROLLBACK ==
  ./switch-flashserver.sh flash-next --yes
EOF
