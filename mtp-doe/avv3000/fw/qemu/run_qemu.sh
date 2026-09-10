#!/usr/bin/env bash
# run_qemu.sh — run one fw test ELF under qemu-system-arm (rung r_fw_qemu).
# Pass condition is the RESULT= line (rc=0, run>0, fail=0 — run==0 means the
# suite never executed). The semihosted exit code is only a backstop; a
# watchdog kills a wedged qemu so a broken tree fails instead of hanging
# the ladder.
set -euo pipefail
elf="${1:?usage: run_qemu.sh <elf>}"
WATCHDOG_S="${WATCHDOG_S:-120}"

out="$(mktemp)"
trap 'rm -f "$out"' EXIT

qemu-system-arm -M lm3s6965evb -cpu cortex-m3 -nographic -semihosting \
    -kernel "$elf" >"$out" 2>&1 &
qpid=$!
(
    # spin-watchdog: no coreutils timeout on macOS
    for _ in $(seq "$WATCHDOG_S"); do
        kill -0 "$qpid" 2>/dev/null || exit 0
        sleep 1
    done
    kill -9 "$qpid" 2>/dev/null || true
) &
wpid=$!

rc=0
wait "$qpid" || rc=$?
kill "$wpid" 2>/dev/null || true
wait "$wpid" 2>/dev/null || true

cat "$out"
if grep -q '^RESULT rc=0 run=[1-9][0-9]* fail=0$' "$out"; then
    exit 0
fi
exit 1
