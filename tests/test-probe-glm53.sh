#!/usr/bin/env bash
set -euo pipefail
# tests/test-probe-glm53.sh — plumbing test for probe-glm53.sh against stubs.
# No Metal, no assistant interruption. Exit 0 = probe script logic verified.
SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
STUB_DIR="$(mktemp -d /tmp/glm53-stub.XXXXXX)"
PORT=31999

cleanup() { rm -rf "$STUB_DIR"; pkill -f "glm53_stub_server.py" 2>/dev/null || true; }
trap cleanup EXIT INT TERM

# stub llama-server: becomes /health-ok after 3s, chat returns 200.
# STUB_DIE=1       -> exits immediately (tests dead-process abort in wait_ready)
# STUB_DIE_NCMOE=N -> exits immediately only when '--n-cpu-moe N' is in argv
#                     (tests that one failed attempt does not abort the ladder)
cat > "$STUB_DIR/fake-llama-server" <<EOF
#!/usr/bin/env bash
[ "\${STUB_DIE:-0}" = "1" ] && { echo "error: simulated early death" >&2; exit 1; }
exec python3 "$STUB_DIR/glm53_stub_server.py" "\$@"
EOF
chmod +x "$STUB_DIR/fake-llama-server"

cat > "$STUB_DIR/glm53_stub_server.py" <<'EOF'
#!/usr/bin/env python3
"""Stub llama-server: /health ok after 3s, chat 200. Ignores model/args."""
import http.server, json, os, re, sys, time

START = time.time()

if os.environ.get("STUB_DIE_NCMOE", "") and \
        f"--n-cpu-moe {os.environ['STUB_DIE_NCMOE']}" in " ".join(sys.argv):
    print("error: simulated death for this n-cpu-moe", file=sys.stderr)
    sys.exit(1)

class H(http.server.BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def do_GET(self):
        ok = time.time() - START > 3
        body = json.dumps({"status": "ok" if ok else "loading"}).encode()
        self.send_response(200 if ok else 503)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(body)
    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0)); self.rfile.read(n)
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps({"choices": [{"message": {"content": "pong"}}],
                                     "usage": {"completion_tokens": 16}}).encode())

port = 31999
m = re.search(r"--port (\d+)", " ".join(sys.argv))
if m: port = int(m.group(1))
print("stub: fake layout line offloaded 46/47 layers KV buffer 512MiB")
print("stub: print_timing: eval time = 1000 ms / 16 tokens = 62.50 ms per token, 16.00 tokens per second")
http.server.HTTPServer(("127.0.0.1", port), H).serve_forever()
EOF

# stub serve: record invocations, succeed
cat > "$STUB_DIR/fake-serve" <<EOF
#!/usr/bin/env bash
echo "stub-serve \$*" >> "$STUB_DIR/serve.log"
EOF
chmod +x "$STUB_DIR/fake-serve"

fail() { echo "TEST FAIL: $*" >&2; exit 1; }

echo "== case 1: success ladder (two attempts) =="
LOG="$STUB_DIR/logs1"; mkdir -p "$LOG"
env SERVER_BIN="$STUB_DIR/fake-llama-server" MODEL="/tmp/whatever.gguf" \
    SERVE_BIN="$STUB_DIR/fake-serve" PORT=$PORT \
    LOG_DIR="$LOG" RESULT="$LOG/result.md" \
    LOAD_TIMEOUT=60 PING_TIMEOUT=30 START_GRACE=0 DRAIN_SECONDS=0 \
    ATTEMPTS="a1:12:32768 a2:6:131072" \
    ASSIST_PORT=$((PORT + 9)) RESTORE_TIMEOUT=10 \
    "$SCRIPT_DIR/probe-glm53.sh" > "$LOG/run.out" 2>&1 \
    || fail "case1: probe exited nonzero"
grep -q "PASS: a1 " "$LOG/result.md" || fail "case1: a1 not PASS"
grep -q "PASS: a2 " "$LOG/result.md" || fail "case1: a2 not PASS"
grep -q "n-cpu-moe=12, ctx=32768" "$LOG/result.md" || fail "case1: params not recorded"
grep -q "stub-serve stop flash-next-mtp" "$STUB_DIR/serve.log" || fail "case1: no stop call"
grep -q "stub-serve start flash-next-mtp" "$STUB_DIR/serve.log" || fail "case1: no start call"
# no stub serving ASSIST_PORT here -> restore is attempted, fails quickly
grep -q "RESTORE FAILED" "$LOG/result.md" || fail "case1: restore not attempted"
grep -q "first good: a1" "$LOG/result.md" || fail "case1: first-good verdict missing"
echo "  ok"

echo "== case 2: server dies instantly -> must abort fast, fail cleanly =="
LOG2="$STUB_DIR/logs2"; mkdir -p "$LOG2"
t0=$(date +%s)
env STUB_DIE=1 \
    SERVER_BIN="$STUB_DIR/fake-llama-server" MODEL="/tmp/whatever.gguf" \
    SERVE_BIN="$STUB_DIR/fake-serve" PORT=$((PORT + 1)) \
    LOG_DIR="$LOG2" RESULT="$LOG2/result.md" \
    LOAD_TIMEOUT=300 PING_TIMEOUT=30 START_GRACE=0 DRAIN_SECONDS=0 \
    ATTEMPTS="a1:12:32768" \
    ASSIST_PORT=$((PORT + 9)) RESTORE_TIMEOUT=10 \
    "$SCRIPT_DIR/probe-glm53.sh" > "$LOG2/run.out" 2>&1 || true
dt=$(( $(date +%s) - t0 ))
[ "$dt" -lt 90 ] || fail "case2: took ${dt}s — fail-fast broken"
grep -q "FAIL: a1 " "$LOG2/result.md" || fail "case2: FAIL not recorded"
grep -q "Verdict: fail" "$LOG2/result.md" || fail "case2: verdict wrong"
echo "  ok (aborted in ${dt}s)"

echo "== case 3: healthy assistant restore is actually detected =="
LOG3="$STUB_DIR/logs3"; mkdir -p "$LOG3"
APORT=$((PORT + 2))
python3 "$STUB_DIR/glm53_stub_server.py" --port "$APORT" > /dev/null 2>&1 &
ASTUB=$!
sleep 4  # stub self-healthy after 3s
env SERVER_BIN="$STUB_DIR/fake-llama-server" MODEL="/tmp/whatever.gguf" \
    SERVE_BIN="$STUB_DIR/fake-serve" PORT=$((PORT + 3)) \
    LOG_DIR="$LOG3" RESULT="$LOG3/result.md" \
    LOAD_TIMEOUT=60 PING_TIMEOUT=30 START_GRACE=0 DRAIN_SECONDS=0 \
    ATTEMPTS="a1:12:32768" \
    ASSIST_PORT="$APORT" RESTORE_TIMEOUT=30 \
    "$SCRIPT_DIR/probe-glm53.sh" > "$LOG3/run.out" 2>&1 \
    || fail "case3: probe exited nonzero"
grep -q "RESTORED and healthy" "$LOG3/result.md" || fail "case3: restore not detected"
kill "$ASTUB" 2>/dev/null || true
echo "  ok"

echo "== case 4: one failed attempt does NOT abort the ladder =="
LOG4="$STUB_DIR/logs4"; mkdir -p "$LOG4"
env STUB_DIE_NCMOE=12 \
    SERVER_BIN="$STUB_DIR/fake-llama-server" MODEL="/tmp/whatever.gguf" \
    SERVE_BIN="$STUB_DIR/fake-serve" PORT=$((PORT + 4)) \
    LOG_DIR="$LOG4" RESULT="$LOG4/result.md" \
    LOAD_TIMEOUT=60 PING_TIMEOUT=30 START_GRACE=0 DRAIN_SECONDS=0 \
    ATTEMPTS="bad:12:32768 good:6:32768" \
    ASSIST_PORT=$((PORT + 9)) RESTORE_TIMEOUT=10 \
    "$SCRIPT_DIR/probe-glm53.sh" > "$LOG4/run.out" 2>&1 || true
grep -q "FAIL: bad " "$LOG4/result.md" || fail "case4: bad attempt not recorded as FAIL"
grep -q "PASS: good " "$LOG4/result.md" || fail "case4: ladder aborted after failure"
grep -q "first good: good" "$LOG4/result.md" || fail "case4: verdict missed recovery"
echo "  ok"

echo "test-probe-glm53: ALL PASS"
