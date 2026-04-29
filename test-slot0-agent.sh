#!/usr/bin/env bash
# Slot 0 agent: builds context through incremental code review + refactoring
# Uses opencode run -s to keep appending to the same session
set -euo pipefail

SESSION="${1:-slot0-test}"
WORKDIR="${HOME}/localllm-test-slot0"
mkdir -p "$WORKDIR"

echo "[SLOT0] Starting session '$SESSION' in $WORKDIR"

# Seed file for the session
cat > "$WORKDIR/README.md" << 'SEED'
This is a test workspace for Slot 0 memory scaling. We will build a small CLI tool incrementally.
SEED

# Initial prompt - creates the foundation
opencode run \
  -m llama-dense/qwen3.6-27b \
  -s "$SESSION" \
  --dir "$WORKDIR" \
  --title "Slot 0: Incremental CLI Builder" \
  "We're building a small CLI tool called 'logrotate-lite' in bash. It should:
   1. Take a directory path as input
   2. Find all .log files older than N days (default 7)
   3. Compress them with gzip
   4. Delete compressed files older than M days (default 30)
   5. Have a dry-run mode
   Start by writing the main script with full comments and argument parsing. Then create a test script that simulates the directory structure. Write both files to disk."

echo "[SLOT0] Initial prompt sent. Waiting for completion..."
sleep 2

# Follow-up prompts that each add ~2-5K tokens of context
follow_ups=(
  "Great, now add a configuration file format. The script should read from a YAML-like config file that specifies: directories to scan, age thresholds, compression level, and retention policy. Parse this config and use it instead of hard-coded defaults. Show the config file format and update the script."
  
  "Now add a 'report' mode that generates a JSON summary of what was found, what would be rotated, and disk space saved. Include per-directory breakdowns and a total summary. The report should be written to stdout or a file path specified by --report-path. Also add a --json flag to make all output machine-readable."
  
  "Add a scheduling mechanism. The script should be able to generate a crontab entry for itself, or run in 'daemon' mode that checks every N minutes. Implement the daemon mode with a PID file, signal handling (SIGTERM for clean shutdown, SIGHUP for reload config), and a status command that shows when it last ran and how much it rotated. Update the argument parser and main loop."
  
  "Now let's add error handling and logging. Create a proper logging subsystem with levels (DEBUG, INFO, WARN, ERROR) that can write to both stdout and a log file. Add retry logic for failed compressions (up to 3 retries with exponential backoff). Add a lock file mechanism to prevent multiple instances from running simultaneously. Update all existing functions to use the new logging system."
  
  "Write comprehensive documentation. Create a detailed README.md that covers: installation, configuration, usage examples, cron setup, daemon mode, troubleshooting, and a comparison with the system logrotate. Include a changelog and a contribution guide with development setup instructions. Make it professional-quality documentation."
  
  "Now let's add integration tests. Create a test harness that: 1) Sets up temporary directories with controlled file ages using 'touch -t', 2) Runs the script in dry-run mode and verifies output matches expected results, 3) Runs the script in live mode and verifies files were rotated, 4) Tests edge cases (empty directories, permission errors, symlinks, very long filenames), 5) Measures execution time and verifies it's under a threshold. Make the test suite runnable with 'bash test.sh'."
  
  "Refactor the script to be modular. Split it into separate functions in a library file (lib.sh) that can be sourced by the main script and the test harness. Organize the code into logical modules: config.sh for configuration parsing, rotate.sh for the core rotation logic, report.sh for reporting, and daemon.sh for daemon mode. Each module should have its own test file. Show the directory structure and the first module (config.sh)."
  
  "Add a web dashboard for monitoring. Create a simple HTTP server in bash (using netcat) that serves a minimal HTML page showing: current status of the rotation daemon, last run results, disk space saved over time (stored in a simple text-based time series file), and the ability to trigger a manual rotation or view the configuration. Keep it minimal but functional."
)

for i in "${!follow_ups[@]}"; do
  echo "[SLOT0] Sending follow-up $((i+1)) of ${#follow_ups[@]}..."
  
  # Take memory snapshot before
  bash "$HOME/localllm/snapshot-memory.sh" "slot0-followup-$((i+1))-before" 2>/dev/null || true
  
  opencode run \
    -m llama-dense/qwen3.6-27b \
    -s "$SESSION" \
    --dir "$WORKDIR" \
    "${follow_ups[$i]}"
  
  # Wait for the response to be generated
  sleep 3
  
  # Take memory snapshot after
  bash "$HOME/localllm/snapshot-memory.sh" "slot0-followup-$((i+1))-after" 2>/dev/null || true
  
  echo "[SLOT0] Follow-up $((i+1)) complete."
done

echo "[SLOT0] All follow-ups complete. Session: $SESSION"
bash "$HOME/localllm/snapshot-memory.sh" "slot0-final" 2>/dev/null || true
