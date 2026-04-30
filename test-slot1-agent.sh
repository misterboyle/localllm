#!/usr/bin/env bash
# Slot 1 agent: builds context through data science analysis
# Different conversation to ensure separate slot
set -euo pipefail

SESSION="${1:-slot1-test}"
WORKDIR="${HOME}/localllm-test-slot1"
mkdir -p "$WORKDIR"

echo "[SLOT1] Starting session '$SESSION' in $WORKDIR"

# Seed file
cat > "$WORKDIR/README.md" << 'SEED'
This is a test workspace for Slot 1 memory scaling. We will build a data analysis pipeline.
SEED

# Initial prompt - very different from Slot 0 to ensure separate slot
opencode run \
  -m mlx-dense/qwen3.6-27b \
  -s "$SESSION" \
  --dir "$WORKDIR" \
  --title "Slot 1: Data Analysis Pipeline" \
  "We're building a data analysis pipeline in Python for analyzing server performance metrics. The pipeline should:
   1. Read CSV files containing timestamped CPU, memory, disk I/O, and network metrics
   2. Calculate rolling averages (5min, 15min, 1hr windows)
   3. Detect anomalies using a simple statistical method (z-score with configurable threshold)
   4. Generate a summary report with key findings
   5. Output results as both human-readable text and JSON
   Start by writing a sample CSV dataset generator that creates realistic server metrics data for 30 days, then write the analysis pipeline. Write both files to disk."

echo "[SLOT1] Initial prompt sent. Waiting for completion..."
sleep 2

# Follow-up prompts
follow_ups=(
  "Now add visualization support. Generate ASCII art charts for the rolling averages and anomaly detections. Include: line charts for CPU and memory over time, a histogram of anomaly distribution by hour of day, and a heat map showing which metrics are most correlated with anomalies. Make the ASCII output colorized for terminal display using ANSI codes."
  
  "Add a comparison mode. The pipeline should be able to load multiple CSV files (e.g., from different servers or different time periods) and compare them. Calculate correlation coefficients between datasets, identify diverging trends, and highlight periods where one server behaved differently from the others. Add a --compare flag that takes multiple input files."
  
  "Now let's add predictive capabilities. Implement a simple exponential smoothing model (Holt-Winters with seasonality) to forecast the next 7 days of metrics. The model should handle daily and weekly seasonality patterns. Include confidence intervals and a 'prediction vs actual' evaluation mode that measures forecast accuracy using MAE, RMSE, and MAPE metrics."
  
  "Add a dashboard export. Generate a complete HTML file with embedded CSS and JavaScript (using Chart.js CDN) that creates an interactive dashboard. The dashboard should include: time series charts with zoom/pan, anomaly markers, comparison views between servers, forecast visualizations with confidence bands, and a summary panel with key metrics. The HTML should be self-contained and work offline."
  
  "Implement a streaming mode. Instead of loading entire CSV files, add a mode that reads data line-by-line and updates statistics incrementally. This should support real-time monitoring where data is appended to the CSV continuously. Use a sliding window approach to maintain rolling statistics without keeping all historical data in memory. Add a --stream flag for this mode."
  
  "Write a comprehensive test suite. Create tests that: 1) Verify the statistical calculations against known correct values, 2) Test anomaly detection with synthetic data containing known anomalies, 3) Test the prediction model's accuracy on seasonal data, 4) Test edge cases (empty files, single-row files, all-zero data, negative values), 5) Benchmark performance with datasets of increasing size (1K, 10K, 100K, 1M rows) and plot scaling behavior."
  
  "Now add a configuration-driven alerting system. Define alert rules in a YAML config file that specify: metric name, operator (gt, lt, gte, lte), threshold value, evaluation window, and notification channels (stdout, file, webhook URL). Implement a rule engine that evaluates all rules against the latest data point and fires alerts. Add cooldown periods to prevent alert fatigue and deduplication logic."
  
  "Create a deployment guide. Write a comprehensive document covering: installation from source, Docker containerization, systemd service setup, configuration management for multi-server environments, monitoring the monitoring system itself, and a runbook for common issues. Include examples of integrating with popular alerting platforms (Slack, PagerDuty, email) and a sample configuration for a 50-server fleet."
)

for i in "${!follow_ups[@]}"; do
  echo "[SLOT1] Sending follow-up $((i+1)) of ${#follow_ups[@]}..."
  
  # Take memory snapshot before
  bash "$HOME/localllm/snapshot-memory.sh" "slot1-followup-$((i+1))-before" 2>/dev/null || true
  
  opencode run \
    -m mlx-dense/qwen3.6-27b \
    -s "$SESSION" \
    --dir "$WORKDIR" \
    "${follow_ups[$i]}"
  
  # Wait for the response
  sleep 3
  
  # Take memory snapshot after
  bash "$HOME/localllm/snapshot-memory.sh" "slot1-followup-$((i+1))-after" 2>/dev/null || true
  
  echo "[SLOT1] Follow-up $((i+1)) complete."
done

echo "[SLOT1] All follow-ups complete. Session: $SESSION"
bash "$HOME/localllm/snapshot-memory.sh" "slot1-final" 2>/dev/null || true
