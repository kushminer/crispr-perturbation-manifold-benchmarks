#!/bin/bash
# Continuous monitoring script - runs status check every 30 minutes
# Press Ctrl+C to stop

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

INTERVAL_MINUTES=30
LOG_FILE="status_updates.log"

echo "Starting continuous monitoring..."
echo "Checking every $INTERVAL_MINUTES minutes"
echo "Log file: $LOG_FILE"
echo "Press Ctrl+C to stop"
echo ""

# Function to log with timestamp
log_status() {
    {
        echo ""
        echo "============================================================"
        echo "Status Update - $(date '+%Y-%m-%d %H:%M:%S')"
        echo "============================================================"
        ./monitor_status.sh
    } | tee -a "$LOG_FILE"
}

# Initial status
log_status

# Loop every 30 minutes
while true; do
    sleep $((INTERVAL_MINUTES * 60))
    log_status
done

