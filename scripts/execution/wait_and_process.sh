#!/bin/bash
# Wait for diagnostic suite to complete, then run post-completion workflow

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

LOG_FILE="full_diagnostic_suite_run.log"
MAX_WAIT_HOURS=24
CHECK_INTERVAL=300  # 5 minutes

echo "============================================================"
echo "Waiting for Diagnostic Suite Completion"
echo "============================================================"
echo ""
echo "Monitoring: $LOG_FILE"
echo "Check interval: $CHECK_INTERVAL seconds (5 minutes)"
echo "Max wait: $MAX_WAIT_HOURS hours"
echo ""

# Function to check if execution is complete
check_completion() {
    # Check if log file ends with completion message
    if tail -20 "$LOG_FILE" 2>/dev/null | grep -q "ALL EPICS COMPLETE\|Complete\|Finished"; then
        return 0
    fi
    
    # Check if no active processes
    active_count=$(ps aux | grep -E "run_all_epics|curvature_sweep|run_epic3|run_epic2" | grep -v grep | wc -l | tr -d ' ')
    if [ "$active_count" -eq 0 ]; then
        # Double-check by looking at recent log activity
        last_activity=$(stat -f "%m" "$LOG_FILE" 2>/dev/null || echo "0")
        current_time=$(date +%s)
        time_since_activity=$((current_time - last_activity))
        
        # If log hasn't been updated in 10 minutes, assume complete
        if [ "$time_since_activity" -gt 600 ]; then
            return 0
        fi
    fi
    
    return 1
}

# Wait for completion
wait_count=0
while [ $wait_count -lt $((MAX_WAIT_HOURS * 3600 / CHECK_INTERVAL)) ]; do
    if check_completion; then
        echo ""
        echo "============================================================"
        echo "✅ Diagnostic Suite Execution Complete!"
        echo "============================================================"
        echo ""
        
        # Run post-completion workflow
        echo "Running post-completion workflow..."
        bash post_completion_workflow.sh
        
        exit 0
    fi
    
    # Show progress
    wait_count=$((wait_count + 1))
    elapsed_minutes=$((wait_count * CHECK_INTERVAL / 60))
    
    echo "[$(date +%H:%M:%S)] Waiting... (${elapsed_minutes} minutes elapsed)"
    echo "   Current progress:"
    ./monitor_progress.sh 2>/dev/null | grep "Overall Progress" || echo "   Checking..."
    
    sleep $CHECK_INTERVAL
done

echo ""
echo "⚠️  Maximum wait time reached. Checking final status..."
./monitor_progress.sh
echo ""
echo "Running post-completion workflow anyway..."
bash post_completion_workflow.sh

