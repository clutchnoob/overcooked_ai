#!/bin/bash
###############################################################################
# JOB QUEUE MANAGER FOR LOCAL PARALLEL EXECUTION
###############################################################################
# Provides functions for managing background jobs with a configurable limit.
# Source this file in other scripts: source "$(dirname "$0")/job_queue.sh"
#
# Compatible with bash 3.x (macOS default) and bash 4+
###############################################################################

# Source configuration if not already loaded
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -z "$MAX_JOBS" ]; then
    source "$SCRIPT_DIR/config.sh"
fi

###############################################################################
# JOB TRACKING (bash 3.x compatible - using indexed arrays and files)
###############################################################################

# Array to track background job PIDs
JOB_PIDS=()

# File-based tracking for job names and times
JOB_TRACKER_FILE="$LOGS_DIR/.job_tracker"
JOB_NAMES_FILE="$LOGS_DIR/.job_names"
JOB_TIMES_FILE="$LOGS_DIR/.job_times"

# Initialize tracking files
init_tracker() {
    rm -f "$JOB_TRACKER_FILE" "$JOB_NAMES_FILE" "$JOB_TIMES_FILE"
    touch "$JOB_TRACKER_FILE" "$JOB_NAMES_FILE" "$JOB_TIMES_FILE"
}

# Get job name by PID
get_job_name() {
    local pid="$1"
    grep "^${pid}:" "$JOB_NAMES_FILE" 2>/dev/null | cut -d: -f2
}

# Save job name for PID
save_job_name() {
    local pid="$1"
    local name="$2"
    echo "${pid}:${name}" >> "$JOB_NAMES_FILE"
}

# Get job start time by PID
get_job_time() {
    local pid="$1"
    grep "^${pid}:" "$JOB_TIMES_FILE" 2>/dev/null | cut -d: -f2
}

# Save job start time for PID
save_job_time() {
    local pid="$1"
    local time="$2"
    echo "${pid}:${time}" >> "$JOB_TIMES_FILE"
}

###############################################################################
# CORE FUNCTIONS
###############################################################################

# Get count of currently running jobs from our tracked PIDs
get_running_count() {
    local count=0
    local new_pids=()
    
    for pid in "${JOB_PIDS[@]}"; do
        if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
            ((count++))
            new_pids+=("$pid")
        fi
    done
    
    # Update the array to only include running PIDs
    JOB_PIDS=("${new_pids[@]}")
    echo "$count"
}

# Wait until there's a slot available (fewer than MAX_JOBS running)
wait_for_slot() {
    local running
    running=$(get_running_count)
    
    while [ "$running" -ge "$MAX_JOBS" ]; do
        sleep "$POLL_INTERVAL"
        running=$(get_running_count)
    done
}

# Wait for all tracked jobs to complete
wait_all() {
    if [ ${#JOB_PIDS[@]} -eq 0 ]; then
        log "No jobs to wait for."
        return
    fi
    
    log "Waiting for ${#JOB_PIDS[@]} jobs to complete..."
    
    for pid in "${JOB_PIDS[@]}"; do
        if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
            wait "$pid" 2>/dev/null || true
        fi
    done
    
    JOB_PIDS=()
    log "All jobs completed."
}

# Launch a job in the background
# Usage: launch_job "job_name" command [args...]
launch_job() {
    local job_name="$1"
    shift
    local cmd=("$@")
    
    # Wait for a slot
    wait_for_slot
    
    # Launch the job
    log "Starting: $job_name"
    "${cmd[@]}" &
    local pid=$!
    
    # Track the job
    JOB_PIDS+=("$pid")
    save_job_name "$pid" "$job_name"
    save_job_time "$pid" "$(date +%s)"
    
    # Save to tracker file
    echo "$pid $job_name $(date +%s)" >> "$JOB_TRACKER_FILE"
    
    log "  PID: $pid, Running: $(get_running_count)/$MAX_JOBS"
}

# Check status of a specific job by PID
check_job() {
    local pid="$1"
    if kill -0 "$pid" 2>/dev/null; then
        echo "running"
    elif wait "$pid" 2>/dev/null; then
        echo "completed"
    else
        echo "failed"
    fi
}

# Print summary of job queue
print_queue_status() {
    local running
    running=$(get_running_count)
    
    section "JOB QUEUE STATUS"
    echo "Running: $running / $MAX_JOBS"
    echo ""
    
    if [ ${#JOB_PIDS[@]} -gt 0 ]; then
        echo "Active Jobs:"
        for pid in "${JOB_PIDS[@]}"; do
            if [ -n "$pid" ]; then
                local name
                name=$(get_job_name "$pid")
                name="${name:-unknown}"
                local start
                start=$(get_job_time "$pid")
                start="${start:-0}"
                local now
                now=$(date +%s)
                local elapsed=$((now - start))
                local mins=$((elapsed / 60))
                local secs=$((elapsed % 60))
                printf "  PID %-6s: %-40s (%dm %ds)\n" "$pid" "$name" "$mins" "$secs"
            fi
        done
    else
        echo "No active jobs."
    fi
}

# Clean up job tracker files
cleanup_tracker() {
    rm -f "$JOB_TRACKER_FILE" "$JOB_NAMES_FILE" "$JOB_TIMES_FILE"
    JOB_PIDS=()
}

###############################################################################
# BATCH JOB FUNCTIONS
###############################################################################

# Run a batch of jobs with parallelization
# Usage: run_batch "stage_name" job_specs...
# Where job_spec is "name|command"
run_batch() {
    local stage_name="$1"
    shift
    local jobs=("$@")
    local total=${#jobs[@]}
    
    section "STAGE: $stage_name"
    log "Total jobs: $total, Max parallel: $MAX_JOBS"
    
    for job_spec in "${jobs[@]}"; do
        # Parse job specification (name|command)
        local job_name="${job_spec%%|*}"
        local job_cmd="${job_spec#*|}"
        
        launch_job "$job_name" bash -c "$job_cmd"
    done
    
    # Wait for all jobs in this batch
    wait_all
    
    log "Stage '$stage_name' complete."
}

###############################################################################
# SIGNAL HANDLING
###############################################################################

# Cleanup function for graceful shutdown
cleanup_jobs() {
    log "Received interrupt signal. Cleaning up..."
    
    for pid in "${JOB_PIDS[@]}"; do
        if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
            local name
            name=$(get_job_name "$pid")
            log "  Terminating PID $pid (${name:-unknown})"
            kill "$pid" 2>/dev/null || true
        fi
    done
    
    cleanup_tracker
    log "Cleanup complete."
    exit 1
}

# Set up signal handlers
trap cleanup_jobs SIGINT SIGTERM

###############################################################################
# INITIALIZATION
###############################################################################

# Initialize tracker files
init_tracker
