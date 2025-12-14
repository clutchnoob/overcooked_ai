#!/bin/bash
###############################################################################
# CONFIGURATION FOR LOCAL PROBABILISTIC MODELS TRAINING
###############################################################################
# This file contains all configurable settings for running ProbMods locally.
# Source this file in other scripts: source "$(dirname "$0")/config.sh"
###############################################################################

# Detect script directory and project paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
OVERCOOKED_ROOT="$(cd "$PROJECT_DIR/.." && pwd)"

# Export paths
export SCRIPT_DIR
export PROJECT_DIR
export OVERCOOKED_ROOT
export PYTHONPATH="${OVERCOOKED_ROOT}/src:${PROJECT_DIR}:${PYTHONPATH}"

###############################################################################
# PARALLELIZATION SETTINGS
###############################################################################

# Maximum number of concurrent background jobs
MAX_JOBS=4

# Polling interval for job queue (seconds)
POLL_INTERVAL=5

###############################################################################
# TRAINING CONFIGURATION
###############################################################################

# Layouts with human demonstration data
LAYOUTS=(
    "cramped_room"
    "asymmetric_advantages"
    "coordination_ring"
)

# All probabilistic models
IMITATION_MODELS=(
    "bayesian_bc"
    "rational_agent"
    "hierarchical_bc"
)

RL_MODELS=(
    "bayesian_gail"
    "bayesian_ppo_bc"
    "bayesian_ppo_gail"
)

ALL_MODELS=("${IMITATION_MODELS[@]}" "${RL_MODELS[@]}")

# Training parameters
IMITATION_EPOCHS=500
RL_TIMESTEPS=200000

# Inverse planning sources
INVERSE_PLANNING_TAGS=(
    "human_demo"
    "ppo_bc"
    "ppo_gail"
)

###############################################################################
# DEVICE DETECTION (Mac MPS / CUDA / CPU)
###############################################################################

detect_device() {
    python3 -c "
import torch
if torch.cuda.is_available():
    print('cuda')
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    print('mps')
else:
    print('cpu')
" 2>/dev/null || echo "cpu"
}

# Set device (can be overridden by environment variable)
if [ -z "$DEVICE" ]; then
    DEVICE=$(detect_device)
fi
export DEVICE

# Enable MPS fallback for operations not yet supported
export PYTORCH_ENABLE_MPS_FALLBACK=1

###############################################################################
# ENVIRONMENT ACTIVATION
###############################################################################

# Conda environment path for this project
CONDA_ENV_PATH="/Users/mahmoudabdelmoneum/Desktop/MIT/Software/Research_Software/conda_envs/CogSciFinalProj"

# Function to activate Python environment
activate_environment() {
    # Activate the CogSciFinalProj conda environment
    if [ -f "$HOME/opt/anaconda3/etc/profile.d/conda.sh" ]; then
        source "$HOME/opt/anaconda3/etc/profile.d/conda.sh"
    elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
        source "$HOME/anaconda3/etc/profile.d/conda.sh"
    elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
        source "$HOME/miniconda3/etc/profile.d/conda.sh"
    fi
    
    # Activate the project environment
    conda activate "$CONDA_ENV_PATH"
    
    # Verify Python is available
    if ! command -v python &> /dev/null; then
        echo "ERROR: Python not found. Please check CONDA_ENV_PATH in config.sh"
        exit 1
    fi
}

###############################################################################
# LOGGING CONFIGURATION
###############################################################################

LOGS_DIR="$SCRIPT_DIR/logs"
RESULTS_DIR="$PROJECT_DIR/results"

# Create directories if they don't exist
mkdir -p "$LOGS_DIR"
mkdir -p "$RESULTS_DIR"

# Generate timestamp for log files
get_timestamp() {
    date +"%Y%m%d_%H%M%S"
}

# Get log file path for a job
get_log_path() {
    local model="$1"
    local layout="$2"
    local timestamp="${3:-$(get_timestamp)}"
    echo "$LOGS_DIR/${model}_${layout}_${timestamp}.log"
}

###############################################################################
# UTILITY FUNCTIONS
###############################################################################

# Print with timestamp
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

# Print error message
error() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: $*" >&2
}

# Print section header
section() {
    echo ""
    echo "=============================================================="
    echo "$*"
    echo "=============================================================="
}

# Check if a command exists
command_exists() {
    command -v "$1" &> /dev/null
}

###############################################################################
# INITIALIZATION
###############################################################################

# Print configuration info (only when sourced with --info flag)
if [ "${1:-}" = "--info" ]; then
    section "LOCAL PROBMODS CONFIGURATION"
    echo "Script Directory:  $SCRIPT_DIR"
    echo "Project Directory: $PROJECT_DIR"
    echo "Overcooked Root:   $OVERCOOKED_ROOT"
    echo "Device:            $DEVICE"
    echo "Max Parallel Jobs: $MAX_JOBS"
    echo "Layouts:           ${LAYOUTS[*]}"
    echo "Imitation Models:  ${IMITATION_MODELS[*]}"
    echo "RL Models:         ${RL_MODELS[*]}"
    echo "Logs Directory:    $LOGS_DIR"
    echo "Results Directory: $RESULTS_DIR"
fi

