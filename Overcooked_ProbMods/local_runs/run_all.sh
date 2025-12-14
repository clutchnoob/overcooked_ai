#!/bin/bash
###############################################################################
# RUN ALL PROBABILISTIC MODELS - LOCAL EXECUTION
###############################################################################
# Main orchestration script to run all 6 probabilistic models and the
# inverse planning pipeline on a local machine with parallel job management.
#
# Usage:
#   ./run_all.sh [options]
#
# Options:
#   --stage <name>     Run only specific stage (imitation, rl, inverse_planning)
#   --layout <name>    Run only for specific layout
#   --model <name>     Run only for specific model
#   --collect          Include trajectory collection in inverse planning
#   --skip-rl          Skip RL models (they require env rollouts)
#   --skip-inverse     Skip inverse planning pipeline
#   --dry-run          Print what would be executed without running
#   --help             Show this help message
#
# Stages:
#   1. Imitation Models (bayesian_bc, rational_agent, hierarchical_bc)
#   2. RL Models (bayesian_gail, bayesian_ppo_bc, bayesian_ppo_gail)
#   3. Inverse Planning Pipeline (collect, train, analyze)
#
# Examples:
#   ./run_all.sh                           # Run everything
#   ./run_all.sh --stage imitation         # Run only imitation models
#   ./run_all.sh --layout cramped_room     # Run all models on one layout
#   ./run_all.sh --skip-rl                 # Skip RL models
#   ./run_all.sh --dry-run                 # Preview execution plan
###############################################################################

set -e

# Source configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/config.sh"

###############################################################################
# ARGUMENT PARSING
###############################################################################

STAGE=""
FILTER_LAYOUT=""
FILTER_MODEL=""
COLLECT_TRAJ=false
SKIP_RL=false
SKIP_INVERSE=false
DRY_RUN=false

print_help() {
    head -40 "$0" | tail -35 | sed 's/^# //' | sed 's/^#//'
    exit 0
}

while [[ $# -gt 0 ]]; do
    case $1 in
        --stage)
            STAGE="$2"
            shift 2
            ;;
        --layout)
            FILTER_LAYOUT="$2"
            shift 2
            ;;
        --model)
            FILTER_MODEL="$2"
            shift 2
            ;;
        --collect)
            COLLECT_TRAJ=true
            shift
            ;;
        --skip-rl)
            SKIP_RL=true
            shift
            ;;
        --skip-inverse)
            SKIP_INVERSE=true
            shift
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --help|-h)
            print_help
            ;;
        *)
            error "Unknown option: $1"
            echo "Use --help for usage information."
            exit 1
            ;;
    esac
done

###############################################################################
# EXECUTION PLAN
###############################################################################

print_execution_plan() {
    section "EXECUTION PLAN"
    
    echo "System Information:"
    echo "  Device:          $DEVICE"
    echo "  Max parallel:    $MAX_JOBS jobs"
    echo "  Python:          $(which python)"
    echo ""
    
    echo "Training Configuration:"
    echo "  Layouts:         ${LAYOUTS[*]}"
    echo "  Imitation epochs: $IMITATION_EPOCHS"
    echo "  RL timesteps:    $TIMESTEPS"
    echo ""
    
    echo "Stages to run:"
    if [ -z "$STAGE" ] || [ "$STAGE" = "imitation" ]; then
        echo "  [1] Imitation Models:"
        for model in "${IMITATION_MODELS[@]}"; do
            if [ -z "$FILTER_MODEL" ] || [ "$model" = "$FILTER_MODEL" ]; then
                for layout in "${LAYOUTS[@]}"; do
                    if [ -z "$FILTER_LAYOUT" ] || [ "$layout" = "$FILTER_LAYOUT" ]; then
                        echo "      - $model on $layout"
                    fi
                done
            fi
        done
    fi
    
    if [ "$SKIP_RL" = false ] && ([ -z "$STAGE" ] || [ "$STAGE" = "rl" ]); then
        echo ""
        echo "  [2] RL Models:"
        for model in "${RL_MODELS[@]}"; do
            if [ -z "$FILTER_MODEL" ] || [ "$model" = "$FILTER_MODEL" ]; then
                for layout in "${LAYOUTS[@]}"; do
                    if [ -z "$FILTER_LAYOUT" ] || [ "$layout" = "$FILTER_LAYOUT" ]; then
                        echo "      - $model on $layout"
                    fi
                done
            fi
        done
    fi
    
    if [ "$SKIP_INVERSE" = false ] && ([ -z "$STAGE" ] || [ "$STAGE" = "inverse_planning" ]); then
        echo ""
        echo "  [3] Inverse Planning:"
        echo "      - Collect trajectories: $COLLECT_TRAJ"
        echo "      - Sources: ${INVERSE_PLANNING_TAGS[*]}"
        echo "      - Layouts: ${LAYOUTS[*]}"
    fi
    
    echo ""
}

###############################################################################
# MAIN EXECUTION
###############################################################################

section "LOCAL PROBABILISTIC MODELS TRAINING"
echo "Started: $(date)"
echo ""

# Activate environment
activate_environment

# Print execution plan
print_execution_plan

if [ "$DRY_RUN" = true ]; then
    echo "[DRY RUN MODE - No actual execution]"
    echo ""
fi

# Build filter arguments
FILTER_ARGS=""
[ -n "$FILTER_LAYOUT" ] && FILTER_ARGS="$FILTER_ARGS --layout $FILTER_LAYOUT"
[ -n "$FILTER_MODEL" ] && FILTER_ARGS="$FILTER_ARGS --model $FILTER_MODEL"
[ "$DRY_RUN" = true ] && FILTER_ARGS="$FILTER_ARGS --dry-run"

START_TIME=$(date +%s)

# Stage 1: Imitation Models
if [ -z "$STAGE" ] || [ "$STAGE" = "imitation" ]; then
    log "Starting Stage 1: Imitation Models..."
    "$SCRIPT_DIR/run_stage.sh" imitation $FILTER_ARGS
fi

# Stage 2: RL Models
if [ "$SKIP_RL" = false ] && ([ -z "$STAGE" ] || [ "$STAGE" = "rl" ]); then
    log "Starting Stage 2: RL Models..."
    "$SCRIPT_DIR/run_stage.sh" rl $FILTER_ARGS
fi

# Stage 3: Inverse Planning
if [ "$SKIP_INVERSE" = false ] && ([ -z "$STAGE" ] || [ "$STAGE" = "inverse_planning" ]); then
    log "Starting Stage 3: Inverse Planning..."
    
    IP_ARGS=""
    [ "$COLLECT_TRAJ" = true ] && IP_ARGS="$IP_ARGS --collect"
    [ -n "$FILTER_LAYOUT" ] && IP_ARGS="$IP_ARGS --layout $FILTER_LAYOUT"
    [ "$DRY_RUN" = true ] && IP_ARGS="$IP_ARGS --dry-run"
    
    "$SCRIPT_DIR/run_inverse_planning_local.sh" $IP_ARGS
fi

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
ELAPSED_HOURS=$((ELAPSED / 3600))
ELAPSED_MINS=$(((ELAPSED % 3600) / 60))
ELAPSED_SECS=$((ELAPSED % 60))

###############################################################################
# SUMMARY
###############################################################################

section "ALL STAGES COMPLETE"
echo "Total duration: ${ELAPSED_HOURS}h ${ELAPSED_MINS}m ${ELAPSED_SECS}s"
echo "Finished:       $(date)"
echo ""
echo "Results saved to:"
echo "  $RESULTS_DIR/"
echo ""
echo "Training logs:"
echo "  $LOGS_DIR/"
echo ""

# Generate summary JSON
if [ "$DRY_RUN" = false ]; then
    SUMMARY_FILE="$SCRIPT_DIR/run_summary.json"
    cat > "$SUMMARY_FILE" << EOF
{
  "timestamp": "$(date -Iseconds)",
  "duration_seconds": $ELAPSED,
  "device": "$DEVICE",
  "max_parallel_jobs": $MAX_JOBS,
  "layouts": $(printf '%s\n' "${LAYOUTS[@]}" | jq -R . | jq -s .),
  "models_trained": {
    "imitation": $(printf '%s\n' "${IMITATION_MODELS[@]}" | jq -R . | jq -s .),
    "rl": $(printf '%s\n' "${RL_MODELS[@]}" | jq -R . | jq -s .)
  },
  "inverse_planning_tags": $(printf '%s\n' "${INVERSE_PLANNING_TAGS[@]}" | jq -R . | jq -s .),
  "results_directory": "$RESULTS_DIR",
  "logs_directory": "$LOGS_DIR"
}
EOF
    log "Summary saved to: $SUMMARY_FILE"
fi

