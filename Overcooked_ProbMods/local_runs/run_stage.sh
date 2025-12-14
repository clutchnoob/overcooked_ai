#!/bin/bash
###############################################################################
# RUN TRAINING STAGE
###############################################################################
# Run a specific stage of training with parallel job management.
#
# Usage:
#   ./run_stage.sh <stage> [options]
#
# Stages:
#   imitation       - Train imitation models (bayesian_bc, rational_agent, hierarchical_bc)
#   rl              - Train RL models (bayesian_gail, bayesian_ppo_bc, bayesian_ppo_gail)
#   inverse_planning - Run inverse planning pipeline
#   all             - Run all stages sequentially
#
# Options:
#   --layout <name>  - Run only for specific layout
#   --model <name>   - Run only for specific model
#   --dry-run        - Print what would be executed without running
###############################################################################

set -e

# Source configuration and job queue
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/config.sh"
source "$SCRIPT_DIR/job_queue.sh"

###############################################################################
# ARGUMENT PARSING
###############################################################################

STAGE="$1"
shift || true

# Optional filters
FILTER_LAYOUT=""
FILTER_MODEL=""
DRY_RUN=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --layout)
            FILTER_LAYOUT="$2"
            shift 2
            ;;
        --model)
            FILTER_MODEL="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        *)
            error "Unknown option: $1"
            exit 1
            ;;
    esac
done

if [ -z "$STAGE" ]; then
    echo "Usage: $0 <stage> [--layout <name>] [--model <name>] [--dry-run]"
    echo ""
    echo "Stages:"
    echo "  imitation        - Train imitation models"
    echo "  rl               - Train RL models"
    echo "  inverse_planning - Run inverse planning pipeline"
    echo "  all              - Run all stages"
    exit 1
fi

###############################################################################
# STAGE EXECUTION FUNCTIONS
###############################################################################

run_imitation_stage() {
    section "STAGE: IMITATION MODELS"
    log "Models: ${IMITATION_MODELS[*]}"
    log "Layouts: ${LAYOUTS[*]}"
    log "Max parallel: $MAX_JOBS"
    echo ""
    
    local job_count=0
    
    for model in "${IMITATION_MODELS[@]}"; do
        # Apply model filter
        if [ -n "$FILTER_MODEL" ] && [ "$model" != "$FILTER_MODEL" ]; then
            continue
        fi
        
        for layout in "${LAYOUTS[@]}"; do
            # Apply layout filter
            if [ -n "$FILTER_LAYOUT" ] && [ "$layout" != "$FILTER_LAYOUT" ]; then
                continue
            fi
            
            local job_name="${model}_${layout}"
            local job_cmd="$SCRIPT_DIR/train_model.sh $model $layout"
            
            if [ "$DRY_RUN" = true ]; then
                echo "[DRY RUN] Would launch: $job_name"
                echo "  Command: $job_cmd"
            else
                launch_job "$job_name" $job_cmd
            fi
            
            ((job_count++))
        done
    done
    
    if [ "$DRY_RUN" = false ]; then
        wait_all
    fi
    
    log "Imitation stage complete. Total jobs: $job_count"
}

run_rl_stage() {
    section "STAGE: RL MODELS"
    log "Models: ${RL_MODELS[*]}"
    log "Layouts: ${LAYOUTS[*]}"
    log "Max parallel: $MAX_JOBS"
    echo ""
    
    local job_count=0
    
    for model in "${RL_MODELS[@]}"; do
        # Apply model filter
        if [ -n "$FILTER_MODEL" ] && [ "$model" != "$FILTER_MODEL" ]; then
            continue
        fi
        
        for layout in "${LAYOUTS[@]}"; do
            # Apply layout filter
            if [ -n "$FILTER_LAYOUT" ] && [ "$layout" != "$FILTER_LAYOUT" ]; then
                continue
            fi
            
            local job_name="${model}_${layout}"
            local job_cmd="$SCRIPT_DIR/train_model.sh $model $layout"
            
            if [ "$DRY_RUN" = true ]; then
                echo "[DRY RUN] Would launch: $job_name"
                echo "  Command: $job_cmd"
            else
                launch_job "$job_name" $job_cmd
            fi
            
            ((job_count++))
        done
    done
    
    if [ "$DRY_RUN" = false ]; then
        wait_all
    fi
    
    log "RL stage complete. Total jobs: $job_count"
}

run_inverse_planning_stage() {
    section "STAGE: INVERSE PLANNING"
    
    if [ "$DRY_RUN" = true ]; then
        echo "[DRY RUN] Would run inverse planning pipeline"
        echo "  Command: $SCRIPT_DIR/run_inverse_planning_local.sh"
    else
        "$SCRIPT_DIR/run_inverse_planning_local.sh"
    fi
}

###############################################################################
# MAIN EXECUTION
###############################################################################

section "LOCAL PROBMODS TRAINING"
echo "Stage:      $STAGE"
echo "Device:     $DEVICE"
echo "Max Jobs:   $MAX_JOBS"
[ -n "$FILTER_LAYOUT" ] && echo "Layout:     $FILTER_LAYOUT"
[ -n "$FILTER_MODEL" ] && echo "Model:      $FILTER_MODEL"
[ "$DRY_RUN" = true ] && echo "Mode:       DRY RUN"
echo "Started:    $(date)"
echo ""

# Activate environment
activate_environment

START_TIME=$(date +%s)

case $STAGE in
    "imitation")
        run_imitation_stage
        ;;
    
    "rl")
        run_rl_stage
        ;;
    
    "inverse_planning")
        run_inverse_planning_stage
        ;;
    
    "all")
        run_imitation_stage
        run_rl_stage
        run_inverse_planning_stage
        ;;
    
    *)
        error "Unknown stage: $STAGE"
        echo "Valid stages: imitation, rl, inverse_planning, all"
        exit 1
        ;;
esac

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
ELAPSED_MIN=$((ELAPSED / 60))
ELAPSED_SEC=$((ELAPSED % 60))

section "STAGE COMPLETE"
echo "Stage:    $STAGE"
echo "Duration: ${ELAPSED_MIN}m ${ELAPSED_SEC}s"
echo "Finished: $(date)"

