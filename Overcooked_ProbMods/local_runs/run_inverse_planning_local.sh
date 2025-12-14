#!/bin/bash
###############################################################################
# INVERSE PLANNING PIPELINE - LOCAL EXECUTION
###############################################################################
# Runs the complete inverse planning pipeline:
#   1. (Optional) Collect trajectories from trained policies
#   2. Train inverse planning models for all layouts and sources
#   3. Run analysis and generate comparison plots
#
# Usage:
#   ./run_inverse_planning_local.sh [options]
#
# Options:
#   --collect       Include trajectory collection step
#   --skip-train    Skip training (only run analysis)
#   --skip-analysis Skip analysis (only run training)
#   --layout <name> Run only for specific layout
#   --tag <name>    Run only for specific tag/source
#   --dry-run       Print what would be executed
###############################################################################

set -e

# Source configuration and job queue
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/config.sh"
source "$SCRIPT_DIR/job_queue.sh"

###############################################################################
# ARGUMENT PARSING
###############################################################################

COLLECT_TRAJECTORIES=false
SKIP_TRAIN=false
SKIP_ANALYSIS=false
FILTER_LAYOUT=""
FILTER_TAG=""
DRY_RUN=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --collect)
            COLLECT_TRAJECTORIES=true
            shift
            ;;
        --skip-train)
            SKIP_TRAIN=true
            shift
            ;;
        --skip-analysis)
            SKIP_ANALYSIS=true
            shift
            ;;
        --layout)
            FILTER_LAYOUT="$2"
            shift 2
            ;;
        --tag)
            FILTER_TAG="$2"
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

###############################################################################
# PIPELINE STEPS
###############################################################################

collect_trajectories() {
    section "STEP 1: COLLECT TRAJECTORIES"
    
    local sources=("human_demo" "bc" "ppo_bc" "ppo_gail")
    local job_count=0
    
    for layout in "${LAYOUTS[@]}"; do
        if [ -n "$FILTER_LAYOUT" ] && [ "$layout" != "$FILTER_LAYOUT" ]; then
            continue
        fi
        
        for source in "${sources[@]}"; do
            local job_name="collect_${layout}_${source}"
            local job_cmd="cd $PROJECT_DIR && python -u scripts/collect_policy_trajectories.py --layout $layout --source $source --output-dir ./results"
            
            if [ "$DRY_RUN" = true ]; then
                echo "[DRY RUN] Would launch: $job_name"
            else
                launch_job "$job_name" bash -c "$job_cmd" 
            fi
            
            ((job_count++))
        done
    done
    
    if [ "$DRY_RUN" = false ]; then
        wait_all
    fi
    
    log "Trajectory collection complete. Jobs: $job_count"
}

train_inverse_planning() {
    section "STEP 2: TRAIN INVERSE PLANNING MODELS"
    
    local job_count=0
    
    for layout in "${LAYOUTS[@]}"; do
        if [ -n "$FILTER_LAYOUT" ] && [ "$layout" != "$FILTER_LAYOUT" ]; then
            continue
        fi
        
        for tag in "${INVERSE_PLANNING_TAGS[@]}"; do
            if [ -n "$FILTER_TAG" ] && [ "$tag" != "$FILTER_TAG" ]; then
                continue
            fi
            
            local job_name="invplan_${layout}_${tag}"
            local use_traj_flag=""
            
            # Use trajectories for non-human sources
            if [ "$tag" != "human_demo" ]; then
                use_traj_flag="--use-trajectories"
            fi
            
            local job_cmd="cd $PROJECT_DIR && python -u scripts/run_inverse_planning.py --layouts $layout --tags $tag --epochs $IMITATION_EPOCHS --results-dir ./results $use_traj_flag"
            
            if [ "$DRY_RUN" = true ]; then
                echo "[DRY RUN] Would launch: $job_name"
                echo "  Command: $job_cmd"
            else
                launch_job "$job_name" bash -c "$job_cmd"
            fi
            
            ((job_count++))
        done
    done
    
    if [ "$DRY_RUN" = false ]; then
        wait_all
    fi
    
    log "Inverse planning training complete. Jobs: $job_count"
}

run_analysis() {
    section "STEP 3: ANALYZE RESULTS"
    
    # Build layout and tag arguments
    local layouts_arg=""
    local tags_arg=""
    
    if [ -n "$FILTER_LAYOUT" ]; then
        layouts_arg="$FILTER_LAYOUT"
    else
        layouts_arg="${LAYOUTS[*]}"
    fi
    
    if [ -n "$FILTER_TAG" ]; then
        tags_arg="$FILTER_TAG"
    else
        tags_arg="${INVERSE_PLANNING_TAGS[*]}"
    fi
    
    local analysis_cmd="cd $PROJECT_DIR && python -u scripts/analyze_inverse_planning.py --layouts $layouts_arg --tags $tags_arg --save-plots --output-json ./results/inverse_planning/analysis_summary.json --results-dir ./results"
    
    if [ "$DRY_RUN" = true ]; then
        echo "[DRY RUN] Would run analysis:"
        echo "  Command: $analysis_cmd"
    else
        log "Running analysis..."
        bash -c "$analysis_cmd"
    fi
    
    log "Analysis complete."
}

###############################################################################
# MAIN EXECUTION
###############################################################################

section "INVERSE PLANNING PIPELINE"
echo "Collect trajectories: $COLLECT_TRAJECTORIES"
echo "Skip training:        $SKIP_TRAIN"
echo "Skip analysis:        $SKIP_ANALYSIS"
echo "Device:               $DEVICE"
echo "Max parallel jobs:    $MAX_JOBS"
[ -n "$FILTER_LAYOUT" ] && echo "Filter layout:        $FILTER_LAYOUT"
[ -n "$FILTER_TAG" ] && echo "Filter tag:           $FILTER_TAG"
[ "$DRY_RUN" = true ] && echo "Mode:                 DRY RUN"
echo "Started:              $(date)"
echo ""

# Activate environment
activate_environment

START_TIME=$(date +%s)

# Step 1: Collect trajectories (optional)
if [ "$COLLECT_TRAJECTORIES" = true ]; then
    collect_trajectories
fi

# Step 2: Train inverse planning models
if [ "$SKIP_TRAIN" = false ]; then
    train_inverse_planning
fi

# Step 3: Run analysis
if [ "$SKIP_ANALYSIS" = false ]; then
    run_analysis
fi

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
ELAPSED_MIN=$((ELAPSED / 60))
ELAPSED_SEC=$((ELAPSED % 60))

section "PIPELINE COMPLETE"
echo "Duration: ${ELAPSED_MIN}m ${ELAPSED_SEC}s"
echo "Finished: $(date)"
echo ""
echo "Outputs:"
echo "  - Posteriors: $RESULTS_DIR/inverse_planning/{layout}/{tag}/"
echo "  - Analysis:   $RESULTS_DIR/inverse_planning/analysis_summary.json"
echo "  - Plots:      $RESULTS_DIR/inverse_planning/plots/"

