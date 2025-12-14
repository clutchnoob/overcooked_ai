#!/bin/bash
###############################################################################
# INVERSE PLANNING FULL PIPELINE
###############################################################################
#
# Submits the complete inverse planning pipeline:
#   1. (Optional) Collect trajectories from trained policies
#   2. Train inverse planning models (array job, parallelized)
#   3. Analyze results (runs after training completes)
#
# Usage:
#   bash hpc/run_inverse_planning_pipeline.sh           # Skip trajectory collection
#   bash hpc/run_inverse_planning_pipeline.sh --collect # Include trajectory collection
#
###############################################################################

set -e

PROJECT_DIR="/om/scratch/Mon/mabdel03/6.S890/overcooked_ai/Overcooked_ProbMods"
COLLECT_TRAJECTORIES=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --collect)
            COLLECT_TRAJECTORIES=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "=============================================================="
echo "INVERSE PLANNING PIPELINE"
echo "=============================================================="
echo "Date: $(date)"
echo "Collect trajectories: $COLLECT_TRAJECTORIES"
echo ""

# Create logs directory
mkdir -p "$PROJECT_DIR/logs"

DEPENDENCY=""

# Step 1: Optionally collect trajectories
if [ "$COLLECT_TRAJECTORIES" = true ]; then
    echo "Step 1: Submitting trajectory collection jobs (array 0-11)..."
    COLLECT_JOB=$(sbatch --parsable "$PROJECT_DIR/hpc/collect_trajectories.sh")
    echo "  Collection job ID: $COLLECT_JOB"
    DEPENDENCY="--dependency=afterok:$COLLECT_JOB"
else
    echo "Step 1: Skipping trajectory collection (use --collect to enable)"
fi

# Step 2: Submit training array job
echo ""
echo "Step 2: Submitting training jobs (array 0-8)..."
if [ -n "$DEPENDENCY" ]; then
    TRAIN_JOB=$(sbatch --parsable $DEPENDENCY "$PROJECT_DIR/hpc/train_inverse_planning.sh")
else
    TRAIN_JOB=$(sbatch --parsable "$PROJECT_DIR/hpc/train_inverse_planning.sh")
fi
echo "  Training job ID: $TRAIN_JOB"

# Step 3: Submit analysis job with dependency on training completion
echo ""
echo "Step 3: Submitting analysis job (will wait for training)..."
ANALYZE_JOB=$(sbatch --parsable --dependency=afterok:$TRAIN_JOB "$PROJECT_DIR/hpc/analyze_inverse_planning.sh")
echo "  Analysis job ID: $ANALYZE_JOB"

echo ""
echo "=============================================================="
echo "PIPELINE SUBMITTED"
echo "=============================================================="
echo ""
if [ "$COLLECT_TRAJECTORIES" = true ]; then
    echo "Collection jobs: $COLLECT_JOB (array 0-11: 3 layouts x 4 sources)"
fi
echo "Training jobs:   $TRAIN_JOB (array 0-8: 3 layouts x 3 sources)"
echo "Analysis job:    $ANALYZE_JOB (depends on training)"
echo ""
echo "Monitor with:"
echo "  squeue -u \$USER"
echo ""
echo "View logs:"
if [ "$COLLECT_TRAJECTORIES" = true ]; then
    echo "  tail -f $PROJECT_DIR/logs/collect_traj_${COLLECT_JOB}_*.out"
fi
echo "  tail -f $PROJECT_DIR/logs/invplan_train_${TRAIN_JOB}_*.out"
echo "  tail -f $PROJECT_DIR/logs/invplan_analyze_${ANALYZE_JOB}.out"
echo ""
echo "Expected outputs:"
echo "  - Posterior params: results/inverse_planning/{layout}/{tag}/params.pt"
echo "  - Analysis JSON:    results/inverse_planning/analysis_summary.json"
echo "  - Plots:            results/inverse_planning/plots/"
echo ""
