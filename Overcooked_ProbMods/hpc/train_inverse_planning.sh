#!/bin/bash
#SBATCH --job-name=invplan_train
#SBATCH --output=/om/scratch/Mon/mabdel03/6.S890/overcooked_ai/Overcooked_ProbMods/logs/invplan_train_%A_%a.out
#SBATCH --error=/om/scratch/Mon/mabdel03/6.S890/overcooked_ai/Overcooked_ProbMods/logs/invplan_train_%A_%a.err
#SBATCH --time=4:00:00
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --array=0-8

###############################################################################
# TRAIN INVERSE PLANNING MODELS (ARRAY JOB)
###############################################################################
#
# Trains LinearInversePlanningModel for Bayesian inverse planning to infer
# feature weights (theta) and rationality (beta) from policy trajectories.
#
# Array job: 3 layouts x 3 sources = 9 jobs
#   Sources: human_demo, ppo_bc, ppo_gail
#
# Usage:
#   sbatch hpc/train_inverse_planning.sh
#
###############################################################################

set -e
export PYTHONUNBUFFERED=1

# Paths
PROJECT_DIR="/om/scratch/Mon/mabdel03/6.S890/overcooked_ai/Overcooked_ProbMods"
OVERCOOKED_ROOT="/om/scratch/Mon/mabdel03/6.S890/overcooked_ai"

# Define layouts and data sources
LAYOUTS=(
    "cramped_room"
    "asymmetric_advantages"
    "coordination_ring"
)

# Tags correspond to different data sources
TAGS=(
    "human_demo"
    "ppo_bc"
    "ppo_gail"
)

# Calculate layout and tag from array task ID
NUM_TAGS=${#TAGS[@]}
LAYOUT_IDX=$((SLURM_ARRAY_TASK_ID / NUM_TAGS))
TAG_IDX=$((SLURM_ARRAY_TASK_ID % NUM_TAGS))

LAYOUT=${LAYOUTS[$LAYOUT_IDX]}
TAG=${TAGS[$TAG_IDX]}

echo "=============================================================="
echo "INVERSE PLANNING TRAINING"
echo "=============================================================="
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Layout:        $LAYOUT"
echo "Tag:           $TAG"
echo "Node:          $(hostname)"
echo "Date:          $(date)"
echo "=============================================================="

# Create logs directory
mkdir -p "$PROJECT_DIR/logs"

# Activate conda environment
source /om2/user/mabdel03/anaconda/bin/activate /om2/user/mabdel03/conda_envs/CompCogSci

# Verify environment
echo ""
echo "=== Environment Check ==="
which python
python --version
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import pyro; print(f'Pyro version: {pyro.__version__}')"

# Set up Python path
export PYTHONPATH="${OVERCOOKED_ROOT}/src:${PROJECT_DIR}:${PYTHONPATH}"

# Change to project directory
cd "$PROJECT_DIR"

echo ""
echo "=== Starting Training ==="

# Run training
# For non-human sources, use --use-trajectories to load from collected data
if [ "$TAG" = "human_demo" ]; then
    python -u scripts/run_inverse_planning.py \
        --layouts "$LAYOUT" \
        --tags "$TAG" \
        --epochs 500 \
        --results-dir ./results
else
    python -u scripts/run_inverse_planning.py \
        --layouts "$LAYOUT" \
        --tags "$TAG" \
        --epochs 500 \
        --results-dir ./results \
        --use-trajectories
fi

echo ""
echo "=============================================================="
echo "TRAINING COMPLETE"
echo "Layout: $LAYOUT, Tag: $TAG"
echo "Time: $(date)"
echo "=============================================================="
