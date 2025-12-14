#!/bin/bash
#SBATCH --job-name=collect_traj
#SBATCH --output=/om/scratch/Mon/mabdel03/6.S890/overcooked_ai/Overcooked_ProbMods/logs/collect_traj_%A_%a.out
#SBATCH --error=/om/scratch/Mon/mabdel03/6.S890/overcooked_ai/Overcooked_ProbMods/logs/collect_traj_%A_%a.err
#SBATCH --time=2:00:00
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --array=0-11

###############################################################################
# COLLECT POLICY TRAJECTORIES (ARRAY JOB)
###############################################################################
#
# Collects state-action trajectories from various policies for inverse planning:
#   - Human demonstrations
#   - BC policies
#   - PPO-BC policies
#   - PPO-GAIL policies
#
# Array job: 3 layouts x 4 sources = 12 jobs
#
# Usage:
#   sbatch hpc/collect_trajectories.sh
#
###############################################################################

set -e
export PYTHONUNBUFFERED=1

# Paths
PROJECT_DIR="/om/scratch/Mon/mabdel03/6.S890/overcooked_ai/Overcooked_ProbMods"
OVERCOOKED_ROOT="/om/scratch/Mon/mabdel03/6.S890/overcooked_ai"

# Define layouts and sources
LAYOUTS=(
    "cramped_room"
    "asymmetric_advantages"
    "coordination_ring"
)

SOURCES=(
    "human_demo"
    "bc"
    "ppo_bc"
    "ppo_gail"
)

# Calculate layout and source from array task ID
NUM_SOURCES=${#SOURCES[@]}
LAYOUT_IDX=$((SLURM_ARRAY_TASK_ID / NUM_SOURCES))
SOURCE_IDX=$((SLURM_ARRAY_TASK_ID % NUM_SOURCES))

LAYOUT=${LAYOUTS[$LAYOUT_IDX]}
SOURCE=${SOURCES[$SOURCE_IDX]}

echo "=============================================================="
echo "TRAJECTORY COLLECTION"
echo "=============================================================="
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Layout:        $LAYOUT"
echo "Source:        $SOURCE"
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

# Set up Python path
export PYTHONPATH="${OVERCOOKED_ROOT}/src:${PROJECT_DIR}:${PYTHONPATH}"

# Change to project directory
cd "$PROJECT_DIR"

# Number of episodes to collect for policy sources
NUM_EPISODES="${NUM_EPISODES:-100}"

echo ""
echo "=== Collecting Trajectories ==="
echo "Episodes per source: $NUM_EPISODES"

# Run collection
python -u scripts/collect_policy_trajectories.py \
    --layout "$LAYOUT" \
    --source "$SOURCE" \
    --num-episodes "$NUM_EPISODES" \
    --output-dir ./results

echo ""
echo "=============================================================="
echo "COLLECTION COMPLETE"
echo "Layout: $LAYOUT, Source: $SOURCE"
echo "Time: $(date)"
echo "=============================================================="
