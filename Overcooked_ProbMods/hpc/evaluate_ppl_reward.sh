#!/bin/bash
#SBATCH --job-name=eval_ppl
#SBATCH --output=/om/scratch/Mon/mabdel03/6.S890/overcooked_ai/Overcooked_ProbMods/logs/eval_ppl_%j.out
#SBATCH --error=/om/scratch/Mon/mabdel03/6.S890/overcooked_ai/Overcooked_ProbMods/logs/eval_ppl_%j.err
#SBATCH --time=4:00:00
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

###############################################################################
# EVALUATE PPL MODELS WITH HUMAN PROXY (REWARD-BASED)
###############################################################################
#
# This script evaluates the trained PPL models by pairing them with a Human
# Proxy model and measuring game rewards - the same evaluation used for the
# RL models in Run 3/4.
#
# Models evaluated:
#   - Rational Agent (softmax-rational Q-learning)
#   - Bayesian BC (Bayesian behavior cloning)
#   - Hierarchical BC (goal-conditioned policy)
#
# Layouts:
#   - cramped_room
#   - asymmetric_advantages
#   - coordination_ring
#
# Usage:
#   sbatch hpc/evaluate_ppl_reward.sh
#
###############################################################################

set -e
export PYTHONUNBUFFERED=1

# Paths
PROJECT_DIR="/om/scratch/Mon/mabdel03/6.S890/overcooked_ai/Overcooked_ProbMods"
OVERCOOKED_ROOT="/om/scratch/Mon/mabdel03/6.S890/overcooked_ai"

echo "=============================================================="
echo "PPL MODEL EVALUATION (REWARD-BASED)"
echo "=============================================================="
echo "Node:        $(hostname)"
echo "Date:        $(date)"
echo "Project Dir: $PROJECT_DIR"
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

# Install missing dependencies
echo ""
echo "=== Installing Dependencies ==="
pip install tensorboard --quiet || echo "tensorboard install skipped"

# Check for GPU (not required for evaluation but helpful)
echo ""
echo "=== Device Check ==="
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# Set up Python path
export PYTHONPATH="${OVERCOOKED_ROOT}/src:${PROJECT_DIR}:${PYTHONPATH}"

# Change to project directory
cd "$PROJECT_DIR"

# Number of games per evaluation (more games = more reliable but slower)
NUM_GAMES="${NUM_GAMES:-10}"

echo ""
echo "=== Starting PPL Evaluation ==="
echo "Games per condition: $NUM_GAMES"
echo ""

# Run evaluation
python -u scripts/evaluate_ppl_reward.py \
    --num_games "$NUM_GAMES" \
    --compare_baselines \
    --verbose

echo ""
echo "=============================================================="
echo "EVALUATION COMPLETE"
echo "Time: $(date)"
echo "=============================================================="
