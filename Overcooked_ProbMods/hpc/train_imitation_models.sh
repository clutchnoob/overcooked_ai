#!/bin/bash
#SBATCH --job-name=probmods_imitation
#SBATCH --output=/om/scratch/Mon/mabdel03/6.S890/overcooked_ai/Overcooked_ProbMods/logs/imitation_%A_%a.out
#SBATCH --error=/om/scratch/Mon/mabdel03/6.S890/overcooked_ai/Overcooked_ProbMods/logs/imitation_%A_%a.err
#SBATCH --time=04:00:00
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --array=0-8

###############################################################################
# PARALLEL TRAINING: IMITATION LEARNING MODELS ONLY
###############################################################################
#
# Trains the 3 imitation-only models (no environment rollouts needed):
#   - Bayesian BC
#   - Rational Agent
#   - Hierarchical BC
#
# On 3 layouts with human data:
#   - cramped_room, asymmetric_advantages, coordination_ring
#
# Total: 3 models x 3 layouts = 9 parallel jobs
#
# These are faster to train (supervised learning on human demos)
#
###############################################################################

set -e
export PYTHONUNBUFFERED=1

PROJECT_DIR="/om/scratch/Mon/mabdel03/6.S890/overcooked_ai/Overcooked_ProbMods"
OVERCOOKED_ROOT="/om/scratch/Mon/mabdel03/6.S890/overcooked_ai"

MODELS=("bayesian_bc" "rational_agent" "hierarchical_bc")
LAYOUTS=("cramped_room" "asymmetric_advantages" "coordination_ring")

NUM_LAYOUTS=${#LAYOUTS[@]}
MODEL_IDX=$((SLURM_ARRAY_TASK_ID / NUM_LAYOUTS))
LAYOUT_IDX=$((SLURM_ARRAY_TASK_ID % NUM_LAYOUTS))

MODEL=${MODELS[$MODEL_IDX]}
LAYOUT=${LAYOUTS[$LAYOUT_IDX]}

echo "=============================================================="
echo "IMITATION MODEL TRAINING"
echo "=============================================================="
echo "Task ID: $SLURM_ARRAY_TASK_ID | Model: $MODEL | Layout: $LAYOUT"
echo "Node: $(hostname) | Date: $(date)"
echo "=============================================================="

source /om2/user/mabdel03/anaconda/bin/activate /om2/user/mabdel03/conda_envs/CompCogSci

nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo "No GPU"
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"

export PYTHONPATH="${OVERCOOKED_ROOT}/src:${PROJECT_DIR}:${PYTHONPATH}"
cd "$PROJECT_DIR"
mkdir -p "$PROJECT_DIR/results/$MODEL/$LAYOUT"

EPOCHS=500

case $MODEL in
    "bayesian_bc")
        python -u scripts/train_bayesian_bc.py --layout "$LAYOUT" --epochs $EPOCHS --results_dir ./results
        ;;
    "rational_agent")
        python -u -c "
from probmods.models.rational_agent import RationalAgentConfig, RationalAgentTrainer
config = RationalAgentConfig(layout_name='$LAYOUT', num_epochs=$EPOCHS, results_dir='./results')
RationalAgentTrainer(config).train()
"
        ;;
    "hierarchical_bc")
        python -u -c "
from probmods.models.hierarchical_bc import HierarchicalBCConfig, HierarchicalBCTrainer
config = HierarchicalBCConfig(layout_name='$LAYOUT', num_epochs=$EPOCHS, num_goals=4, results_dir='./results')
HierarchicalBCTrainer(config).train()
"
        ;;
esac

echo "=== COMPLETE: $MODEL on $LAYOUT ==="
