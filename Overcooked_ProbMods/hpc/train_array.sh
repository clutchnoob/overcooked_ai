#!/bin/bash
#SBATCH --job-name=probmods_array
#SBATCH --output=/om/scratch/Mon/mabdel03/6.S890/overcooked_ai/Overcooked_ProbMods/logs/train_%A_%a.out
#SBATCH --error=/om/scratch/Mon/mabdel03/6.S890/overcooked_ai/Overcooked_ProbMods/logs/train_%A_%a.err
#SBATCH --time=12:00:00
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --array=0-29

# Array job to train all model/layout combinations in parallel
# Usage: sbatch train_array.sh
# Total jobs: 6 models x 5 layouts = 30

set -e
export PYTHONUNBUFFERED=1

PROJECT_DIR="/om/scratch/Mon/mabdel03/6.S890/overcooked_ai/Overcooked_ProbMods"

# Define models and layouts
MODELS=(
    "bayesian_bc"
    "rational_agent"
    "hierarchical_bc"
    "bayesian_gail"
    "bayesian_ppo_bc"
    "bayesian_ppo_gail"
)

LAYOUTS=(
    "cramped_room"
    "asymmetric_advantages"
    "coordination_ring"
    "forced_coordination"
    "counter_circuit_o_1order"
)

# Calculate model and layout from array task ID
NUM_LAYOUTS=${#LAYOUTS[@]}
MODEL_IDX=$((SLURM_ARRAY_TASK_ID / NUM_LAYOUTS))
LAYOUT_IDX=$((SLURM_ARRAY_TASK_ID % NUM_LAYOUTS))

MODEL=${MODELS[$MODEL_IDX]}
LAYOUT=${LAYOUTS[$LAYOUT_IDX]}

echo "=========================================="
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Model: $MODEL"
echo "Layout: $LAYOUT"
echo "Node: $(hostname)"
echo "=========================================="

# Activate environment (use conda directly to avoid CUDA issues)
source /om2/user/mabdel03/anaconda/bin/activate /om2/user/mabdel03/conda_envs/CompCogSci

# Check GPU
nvidia-smi --query-gpu=name --format=csv,noheader || echo "No GPU"
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"

cd "$PROJECT_DIR"

# Run the appropriate training script
echo "Starting training for $MODEL on $LAYOUT..."

case $MODEL in
    "bayesian_bc")
        python -u scripts/train_bayesian_bc.py --layout "$LAYOUT" --epochs 500 --results_dir ./results
        ;;
    "rational_agent")
        python -u -c "
from probmods.models.rational_agent import RationalAgentConfig, RationalAgentTrainer
config = RationalAgentConfig(layout_name='$LAYOUT', num_epochs=500, results_dir='./results')
trainer = RationalAgentTrainer(config)
trainer.train()
"
        ;;
    "hierarchical_bc")
        python -u -c "
from probmods.models.hierarchical_bc import HierarchicalBCConfig, HierarchicalBCTrainer
config = HierarchicalBCConfig(layout_name='$LAYOUT', num_epochs=500, results_dir='./results')
trainer = HierarchicalBCTrainer(config)
trainer.train()
"
        ;;
    "bayesian_gail"|"bayesian_ppo_bc"|"bayesian_ppo_gail")
        echo "Skipping $MODEL - requires RL environment setup"
        ;;
    *)
        echo "Unknown model: $MODEL"
        exit 1
        ;;
esac

echo "Training complete for $MODEL on $LAYOUT"
