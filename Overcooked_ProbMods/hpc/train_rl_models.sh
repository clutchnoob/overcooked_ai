#!/bin/bash
#SBATCH --job-name=probmods_rl
#SBATCH --output=/om/scratch/Mon/mabdel03/6.S890/overcooked_ai/Overcooked_ProbMods/logs/rl_%A_%a.out
#SBATCH --error=/om/scratch/Mon/mabdel03/6.S890/overcooked_ai/Overcooked_ProbMods/logs/rl_%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --array=0-8

###############################################################################
# PARALLEL TRAINING: REINFORCEMENT LEARNING MODELS
###############################################################################
#
# Trains the 3 RL-based models (require environment rollouts):
#   - Bayesian GAIL
#   - Bayesian PPO+BC
#   - Bayesian PPO+GAIL
#
# On 3 layouts with human data:
#   - cramped_room, asymmetric_advantages, coordination_ring
#
# Total: 3 models x 3 layouts = 9 parallel jobs
#
# These take longer (environment interaction + adversarial training)
#
###############################################################################

set -e
export PYTHONUNBUFFERED=1

PROJECT_DIR="/om/scratch/Mon/mabdel03/6.S890/overcooked_ai/Overcooked_ProbMods"
OVERCOOKED_ROOT="/om/scratch/Mon/mabdel03/6.S890/overcooked_ai"

MODELS=("bayesian_gail" "bayesian_ppo_bc" "bayesian_ppo_gail")
LAYOUTS=("cramped_room" "asymmetric_advantages" "coordination_ring")

NUM_LAYOUTS=${#LAYOUTS[@]}
MODEL_IDX=$((SLURM_ARRAY_TASK_ID / NUM_LAYOUTS))
LAYOUT_IDX=$((SLURM_ARRAY_TASK_ID % NUM_LAYOUTS))

MODEL=${MODELS[$MODEL_IDX]}
LAYOUT=${LAYOUTS[$LAYOUT_IDX]}

echo "=============================================================="
echo "RL MODEL TRAINING"
echo "=============================================================="
echo "Task ID: $SLURM_ARRAY_TASK_ID | Model: $MODEL | Layout: $LAYOUT"
echo "Node: $(hostname) | Date: $(date)"
echo "=============================================================="

source /om2/user/mabdel03/anaconda/bin/activate /om2/user/mabdel03/conda_envs/CompCogSci

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "No GPU"
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"

export PYTHONPATH="${OVERCOOKED_ROOT}/src:${PROJECT_DIR}:${PYTHONPATH}"
cd "$PROJECT_DIR"
mkdir -p "$PROJECT_DIR/results/$MODEL/$LAYOUT"

TIMESTEPS=200000

case $MODEL in
    "bayesian_gail")
        python -u -c "
from probmods.models.bayesian_gail import BayesianGAILConfig, BayesianGAILTrainer
config = BayesianGAILConfig(layout_name='$LAYOUT', total_timesteps=$TIMESTEPS, results_dir='./results')
BayesianGAILTrainer(config).train()
"
        ;;
    "bayesian_ppo_bc")
        python -u -c "
from probmods.models.bayesian_ppo_bc import BayesianPPOBCConfig, BayesianPPOBCTrainer
config = BayesianPPOBCConfig(layout_name='$LAYOUT', total_timesteps=$TIMESTEPS, results_dir='./results')
BayesianPPOBCTrainer(config).train()
"
        ;;
    "bayesian_ppo_gail")
        python -u -c "
from probmods.models.bayesian_ppo_gail import BayesianPPOGAILConfig, BayesianPPOGAILTrainer
config = BayesianPPOGAILConfig(layout_name='$LAYOUT', total_timesteps=$TIMESTEPS, results_dir='./results')
BayesianPPOGAILTrainer(config).train()
"
        ;;
esac

echo "=== COMPLETE: $MODEL on $LAYOUT ==="
