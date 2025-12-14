#!/bin/bash
#SBATCH --job-name=probmods_all
#SBATCH --output=/om/scratch/Mon/mabdel03/6.S890/overcooked_ai/Overcooked_ProbMods/logs/train_all_%A_%a.out
#SBATCH --error=/om/scratch/Mon/mabdel03/6.S890/overcooked_ai/Overcooked_ProbMods/logs/train_all_%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --array=0-17

###############################################################################
# PARALLEL TRAINING ARRAY JOB FOR ALL PROBABILISTIC MODELS
###############################################################################
#
# This script trains ALL 6 probabilistic models on ALL 3 layouts with human data.
#
# Models (6):
#   0. bayesian_bc        - Bayesian Behavioral Cloning (imitation only)
#   1. rational_agent     - Softmax-rational agent (imitation only)
#   2. hierarchical_bc    - Hierarchical goal-conditioned BC (imitation only)
#   3. bayesian_gail      - Bayesian GAIL (requires env rollouts)
#   4. bayesian_ppo_bc    - Bayesian PPO with BC anchor (requires env rollouts)
#   5. bayesian_ppo_gail  - Bayesian PPO + GAIL (requires env rollouts)
#
# Layouts with human data (3):
#   0. cramped_room
#   1. asymmetric_advantages
#   2. coordination_ring
#
# Total jobs: 6 models x 3 layouts = 18 parallel tasks
#
# Usage:
#   sbatch hpc/train_all_models.sh
#
# Monitor:
#   squeue -u $USER
#   sacct -j <JOB_ID> --format=JobID,JobName,State,ExitCode,Elapsed
#
###############################################################################

set -e
export PYTHONUNBUFFERED=1

# Absolute paths
PROJECT_DIR="/om/scratch/Mon/mabdel03/6.S890/overcooked_ai/Overcooked_ProbMods"
OVERCOOKED_ROOT="/om/scratch/Mon/mabdel03/6.S890/overcooked_ai"

# Define all 6 models
MODELS=(
    "bayesian_bc"
    "rational_agent"
    "hierarchical_bc"
    "bayesian_gail"
    "bayesian_ppo_bc"
    "bayesian_ppo_gail"
)

# Only layouts with human demonstration data
LAYOUTS=(
    "cramped_room"
    "asymmetric_advantages"
    "coordination_ring"
)

# Calculate model and layout indices from array task ID
NUM_LAYOUTS=${#LAYOUTS[@]}
MODEL_IDX=$((SLURM_ARRAY_TASK_ID / NUM_LAYOUTS))
LAYOUT_IDX=$((SLURM_ARRAY_TASK_ID % NUM_LAYOUTS))

MODEL=${MODELS[$MODEL_IDX]}
LAYOUT=${LAYOUTS[$LAYOUT_IDX]}

echo "=============================================================="
echo "PROBABILISTIC MODEL TRAINING"
echo "=============================================================="
echo "Array Task ID:    $SLURM_ARRAY_TASK_ID"
echo "Model:            $MODEL"
echo "Layout:           $LAYOUT"
echo "Node:             $(hostname)"
echo "Date:             $(date)"
echo "Project Dir:      $PROJECT_DIR"
echo "=============================================================="

# Activate conda environment directly (avoids CUDA issues)
source /om2/user/mabdel03/anaconda/bin/activate /om2/user/mabdel03/conda_envs/CompCogSci

# Verify environment
echo ""
echo "=== Environment Check ==="
which python
python --version

# Check GPU availability
echo ""
echo "=== GPU Check ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || echo "No GPU detected"
python -c "import torch; print(f'PyTorch CUDA available: {torch.cuda.is_available()}'); print(f'CUDA device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"

# Set up Python path
export PYTHONPATH="${OVERCOOKED_ROOT}/src:${PROJECT_DIR}:${PYTHONPATH}"

# Change to project directory
cd "$PROJECT_DIR"

# Create results directory
mkdir -p "$PROJECT_DIR/results/$MODEL/$LAYOUT"

echo ""
echo "=== Starting Training: $MODEL on $LAYOUT ==="
echo ""

# Training parameters
# - Imitation models (BC, rational, hierarchical): epochs
# - RL models (GAIL, PPO): timesteps
IMITATION_EPOCHS=500
RL_TIMESTEPS=200000

case $MODEL in
    "bayesian_bc")
        echo "Training Bayesian Behavioral Cloning..."
        python -u scripts/train_bayesian_bc.py \
            --layout "$LAYOUT" \
            --epochs $IMITATION_EPOCHS \
            --results_dir ./results
        ;;
    
    "rational_agent")
        echo "Training Rational Agent Model..."
        python -u -c "
import sys
sys.path.insert(0, '${PROJECT_DIR}')
from probmods.models.rational_agent import RationalAgentConfig, RationalAgentTrainer

config = RationalAgentConfig(
    layout_name='$LAYOUT',
    num_epochs=$IMITATION_EPOCHS,
    results_dir='./results'
)
trainer = RationalAgentTrainer(config)
trainer.train()
print('Training complete!')
"
        ;;
    
    "hierarchical_bc")
        echo "Training Hierarchical BC Model..."
        python -u -c "
import sys
sys.path.insert(0, '${PROJECT_DIR}')
from probmods.models.hierarchical_bc import HierarchicalBCConfig, HierarchicalBCTrainer

config = HierarchicalBCConfig(
    layout_name='$LAYOUT',
    num_epochs=$IMITATION_EPOCHS,
    num_goals=4,
    results_dir='./results'
)
trainer = HierarchicalBCTrainer(config)
trainer.train()
print('Training complete!')
"
        ;;
    
    "bayesian_gail")
        echo "Training Bayesian GAIL..."
        python -u -c "
import sys
sys.path.insert(0, '${PROJECT_DIR}')
from probmods.models.bayesian_gail import BayesianGAILConfig, BayesianGAILTrainer

config = BayesianGAILConfig(
    layout_name='$LAYOUT',
    total_timesteps=$RL_TIMESTEPS,
    results_dir='./results'
)
trainer = BayesianGAILTrainer(config)
trainer.train()
print('Training complete!')
"
        ;;
    
    "bayesian_ppo_bc")
        echo "Training Bayesian PPO+BC..."
        python -u -c "
import sys
sys.path.insert(0, '${PROJECT_DIR}')
from probmods.models.bayesian_ppo_bc import BayesianPPOBCConfig, BayesianPPOBCTrainer

config = BayesianPPOBCConfig(
    layout_name='$LAYOUT',
    total_timesteps=$RL_TIMESTEPS,
    results_dir='./results'
)
trainer = BayesianPPOBCTrainer(config)
trainer.train()
print('Training complete!')
"
        ;;
    
    "bayesian_ppo_gail")
        echo "Training Bayesian PPO+GAIL..."
        python -u -c "
import sys
sys.path.insert(0, '${PROJECT_DIR}')
from probmods.models.bayesian_ppo_gail import BayesianPPOGAILConfig, BayesianPPOGAILTrainer

config = BayesianPPOGAILConfig(
    layout_name='$LAYOUT',
    total_timesteps=$RL_TIMESTEPS,
    results_dir='./results'
)
trainer = BayesianPPOGAILTrainer(config)
trainer.train()
print('Training complete!')
"
        ;;
    
    *)
        echo "ERROR: Unknown model: $MODEL"
        exit 1
        ;;
esac

echo ""
echo "=============================================================="
echo "TRAINING COMPLETE"
echo "Model:   $MODEL"
echo "Layout:  $LAYOUT"
echo "Results: $PROJECT_DIR/results/$MODEL/$LAYOUT/"
echo "Time:    $(date)"
echo "=============================================================="
