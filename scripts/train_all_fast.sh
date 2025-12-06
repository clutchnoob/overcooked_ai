#!/bin/bash
#SBATCH --job-name=overcooked_fast
#SBATCH --output=logs/train_all_fast_%j.out
#SBATCH --error=logs/train_all_fast_%j.err
#SBATCH --time=04:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=8

# Fast training version (~2-4 hours total instead of 48+)
# Uses 1M timesteps and early stopping

# Create logs directory
mkdir -p logs

# Activate conda environment
source ~/.bashrc
conda activate overcooked

# Navigate to project
cd $SLURM_SUBMIT_DIR/src/human_aware_rl

echo "========================================="
echo "Step 1: Training BC models"
echo "========================================="
python -m human_aware_rl.imitation.train_bc_models --all_layouts

echo "========================================="
echo "Step 2: Training PPO Self-Play (FAST)"
echo "========================================="
python -m human_aware_rl.ppo.train_ppo_sp --all_layouts --seeds 0 --fast

echo "========================================="
echo "Step 3: Training PPO with BC partner (FAST)"
echo "========================================="
python -m human_aware_rl.ppo.train_ppo_bc --all_layouts --seeds 0 --fast

echo "========================================="
echo "All training complete!"
echo "========================================="

