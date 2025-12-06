#!/bin/bash
#SBATCH --job-name=ppo_gail_asym
#SBATCH --output=logs/ppo_gail_asym_%j.out
#SBATCH --error=logs/ppo_gail_asym_%j.err
#SBATCH --time=04:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4

# Navigate to project root
cd "$SLURM_SUBMIT_DIR/.."

# Create logs directory
mkdir -p "$SLURM_SUBMIT_DIR/logs"

# Activate conda environment
source ~/.bashrc
conda activate MAL_env

# Navigate to human_aware_rl
cd src/human_aware_rl

# Run PPO_GAIL training (fast mode, single seed)
python -m human_aware_rl.ppo.train_ppo_gail --layout asymmetric_advantages --seed 0 --fast

