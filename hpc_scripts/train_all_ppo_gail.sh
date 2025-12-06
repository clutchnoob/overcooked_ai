#!/bin/bash
#SBATCH --job-name=ppo_gail_all
#SBATCH --output=logs/ppo_gail_all_%j.out
#SBATCH --error=logs/ppo_gail_all_%j.err
#SBATCH --time=24:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8

# Navigate to project root
cd "$SLURM_SUBMIT_DIR/.."

# Create logs directory
mkdir -p "$SLURM_SUBMIT_DIR/logs"

# Activate conda environment
source ~/.bashrc
conda activate MAL_env

# Navigate to human_aware_rl
cd src/human_aware_rl

echo "=========================================="
echo "PPO_GAIL Training - All Layouts (Fast Mode)"
echo "=========================================="

# Train all layouts with 5 seeds each
python -m human_aware_rl.ppo.train_ppo_gail --all_layouts --fast --seeds 0,10,20,30,40

echo "=========================================="
echo "All PPO_GAIL training complete!"
echo "=========================================="

