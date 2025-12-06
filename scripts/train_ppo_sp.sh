#!/bin/bash
#SBATCH --job-name=ppo_sp
#SBATCH --output=logs/ppo_sp_%j.out
#SBATCH --error=logs/ppo_sp_%j.err
#SBATCH --time=24:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8

# Create logs directory
mkdir -p logs

# Activate conda environment
source ~/.bashrc
conda activate overcooked

# Navigate to project
cd $SLURM_SUBMIT_DIR/src/human_aware_rl

# Train PPO Self-Play for all layouts with 5 seeds
python -m human_aware_rl.ppo.train_ppo_sp --all_layouts --seeds 0,10,20,30,40

echo "PPO Self-Play training complete!"

