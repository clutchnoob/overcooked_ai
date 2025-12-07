#!/bin/bash
#SBATCH --job-name=sp_cr_s0
#SBATCH --output=../logs/ppo_sp_cramped_room_seed0_%j.out
#SBATCH --error=../logs/ppo_sp_cramped_room_seed0_%j.err
#SBATCH --time=12:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8

# Navigate to project root
cd "$SLURM_SUBMIT_DIR/../.."

# Create logs directory
mkdir -p hpc_scripts/logs

# Activate conda environment
source ~/.bashrc
conda activate MAL_env

# Navigate to human_aware_rl
cd src/human_aware_rl

# Run PPO Self-Play training (FULL PAPER PARAMS)
python -m human_aware_rl.ppo.train_ppo_sp --layout cramped_room --seed 0

