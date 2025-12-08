#!/bin/bash
#SBATCH --job-name=ppo_gail_cramped_room_s40
#SBATCH --output=logs/ppo_gail_cramped_room_seed40_%j.out
#SBATCH --error=logs/ppo_gail_cramped_room_seed40_%j.err
#SBATCH --time=24:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8

# PPO_GAIL Training (PPO with GAIL partner instead of BC)
# Layout: cramped_room, Seed: 40
# Training iterations: 550 (paper value)

# Navigate to project root
cd "$SLURM_SUBMIT_DIR/.."

# Create logs directory
mkdir -p "$SLURM_SUBMIT_DIR/logs"

# Activate conda environment
source /om2/user/mabdel03/anaconda/etc/profile.d/conda.sh
conda activate /om/scratch/Mon/mabdel03/conda_envs/MAL_env

# Navigate to src for Python modules
cd src

echo "Training PPO_GAIL for cramped_room seed 40..."
echo "Paper iterations: 550"

python -m human_aware_rl.ppo.train_ppo_gail --layout cramped_room --seed 40

echo "PPO_GAIL training complete for cramped_room seed 40"
