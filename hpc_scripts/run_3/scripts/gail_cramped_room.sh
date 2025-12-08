#!/bin/bash
#SBATCH --job-name=gail_cramped_room
#SBATCH --output=logs/gail_cramped_room_%j.out
#SBATCH --error=logs/gail_cramped_room_%j.err
#SBATCH --time=08:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8

# GAIL Training for cramped_room

# Navigate to project root
cd "$SLURM_SUBMIT_DIR/.."

# Create logs directory
mkdir -p "$SLURM_SUBMIT_DIR/logs"

# Activate conda environment
source /om2/user/mabdel03/anaconda/etc/profile.d/conda.sh
conda activate /om/scratch/Mon/mabdel03/conda_envs/MAL_env

# Navigate to src for Python modules
cd src

echo "Training GAIL model for cramped_room..."

python -m human_aware_rl.imitation.gail --layout cramped_room

echo "GAIL training complete for cramped_room"
