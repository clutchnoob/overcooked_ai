#!/bin/bash
#SBATCH -n 16
#SBATCH -t 47:00:00
#SBATCH --mem=32G
#SBATCH --job-name=gail_cramped
#SBATCH --output=logs/gail_cramped_%j.out
#SBATCH --error=logs/gail_cramped_%j.err

source /om2/user/mabdel03/anaconda/etc/profile.d/conda.sh
conda activate /om/scratch/Mon/mabdel03/conda_envs/MAL_env

# Navigate to project directory (SLURM_SUBMIT_DIR is where sbatch was called)
cd "$SLURM_SUBMIT_DIR/.."
cd src

mkdir -p "$SLURM_SUBMIT_DIR/logs"

python -m human_aware_rl.imitation.gail --layout cramped_room
