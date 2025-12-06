#!/bin/bash
#SBATCH -n 16
#SBATCH -t 12:00:00
#SBATCH --mem=32G
#SBATCH --job-name=bc_ppo_coord
#SBATCH --output=logs/ppo_bc_coordination_%j.out
#SBATCH --error=logs/ppo_bc_coordination_%j.err

source /om2/user/mabdel03/anaconda/etc/profile.d/conda.sh
conda activate /om/scratch/Mon/mabdel03/conda_envs/MAL_env

mkdir -p logs

python -m human_aware_rl.ppo.train_ppo_bc --layout coordination_ring --seed 0 --fast

