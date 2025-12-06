#!/bin/bash
#SBATCH -n 16
#SBATCH -t 12:00:00
#SBATCH --mem=32G
#SBATCH --job-name=sp_coord
#SBATCH --output=logs/ppo_sp_coordination_%j.out
#SBATCH --error=logs/ppo_sp_coordination_%j.err

source /om2/user/mabdel03/anaconda/etc/profile.d/conda.sh
conda activate /om/scratch/Mon/mabdel03/conda_envs/MAL_env

# Navigate to project directory
cd "$(dirname "$0")/.."
cd src/human_aware_rl

mkdir -p ../../hpc_scripts/logs

python -m human_aware_rl.ppo.train_ppo_sp --layout coordination_ring --seed 0 --fast

