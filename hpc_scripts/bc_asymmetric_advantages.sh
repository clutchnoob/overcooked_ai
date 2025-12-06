#!/bin/bash
#SBATCH -n 16
#SBATCH -t 02:00:00
#SBATCH --mem=16G
#SBATCH --job-name=bc_asymm
#SBATCH --output=logs/bc_asymmetric_%j.out
#SBATCH --error=logs/bc_asymmetric_%j.err

source /om2/user/mabdel03/anaconda/etc/profile.d/conda.sh
conda activate /om/scratch/Mon/mabdel03/conda_envs/MAL_env

# Navigate to project directory
cd "$(dirname "$0")/.."
cd src/human_aware_rl

mkdir -p ../../hpc_scripts/logs

python -m human_aware_rl.imitation.train_bc_models --layout asymmetric_advantages

