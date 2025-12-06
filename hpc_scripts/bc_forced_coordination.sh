#!/bin/bash
#SBATCH -n 16
#SBATCH -t 02:00:00
#SBATCH --mem=16G
#SBATCH --job-name=bc_forced
#SBATCH --output=logs/bc_forced_%j.out
#SBATCH --error=logs/bc_forced_%j.err

source /om2/user/mabdel03/anaconda/etc/profile.d/conda.sh
conda activate /om/scratch/Mon/mabdel03/conda_envs/MAL_env

# Navigate to project directory (SLURM_SUBMIT_DIR is where sbatch was called)
cd "$SLURM_SUBMIT_DIR/.."
cd src/human_aware_rl

mkdir -p "$SLURM_SUBMIT_DIR/logs"

python -m human_aware_rl.imitation.train_bc_models --layout forced_coordination

