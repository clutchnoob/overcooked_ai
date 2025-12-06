#!/bin/bash
#SBATCH -n 16
#SBATCH -t 12:00:00
#SBATCH --mem=32G
#SBATCH --job-name=airl_forced
#SBATCH --output=logs/airl_forced_coordination_%j.out
#SBATCH --error=logs/airl_forced_coordination_%j.err

source /om2/user/mabdel03/anaconda/etc/profile.d/conda.sh
conda activate /om/scratch/Mon/mabdel03/conda_envs/MAL_env

# Navigate to project directory (SLURM_SUBMIT_DIR is where sbatch was called)
cd "$SLURM_SUBMIT_DIR/.."
cd src/human_aware_rl

mkdir -p "$SLURM_SUBMIT_DIR/logs"

# Full AIRL training (5M timesteps, ~8-12 hours)
python -m human_aware_rl.imitation.train_airl --layout forced_coordination

