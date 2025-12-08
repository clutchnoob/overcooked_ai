#!/bin/bash
#SBATCH --job-name=sp_coord_s0
#SBATCH --output=../logs/ppo_sp_coordination_ring_seed0_%j.out
#SBATCH --error=../logs/ppo_sp_coordination_ring_seed0_%j.err
#SBATCH --time=12:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8

# Navigate to project root
cd "$SLURM_SUBMIT_DIR/../.."

# Create logs directory
mkdir -p hpc_scripts/logs

# Activate conda environment
source /om2/user/mabdel03/anaconda/etc/profile.d/conda.sh
conda activate /om/scratch/Mon/mabdel03/conda_envs/MAL_env

# Navigate to human_aware_rl
cd src

# Run PPO Self-Play training (FULL PAPER PARAMS - 650 iterations)
python -m human_aware_rl.ppo.train_ppo_sp --layout coordination_ring --seed 0
