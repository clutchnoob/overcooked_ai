#!/bin/bash
#SBATCH --job-name=bc_coord
#SBATCH --output=../logs/bc_coordination_ring_%j.out
#SBATCH --error=../logs/bc_coordination_ring_%j.err
#SBATCH --time=02:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4

# Navigate to project root
cd "$SLURM_SUBMIT_DIR/../.."

# Create logs directory
mkdir -p hpc_scripts/logs

# Activate conda environment
source /om2/user/mabdel03/anaconda/etc/profile.d/conda.sh
conda activate /om/scratch/Mon/mabdel03/conda_envs/MAL_env

# Navigate to human_aware_rl
cd src

# Run BC training (train + test models)
python -m human_aware_rl.imitation.train_bc_models --layout coordination_ring
