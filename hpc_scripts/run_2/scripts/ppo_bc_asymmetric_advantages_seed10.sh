#!/bin/bash
#SBATCH --job-name=bc_aa_s10
#SBATCH --output=../logs/ppo_bc_asymmetric_advantages_seed10_%j.out
#SBATCH --error=../logs/ppo_bc_asymmetric_advantages_seed10_%j.err
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

# Run PPO with BC partner training (FULL PAPER PARAMS - 650 iterations)
python -m human_aware_rl.ppo.train_ppo_bc --layout asymmetric_advantages --seed 10
