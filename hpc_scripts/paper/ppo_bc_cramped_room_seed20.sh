#!/bin/bash
#SBATCH --job-name=bc_cr_s20
#SBATCH --output=../logs/ppo_bc_cramped_room_seed20_%j.out
#SBATCH --error=../logs/ppo_bc_cramped_room_seed20_%j.err
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

# Run PPO with BC partner training (FULL PAPER PARAMS - 550 iterations)
python -m human_aware_rl.ppo.train_ppo_bc --layout cramped_room --seed 20
