#!/bin/bash
#SBATCH --job-name=ppo_sp_asymmetric_advantages_s10
#SBATCH --output=logs/ppo_sp_asymmetric_advantages_seed10_%j.out
#SBATCH --error=logs/ppo_sp_asymmetric_advantages_seed10_%j.err
#SBATCH --time=24:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8

# PPO Self-Play Training
# Layout: asymmetric_advantages, Seed: 10
# Training iterations: 650 (paper value)

# Navigate to project root
cd "$SLURM_SUBMIT_DIR/.."

# Create logs directory
mkdir -p "$SLURM_SUBMIT_DIR/logs"

# Activate conda environment
source /om2/user/mabdel03/anaconda/etc/profile.d/conda.sh
conda activate /om/scratch/Mon/mabdel03/conda_envs/MAL_env

# Navigate to src for Python modules
cd src

echo "Training PPO_SP for asymmetric_advantages seed 10..."
echo "Paper iterations: 650"

python -m human_aware_rl.ppo.train_ppo_sp --layout asymmetric_advantages --seed 10

echo "PPO_SP training complete for asymmetric_advantages seed 10"
