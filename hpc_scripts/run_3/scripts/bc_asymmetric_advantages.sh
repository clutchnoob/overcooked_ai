#!/bin/bash
#SBATCH --job-name=bc_asymmetric_advantages
#SBATCH --output=logs/bc_asymmetric_advantages_%j.out
#SBATCH --error=logs/bc_asymmetric_advantages_%j.err
#SBATCH --time=04:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4

# BC Training for asymmetric_advantages
# Trains both train (for PPO partner) and test (for Human Proxy) models

# Navigate to project root
cd "$SLURM_SUBMIT_DIR/.."

# Create logs directory
mkdir -p "$SLURM_SUBMIT_DIR/logs"

# Activate conda environment
source /om2/user/mabdel03/anaconda/etc/profile.d/conda.sh
conda activate /om/scratch/Mon/mabdel03/conda_envs/MAL_env

# Navigate to src for Python modules
cd src

echo "Training BC models for asymmetric_advantages..."

# Train BC models (both train and test)
python -m human_aware_rl.imitation.train_bc_models --layout asymmetric_advantages

echo "BC training complete for asymmetric_advantages"
