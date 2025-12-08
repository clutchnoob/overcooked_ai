#!/bin/bash
#SBATCH --job-name=bc_forced_coordination
#SBATCH --output=logs/bc_forced_coordination_%j.out
#SBATCH --error=logs/bc_forced_coordination_%j.err
#SBATCH --time=04:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4

# BC Training for forced_coordination
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

echo "Training BC models for forced_coordination..."

# Train BC models (both train and test)
python -m human_aware_rl.imitation.train_bc_models --layout forced_coordination

echo "BC training complete for forced_coordination"
