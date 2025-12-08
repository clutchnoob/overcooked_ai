#!/bin/bash
#SBATCH --job-name=ppo_bc_forced_coordination_s40
#SBATCH --output=logs/ppo_bc_forced_coordination_seed40_%j.out
#SBATCH --error=logs/ppo_bc_forced_coordination_seed40_%j.err
#SBATCH --time=24:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8

# PPO_BC Training (PPO with BC partner)
# Layout: forced_coordination, Seed: 40
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

echo "Training PPO_BC for forced_coordination seed 40..."
echo "Paper iterations: 650"

python -m human_aware_rl.ppo.train_ppo_bc --layout forced_coordination --seed 40

echo "PPO_BC training complete for forced_coordination seed 40"
