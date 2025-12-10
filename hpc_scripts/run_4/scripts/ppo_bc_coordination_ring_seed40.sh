#!/bin/bash
#SBATCH --job-name=ppo_bc_coordination_ring_s40
#SBATCH --output=../logs/ppo_bc_coordination_ring_seed40_%j.out
#SBATCH --error=../logs/ppo_bc_coordination_ring_seed40_%j.err
#SBATCH --time=24:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8

# PPO_BC Training (PPO with BC partner)
# Run: 4
# Layout: coordination_ring, Seed: 40
# Training iterations: 650 (paper value)

# Navigate to project root
cd "$SLURM_SUBMIT_DIR/../../.."

# Create logs directory
mkdir -p "$SLURM_SUBMIT_DIR/../logs"

# Activate conda environment
source /om2/user/mabdel03/anaconda/etc/profile.d/conda.sh
conda activate /om/scratch/Mon/mabdel03/conda_envs/MAL_env

# Navigate to src for Python modules
cd src

echo "Run 4: Training PPO_BC for coordination_ring seed 40..."
echo "Paper iterations: 650"

# Uses BC models from bc_runs_run4/train/coordination_ring/
python -m human_aware_rl.ppo.train_ppo_bc --layout coordination_ring --seed 40 --bc_model_base_dir human_aware_rl/bc_runs_run4/train --results_dir results/ppo_bc_run4

echo "PPO_BC training complete for coordination_ring seed 40"
