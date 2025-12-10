#!/bin/bash
#SBATCH --job-name=ppo_sp_counter_circuit_s40
#SBATCH --output=../logs/ppo_sp_counter_circuit_seed40_%j.out
#SBATCH --error=../logs/ppo_sp_counter_circuit_seed40_%j.err
#SBATCH --time=24:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8

# PPO Self-Play Training
# Run: 4
# Layout: counter_circuit, Seed: 40
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

echo "Run 4: Training PPO_SP for counter_circuit seed 40..."
echo "Paper iterations: 650"

python -m human_aware_rl.ppo.train_ppo_sp --layout counter_circuit --seed 40 --results_dir results/ppo_sp_run4

echo "PPO_SP training complete for counter_circuit seed 40"
