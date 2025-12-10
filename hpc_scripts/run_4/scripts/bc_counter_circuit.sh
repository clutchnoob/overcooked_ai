#!/bin/bash
#SBATCH --job-name=bc_counter_circuit
#SBATCH --output=../logs/bc_counter_circuit_%j.out
#SBATCH --error=../logs/bc_counter_circuit_%j.err
#SBATCH --time=04:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4

# BC Training for counter_circuit
# Run: 4
# Trains both train (for PPO partner) and test (for Human Proxy) models

# Navigate to project root (scripts/ -> run_N/ -> hpc_scripts/ -> project root)
cd "$SLURM_SUBMIT_DIR/../../.."

# Create logs directory
mkdir -p "$SLURM_SUBMIT_DIR/../logs"

# Activate conda environment
source /om2/user/mabdel03/anaconda/etc/profile.d/conda.sh
conda activate /om/scratch/Mon/mabdel03/conda_envs/MAL_env

# Navigate to src for Python modules
cd src

echo "Run 4: Training BC models for counter_circuit..."

# Train BC models (both train and test) - outputs to bc_runs_run4/
# Note: BC trains both train and test splits; PPO_BC uses train/counter_circuit/
python -m human_aware_rl.imitation.train_bc_models --layout counter_circuit --output_base_dir human_aware_rl/bc_runs_run4

echo "BC training complete for counter_circuit"
