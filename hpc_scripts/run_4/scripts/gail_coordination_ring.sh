#!/bin/bash
#SBATCH --job-name=gail_coordination_ring
#SBATCH --output=../logs/gail_coordination_ring_%j.out
#SBATCH --error=../logs/gail_coordination_ring_%j.err
#SBATCH --time=08:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8

# GAIL Training for coordination_ring
# Run: 4

# Navigate to project root
cd "$SLURM_SUBMIT_DIR/../../.."

# Create logs directory
mkdir -p "$SLURM_SUBMIT_DIR/../logs"

# Activate conda environment
source /om2/user/mabdel03/anaconda/etc/profile.d/conda.sh
conda activate /om/scratch/Mon/mabdel03/conda_envs/MAL_env

# Navigate to src for Python modules
cd src

echo "Run 4: Training GAIL model for coordination_ring..."

python -m human_aware_rl.imitation.gail --layout coordination_ring --results_dir human_aware_rl/gail_runs_run4

echo "GAIL training complete for coordination_ring"
