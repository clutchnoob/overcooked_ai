#!/bin/bash
#SBATCH --job-name=ppo_gail_forced_coordination_s20
#SBATCH --output=../logs/ppo_gail_run4/ppo_gail_forced_coordination_seed20_%j.out
#SBATCH --error=../logs/ppo_gail_run4/ppo_gail_forced_coordination_seed20_%j.err
#SBATCH --time=24:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8

# PPO_GAIL Training (PPO with GAIL partner instead of BC)
# Run: 4
# Layout: forced_coordination, Seed: 20
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

echo "Run 4: Training PPO_GAIL for forced_coordination seed 20..."
echo "Paper iterations: 650"

# Uses GAIL models from gail_runs_run4/forced_coordination/
python -m human_aware_rl.ppo.train_ppo_gail --layout forced_coordination --seed 20 --gail_model_base_dir human_aware_rl/gail_runs_run4 --results_dir results/ppo_gail_run4

echo "PPO_GAIL training complete for forced_coordination seed 20"
