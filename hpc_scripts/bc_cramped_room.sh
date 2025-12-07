#!/bin/bash
#SBATCH --job-name=bc_cramped_room
#SBATCH --output=logs/bc_cramped_room_%j.out
#SBATCH --error=logs/bc_cramped_room_%j.err
#SBATCH --time=04:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4

# BC Training for cramped_room
# Trains both train (for PPO partner) and test (for Human Proxy) models

cd "$SLURM_SUBMIT_DIR/.."
source .venv/bin/activate || conda activate overcooked

mkdir -p "$SLURM_SUBMIT_DIR/logs"

echo "Training BC models for cramped_room..."

# Train BC on training data (for PPO_BC partner)
python -m human_aware_rl.imitation.behavior_cloning \
    --layout cramped_room \
    --data_type train \
    --epochs 100

# Train BC on test data (for Human Proxy evaluation)
python -m human_aware_rl.imitation.behavior_cloning \
    --layout cramped_room \
    --data_type test \
    --epochs 100

echo "BC training complete for cramped_room"
