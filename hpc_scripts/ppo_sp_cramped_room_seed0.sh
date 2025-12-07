#!/bin/bash
#SBATCH --job-name=ppo_sp_cramped_s0
#SBATCH --output=logs/ppo_sp_cramped_room_seed0_%j.out
#SBATCH --error=logs/ppo_sp_cramped_room_seed0_%j.err
#SBATCH --time=24:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8

# PPO Self-Play Training
# Layout: cramped_room, Seed: 0
# Training iterations: 550 (paper value)

cd "$SLURM_SUBMIT_DIR/.."
source .venv/bin/activate || conda activate overcooked

mkdir -p "$SLURM_SUBMIT_DIR/logs"

echo "Training PPO_SP for cramped_room seed 0..."
echo "Paper iterations: 550"

python -m human_aware_rl.ppo.train_ppo_sp \
    --layout cramped_room \
    --seed 0 \
    --num_training_iters 550 \
    --results_dir results/ppo_sp

echo "PPO_SP training complete for cramped_room seed 0"

