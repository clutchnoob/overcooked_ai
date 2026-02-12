#!/bin/bash
#SBATCH --job-name=ppo_sp_coordination_ring_s0
#SBATCH --output=../logs/ppo_sp_coordination_ring_seed0_%j.out
#SBATCH --error=../logs/ppo_sp_coordination_ring_seed0_%j.err
#SBATCH --time=12:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=16
#SBATCH --partition=sched_mit_hill,mit_normal,newnodes

# PPO Self-Play Training: coordination_ring (seed=0)

source "$(dirname "$0")/../config.sh"

log_start

echo "Layout: coordination_ring"
echo "Seed: 0"
echo "Training PPO Self-Play..."
echo ""

python -m human_aware_rl.ppo.train_ppo_sp \
    --layout coordination_ring \
    --seed 0 \
    --results_dir "${RESULTS_DIR}/ppo_sp"

EXIT_CODE=$?

log_end $EXIT_CODE
exit $EXIT_CODE
