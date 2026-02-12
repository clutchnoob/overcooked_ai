#!/bin/bash
#SBATCH --job-name=ppo_sp_counter_circuit_s10
#SBATCH --output=../logs/ppo_sp_counter_circuit_seed10_%j.out
#SBATCH --error=../logs/ppo_sp_counter_circuit_seed10_%j.err
#SBATCH --time=12:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=16
#SBATCH --partition=sched_mit_hill,mit_normal,newnodes

# PPO Self-Play Training: counter_circuit (seed=10)

source "$(dirname "$0")/../config.sh"

log_start

echo "Layout: counter_circuit"
echo "Seed: 10"
echo "Training PPO Self-Play..."
echo ""

python -m human_aware_rl.ppo.train_ppo_sp \
    --layout counter_circuit \
    --seed 10 \
    --results_dir "${RESULTS_DIR}/ppo_sp"

EXIT_CODE=$?

log_end $EXIT_CODE
exit $EXIT_CODE
