#!/bin/bash
#SBATCH --job-name=ppo_sp_forced_coordination_s10
#SBATCH --output=../logs/ppo_sp_forced_coordination_seed10_%j.out
#SBATCH --error=../logs/ppo_sp_forced_coordination_seed10_%j.err
#SBATCH --time=12:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=16
#SBATCH --partition=sched_mit_hill,mit_normal,newnodes

# PPO Self-Play Training: forced_coordination (seed=10)

source "$(dirname "$0")/../config.sh"

log_start

echo "Layout: forced_coordination"
echo "Seed: 10"
echo "Training PPO Self-Play..."
echo ""

python -m human_aware_rl.ppo.train_ppo_sp \
    --layout forced_coordination \
    --seed 10 \
    --results_dir "${RESULTS_DIR}/ppo_sp"

EXIT_CODE=$?

log_end $EXIT_CODE
exit $EXIT_CODE
