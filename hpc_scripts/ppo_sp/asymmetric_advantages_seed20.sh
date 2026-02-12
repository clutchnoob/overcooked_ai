#!/bin/bash
#SBATCH --job-name=ppo_sp_asymmetric_advantages_s20
#SBATCH --output=../logs/ppo_sp_asymmetric_advantages_seed20_%j.out
#SBATCH --error=../logs/ppo_sp_asymmetric_advantages_seed20_%j.err
#SBATCH --time=12:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=16
#SBATCH --partition=sched_mit_hill,mit_normal,newnodes

# PPO Self-Play Training: asymmetric_advantages (seed=20)

source "$(dirname "$0")/../config.sh"

log_start

echo "Layout: asymmetric_advantages"
echo "Seed: 20"
echo "Training PPO Self-Play..."
echo ""

python -m human_aware_rl.ppo.train_ppo_sp \
    --layout asymmetric_advantages \
    --seed 20 \
    --results_dir "${RESULTS_DIR}/ppo_sp"

EXIT_CODE=$?

log_end $EXIT_CODE
exit $EXIT_CODE
