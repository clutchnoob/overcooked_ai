#!/bin/bash
#SBATCH --job-name=ppo_bc_cramped_room_s40
#SBATCH --output=../logs/ppo_bc_cramped_room_seed40_%j.out
#SBATCH --error=../logs/ppo_bc_cramped_room_seed40_%j.err
#SBATCH --time=12:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=16
#SBATCH --partition=sched_mit_hill,mit_normal,newnodes

# PPO with BC Partner Training: cramped_room (seed=40)

source "$(dirname "$0")/../config.sh"

log_start

echo "Layout: cramped_room -> legacy version"
echo "Seed: 40"
echo "Training PPO with BC partner..."
echo ""

python -m human_aware_rl.ppo.train_ppo_bc \
    --layout cramped_room \
    --seed 40 \
    --results_dir "${RESULTS_DIR}/ppo_bc"

EXIT_CODE=$?

log_end $EXIT_CODE
exit $EXIT_CODE
