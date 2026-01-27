#!/bin/bash
#SBATCH --job-name=ppo_airl_cramped_room_s20
#SBATCH --output=/om/scratch/Mon/mabdel03/6.S890/overcooked_ai/hpc_scripts/logs/%x_%j.out
#SBATCH --error=/om/scratch/Mon/mabdel03/6.S890/overcooked_ai/hpc_scripts/logs/%x_%j.err
#SBATCH --time=48:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=16
#SBATCH --partition=normal

# Train PPO with AIRL partner for cramped_room layout with seed 20

# Use absolute path for config
source /om/scratch/Mon/mabdel03/6.S890/overcooked_ai/hpc_scripts/config.sh

log_start

python -m human_aware_rl.ppo.train_ppo_airl \
    --layout cramped_room \
    --seed 20 \
    --results_dir "${RESULTS_DIR}/ppo_airl"

EXIT_CODE=$?
log_end $EXIT_CODE
exit $EXIT_CODE
