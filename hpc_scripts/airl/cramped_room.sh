#!/bin/bash
#SBATCH --job-name=airl_cramped_room
#SBATCH --output=/om/scratch/Mon/mabdel03/6.S890/overcooked_ai/hpc_scripts/logs/%x_%j.out
#SBATCH --error=/om/scratch/Mon/mabdel03/6.S890/overcooked_ai/hpc_scripts/logs/%x_%j.err
#SBATCH --time=08:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --partition=normal

# Train AIRL model for cramped_room layout

# Use absolute path for config
source /om/scratch/Mon/mabdel03/6.S890/overcooked_ai/hpc_scripts/config.sh

log_start

python -m human_aware_rl.imitation.train_airl \
    --layout cramped_room

EXIT_CODE=$?
log_end $EXIT_CODE
exit $EXIT_CODE
