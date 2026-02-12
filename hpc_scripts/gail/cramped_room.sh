#!/bin/bash
#SBATCH --job-name=gail_cramped_room
#SBATCH --output=../logs/%x_%j.out
#SBATCH --error=../logs/%x_%j.err
#SBATCH --time=08:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8

# Train GAIL model for cramped_room layout

source "$(dirname "$0")/../config.sh"

log_start

python -m human_aware_rl.imitation.gail \
    --layout cramped_room

EXIT_CODE=$?
log_end $EXIT_CODE
exit $EXIT_CODE
