#!/bin/bash
#SBATCH --job-name=gail_forced_coordination
#SBATCH --output=../logs/%x_%j.out
#SBATCH --error=../logs/%x_%j.err
#SBATCH --time=08:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --partition=sched_mit_hill,mit_normal,newnodes

# Train GAIL model for forced_coordination layout

source "$(dirname "$0")/../config.sh"

log_start

python -m human_aware_rl.imitation.gail \
    --layout forced_coordination

EXIT_CODE=$?
log_end $EXIT_CODE
exit $EXIT_CODE
