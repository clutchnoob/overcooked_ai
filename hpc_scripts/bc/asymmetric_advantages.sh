#!/bin/bash
#SBATCH --job-name=bc_asymmetric_advantages
#SBATCH --output=../logs/%x_%j.out
#SBATCH --error=../logs/%x_%j.err
#SBATCH --time=04:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=8
#SBATCH --partition=sched_mit_hill,mit_normal,newnodes

# Train BC model for asymmetric_advantages layout

source "$(dirname "$0")/../config.sh"

log_start

python -m human_aware_rl.imitation.train_bc_models \
    --layout asymmetric_advantages

EXIT_CODE=$?
log_end $EXIT_CODE
exit $EXIT_CODE
