#!/bin/bash
#SBATCH --job-name=bc_coordination_ring
#SBATCH --output=../logs/%x_%j.out
#SBATCH --error=../logs/%x_%j.err
#SBATCH --time=04:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=8

# Train BC model for coordination_ring layout

source "$(dirname "$0")/../config.sh"

log_start

python -m human_aware_rl.imitation.train_bc_models \
    --layout coordination_ring

EXIT_CODE=$?
log_end $EXIT_CODE
exit $EXIT_CODE
