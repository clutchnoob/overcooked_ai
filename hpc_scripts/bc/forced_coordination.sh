#!/bin/bash
#SBATCH --job-name=bc_forced_coordination
#SBATCH --output=/om/scratch/Mon/mabdel03/6.S890/overcooked_ai/hpc_scripts/logs/%x_%j.out
#SBATCH --error=/om/scratch/Mon/mabdel03/6.S890/overcooked_ai/hpc_scripts/logs/%x_%j.err
#SBATCH --time=04:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=8
#SBATCH --partition=normal

# Train BC model for forced_coordination layout

# Use absolute path for config
source /om/scratch/Mon/mabdel03/6.S890/overcooked_ai/hpc_scripts/config.sh

log_start

python -m human_aware_rl.imitation.train_bc_models \
    --layout forced_coordination

EXIT_CODE=$?
log_end $EXIT_CODE
exit $EXIT_CODE
