#!/bin/bash
#SBATCH --job-name=ppo_gail_counter_circuit_s20
#SBATCH --output=../logs/ppo_gail_counter_circuit_seed20_%j.out
#SBATCH --error=../logs/ppo_gail_counter_circuit_seed20_%j.err
#SBATCH --time=48:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=16

# ============================================================================
# PPO with GAIL Partner Training: counter_circuit (seed=20)
# ============================================================================
# Trains PPO agent with a GAIL partner trained on human demonstrations.
# GAIL models must be trained first (see hpc_scripts/gail/).
# Uses LEGACY layouts for consistency with PPO SP.
# ============================================================================

source "$(dirname "$0")/../config.sh"

log_start

echo "Layout: counter_circuit -> legacy version"
echo "Seed: 20"
echo "Training PPO with GAIL partner (using legacy layout for consistency)..."
echo ""

python -m human_aware_rl.ppo.train_ppo_gail \
    --layout counter_circuit \
    --seed 20 \
    --results_dir "${RESULTS_DIR}/ppo_gail"

EXIT_CODE=$?

log_end $EXIT_CODE
exit $EXIT_CODE
