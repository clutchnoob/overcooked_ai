#!/bin/bash
#SBATCH --job-name=ppo_gail_asymmetric_advantages_s10
#SBATCH --output=../logs/ppo_gail_asymmetric_advantages_seed10_%j.out
#SBATCH --error=../logs/ppo_gail_asymmetric_advantages_seed10_%j.err
#SBATCH --time=48:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=16

# ============================================================================
# PPO with GAIL Partner Training: asymmetric_advantages (seed=10)
# ============================================================================
# Trains PPO agent with a GAIL partner trained on human demonstrations.
# GAIL models must be trained first (see hpc_scripts/gail/).
# Uses LEGACY layouts for consistency with PPO SP.
# ============================================================================

source "$(dirname "$0")/../config.sh"

log_start

echo "Layout: asymmetric_advantages -> legacy version"
echo "Seed: 10"
echo "Training PPO with GAIL partner (using legacy layout for consistency)..."
echo ""

python -m human_aware_rl.ppo.train_ppo_gail \
    --layout asymmetric_advantages \
    --seed 10 \
    --results_dir "${RESULTS_DIR}/ppo_gail"

EXIT_CODE=$?

log_end $EXIT_CODE
exit $EXIT_CODE
