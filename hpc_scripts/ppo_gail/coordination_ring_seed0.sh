#!/bin/bash
#SBATCH --job-name=ppo_gail_coordination_ring_s0
#SBATCH --output=../logs/ppo_gail_coordination_ring_seed0_%j.out
#SBATCH --error=../logs/ppo_gail_coordination_ring_seed0_%j.err
#SBATCH --time=48:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=16
#SBATCH --partition=normal

# ============================================================================
# PPO with GAIL Partner Training: coordination_ring (seed=0)
# ============================================================================
# PPO GAIL trains an agent with a GAIL partner trained on human demonstrations.
#
# IMPORTANT: Uses LEGACY layouts (same as PPO SP) for consistency!
# All experiments in the original paper used the same layout for all methods.
# Legacy layouts have explicit MDP params: cook_time=20, num_items=3, delivery=20
# ============================================================================

# Source config (sets up conda, paths, etc.)
source "$(dirname "$0")/../config.sh"

log_start

echo "Layout: coordination_ring -> legacy version"
echo "Seed: 0"
echo "Training PPO with GAIL partner (using legacy layout for consistency)..."
echo ""

python -m human_aware_rl.ppo.train_ppo_gail \
    --layout coordination_ring \
    --seed 0 \
    --results_dir "${RESULTS_DIR}/ppo_gail"

EXIT_CODE=$?

log_end $EXIT_CODE
exit $EXIT_CODE
