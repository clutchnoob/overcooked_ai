#!/bin/bash
# RERUN: Submit all PPO_GAIL training jobs for Run 4 (Bug Fix Applied)
# Total jobs: 25 (5 layouts x 5 seeds)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Change to script directory so sbatch submission context is correct
cd "$SCRIPT_DIR"

echo "============================================="
echo "Run 4: RERUN PPO_GAIL Training (Bug Fix)"
echo "============================================="
echo ""
echo "Submitting 25 PPO_GAIL jobs..."
echo ""

# cramped_room (5 seeds)
sbatch "ppo_gail_cramped_room_seed0.sh"
sbatch "ppo_gail_cramped_room_seed10.sh"
sbatch "ppo_gail_cramped_room_seed20.sh"
sbatch "ppo_gail_cramped_room_seed30.sh"
sbatch "ppo_gail_cramped_room_seed40.sh"

# asymmetric_advantages (5 seeds)
sbatch "ppo_gail_asymmetric_advantages_seed0.sh"
sbatch "ppo_gail_asymmetric_advantages_seed10.sh"
sbatch "ppo_gail_asymmetric_advantages_seed20.sh"
sbatch "ppo_gail_asymmetric_advantages_seed30.sh"
sbatch "ppo_gail_asymmetric_advantages_seed40.sh"

# coordination_ring (5 seeds)
sbatch "ppo_gail_coordination_ring_seed0.sh"
sbatch "ppo_gail_coordination_ring_seed10.sh"
sbatch "ppo_gail_coordination_ring_seed20.sh"
sbatch "ppo_gail_coordination_ring_seed30.sh"
sbatch "ppo_gail_coordination_ring_seed40.sh"

# forced_coordination (5 seeds)
sbatch "ppo_gail_forced_coordination_seed0.sh"
sbatch "ppo_gail_forced_coordination_seed10.sh"
sbatch "ppo_gail_forced_coordination_seed20.sh"
sbatch "ppo_gail_forced_coordination_seed30.sh"
sbatch "ppo_gail_forced_coordination_seed40.sh"

# counter_circuit (5 seeds)
sbatch "ppo_gail_counter_circuit_seed0.sh"
sbatch "ppo_gail_counter_circuit_seed10.sh"
sbatch "ppo_gail_counter_circuit_seed20.sh"
sbatch "ppo_gail_counter_circuit_seed30.sh"
sbatch "ppo_gail_counter_circuit_seed40.sh"

echo ""
echo "============================================="
echo "Submitted 25 PPO_GAIL jobs for Run 4 RERUN"
echo "============================================="
echo ""
echo "Monitor with: squeue -u \$USER | grep ppo_gail"

