#!/bin/bash
# Submit all PPO_SP training jobs for Run 4
# Total jobs: 25

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Run 4: Submitting PPO_SP training jobs..."

sbatch "$SCRIPT_DIR/ppo_sp_cramped_room_seed0.sh"
sbatch "$SCRIPT_DIR/ppo_sp_cramped_room_seed10.sh"
sbatch "$SCRIPT_DIR/ppo_sp_cramped_room_seed20.sh"
sbatch "$SCRIPT_DIR/ppo_sp_cramped_room_seed30.sh"
sbatch "$SCRIPT_DIR/ppo_sp_cramped_room_seed40.sh"
sbatch "$SCRIPT_DIR/ppo_sp_asymmetric_advantages_seed0.sh"
sbatch "$SCRIPT_DIR/ppo_sp_asymmetric_advantages_seed10.sh"
sbatch "$SCRIPT_DIR/ppo_sp_asymmetric_advantages_seed20.sh"
sbatch "$SCRIPT_DIR/ppo_sp_asymmetric_advantages_seed30.sh"
sbatch "$SCRIPT_DIR/ppo_sp_asymmetric_advantages_seed40.sh"
sbatch "$SCRIPT_DIR/ppo_sp_coordination_ring_seed0.sh"
sbatch "$SCRIPT_DIR/ppo_sp_coordination_ring_seed10.sh"
sbatch "$SCRIPT_DIR/ppo_sp_coordination_ring_seed20.sh"
sbatch "$SCRIPT_DIR/ppo_sp_coordination_ring_seed30.sh"
sbatch "$SCRIPT_DIR/ppo_sp_coordination_ring_seed40.sh"
sbatch "$SCRIPT_DIR/ppo_sp_forced_coordination_seed0.sh"
sbatch "$SCRIPT_DIR/ppo_sp_forced_coordination_seed10.sh"
sbatch "$SCRIPT_DIR/ppo_sp_forced_coordination_seed20.sh"
sbatch "$SCRIPT_DIR/ppo_sp_forced_coordination_seed30.sh"
sbatch "$SCRIPT_DIR/ppo_sp_forced_coordination_seed40.sh"
sbatch "$SCRIPT_DIR/ppo_sp_counter_circuit_seed0.sh"
sbatch "$SCRIPT_DIR/ppo_sp_counter_circuit_seed10.sh"
sbatch "$SCRIPT_DIR/ppo_sp_counter_circuit_seed20.sh"
sbatch "$SCRIPT_DIR/ppo_sp_counter_circuit_seed30.sh"
sbatch "$SCRIPT_DIR/ppo_sp_counter_circuit_seed40.sh"

echo ""
echo "Submitted 25 PPO_SP jobs for Run 4"
