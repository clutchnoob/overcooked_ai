#!/bin/bash
# Submit all paper reproduction jobs

cd "$(dirname "$0")/paper"

echo '=========================================='
echo 'Submitting all paper reproduction jobs'
echo '=========================================='

# BC jobs (must complete first - no dependencies for now)
echo 'Submitting BC jobs...'
sbatch bc_cramped_room.sh
sbatch bc_asymmetric_advantages.sh
sbatch bc_coordination_ring.sh
sbatch bc_forced_coordination.sh
sbatch bc_counter_circuit.sh

# PPO Self-Play jobs
echo 'Submitting PPO Self-Play jobs...'
sbatch ppo_sp_cramped_room_seed0.sh
sbatch ppo_sp_cramped_room_seed10.sh
sbatch ppo_sp_cramped_room_seed20.sh
sbatch ppo_sp_cramped_room_seed30.sh
sbatch ppo_sp_cramped_room_seed40.sh
sbatch ppo_sp_asymmetric_advantages_seed0.sh
sbatch ppo_sp_asymmetric_advantages_seed10.sh
sbatch ppo_sp_asymmetric_advantages_seed20.sh
sbatch ppo_sp_asymmetric_advantages_seed30.sh
sbatch ppo_sp_asymmetric_advantages_seed40.sh
sbatch ppo_sp_coordination_ring_seed0.sh
sbatch ppo_sp_coordination_ring_seed10.sh
sbatch ppo_sp_coordination_ring_seed20.sh
sbatch ppo_sp_coordination_ring_seed30.sh
sbatch ppo_sp_coordination_ring_seed40.sh
sbatch ppo_sp_forced_coordination_seed0.sh
sbatch ppo_sp_forced_coordination_seed10.sh
sbatch ppo_sp_forced_coordination_seed20.sh
sbatch ppo_sp_forced_coordination_seed30.sh
sbatch ppo_sp_forced_coordination_seed40.sh
sbatch ppo_sp_counter_circuit_seed0.sh
sbatch ppo_sp_counter_circuit_seed10.sh
sbatch ppo_sp_counter_circuit_seed20.sh
sbatch ppo_sp_counter_circuit_seed30.sh
sbatch ppo_sp_counter_circuit_seed40.sh

# PPO with BC partner jobs
echo 'Submitting PPO_BC jobs...'
sbatch ppo_bc_cramped_room_seed0.sh
sbatch ppo_bc_cramped_room_seed10.sh
sbatch ppo_bc_cramped_room_seed20.sh
sbatch ppo_bc_cramped_room_seed30.sh
sbatch ppo_bc_cramped_room_seed40.sh
sbatch ppo_bc_asymmetric_advantages_seed0.sh
sbatch ppo_bc_asymmetric_advantages_seed10.sh
sbatch ppo_bc_asymmetric_advantages_seed20.sh
sbatch ppo_bc_asymmetric_advantages_seed30.sh
sbatch ppo_bc_asymmetric_advantages_seed40.sh
sbatch ppo_bc_coordination_ring_seed0.sh
sbatch ppo_bc_coordination_ring_seed10.sh
sbatch ppo_bc_coordination_ring_seed20.sh
sbatch ppo_bc_coordination_ring_seed30.sh
sbatch ppo_bc_coordination_ring_seed40.sh
sbatch ppo_bc_forced_coordination_seed0.sh
sbatch ppo_bc_forced_coordination_seed10.sh
sbatch ppo_bc_forced_coordination_seed20.sh
sbatch ppo_bc_forced_coordination_seed30.sh
sbatch ppo_bc_forced_coordination_seed40.sh
sbatch ppo_bc_counter_circuit_seed0.sh
sbatch ppo_bc_counter_circuit_seed10.sh
sbatch ppo_bc_counter_circuit_seed20.sh
sbatch ppo_bc_counter_circuit_seed30.sh
sbatch ppo_bc_counter_circuit_seed40.sh

echo '=========================================='
echo 'All jobs submitted!'
echo 'Total: 5 BC + 25 PPO_SP + 25 PPO_BC = 55 jobs'
echo 'Check status with: squeue -u $USER'
echo '=========================================='