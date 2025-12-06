#!/bin/bash
# Submits all 5 PPO_BC jobs in parallel (each on separate resources)
cd "$(dirname "$0")"

echo "Submitting PPO BC training jobs..."
sbatch ppo_bc_cramped_room.sh
sbatch ppo_bc_asymmetric_advantages.sh
sbatch ppo_bc_coordination_ring.sh
sbatch ppo_bc_forced_coordination.sh
sbatch ppo_bc_counter_circuit.sh

echo ""
echo "Submitted 5 PPO_BC jobs (each on separate resources)"
echo "Check status with: squeue -u \$USER"

