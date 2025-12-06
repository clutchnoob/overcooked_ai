#!/bin/bash
# Submits all 5 PPO Self-Play jobs in parallel (each on separate resources)
cd "$(dirname "$0")"

echo "Submitting PPO Self-Play training jobs..."
sbatch ppo_sp_cramped_room.sh
sbatch ppo_sp_asymmetric_advantages.sh
sbatch ppo_sp_coordination_ring.sh
sbatch ppo_sp_forced_coordination.sh
sbatch ppo_sp_counter_circuit.sh

echo ""
echo "Submitted 5 PPO_SP jobs (each on separate resources)"
echo "Check status with: squeue -u \$USER"

