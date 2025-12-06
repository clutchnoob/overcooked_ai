#!/bin/bash
# Submit all PPO_GAIL training jobs

# Create logs directory
mkdir -p logs

echo "Submitting PPO_GAIL training jobs..."

sbatch ppo_gail_cramped_room.sh
sbatch ppo_gail_asymmetric_advantages.sh
sbatch ppo_gail_coordination_ring.sh
sbatch ppo_gail_forced_coordination.sh
sbatch ppo_gail_counter_circuit.sh

echo "All PPO_GAIL jobs submitted!"
echo "Check status with: squeue -u \$USER"

