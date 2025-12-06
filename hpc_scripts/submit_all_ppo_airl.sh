#!/bin/bash
# Submit all PPO_AIRL training jobs
# NOTE: Run this AFTER AIRL training is complete

echo "Submitting PPO_AIRL training jobs..."
echo "NOTE: AIRL models must be trained first!"

sbatch ppo_airl_cramped_room.sh
sbatch ppo_airl_asymmetric_advantages.sh
sbatch ppo_airl_coordination_ring.sh
sbatch ppo_airl_forced_coordination.sh
sbatch ppo_airl_counter_circuit.sh

echo "All PPO_AIRL jobs submitted!"
echo "Check status with: squeue -u \$USER"

