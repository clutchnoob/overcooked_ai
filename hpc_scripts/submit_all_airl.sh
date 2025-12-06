#!/bin/bash
# Submit all AIRL training jobs

echo "Submitting AIRL training jobs..."

sbatch airl_cramped_room.sh
sbatch airl_asymmetric_advantages.sh
sbatch airl_coordination_ring.sh
sbatch airl_forced_coordination.sh
sbatch airl_counter_circuit.sh

echo "All AIRL jobs submitted!"
echo "Check status with: squeue -u \$USER"

