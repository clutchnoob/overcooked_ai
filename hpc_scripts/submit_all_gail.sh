#!/bin/bash
# Submit all GAIL training jobs

# Create logs directory
mkdir -p logs

echo "Submitting GAIL training jobs..."

sbatch gail_cramped_room.sh
sbatch gail_asymmetric_advantages.sh
sbatch gail_coordination_ring.sh
sbatch gail_forced_coordination.sh
sbatch gail_counter_circuit.sh

echo "All GAIL jobs submitted!"
echo "Check status with: squeue -u \$USER"
