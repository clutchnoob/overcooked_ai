#!/bin/bash
# Submits all 5 BC jobs in parallel (each on separate resources)
cd "$(dirname "$0")"

echo "Submitting BC training jobs..."
sbatch bc_cramped_room.sh
sbatch bc_asymmetric_advantages.sh
sbatch bc_coordination_ring.sh
sbatch bc_forced_coordination.sh
sbatch bc_counter_circuit.sh

echo ""
echo "Submitted 5 BC jobs (each on separate resources)"
echo "Check status with: squeue -u \$USER"

