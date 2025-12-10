#!/bin/bash
# Submit all BC training jobs for Run 4
# Total jobs: 5

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Run 4: Submitting BC training jobs..."

sbatch "$SCRIPT_DIR/bc_cramped_room.sh"
sbatch "$SCRIPT_DIR/bc_asymmetric_advantages.sh"
sbatch "$SCRIPT_DIR/bc_coordination_ring.sh"
sbatch "$SCRIPT_DIR/bc_forced_coordination.sh"
sbatch "$SCRIPT_DIR/bc_counter_circuit.sh"

echo ""
echo "Submitted 5 BC jobs for Run 4"
