#!/bin/bash
# Submit all GAIL training jobs for Run 4
# Total jobs: 5

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Run 4: Submitting GAIL training jobs..."

sbatch "$SCRIPT_DIR/gail_cramped_room.sh"
sbatch "$SCRIPT_DIR/gail_asymmetric_advantages.sh"
sbatch "$SCRIPT_DIR/gail_coordination_ring.sh"
sbatch "$SCRIPT_DIR/gail_forced_coordination.sh"
sbatch "$SCRIPT_DIR/gail_counter_circuit.sh"

echo ""
echo "Submitted 5 GAIL jobs for Run 4"
