#!/bin/bash
# ============================================================================
# Submit all BC training jobs
# ============================================================================
# Usage: ./submit_bc.sh
# Returns: Prints job IDs for dependency tracking
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HPC_DIR="$(dirname "${SCRIPT_DIR}")"

# Ensure logs directory exists
mkdir -p "${HPC_DIR}/logs"

echo "Submitting BC training jobs..."
echo "============================================================================"

# Array to store job IDs
declare -a BC_JOB_IDS

# Submit all BC jobs
for layout in cramped_room asymmetric_advantages coordination_ring forced_coordination counter_circuit; do
    JOB_ID=$(sbatch --parsable "${SCRIPT_DIR}/${layout}.sh")
    BC_JOB_IDS+=("$JOB_ID")
    echo "Submitted bc_${layout}: Job ID ${JOB_ID}"
done

echo "============================================================================"
echo "Total BC jobs submitted: ${#BC_JOB_IDS[@]}"
echo ""
echo "Job IDs (for dependency tracking):"
echo "${BC_JOB_IDS[@]}"
echo ""
echo "Dependency string for downstream jobs:"
echo "afterok:$(IFS=:; echo "${BC_JOB_IDS[*]}")"
echo "============================================================================"

# Export for use in master script
export BC_JOB_IDS
