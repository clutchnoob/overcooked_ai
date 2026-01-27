#!/bin/bash
# ============================================================================
# Submit all GAIL training jobs
# ============================================================================
# Usage: ./submit_gail.sh
# Returns: Prints job IDs for dependency tracking
#
# GAIL models are a prerequisite for PPO_GAIL training.
# After GAIL jobs complete, you can submit PPO_GAIL jobs.
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HPC_DIR="$(dirname "${SCRIPT_DIR}")"

# Ensure logs directory exists
mkdir -p "${HPC_DIR}/logs"

echo "Submitting GAIL training jobs..."
echo "============================================================================"

# Array to store job IDs
declare -a GAIL_JOB_IDS

# Submit all GAIL jobs
for layout in cramped_room asymmetric_advantages coordination_ring forced_coordination counter_circuit; do
    JOB_ID=$(sbatch --parsable "${SCRIPT_DIR}/${layout}.sh")
    GAIL_JOB_IDS+=("$JOB_ID")
    echo "Submitted gail_${layout}: Job ID ${JOB_ID}"
done

echo "============================================================================"
echo "Total GAIL jobs submitted: ${#GAIL_JOB_IDS[@]}"
echo ""
echo "Job IDs (for dependency tracking):"
echo "${GAIL_JOB_IDS[@]}"
echo ""
echo "After GAIL jobs complete, submit PPO_GAIL with:"
echo "  cd ${HPC_DIR}/ppo_gail && ./submit_ppo_gail.sh"
echo "============================================================================"

# Export for use in master script
export GAIL_JOB_IDS
