#!/bin/bash
# ============================================================================
# Submit all AIRL training jobs
# ============================================================================
# Usage: ./submit_airl.sh
# Returns: Prints job IDs for dependency tracking
#
# AIRL models are a prerequisite for PPO_AIRL training.
# After AIRL jobs complete, you can submit PPO_AIRL jobs.
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HPC_DIR="$(dirname "${SCRIPT_DIR}")"

# Ensure logs directory exists
mkdir -p "${HPC_DIR}/logs"

echo "Submitting AIRL training jobs..."
echo "============================================================================"

# Array to store job IDs
declare -a AIRL_JOB_IDS

# Submit all AIRL jobs
for layout in cramped_room asymmetric_advantages coordination_ring forced_coordination counter_circuit; do
    JOB_ID=$(sbatch --parsable "${SCRIPT_DIR}/${layout}.sh")
    AIRL_JOB_IDS+=("$JOB_ID")
    echo "Submitted airl_${layout}: Job ID ${JOB_ID}"
done

echo "============================================================================"
echo "Total AIRL jobs submitted: ${#AIRL_JOB_IDS[@]}"
echo ""
echo "Job IDs (for dependency tracking):"
echo "${AIRL_JOB_IDS[@]}"
echo ""
echo "After AIRL jobs complete, submit PPO_AIRL with:"
echo "  cd ${HPC_DIR}/ppo_airl && ./submit_ppo_airl.sh"
echo "============================================================================"

# Export for use in master script
export AIRL_JOB_IDS
