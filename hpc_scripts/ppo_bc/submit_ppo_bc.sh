#!/bin/bash
# ============================================================================
# Submit all PPO+BC training jobs (25 = 5 layouts × 5 seeds)
# ============================================================================
# Usage: ./submit_ppo_bc.sh
#
# NOTE: BC models must be trained first! These jobs should be submitted
# after BC training completes, or with --dependency=afterok:<bc_job_ids>
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HPC_DIR="$(dirname "${SCRIPT_DIR}")"

# Ensure logs directory exists
mkdir -p "${HPC_DIR}/logs"

echo "Submitting PPO+BC jobs (25 = 5 layouts × 5 seeds)..."
echo "============================================================================"

COUNT=0
for layout in cramped_room asymmetric_advantages coordination_ring forced_coordination counter_circuit; do
    for seed in 0 10 20 30 40; do
        JOB_ID=$(sbatch --parsable "${SCRIPT_DIR}/${layout}_seed${seed}.sh")
        echo "  ppo_bc_${layout}_s${seed}: Job ${JOB_ID}"
        ((COUNT++))
    done
done

echo "============================================================================"
echo "PPO_BC jobs submitted: ${COUNT}"
echo "============================================================================"
