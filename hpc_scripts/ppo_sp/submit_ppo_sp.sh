#!/bin/bash
# ============================================================================
# Submit all PPO Self-Play training jobs (25 = 5 layouts × 5 seeds)
# ============================================================================
# Usage: ./submit_ppo_sp.sh
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HPC_DIR="$(dirname "${SCRIPT_DIR}")"

# Ensure logs directory exists
mkdir -p "${HPC_DIR}/logs"

echo "Submitting PPO Self-Play jobs (25 = 5 layouts × 5 seeds)..."
echo "============================================================================"

COUNT=0
for layout in cramped_room asymmetric_advantages coordination_ring forced_coordination counter_circuit; do
    for seed in 0 10 20 30 40; do
        JOB_ID=$(sbatch --parsable "${SCRIPT_DIR}/${layout}_seed${seed}.sh")
        echo "  ppo_sp_${layout}_s${seed}: Job ${JOB_ID}"
        ((COUNT++))
    done
done

echo "============================================================================"
echo "PPO_SP jobs submitted: ${COUNT}"
echo "============================================================================"
