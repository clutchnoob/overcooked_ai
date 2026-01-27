#!/bin/bash
# ============================================================================
# Submit all PPO with AIRL partner training jobs
# ============================================================================
# Usage: ./submit_ppo_airl.sh [--dependency JOB_IDS]
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HPC_DIR="$(dirname "${SCRIPT_DIR}")"

# Parse arguments
DEPENDENCY=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --dependency)
            DEPENDENCY="--dependency=afterok:$2"
            shift 2
            ;;
        *)
            shift
            ;;
    esac
done

mkdir -p "${HPC_DIR}/logs"

echo "Submitting PPO with AIRL partner training jobs..."
if [ -n "$DEPENDENCY" ]; then
    echo "Dependency: $DEPENDENCY"
fi
echo "============================================================================"

LAYOUTS="cramped_room asymmetric_advantages coordination_ring forced_coordination counter_circuit"
SEEDS="0 10 20 30 40"

declare -a PPO_AIRL_JOB_IDS
COUNT=0

for layout in $LAYOUTS; do
    for seed in $SEEDS; do
        JOB_ID=$(sbatch --parsable $DEPENDENCY "${SCRIPT_DIR}/${layout}_seed${seed}.sh")
        PPO_AIRL_JOB_IDS+=("$JOB_ID")
        echo "Submitted ppo_airl_${layout}_s${seed}: Job ID ${JOB_ID}"
        ((COUNT++))
    done
done

echo "============================================================================"
echo "Total PPO_AIRL jobs submitted: ${COUNT}"
echo "Job IDs: ${PPO_AIRL_JOB_IDS[@]}"
echo "============================================================================"

export PPO_AIRL_JOB_IDS
