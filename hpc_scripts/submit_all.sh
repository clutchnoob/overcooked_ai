#!/bin/bash
# ============================================================================
# Master Submission Script - Submit All Training Jobs
# ============================================================================
# This script submits all training jobs with proper dependencies:
# 1. BC models (no dependencies)
# 2. PPO_SP (no dependencies, runs in parallel with BC)
# 3. PPO_BC, PPO_GAIL (depend on BC completion)
#
# Usage:
#   ./submit_all.sh              # Submit all jobs
#   ./submit_all.sh --bc-only    # Submit only BC jobs
#   ./submit_all.sh --ppo-only   # Submit only PPO jobs (assumes BC done)
#   ./submit_all.sh --dry-run    # Show what would be submitted
# ============================================================================

# Note: Not using set -e because ((COUNT++)) returns 1 when COUNT=0

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Parse arguments
BC_ONLY=false
PPO_ONLY=false
DRY_RUN=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --bc-only)
            BC_ONLY=true
            shift
            ;;
        --ppo-only)
            PPO_ONLY=true
            shift
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--bc-only|--ppo-only|--dry-run]"
            exit 1
            ;;
    esac
done

# Ensure logs directory exists
mkdir -p "${SCRIPT_DIR}/logs"

echo "============================================================================"
echo "Overcooked AI - HPC Training Job Submission"
echo "============================================================================"
echo "Date: $(date)"
echo "Script directory: ${SCRIPT_DIR}"
echo ""

# Function to submit BC jobs
submit_bc() {
    echo "Submitting BC training jobs (5 layouts)..."
    echo "--------------------------------------------"
    
    declare -a BC_JOB_IDS
    for layout in cramped_room asymmetric_advantages coordination_ring forced_coordination counter_circuit; do
        if [ "$DRY_RUN" = true ]; then
            echo "[DRY RUN] Would submit: bc/${layout}.sh"
            BC_JOB_IDS+=("DRY_RUN_${layout}")
        else
            JOB_ID=$(sbatch --parsable "${SCRIPT_DIR}/bc/${layout}.sh")
            BC_JOB_IDS+=("$JOB_ID")
            echo "  bc_${layout}: Job ${JOB_ID}"
        fi
    done
    
    echo ""
    echo "BC jobs submitted: ${#BC_JOB_IDS[@]}"
    
    # Return job IDs as colon-separated string
    BC_DEPENDENCY=$(IFS=:; echo "${BC_JOB_IDS[*]}")
    export BC_DEPENDENCY
}

# Function to submit PPO_SP jobs (no dependency)
submit_ppo_sp() {
    echo "Submitting PPO Self-Play jobs (25 = 5 layouts × 5 seeds)..."
    echo "--------------------------------------------"
    
    COUNT=0
    for layout in cramped_room asymmetric_advantages coordination_ring forced_coordination counter_circuit; do
        for seed in 0 10 20 30 40; do
            if [ "$DRY_RUN" = true ]; then
                echo "[DRY RUN] Would submit: ppo_sp/${layout}_seed${seed}.sh"
            else
                JOB_ID=$(sbatch --parsable "${SCRIPT_DIR}/ppo_sp/${layout}_seed${seed}.sh")
                echo "  ppo_sp_${layout}_s${seed}: Job ${JOB_ID}"
            fi
            ((COUNT++))
        done
    done
    
    echo ""
    echo "PPO_SP jobs submitted: ${COUNT}"
}

# Function to submit PPO with partner jobs (with BC dependency)
submit_ppo_with_partner() {
    local MODEL_TYPE=$1
    local DEPENDENCY=$2
    
    echo "Submitting PPO_${MODEL_TYPE} jobs (25 = 5 layouts × 5 seeds)..."
    if [ -n "$DEPENDENCY" ]; then
        echo "  Dependency: afterok:${DEPENDENCY}"
    fi
    echo "--------------------------------------------"
    
    local DEP_FLAG=""
    if [ -n "$DEPENDENCY" ]; then
        DEP_FLAG="--dependency=afterok:${DEPENDENCY}"
    fi
    
    COUNT=0
    for layout in cramped_room asymmetric_advantages coordination_ring forced_coordination counter_circuit; do
        for seed in 0 10 20 30 40; do
            SCRIPT_PATH="${SCRIPT_DIR}/ppo_$(echo ${MODEL_TYPE} | tr '[:upper:]' '[:lower:]')/${layout}_seed${seed}.sh"
            if [ "$DRY_RUN" = true ]; then
                echo "[DRY RUN] Would submit: ${SCRIPT_PATH}"
            else
                JOB_ID=$(sbatch --parsable $DEP_FLAG "${SCRIPT_PATH}")
                echo "  ppo_${MODEL_TYPE}_${layout}_s${seed}: Job ${JOB_ID}"
            fi
            ((COUNT++))
        done
    done
    
    echo ""
    echo "PPO_${MODEL_TYPE} jobs submitted: ${COUNT}"
}

# Main submission logic
TOTAL_JOBS=0

if [ "$PPO_ONLY" = false ]; then
    submit_bc
    TOTAL_JOBS=$((TOTAL_JOBS + 5))
    echo ""
fi

if [ "$BC_ONLY" = false ]; then
    # PPO_SP runs independently
    submit_ppo_sp
    TOTAL_JOBS=$((TOTAL_JOBS + 25))
    echo ""
    
    # PPO_BC and PPO_GAIL depend on BC
    if [ "$PPO_ONLY" = true ]; then
        # No dependency if running PPO only (assumes BC done)
        BC_DEPENDENCY=""
    fi
    
    submit_ppo_with_partner "BC" "${BC_DEPENDENCY:-}"
    TOTAL_JOBS=$((TOTAL_JOBS + 25))
    echo ""
    
    submit_ppo_with_partner "GAIL" "${BC_DEPENDENCY:-}"
    TOTAL_JOBS=$((TOTAL_JOBS + 25))
    echo ""
fi

echo "============================================================================"
echo "Submission Summary"
echo "============================================================================"
if [ "$DRY_RUN" = true ]; then
    echo "[DRY RUN] Would have submitted ${TOTAL_JOBS} jobs"
else
    echo "Total jobs submitted: ${TOTAL_JOBS}"
    echo ""
    echo "Monitor jobs with:"
    echo "  squeue -u \$USER"
    echo ""
    echo "Check logs in:"
    echo "  ${SCRIPT_DIR}/logs/"
fi
echo "============================================================================"
