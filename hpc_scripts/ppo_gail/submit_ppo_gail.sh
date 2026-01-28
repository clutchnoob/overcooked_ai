#!/bin/bash
# ============================================================================
# Submit all PPO GAIL training jobs
# ============================================================================
# This script submits SLURM jobs for PPO with GAIL partner training.
# PPO GAIL uses GAIL (Generative Adversarial Imitation Learning) models
# as training partners.
#
# Usage:
#   cd hpc_scripts/ppo_gail
#   ./submit_ppo_gail.sh
#
# Total jobs: 25 (5 layouts × 5 seeds)
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=============================================="
echo "Submitting PPO GAIL Training Jobs"
echo "=============================================="
echo ""

LAYOUTS=("cramped_room" "asymmetric_advantages" "coordination_ring" "forced_coordination" "counter_circuit")
SEEDS=(0 10 20 30 40)

total_jobs=0
submitted_jobs=0

for layout in "${LAYOUTS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        script="${layout}_seed${seed}.sh"
        if [ -f "$script" ]; then
            echo "Submitting: $script"
            sbatch "$script"
            ((submitted_jobs++))
        else
            echo "WARNING: Script not found: $script"
        fi
        ((total_jobs++))
    done
done

echo ""
echo "=============================================="
echo "Submitted $submitted_jobs / $total_jobs jobs"
echo "=============================================="
echo ""
echo "To check job status: squeue -u \$USER"
echo "To cancel all jobs: scancel -u \$USER"
