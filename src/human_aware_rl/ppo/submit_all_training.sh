#!/usr/bin/env bash
#
# Master launcher: submits one SLURM job per (experiment, layout, seed).
# Each job gets its own allocation so all runs train in parallel.
#
# Usage:
#   bash submit_all_training.sh                    # submit all experiments
#   bash submit_all_training.sh --dry-run          # print commands without submitting
#   bash submit_all_training.sh --only sp          # only PPO_SP
#   bash submit_all_training.sh --only bc          # only PPO_BC
#   bash submit_all_training.sh --only gail        # only PPO_GAIL

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
RESULTS_DIR="${RESULTS_DIR:-$REPO_ROOT/results}"
BC_MODEL_BASE="${BC_MODEL_BASE:-$REPO_ROOT/src/human_aware_rl/imitation/bc_runs/train}"
GAIL_MODEL_BASE="${GAIL_MODEL_BASE:-$REPO_ROOT/src/human_aware_rl/gail/gail_runs}"

LAYOUTS="cramped_room asymmetric_advantages coordination_ring forced_coordination counter_circuit"
SEEDS="0 10 20 30 40"

# SLURM resource defaults (override via env vars)
PARTITION="${PARTITION:-normal}"
MEM="${MEM:-48G}"
CPUS="${CPUS:-16}"
TIME="${TIME:-24:00:00}"
LOG_DIR="${LOG_DIR:-$RESULTS_DIR/slurm_logs}"

DRY_RUN=false
ONLY=""

for arg in "$@"; do
    case $arg in
        --dry-run) DRY_RUN=true ;;
        --only)    shift; ONLY="$1" ;;
        sp|bc|gail) ONLY="$arg" ;;
    esac
done

mkdir -p "$LOG_DIR"

submit_job() {
    local job_name="$1"
    local python_cmd="$2"

    local sbatch_script
    sbatch_script=$(cat <<SBATCH_EOF
#!/usr/bin/env bash
#SBATCH --job-name=${job_name}
#SBATCH --output=${LOG_DIR}/${job_name}_%j.out
#SBATCH --error=${LOG_DIR}/${job_name}_%j.err
#SBATCH --partition=${PARTITION}
#SBATCH --mem=${MEM}
#SBATCH --cpus-per-task=${CPUS}
#SBATCH --time=${TIME}
#SBATCH --nodes=1
#SBATCH --ntasks=1

echo "=========================================="
echo "Job:       \${SLURM_JOB_NAME}"
echo "Job ID:    \${SLURM_JOB_ID}"
echo "Node:      \${SLURM_NODELIST}"
echo "CPUs:      \${SLURM_CPUS_PER_TASK}"
echo "Memory:    ${MEM}"
echo "Started:   \$(date)"
echo "=========================================="

cd ${REPO_ROOT}

# Activate conda/venv if needed (uncomment and edit):
# source activate overcooked
# source ${REPO_ROOT}/venv/bin/activate

export PYTHONPATH="${REPO_ROOT}/src:\${PYTHONPATH:-}"
export OMP_NUM_THREADS=${CPUS}
export MKL_NUM_THREADS=${CPUS}
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=true intra_op_parallelism_threads=${CPUS}"

${python_cmd}

echo "=========================================="
echo "Finished:  \$(date)"
echo "=========================================="
SBATCH_EOF
)

    if [ "$DRY_RUN" = true ]; then
        echo "[DRY RUN] Would submit: $job_name"
        echo "  Command: $python_cmd"
        echo ""
    else
        echo "$sbatch_script" | sbatch
        echo "Submitted: $job_name"
    fi
}

job_count=0

# --- PPO Self-Play ---
if [ -z "$ONLY" ] || [ "$ONLY" = "sp" ]; then
    echo "=== Submitting PPO_SP jobs ==="
    for layout in $LAYOUTS; do
        for seed in $SEEDS; do
            job_name="ppo_sp_${layout}_s${seed}"
            cmd="python -m human_aware_rl.ppo.train_ppo_sp \
                --layout ${layout} \
                --seed ${seed} \
                --results_dir ${RESULTS_DIR}"
            submit_job "$job_name" "$cmd"
            job_count=$((job_count + 1))
        done
    done
    echo ""
fi

# --- PPO_BC ---
if [ -z "$ONLY" ] || [ "$ONLY" = "bc" ]; then
    echo "=== Submitting PPO_BC jobs ==="
    for layout in $LAYOUTS; do
        bc_dir="${BC_MODEL_BASE}/${layout}"
        for seed in $SEEDS; do
            job_name="ppo_bc_${layout}_s${seed}"
            cmd="python -m human_aware_rl.ppo.train_ppo_bc \
                --layout ${layout} \
                --seed ${seed} \
                --bc_model_dir ${bc_dir} \
                --results_dir ${RESULTS_DIR}"
            submit_job "$job_name" "$cmd"
            job_count=$((job_count + 1))
        done
    done
    echo ""
fi

# --- PPO_GAIL ---
if [ -z "$ONLY" ] || [ "$ONLY" = "gail" ]; then
    echo "=== Submitting PPO_GAIL jobs ==="
    for layout in $LAYOUTS; do
        gail_dir="${GAIL_MODEL_BASE}/${layout}"
        for seed in $SEEDS; do
            job_name="ppo_gail_${layout}_s${seed}"
            cmd="python -m human_aware_rl.ppo.train_ppo_gail \
                --layout ${layout} \
                --seed ${seed} \
                --controlled \
                --gail_model_dir ${gail_dir} \
                --results_dir ${RESULTS_DIR}"
            submit_job "$job_name" "$cmd"
            job_count=$((job_count + 1))
        done
    done
    echo ""
fi

echo "Total jobs submitted: $job_count"
if [ "$DRY_RUN" = true ]; then
    echo "(dry run -- nothing was actually submitted)"
fi
