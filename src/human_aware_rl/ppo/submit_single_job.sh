#!/usr/bin/env bash
#
# Submit a single training job to SLURM.
#
# Usage:
#   bash submit_single_job.sh sp cramped_room 0
#   bash submit_single_job.sh bc asymmetric_advantages 10
#   bash submit_single_job.sh gail coordination_ring 20

set -euo pipefail

if [ $# -lt 3 ]; then
    echo "Usage: $0 <sp|bc|gail> <layout> <seed> [extra_args...]"
    echo ""
    echo "Examples:"
    echo "  $0 sp cramped_room 0"
    echo "  $0 bc asymmetric_advantages 10"
    echo "  $0 gail coordination_ring 20"
    exit 1
fi

EXP_TYPE="$1"
LAYOUT="$2"
SEED="$3"
shift 3
EXTRA_ARGS="$*"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
RESULTS_DIR="${RESULTS_DIR:-$REPO_ROOT/results}"
BC_MODEL_BASE="${BC_MODEL_BASE:-$REPO_ROOT/src/human_aware_rl/imitation/bc_runs/train}"
GAIL_MODEL_BASE="${GAIL_MODEL_BASE:-$REPO_ROOT/src/human_aware_rl/gail/gail_runs}"

PARTITION="${PARTITION:-normal}"
MEM="${MEM:-48G}"
CPUS="${CPUS:-16}"
TIME="${TIME:-24:00:00}"
LOG_DIR="${LOG_DIR:-$RESULTS_DIR/slurm_logs}"

mkdir -p "$LOG_DIR"

JOB_NAME="ppo_${EXP_TYPE}_${LAYOUT}_s${SEED}"

case "$EXP_TYPE" in
    sp)
        PYTHON_CMD="python -m human_aware_rl.ppo.train_ppo_sp \
            --layout ${LAYOUT} --seed ${SEED} \
            --results_dir ${RESULTS_DIR} ${EXTRA_ARGS}"
        ;;
    bc)
        PYTHON_CMD="python -m human_aware_rl.ppo.train_ppo_bc \
            --layout ${LAYOUT} --seed ${SEED} \
            --bc_model_dir ${BC_MODEL_BASE}/${LAYOUT} \
            --results_dir ${RESULTS_DIR} ${EXTRA_ARGS}"
        ;;
    gail)
        PYTHON_CMD="python -m human_aware_rl.ppo.train_ppo_gail \
            --layout ${LAYOUT} --seed ${SEED} \
            --controlled \
            --gail_model_dir ${GAIL_MODEL_BASE}/${LAYOUT} \
            --results_dir ${RESULTS_DIR} ${EXTRA_ARGS}"
        ;;
    *)
        echo "Error: Unknown experiment type '$EXP_TYPE'. Use sp, bc, or gail."
        exit 1
        ;;
esac

sbatch <<SBATCH_EOF
#!/usr/bin/env bash
#SBATCH --job-name=${JOB_NAME}
#SBATCH --output=${LOG_DIR}/${JOB_NAME}_%j.out
#SBATCH --error=${LOG_DIR}/${JOB_NAME}_%j.err
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

${PYTHON_CMD}

echo "=========================================="
echo "Finished:  \$(date)"
echo "=========================================="
SBATCH_EOF

echo "Submitted: $JOB_NAME"
