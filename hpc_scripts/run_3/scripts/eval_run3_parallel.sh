#!/bin/bash
#SBATCH --job-name=eval_run3_par
#SBATCH --job-name=eval_run3_par
#SBATCH --array=0-4
#SBATCH --output=../logs/eval_run3/eval_run3_par_%A_%a.out
#SBATCH --error=../logs/eval_run3/eval_run3_par_%A_%a.err
#SBATCH --time=3:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8

set -euo pipefail

# Layout list (one per array index)
LAYOUTS=(
  "cramped_room"
  "asymmetric_advantages"
  "coordination_ring"
  "forced_coordination"
  "counter_circuit"
)

# Validate array index
if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  echo "SLURM_ARRAY_TASK_ID is not set; this script must be run as a job array."
  exit 1
fi
if (( SLURM_ARRAY_TASK_ID < 0 || SLURM_ARRAY_TASK_ID >= ${#LAYOUTS[@]} )); then
  echo "Invalid SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}; expected 0..$(( ${#LAYOUTS[@]} - 1 ))"
  exit 1
fi

LAYOUT="${LAYOUTS[$SLURM_ARRAY_TASK_ID]}"

# Move to repo root (scripts are in hpc_scripts/run_3/scripts)
cd "$SLURM_SUBMIT_DIR/../../.." || exit 1
PROJECT_ROOT="$(pwd)"

# Prepare log and result directories
mkdir -p "$SLURM_SUBMIT_DIR/../logs/eval_run3"
mkdir -p "$PROJECT_ROOT/eval_results/run3"

# Activate conda environment
source /om2/user/mabdel03/anaconda/etc/profile.d/conda.sh
conda activate /om/scratch/Mon/mabdel03/conda_envs/MAL_env

# Ensure local sources are importable
export PYTHONPATH="$PROJECT_ROOT/src"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

# Configurable games per seed (default 5); override via env NUM_GAMES
NUM_GAMES=${NUM_GAMES:-5}

RESULTS_PATH="$PROJECT_ROOT/eval_results/run3/run3_results_${LAYOUT}_$(date +%Y%m%d_%H%M%S).json"

echo "Running Run 3 evaluation (parallel) for layout: ${LAYOUT}"
echo "Results -> $RESULTS_PATH"
echo "Num games per seed: $NUM_GAMES"

python -m human_aware_rl.evaluation.evaluate_run \
  --run_number 3 \
  --layouts "$LAYOUT" \
  --num_games "$NUM_GAMES" \
  --save_results "$RESULTS_PATH" \
  --no_plot \
  --verbose

echo "Evaluation complete for layout ${LAYOUT}."
