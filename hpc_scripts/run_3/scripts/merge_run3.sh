#!/bin/bash
#SBATCH --job-name=merge_run3
#SBATCH --job-name=merge_run3
#SBATCH --output=../logs/eval_run3/merge_run3_%j.out
#SBATCH --error=../logs/eval_run3/merge_run3_%j.err
#SBATCH --time=00:20:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2

set -euo pipefail

# Move to repo root (scripts are in hpc_scripts/run_3/scripts)
cd "$SLURM_SUBMIT_DIR/../../.." || exit 1
PROJECT_ROOT="$(pwd)"

# Prepare log and result directories
mkdir -p "$SLURM_SUBMIT_DIR/../logs/eval_run3"
mkdir -p "$PROJECT_ROOT/eval_results/run3"

# Activate conda environment
export MKL_INTERFACE_LAYER=GNU
export MKL_THREADING_LAYER=GNU
source /om2/user/mabdel03/anaconda/etc/profile.d/conda.sh
conda activate /om/scratch/Mon/mabdel03/conda_envs/MAL_env

# Ensure local sources are importable
export PYTHONPATH="$PROJECT_ROOT/src"

INPUT_DIR="$PROJECT_ROOT/eval_results/run3"
OUTPUT_PATH="$PROJECT_ROOT/eval_results/run3/run3_results_merged.json"

echo "Merging run3 eval results from $INPUT_DIR"

python -m human_aware_rl.evaluation.merge_results \
  --input_dir "$INPUT_DIR" \
  --output "$OUTPUT_PATH"

echo "Merged results written to $OUTPUT_PATH"
