#!/bin/bash
#SBATCH --job-name=eval_run3
#SBATCH --output=../logs/eval_run3/eval_run3_%j.out
#SBATCH --error=../logs/eval_run3/eval_run3_%j.err
#SBATCH --time=6:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4

# Evaluate all Run 3 models (SP, BC, GAIL, PPO_SP, PPO_BC, PPO_GAIL)

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

# Configurable games per seed (default 5); override via env NUM_GAMES
NUM_GAMES=${NUM_GAMES:-5}

RESULTS_PATH="$PROJECT_ROOT/eval_results/run3/run3_results_$(date +%Y%m%d_%H%M%S).json"

echo "Running Run 3 evaluation..."
echo "Results -> $RESULTS_PATH"
echo "Num games per seed: $NUM_GAMES"

python -m human_aware_rl.evaluation.evaluate_run \
  --run_number 3 \
  --num_games "$NUM_GAMES" \
  --save_results "$RESULTS_PATH" \
  --no_plot \
  --verbose

echo "Evaluation complete."
