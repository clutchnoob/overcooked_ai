#!/bin/bash
# ============================================================================
# HPC Evaluation Script - Human Proxy Evaluation
# ============================================================================
# Runs evaluation for BC, PPO_SP, PPO_BC, PPO_GAIL paired with Human Proxy.
# Generates JSON results and Figure 4-style plot.
#
# Usage (SLURM):
#   sbatch hpc_scripts/evaluation/run_evaluation.sh
#
# Usage (Interactive/Background):
#   nohup bash hpc_scripts/evaluation/run_evaluation.sh > hpc_scripts/logs/eval_$(date +%Y%m%d_%H%M%S).log 2>&1 &
# ============================================================================

#SBATCH --job-name=overcooked_eval
#SBATCH --time=08:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --partition=normal
#SBATCH --output=/om/scratch/Mon/mabdel03/6.S890/overcooked_ai/hpc_scripts/logs/eval_%j.out
#SBATCH --error=/om/scratch/Mon/mabdel03/6.S890/overcooked_ai/hpc_scripts/logs/eval_%j.err

set -eo pipefail

# ============================================================================
# Configuration
# ============================================================================
PROJECT_ROOT="/om/scratch/Mon/mabdel03/6.S890/overcooked_ai"
HUMAN_AWARE_RL_DIR="${PROJECT_ROOT}/src/human_aware_rl"
CONDA_ENV="/om/scratch/Mon/mabdel03/conda_envs/MAL_env"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Create output directories
mkdir -p "${HUMAN_AWARE_RL_DIR}/results"
mkdir -p "${PROJECT_ROOT}/hpc_scripts/logs"

# Output paths with timestamp
RESULTS_JSON="${HUMAN_AWARE_RL_DIR}/results/hpc_eval_results_${TIMESTAMP}.json"
FIGURE_PATH="${HUMAN_AWARE_RL_DIR}/results/hpc_eval_figure4_${TIMESTAMP}.png"

# ============================================================================
# Environment Setup
# ============================================================================
echo "============================================================"
echo "HPC Model Evaluation - Started at $(date)"
echo "============================================================"
echo "Project root: ${PROJECT_ROOT}"
echo "Results will be saved to:"
echo "  - JSON: ${RESULTS_JSON}"
echo "  - Figure: ${FIGURE_PATH}"
echo "============================================================"

# Setup conda (disable unbound variable check for conda activation scripts)
set +u
source /om2/user/mabdel03/anaconda/etc/profile.d/conda.sh
conda activate "${CONDA_ENV}"
set -u

# Set PYTHONPATH
export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH:-}"

# Change to working directory
cd "${HUMAN_AWARE_RL_DIR}"

echo "Python: $(which python)"
echo "Working directory: $(pwd)"
echo "PYTHONPATH: ${PYTHONPATH}"
echo "============================================================"

# ============================================================================
# Run Evaluation
# ============================================================================
echo ""
echo "Starting evaluation..."
echo ""

python -m human_aware_rl.evaluation.evaluate_hpc_models \
    --num_games 10 \
    --output "${RESULTS_JSON}"

echo ""
echo "============================================================"
echo "Evaluation complete. Generating plot..."
echo "============================================================"
echo ""

# ============================================================================
# Generate Plot
# ============================================================================
python -m human_aware_rl.evaluation.plot_hpc_results \
    --input "${RESULTS_JSON}" \
    --output "${FIGURE_PATH}" \
    --include_gail

# Also copy to a "latest" file for easy access
cp "${RESULTS_JSON}" "${HUMAN_AWARE_RL_DIR}/results/hpc_eval_results_latest.json"
cp "${FIGURE_PATH}" "${HUMAN_AWARE_RL_DIR}/results/hpc_eval_figure4_latest.png"

echo ""
echo "============================================================"
echo "HPC Model Evaluation - Completed at $(date)"
echo "============================================================"
echo "Results saved to:"
echo "  - JSON: ${RESULTS_JSON}"
echo "  - Figure: ${FIGURE_PATH}"
echo "  - Latest JSON: ${HUMAN_AWARE_RL_DIR}/results/hpc_eval_results_latest.json"
echo "  - Latest Figure: ${HUMAN_AWARE_RL_DIR}/results/hpc_eval_figure4_latest.png"
echo "============================================================"
