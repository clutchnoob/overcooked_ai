#!/bin/bash
#SBATCH --job-name=invplan_analyze
#SBATCH --output=/om/scratch/Mon/mabdel03/6.S890/overcooked_ai/Overcooked_ProbMods/logs/invplan_analyze_%j.out
#SBATCH --error=/om/scratch/Mon/mabdel03/6.S890/overcooked_ai/Overcooked_ProbMods/logs/invplan_analyze_%j.err
#SBATCH --time=1:00:00
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

###############################################################################
# ANALYZE INVERSE PLANNING RESULTS
###############################################################################
#
# Runs analysis on trained inverse planning models:
#   - Computes posterior statistics (means, credible intervals for theta & beta)
#   - Generates visualizations (feature weight plots, algorithm comparisons)
#   - Outputs summary JSON with cognitive parameter comparisons
#   - Computes similarity metrics (cosine similarity, KL divergence)
#
# Run this AFTER train_inverse_planning.sh completes.
#
# Usage:
#   sbatch hpc/analyze_inverse_planning.sh
#
###############################################################################

set -e
export PYTHONUNBUFFERED=1

# Paths
PROJECT_DIR="/om/scratch/Mon/mabdel03/6.S890/overcooked_ai/Overcooked_ProbMods"
OVERCOOKED_ROOT="/om/scratch/Mon/mabdel03/6.S890/overcooked_ai"

echo "=============================================================="
echo "INVERSE PLANNING ANALYSIS"
echo "=============================================================="
echo "Node:        $(hostname)"
echo "Date:        $(date)"
echo "Project Dir: $PROJECT_DIR"
echo "=============================================================="

# Create directories
mkdir -p "$PROJECT_DIR/logs"
mkdir -p "$PROJECT_DIR/results/inverse_planning/plots"

# Activate conda environment
source /om2/user/mabdel03/anaconda/bin/activate /om2/user/mabdel03/conda_envs/CompCogSci

# Verify environment
echo ""
echo "=== Environment Check ==="
which python
python --version

# Set up Python path
export PYTHONPATH="${OVERCOOKED_ROOT}/src:${PROJECT_DIR}:${PYTHONPATH}"

# Change to project directory
cd "$PROJECT_DIR"

# Define what to analyze
LAYOUTS="cramped_room asymmetric_advantages coordination_ring"
TAGS="human_demo ppo_bc ppo_gail"

echo ""
echo "=== Running Analysis ==="
echo "Layouts: $LAYOUTS"
echo "Tags:    $TAGS"

# Run analysis with all options
python -u scripts/analyze_inverse_planning.py \
    --layouts $LAYOUTS \
    --tags $TAGS \
    --save-plots \
    --output-json ./results/inverse_planning/analysis_summary.json \
    --results-dir ./results

echo ""
echo "=============================================================="
echo "ANALYSIS COMPLETE"
echo "Time: $(date)"
echo ""
echo "Outputs:"
echo "  - JSON: results/inverse_planning/analysis_summary.json"
echo "  - Plots: results/inverse_planning/plots/"
echo "=============================================================="
