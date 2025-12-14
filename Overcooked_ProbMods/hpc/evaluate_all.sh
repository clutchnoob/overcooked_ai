#!/bin/bash
#SBATCH --job-name=probmods_eval
#SBATCH --output=logs/eval_%j.out
#SBATCH --error=logs/eval_%j.err
#SBATCH --time=4:00:00
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1

# Evaluate all trained models and generate comparison
# Usage: sbatch evaluate_all.sh

set -e

echo "=========================================="
echo "Evaluating All Probabilistic Models"
echo "=========================================="

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
ROOT_DIR="$(dirname "$PROJECT_DIR")"

# Create directories
mkdir -p "$PROJECT_DIR/logs"
mkdir -p "$PROJECT_DIR/results/evaluations"

# Activate environment
source /om2/user/mabdel03/conda_envs/activate_CompCogSci.sh

cd "$PROJECT_DIR"

# Layouts to evaluate
LAYOUTS=(
    "cramped_room"
    "asymmetric_advantages"
    "coordination_ring"
    "forced_coordination"
    "counter_circuit_o_1order"
)

# Evaluate each layout
for layout in "${LAYOUTS[@]}"; do
    echo ""
    echo "Evaluating on layout: $layout"
    
    python scripts/evaluate.py \
        --model all \
        --layout "$layout" \
        --results-dir "$PROJECT_DIR/results" \
        --output "$PROJECT_DIR/results/evaluations/${layout}_eval.json" \
        --num-samples 100
done

# Run comparison across all layouts
echo ""
echo "Running model comparison..."
python scripts/compare.py \
    --layouts "${LAYOUTS[@]}" \
    --results-dir "$PROJECT_DIR/results" \
    --output "$PROJECT_DIR/results/evaluations/comparison.json" \
    --output-md "$PROJECT_DIR/results/evaluations/comparison.md" \
    --num-samples 100

echo ""
echo "=========================================="
echo "Evaluation complete!"
echo "Results saved to: $PROJECT_DIR/results/evaluations/"
echo "=========================================="
