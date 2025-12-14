#!/bin/bash
#SBATCH --job-name=probmods_all
#SBATCH --output=/om/scratch/Mon/mabdel03/6.S890/overcooked_ai/Overcooked_ProbMods/logs/train_all_%j.out
#SBATCH --error=/om/scratch/Mon/mabdel03/6.S890/overcooked_ai/Overcooked_ProbMods/logs/train_all_%j.err
#SBATCH --time=48:00:00
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:1

# Train all probabilistic models on all layouts
# Usage: sbatch train_all_layouts.sh

set -e

# Use absolute paths
PROJECT_DIR="/om/scratch/Mon/mabdel03/6.S890/overcooked_ai/Overcooked_ProbMods"

# Force unbuffered Python output
export PYTHONUNBUFFERED=1

echo "=========================================="
echo "Training All Probabilistic Models"
echo "Project dir: $PROJECT_DIR"
echo "=========================================="

# Create log directory
mkdir -p "$PROJECT_DIR/logs"

# Activate environment
source /om2/user/mabdel03/conda_envs/activate_CompCogSci.sh

# Install dependencies
cd "$PROJECT_DIR"
pip install -e . --quiet

# Layouts to train on
LAYOUTS=(
    "cramped_room"
    "asymmetric_advantages"
    "coordination_ring"
    "forced_coordination"
    "counter_circuit_o_1order"
)

# Train on each layout
for layout in "${LAYOUTS[@]}"; do
    echo ""
    echo "=========================================="
    echo "Training on layout: $layout"
    echo "=========================================="
    
    python scripts/train_all.py \
        --layout "$layout" \
        --results_dir "$PROJECT_DIR/results"
    
    echo "Completed training for $layout"
done

echo ""
echo "=========================================="
echo "All training complete!"
echo "=========================================="
