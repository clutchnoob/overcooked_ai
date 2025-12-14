#!/bin/bash
#SBATCH --job-name=probmods_train
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err
#SBATCH --time=12:00:00
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1

# Train a single probabilistic model
# Usage: sbatch train_single.sh --export=MODEL=bayesian_bc,LAYOUT=cramped_room

set -e

# Default values
MODEL=${MODEL:-bayesian_bc}
LAYOUT=${LAYOUT:-cramped_room}
EPOCHS=${EPOCHS:-1000}

echo "=========================================="
echo "Training Probabilistic Model"
echo "Model: $MODEL"
echo "Layout: $LAYOUT"
echo "Epochs: $EPOCHS"
echo "=========================================="

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
ROOT_DIR="$(dirname "$PROJECT_DIR")"

# Create log directory
mkdir -p "$PROJECT_DIR/logs"

# Activate environment
source /om2/user/mabdel03/conda_envs/activate_CompCogSci.sh

# Install dependencies if needed
cd "$PROJECT_DIR"
pip install -e . --quiet

# Run training
echo "Starting training..."
python scripts/train_${MODEL}.py \
    --layout "$LAYOUT" \
    --results-dir "$PROJECT_DIR/results" \
    --num-epochs "$EPOCHS"

echo "Training complete!"
echo "Results saved to: $PROJECT_DIR/results/$MODEL/$LAYOUT"
