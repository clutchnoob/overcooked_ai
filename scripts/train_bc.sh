#!/bin/bash
#SBATCH --job-name=bc_train
#SBATCH --output=logs/bc_train_%j.out
#SBATCH --error=logs/bc_train_%j.err
#SBATCH --time=01:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4

# Create logs directory
mkdir -p logs

# Activate conda environment
source ~/.bashrc
conda activate overcooked

# Navigate to project
cd $SLURM_SUBMIT_DIR/src/human_aware_rl

# Train BC models for all layouts
python -m human_aware_rl.imitation.train_bc_models --all_layouts

echo "BC training complete!"

