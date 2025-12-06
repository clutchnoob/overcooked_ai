#!/bin/bash
#SBATCH --job-name=ppo_gail_once
#SBATCH --output=logs/ppo_gail_once_%j.out
#SBATCH --error=logs/ppo_gail_once_%j.err
#SBATCH --time=08:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4

# Navigate to project root
cd "$SLURM_SUBMIT_DIR/.."

# Create logs directory
mkdir -p "$SLURM_SUBMIT_DIR/logs"

# Activate conda environment
source ~/.bashrc
conda activate MAL_env

# Navigate to human_aware_rl
cd src/human_aware_rl

LAYOUTS=("cramped_room" "asymmetric_advantages" "coordination_ring" "forced_coordination" "counter_circuit")

echo "=========================================="
echo "PPO_GAIL Training - One Seed Per Layout"
echo "=========================================="

for layout in "${LAYOUTS[@]}"; do
    OUTPUT_DIR="ppo_gail_runs/${layout}/seed_0"
    
    if [ -d "$OUTPUT_DIR" ] && [ -f "$OUTPUT_DIR/config.json" ]; then
        echo "SKIP: $layout already trained (found $OUTPUT_DIR)"
    else
        echo "TRAIN: $layout"
        python -m human_aware_rl.ppo.train_ppo_gail --layout "$layout" --seed 0 --fast
    fi
done

echo "=========================================="
echo "Done!"
echo "=========================================="

