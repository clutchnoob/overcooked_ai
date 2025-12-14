#!/bin/bash
#SBATCH --job-name=bayesian_bc
#SBATCH --output=/om/scratch/Mon/mabdel03/6.S890/overcooked_ai/Overcooked_ProbMods/logs/train_bc_%j.out
#SBATCH --error=/om/scratch/Mon/mabdel03/6.S890/overcooked_ai/Overcooked_ProbMods/logs/train_bc_%j.err
#SBATCH --time=4:00:00
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1

set -e
export PYTHONUNBUFFERED=1

echo "Starting Bayesian BC training..."
echo "Node: $(hostname)"

# Activate conda directly without the activation script
source /om2/user/mabdel03/anaconda/bin/activate /om2/user/mabdel03/conda_envs/CompCogSci

# Check GPU
echo "CUDA visible devices: $CUDA_VISIBLE_DEVICES"
nvidia-smi

python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'Device count: {torch.cuda.device_count()}')" || true

cd /om/scratch/Mon/mabdel03/6.S890/overcooked_ai/Overcooked_ProbMods

echo "Running training..."
python -u scripts/train_bayesian_bc.py --layout cramped_room --epochs 500 --results_dir ./results

echo "Done!"
