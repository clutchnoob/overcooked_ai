#!/bin/bash
# Submit all paper reproduction jobs
# Full training with paper hyperparameters

cd "$(dirname "$0")"

echo '=========================================='
echo 'Paper Reproduction - Full Training'
echo '=========================================='

LAYOUTS=("cramped_room" "asymmetric_advantages" "coordination_ring" "forced_coordination" "counter_circuit")
SEEDS=(0 10 20 30 40)

# Create logs directory
mkdir -p ../logs

# Submit BC jobs first
echo ''
echo 'Submitting BC jobs...'
for layout in "${LAYOUTS[@]}"; do
    if [ -f "bc_${layout}.sh" ]; then
        sbatch "bc_${layout}.sh"
    fi
done

# Submit PPO Self-Play jobs
echo ''
echo 'Submitting PPO Self-Play jobs...'
for layout in "${LAYOUTS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        if [ -f "ppo_sp_${layout}_seed${seed}.sh" ]; then
            sbatch "ppo_sp_${layout}_seed${seed}.sh"
        fi
    done
done

# Submit PPO_BC jobs
echo ''
echo 'Submitting PPO_BC jobs...'
for layout in "${LAYOUTS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        if [ -f "ppo_bc_${layout}_seed${seed}.sh" ]; then
            sbatch "ppo_bc_${layout}_seed${seed}.sh"
        fi
    done
done

echo ''
echo '=========================================='
echo 'All jobs submitted!'
echo 'Check status with: squeue -u $USER'
echo '=========================================='

