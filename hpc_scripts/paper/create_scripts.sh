#!/bin/bash
# Create all paper reproduction scripts

LAYOUTS=("cramped_room" "asymmetric_advantages" "coordination_ring" "forced_coordination" "counter_circuit")
SEEDS=(0 10 20 30 40)

# Training iterations per layout (from paper)
declare -A ITERS
ITERS[cramped_room]=550
ITERS[asymmetric_advantages]=650
ITERS[coordination_ring]=650
ITERS[forced_coordination]=650
ITERS[counter_circuit]=650

# Short names for job IDs
declare -A SHORT
SHORT[cramped_room]=cr
SHORT[asymmetric_advantages]=aa
SHORT[coordination_ring]=coord
SHORT[forced_coordination]=fc
SHORT[counter_circuit]=cc

echo "Creating paper reproduction scripts..."

# BC scripts
for layout in "${LAYOUTS[@]}"; do
    short=${SHORT[$layout]}
    cat > "bc_${layout}.sh" << EOF
#!/bin/bash
#SBATCH --job-name=bc_${short}
#SBATCH --output=../logs/bc_${layout}_%j.out
#SBATCH --error=../logs/bc_${layout}_%j.err
#SBATCH --time=02:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4

cd "\$SLURM_SUBMIT_DIR/../.."
mkdir -p hpc_scripts/logs
source ~/.bashrc
conda activate MAL_env
cd src/human_aware_rl

python -m human_aware_rl.imitation.train_bc_models --layout ${layout}
EOF
    echo "Created: bc_${layout}.sh"
done

# PPO Self-Play scripts
for layout in "${LAYOUTS[@]}"; do
    short=${SHORT[$layout]}
    iters=${ITERS[$layout]}
    for seed in "${SEEDS[@]}"; do
        cat > "ppo_sp_${layout}_seed${seed}.sh" << EOF
#!/bin/bash
#SBATCH --job-name=sp_${short}_s${seed}
#SBATCH --output=../logs/ppo_sp_${layout}_seed${seed}_%j.out
#SBATCH --error=../logs/ppo_sp_${layout}_seed${seed}_%j.err
#SBATCH --time=12:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8

cd "\$SLURM_SUBMIT_DIR/../.."
mkdir -p hpc_scripts/logs
source ~/.bashrc
conda activate MAL_env
cd src/human_aware_rl

# Full paper params: ${iters} iterations
python -m human_aware_rl.ppo.train_ppo_sp --layout ${layout} --seed ${seed}
EOF
        echo "Created: ppo_sp_${layout}_seed${seed}.sh"
    done
done

# PPO_BC scripts
for layout in "${LAYOUTS[@]}"; do
    short=${SHORT[$layout]}
    iters=${ITERS[$layout]}
    for seed in "${SEEDS[@]}"; do
        cat > "ppo_bc_${layout}_seed${seed}.sh" << EOF
#!/bin/bash
#SBATCH --job-name=bc_${short}_s${seed}
#SBATCH --output=../logs/ppo_bc_${layout}_seed${seed}_%j.out
#SBATCH --error=../logs/ppo_bc_${layout}_seed${seed}_%j.err
#SBATCH --time=12:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8

cd "\$SLURM_SUBMIT_DIR/../.."
mkdir -p hpc_scripts/logs
source ~/.bashrc
conda activate MAL_env
cd src/human_aware_rl

# Full paper params: ${iters} iterations
python -m human_aware_rl.ppo.train_ppo_bc --layout ${layout} --seed ${seed}
EOF
        echo "Created: ppo_bc_${layout}_seed${seed}.sh"
    done
done

echo ""
echo "=========================================="
echo "Created:"
echo "  - 5 BC scripts"
echo "  - 25 PPO_SP scripts (5 layouts x 5 seeds)"
echo "  - 25 PPO_BC scripts (5 layouts x 5 seeds)"
echo "  - Total: 55 scripts"
echo "=========================================="
echo ""
echo "To submit all: bash submit_all_paper.sh"

