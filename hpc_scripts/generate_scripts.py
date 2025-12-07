#!/usr/bin/env python3
"""
Generate HPC SLURM scripts for training all models.

This script generates batch scripts for:
- BC training (per layout, train + test)
- GAIL training (per layout)
- PPO_SP training (per layout, per seed)
- PPO_BC training (per layout, per seed)
- PPO_GAIL training (per layout, per seed)

Usage:
    python generate_scripts.py

The generated scripts use paper hyperparameters by default.
"""

import os
import stat

LAYOUTS = [
    "cramped_room",
    "asymmetric_advantages", 
    "coordination_ring",
    "forced_coordination",
    "counter_circuit",
]

SEEDS = [0, 10, 20, 30, 40]

# Paper training iterations
PAPER_ITERS = {
    "cramped_room": 550,
    "asymmetric_advantages": 650,
    "coordination_ring": 650,
    "forced_coordination": 650,
    "counter_circuit": 650,
}

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def create_bc_script(layout: str) -> str:
    """Create BC training script for a layout (trains both train and test)."""
    return f'''#!/bin/bash
#SBATCH --job-name=bc_{layout}
#SBATCH --output=logs/bc_{layout}_%j.out
#SBATCH --error=logs/bc_{layout}_%j.err
#SBATCH --time=04:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4

# BC Training for {layout}
# Trains both train (for PPO partner) and test (for Human Proxy) models

cd "$SLURM_SUBMIT_DIR/.."
source .venv/bin/activate || conda activate overcooked

mkdir -p "$SLURM_SUBMIT_DIR/logs"

echo "Training BC models for {layout}..."

# Train BC on training data (for PPO_BC partner)
python -m human_aware_rl.imitation.behavior_cloning \\
    --layout {layout} \\
    --data_type train \\
    --epochs 100

# Train BC on test data (for Human Proxy evaluation)
python -m human_aware_rl.imitation.behavior_cloning \\
    --layout {layout} \\
    --data_type test \\
    --epochs 100

echo "BC training complete for {layout}"
'''


def create_gail_script(layout: str) -> str:
    """Create GAIL training script for a layout."""
    return f'''#!/bin/bash
#SBATCH --job-name=gail_{layout}
#SBATCH --output=logs/gail_{layout}_%j.out
#SBATCH --error=logs/gail_{layout}_%j.err
#SBATCH --time=08:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4

# GAIL Training for {layout}

cd "$SLURM_SUBMIT_DIR/.."
source .venv/bin/activate || conda activate overcooked

mkdir -p "$SLURM_SUBMIT_DIR/logs"

echo "Training GAIL model for {layout}..."

python -m human_aware_rl.imitation.gail \\
    --layout {layout} \\
    --epochs 200

echo "GAIL training complete for {layout}"
'''


def create_ppo_sp_script(layout: str, seed: int) -> str:
    """Create PPO Self-Play training script."""
    iters = PAPER_ITERS[layout]
    return f'''#!/bin/bash
#SBATCH --job-name=ppo_sp_{layout}_s{seed}
#SBATCH --output=logs/ppo_sp_{layout}_seed{seed}_%j.out
#SBATCH --error=logs/ppo_sp_{layout}_seed{seed}_%j.err
#SBATCH --time=24:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8

# PPO Self-Play Training
# Layout: {layout}, Seed: {seed}
# Training iterations: {iters} (paper value)

cd "$SLURM_SUBMIT_DIR/.."
source .venv/bin/activate || conda activate overcooked

mkdir -p "$SLURM_SUBMIT_DIR/logs"

echo "Training PPO_SP for {layout} seed {seed}..."
echo "Paper iterations: {iters}"

python -m human_aware_rl.ppo.train_ppo_sp \\
    --layout {layout} \\
    --seed {seed} \\
    --num_training_iters {iters} \\
    --results_dir results/ppo_sp

echo "PPO_SP training complete for {layout} seed {seed}"
'''


def create_ppo_bc_script(layout: str, seed: int) -> str:
    """Create PPO_BC training script."""
    iters = PAPER_ITERS[layout]
    return f'''#!/bin/bash
#SBATCH --job-name=ppo_bc_{layout}_s{seed}
#SBATCH --output=logs/ppo_bc_{layout}_seed{seed}_%j.out
#SBATCH --error=logs/ppo_bc_{layout}_seed{seed}_%j.err
#SBATCH --time=24:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8

# PPO_BC Training (PPO with BC partner)
# Layout: {layout}, Seed: {seed}
# Training iterations: {iters} (paper value)

cd "$SLURM_SUBMIT_DIR/.."
source .venv/bin/activate || conda activate overcooked

mkdir -p "$SLURM_SUBMIT_DIR/logs"

echo "Training PPO_BC for {layout} seed {seed}..."
echo "Paper iterations: {iters}"

python -m human_aware_rl.ppo.train_ppo_bc \\
    --layout {layout} \\
    --seed {seed} \\
    --num_training_iters {iters} \\
    --use_default_bc_models \\
    --results_dir results/ppo_bc

echo "PPO_BC training complete for {layout} seed {seed}"
'''


def create_ppo_gail_script(layout: str, seed: int) -> str:
    """Create PPO_GAIL training script (PPO with GAIL partner)."""
    iters = PAPER_ITERS[layout]
    return f'''#!/bin/bash
#SBATCH --job-name=ppo_gail_{layout}_s{seed}
#SBATCH --output=logs/ppo_gail_{layout}_seed{seed}_%j.out
#SBATCH --error=logs/ppo_gail_{layout}_seed{seed}_%j.err
#SBATCH --time=24:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8

# PPO_GAIL Training (PPO with GAIL partner instead of BC)
# Layout: {layout}, Seed: {seed}
# Training iterations: {iters} (paper value)

cd "$SLURM_SUBMIT_DIR/.."
source .venv/bin/activate || conda activate overcooked

mkdir -p "$SLURM_SUBMIT_DIR/logs"

echo "Training PPO_GAIL for {layout} seed {seed}..."
echo "Paper iterations: {iters}"

python -m human_aware_rl.ppo.train_ppo_gail \\
    --layout {layout} \\
    --seed {seed} \\
    --num_training_iters {iters} \\
    --results_dir results/ppo_gail

echo "PPO_GAIL training complete for {layout} seed {seed}"
'''


def create_submit_all_script(model_type: str, scripts: list) -> str:
    """Create a submit_all script."""
    return f'''#!/bin/bash
# Submit all {model_type.upper()} training jobs
# Total jobs: {len(scripts)}

SCRIPT_DIR="$(cd "$(dirname "${{BASH_SOURCE[0]}}")" && pwd)"

echo "Submitting {model_type.upper()} training jobs..."

''' + '\n'.join([f'sbatch "$SCRIPT_DIR/{s}"' for s in scripts]) + f'''

echo ""
echo "Submitted {len(scripts)} {model_type.upper()} jobs"
'''


def write_script(path: str, content: str):
    """Write script file and make it executable."""
    with open(path, 'w') as f:
        f.write(content)
    # Make executable
    st = os.stat(path)
    os.chmod(path, st.st_mode | stat.S_IEXEC)


def main():
    """Generate all HPC scripts."""
    os.makedirs(SCRIPT_DIR, exist_ok=True)
    
    # Track scripts for submit_all
    bc_scripts = []
    gail_scripts = []
    ppo_sp_scripts = []
    ppo_bc_scripts = []
    ppo_gail_scripts = []
    
    print("Generating HPC scripts...")
    print(f"Output directory: {SCRIPT_DIR}")
    print()
    
    # BC scripts (one per layout)
    print("BC scripts (1 per layout, trains train+test):")
    for layout in LAYOUTS:
        filename = f"bc_{layout}.sh"
        write_script(os.path.join(SCRIPT_DIR, filename), create_bc_script(layout))
        bc_scripts.append(filename)
        print(f"  {filename}")
    print()
    
    # GAIL scripts (one per layout)
    print("GAIL scripts (1 per layout):")
    for layout in LAYOUTS:
        filename = f"gail_{layout}.sh"
        write_script(os.path.join(SCRIPT_DIR, filename), create_gail_script(layout))
        gail_scripts.append(filename)
        print(f"  {filename}")
    print()
    
    # PPO_SP scripts (per layout, per seed)
    print("PPO_SP scripts (5 layouts × 5 seeds = 25):")
    for layout in LAYOUTS:
        for seed in SEEDS:
            filename = f"ppo_sp_{layout}_seed{seed}.sh"
            write_script(os.path.join(SCRIPT_DIR, filename), create_ppo_sp_script(layout, seed))
            ppo_sp_scripts.append(filename)
    print(f"  Created {len(ppo_sp_scripts)} scripts")
    print()
    
    # PPO_BC scripts (per layout, per seed)
    print("PPO_BC scripts (5 layouts × 5 seeds = 25):")
    for layout in LAYOUTS:
        for seed in SEEDS:
            filename = f"ppo_bc_{layout}_seed{seed}.sh"
            write_script(os.path.join(SCRIPT_DIR, filename), create_ppo_bc_script(layout, seed))
            ppo_bc_scripts.append(filename)
    print(f"  Created {len(ppo_bc_scripts)} scripts")
    print()
    
    # PPO_GAIL scripts (per layout, per seed)
    print("PPO_GAIL scripts (5 layouts × 5 seeds = 25):")
    for layout in LAYOUTS:
        for seed in SEEDS:
            filename = f"ppo_gail_{layout}_seed{seed}.sh"
            write_script(os.path.join(SCRIPT_DIR, filename), create_ppo_gail_script(layout, seed))
            ppo_gail_scripts.append(filename)
    print(f"  Created {len(ppo_gail_scripts)} scripts")
    print()
    
    # Submit all scripts
    print("Submit scripts:")
    for model_type, scripts in [
        ("bc", bc_scripts),
        ("gail", gail_scripts),
        ("ppo_sp", ppo_sp_scripts),
        ("ppo_bc", ppo_bc_scripts),
        ("ppo_gail", ppo_gail_scripts),
    ]:
        filename = f"submit_all_{model_type}.sh"
        write_script(os.path.join(SCRIPT_DIR, filename), create_submit_all_script(model_type, scripts))
        print(f"  {filename} ({len(scripts)} jobs)")
    print()
    
    total_scripts = len(bc_scripts) + len(gail_scripts) + len(ppo_sp_scripts) + len(ppo_bc_scripts) + len(ppo_gail_scripts)
    print(f"Total: {total_scripts} training scripts + 5 submit scripts")
    print()
    print("Training order:")
    print("  1. BC (needed by PPO_BC): ./submit_all_bc.sh")
    print("  2. GAIL (needed by PPO_GAIL): ./submit_all_gail.sh")
    print("  3. After BC/GAIL complete:")
    print("     - PPO_SP: ./submit_all_ppo_sp.sh")
    print("     - PPO_BC: ./submit_all_ppo_bc.sh")
    print("     - PPO_GAIL: ./submit_all_ppo_gail.sh")


if __name__ == "__main__":
    main()
