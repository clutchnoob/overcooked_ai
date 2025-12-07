#!/usr/bin/env python3
"""
Generate HPC batch scripts for full paper reproduction.

Creates separate scripts for each layout and seed combination.
"""

import os

LAYOUTS = [
    "cramped_room",
    "asymmetric_advantages", 
    "coordination_ring",
    "forced_coordination",
    "counter_circuit",
]

SEEDS = [0, 10, 20, 30, 40]

# Training iterations from paper
TRAINING_ITERS = {
    "cramped_room": 550,
    "asymmetric_advantages": 650,
    "coordination_ring": 650,
    "forced_coordination": 650,
    "counter_circuit": 650,
}

# Short names for job names
SHORT_NAMES = {
    "cramped_room": "cr",
    "asymmetric_advantages": "aa",
    "coordination_ring": "coord",
    "forced_coordination": "fc",
    "counter_circuit": "cc",
}

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PAPER_DIR = os.path.join(SCRIPT_DIR, "paper")
os.makedirs(PAPER_DIR, exist_ok=True)


def generate_ppo_sp_script(layout: str, seed: int):
    """Generate PPO Self-Play script."""
    short = SHORT_NAMES[layout]
    
    script = f'''#!/bin/bash
#SBATCH --job-name=sp_{short}_s{seed}
#SBATCH --output=../logs/ppo_sp_{layout}_seed{seed}_%j.out
#SBATCH --error=../logs/ppo_sp_{layout}_seed{seed}_%j.err
#SBATCH --time=12:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8

# Navigate to project root
cd "$SLURM_SUBMIT_DIR/../.."

# Create logs directory
mkdir -p hpc_scripts/logs

# Activate conda environment
source ~/.bashrc
conda activate MAL_env

# Navigate to human_aware_rl
cd src/human_aware_rl

# Run PPO Self-Play training (FULL PAPER PARAMS - {TRAINING_ITERS[layout]} iterations)
python -m human_aware_rl.ppo.train_ppo_sp --layout {layout} --seed {seed}
'''
    
    filepath = os.path.join(PAPER_DIR, f"ppo_sp_{layout}_seed{seed}.sh")
    with open(filepath, "w") as f:
        f.write(script)
    print(f"Created: {filepath}")


def generate_ppo_bc_script(layout: str, seed: int):
    """Generate PPO with BC partner script."""
    short = SHORT_NAMES[layout]
    
    script = f'''#!/bin/bash
#SBATCH --job-name=bc_{short}_s{seed}
#SBATCH --output=../logs/ppo_bc_{layout}_seed{seed}_%j.out
#SBATCH --error=../logs/ppo_bc_{layout}_seed{seed}_%j.err
#SBATCH --time=12:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8

# Navigate to project root
cd "$SLURM_SUBMIT_DIR/../.."

# Create logs directory
mkdir -p hpc_scripts/logs

# Activate conda environment
source ~/.bashrc
conda activate MAL_env

# Navigate to human_aware_rl
cd src/human_aware_rl

# Run PPO with BC partner training (FULL PAPER PARAMS - {TRAINING_ITERS[layout]} iterations)
python -m human_aware_rl.ppo.train_ppo_bc --layout {layout} --seed {seed}
'''
    
    filepath = os.path.join(PAPER_DIR, f"ppo_bc_{layout}_seed{seed}.sh")
    with open(filepath, "w") as f:
        f.write(script)
    print(f"Created: {filepath}")


def generate_bc_script(layout: str):
    """Generate BC training script."""
    short = SHORT_NAMES[layout]
    
    script = f'''#!/bin/bash
#SBATCH --job-name=bc_{short}
#SBATCH --output=../logs/bc_{layout}_%j.out
#SBATCH --error=../logs/bc_{layout}_%j.err
#SBATCH --time=02:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4

# Navigate to project root
cd "$SLURM_SUBMIT_DIR/../.."

# Create logs directory
mkdir -p hpc_scripts/logs

# Activate conda environment
source ~/.bashrc
conda activate MAL_env

# Navigate to human_aware_rl
cd src/human_aware_rl

# Run BC training (train + test models)
python -m human_aware_rl.imitation.train_bc_models --layout {layout}
'''
    
    filepath = os.path.join(PAPER_DIR, f"bc_{layout}.sh")
    with open(filepath, "w") as f:
        f.write(script)
    print(f"Created: {filepath}")


def generate_submit_all_script():
    """Generate script to submit all jobs."""
    
    lines = [
        "#!/bin/bash",
        "# Submit all paper reproduction jobs",
        "",
        "cd \"$(dirname \"$0\")/paper\"",
        "",
        "echo '=========================================='",
        "echo 'Submitting all paper reproduction jobs'",
        "echo '=========================================='",
        "",
        "# BC jobs (must complete first - no dependencies for now)",
        "echo 'Submitting BC jobs...'",
    ]
    
    for layout in LAYOUTS:
        lines.append(f"sbatch bc_{layout}.sh")
    
    lines.extend([
        "",
        "# PPO Self-Play jobs",
        "echo 'Submitting PPO Self-Play jobs...'",
    ])
    
    for layout in LAYOUTS:
        for seed in SEEDS:
            lines.append(f"sbatch ppo_sp_{layout}_seed{seed}.sh")
    
    lines.extend([
        "",
        "# PPO with BC partner jobs",
        "echo 'Submitting PPO_BC jobs...'",
    ])
    
    for layout in LAYOUTS:
        for seed in SEEDS:
            lines.append(f"sbatch ppo_bc_{layout}_seed{seed}.sh")
    
    lines.extend([
        "",
        "echo '=========================================='",
        "echo 'All jobs submitted!'",
        "echo 'Total: 5 BC + 25 PPO_SP + 25 PPO_BC = 55 jobs'",
        "echo 'Check status with: squeue -u $USER'",
        "echo '=========================================='",
    ])
    
    filepath = os.path.join(SCRIPT_DIR, "submit_paper_all.sh")
    with open(filepath, "w") as f:
        f.write("\n".join(lines))
    print(f"\nCreated: {filepath}")


def main():
    print("Generating paper reproduction scripts...\n")
    
    # BC scripts
    print("=== BC Scripts ===")
    for layout in LAYOUTS:
        generate_bc_script(layout)
    
    # PPO Self-Play scripts
    print("\n=== PPO Self-Play Scripts ===")
    for layout in LAYOUTS:
        for seed in SEEDS:
            generate_ppo_sp_script(layout, seed)
    
    # PPO with BC partner scripts
    print("\n=== PPO_BC Scripts ===")
    for layout in LAYOUTS:
        for seed in SEEDS:
            generate_ppo_bc_script(layout, seed)
    
    # Submit all script
    print("\n=== Submit Script ===")
    generate_submit_all_script()
    
    print("\n" + "="*50)
    print("Generated scripts:")
    print(f"  - BC: 5 scripts")
    print(f"  - PPO_SP: {len(LAYOUTS) * len(SEEDS)} scripts")
    print(f"  - PPO_BC: {len(LAYOUTS) * len(SEEDS)} scripts")
    print(f"  - Total: {5 + 2 * len(LAYOUTS) * len(SEEDS)} scripts")
    print("="*50)
    print("\nTo submit all jobs:")
    print("  cd hpc_scripts && bash submit_paper_all.sh")


if __name__ == "__main__":
    main()

