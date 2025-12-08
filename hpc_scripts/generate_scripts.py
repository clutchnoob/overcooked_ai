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
    python generate_scripts.py --run N

Where N is the run number (e.g., 4 for run_4/).
Scripts are output to hpc_scripts/run_N/scripts/ with logs in run_N/logs/.
Model outputs are saved to versioned directories (e.g., bc_runs_runN/).

The generated scripts use paper hyperparameters by default.
"""

import argparse
import os
import stat
from datetime import datetime

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


def create_bc_script(layout: str, run: int) -> str:
    """Create BC training script for a layout (trains both train and test)."""
    return f'''#!/bin/bash
#SBATCH --job-name=bc_{layout}
#SBATCH --output=../logs/bc_{layout}_%j.out
#SBATCH --error=../logs/bc_{layout}_%j.err
#SBATCH --time=04:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4

# BC Training for {layout}
# Run: {run}
# Trains both train (for PPO partner) and test (for Human Proxy) models

# Navigate to project root (scripts/ -> run_N/ -> hpc_scripts/ -> project root)
cd "$SLURM_SUBMIT_DIR/../../.."

# Create logs directory
mkdir -p "$SLURM_SUBMIT_DIR/../logs"

# Activate conda environment
source /om2/user/mabdel03/anaconda/etc/profile.d/conda.sh
conda activate /om/scratch/Mon/mabdel03/conda_envs/MAL_env

# Navigate to src for Python modules
cd src

echo "Run {run}: Training BC models for {layout}..."

# Train BC models (both train and test) - outputs to bc_runs_run{run}/
# Note: BC trains both train and test splits; PPO_BC uses train/{layout}/
python -m human_aware_rl.imitation.train_bc_models --layout {layout} --output_base_dir human_aware_rl/bc_runs_run{run}

echo "BC training complete for {layout}"
'''


def create_gail_script(layout: str, run: int) -> str:
    """Create GAIL training script for a layout."""
    return f'''#!/bin/bash
#SBATCH --job-name=gail_{layout}
#SBATCH --output=../logs/gail_{layout}_%j.out
#SBATCH --error=../logs/gail_{layout}_%j.err
#SBATCH --time=08:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8

# GAIL Training for {layout}
# Run: {run}

# Navigate to project root
cd "$SLURM_SUBMIT_DIR/../../.."

# Create logs directory
mkdir -p "$SLURM_SUBMIT_DIR/../logs"

# Activate conda environment
source /om2/user/mabdel03/anaconda/etc/profile.d/conda.sh
conda activate /om/scratch/Mon/mabdel03/conda_envs/MAL_env

# Navigate to src for Python modules
cd src

echo "Run {run}: Training GAIL model for {layout}..."

python -m human_aware_rl.imitation.gail --layout {layout} --results_dir human_aware_rl/gail_runs_run{run}

echo "GAIL training complete for {layout}"
'''


def create_ppo_sp_script(layout: str, seed: int, run: int) -> str:
    """Create PPO Self-Play training script."""
    iters = PAPER_ITERS[layout]
    return f'''#!/bin/bash
#SBATCH --job-name=ppo_sp_{layout}_s{seed}
#SBATCH --output=../logs/ppo_sp_{layout}_seed{seed}_%j.out
#SBATCH --error=../logs/ppo_sp_{layout}_seed{seed}_%j.err
#SBATCH --time=24:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8

# PPO Self-Play Training
# Run: {run}
# Layout: {layout}, Seed: {seed}
# Training iterations: {iters} (paper value)

# Navigate to project root
cd "$SLURM_SUBMIT_DIR/../../.."

# Create logs directory
mkdir -p "$SLURM_SUBMIT_DIR/../logs"

# Activate conda environment
source /om2/user/mabdel03/anaconda/etc/profile.d/conda.sh
conda activate /om/scratch/Mon/mabdel03/conda_envs/MAL_env

# Navigate to src for Python modules
cd src

echo "Run {run}: Training PPO_SP for {layout} seed {seed}..."
echo "Paper iterations: {iters}"

python -m human_aware_rl.ppo.train_ppo_sp --layout {layout} --seed {seed} --results_dir results/ppo_sp_run{run}

echo "PPO_SP training complete for {layout} seed {seed}"
'''


def create_ppo_bc_script(layout: str, seed: int, run: int) -> str:
    """Create PPO_BC training script."""
    iters = PAPER_ITERS[layout]
    return f'''#!/bin/bash
#SBATCH --job-name=ppo_bc_{layout}_s{seed}
#SBATCH --output=../logs/ppo_bc_{layout}_seed{seed}_%j.out
#SBATCH --error=../logs/ppo_bc_{layout}_seed{seed}_%j.err
#SBATCH --time=24:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8

# PPO_BC Training (PPO with BC partner)
# Run: {run}
# Layout: {layout}, Seed: {seed}
# Training iterations: {iters} (paper value)

# Navigate to project root
cd "$SLURM_SUBMIT_DIR/../../.."

# Create logs directory
mkdir -p "$SLURM_SUBMIT_DIR/../logs"

# Activate conda environment
source /om2/user/mabdel03/anaconda/etc/profile.d/conda.sh
conda activate /om/scratch/Mon/mabdel03/conda_envs/MAL_env

# Navigate to src for Python modules
cd src

echo "Run {run}: Training PPO_BC for {layout} seed {seed}..."
echo "Paper iterations: {iters}"

# Uses BC models from bc_runs_run{run}/train/{layout}/
python -m human_aware_rl.ppo.train_ppo_bc --layout {layout} --seed {seed} --bc_model_base_dir human_aware_rl/bc_runs_run{run}/train --results_dir results/ppo_bc_run{run}

echo "PPO_BC training complete for {layout} seed {seed}"
'''


def create_ppo_gail_script(layout: str, seed: int, run: int) -> str:
    """Create PPO_GAIL training script (PPO with GAIL partner)."""
    iters = PAPER_ITERS[layout]
    return f'''#!/bin/bash
#SBATCH --job-name=ppo_gail_{layout}_s{seed}
#SBATCH --output=../logs/ppo_gail_{layout}_seed{seed}_%j.out
#SBATCH --error=../logs/ppo_gail_{layout}_seed{seed}_%j.err
#SBATCH --time=24:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8

# PPO_GAIL Training (PPO with GAIL partner instead of BC)
# Run: {run}
# Layout: {layout}, Seed: {seed}
# Training iterations: {iters} (paper value)

# Navigate to project root
cd "$SLURM_SUBMIT_DIR/../../.."

# Create logs directory
mkdir -p "$SLURM_SUBMIT_DIR/../logs"

# Activate conda environment
source /om2/user/mabdel03/anaconda/etc/profile.d/conda.sh
conda activate /om/scratch/Mon/mabdel03/conda_envs/MAL_env

# Navigate to src for Python modules
cd src

echo "Run {run}: Training PPO_GAIL for {layout} seed {seed}..."
echo "Paper iterations: {iters}"

# Uses GAIL models from gail_runs_run{run}/{layout}/
python -m human_aware_rl.ppo.train_ppo_gail --layout {layout} --seed {seed} --gail_model_base_dir human_aware_rl/gail_runs_run{run} --results_dir results/ppo_gail_run{run}

echo "PPO_GAIL training complete for {layout} seed {seed}"
'''


def create_submit_all_script(model_type: str, scripts: list, run: int) -> str:
    """Create a submit_all script."""
    return f'''#!/bin/bash
# Submit all {model_type.upper()} training jobs for Run {run}
# Total jobs: {len(scripts)}

SCRIPT_DIR="$(cd "$(dirname "${{BASH_SOURCE[0]}}")" && pwd)"

echo "Run {run}: Submitting {model_type.upper()} training jobs..."

''' + '\n'.join([f'sbatch "$SCRIPT_DIR/{s}"' for s in scripts]) + f'''

echo ""
echo "Submitted {len(scripts)} {model_type.upper()} jobs for Run {run}"
'''


def create_upload_script(run: int) -> str:
    """Create Dropbox upload script for a run."""
    return f'''#!/bin/bash
# Upload Run {run} results to Dropbox
# Usage: ./upload_to_dropbox.sh
#
# Prerequisites:
#   - dbxcli must be installed and authenticated
#   - Activate the dropbox_cli conda environment first:
#     source /om2/user/mabdel03/anaconda/etc/profile.d/conda.sh
#     conda activate /om2/user/mabdel03/conda_envs/dropbox_cli

set -e

RUN_NUM={run}
DROPBOX_BASE="/All files/Mahmoud Abdelmoneum/6.S890/Test_Runs"
DROPBOX_RUN_DIR="$DROPBOX_BASE/Run_$RUN_NUM"

# Project paths
SCRIPT_DIR="$(cd "$(dirname "${{BASH_SOURCE[0]}}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR/../.."
SRC_DIR="$PROJECT_ROOT/src"

echo "=========================================="
echo "Uploading Run $RUN_NUM to Dropbox"
echo "=========================================="
echo "Dropbox destination: $DROPBOX_RUN_DIR"
echo ""

# Create Dropbox directory structure
echo "Creating Dropbox directories..."
dbxcli mkdir "$DROPBOX_RUN_DIR" 2>/dev/null || true
dbxcli mkdir "$DROPBOX_RUN_DIR/logs" 2>/dev/null || true
dbxcli mkdir "$DROPBOX_RUN_DIR/models" 2>/dev/null || true
dbxcli mkdir "$DROPBOX_RUN_DIR/models/bc_runs" 2>/dev/null || true
dbxcli mkdir "$DROPBOX_RUN_DIR/models/gail_runs" 2>/dev/null || true
dbxcli mkdir "$DROPBOX_RUN_DIR/models/ppo_sp" 2>/dev/null || true
dbxcli mkdir "$DROPBOX_RUN_DIR/models/ppo_bc" 2>/dev/null || true
dbxcli mkdir "$DROPBOX_RUN_DIR/models/ppo_gail" 2>/dev/null || true
echo "Directories created."
echo ""

# Upload logs
echo "=== Uploading logs ==="
if [ -d "$SCRIPT_DIR/logs" ]; then
    for f in "$SCRIPT_DIR/logs"/*.out "$SCRIPT_DIR/logs"/*.err; do
        [ -f "$f" ] && dbxcli put "$f" "$DROPBOX_RUN_DIR/logs/$(basename "$f")" 2>/dev/null && echo "  Uploaded: $(basename "$f")"
    done
else
    echo "  No logs directory found"
fi
echo ""

# Upload README and Results
echo "=== Uploading documentation ==="
[ -f "$SCRIPT_DIR/README.md" ] && dbxcli put "$SCRIPT_DIR/README.md" "$DROPBOX_RUN_DIR/README.md" && echo "  Uploaded: README.md"
[ -f "$SCRIPT_DIR/Run_${{RUN_NUM}}_Results.md" ] && dbxcli put "$SCRIPT_DIR/Run_${{RUN_NUM}}_Results.md" "$DROPBOX_RUN_DIR/Run_${{RUN_NUM}}_Results.md" && echo "  Uploaded: Run_${{RUN_NUM}}_Results.md"
echo ""

# Upload BC models
echo "=== Uploading BC models ==="
BC_DIR="$SRC_DIR/human_aware_rl/bc_runs_run${{RUN_NUM}}"
if [ -d "$BC_DIR" ]; then
    for split in train test; do
        if [ -d "$BC_DIR/$split" ]; then
            for layout_dir in "$BC_DIR/$split"/*/; do
                layout=$(basename "$layout_dir")
                dbxcli mkdir "$DROPBOX_RUN_DIR/models/bc_runs/$split/$layout" 2>/dev/null || true
                for f in "$layout_dir"/*; do
                    [ -f "$f" ] && dbxcli put "$f" "$DROPBOX_RUN_DIR/models/bc_runs/$split/$layout/$(basename "$f")" 2>/dev/null
                done
                echo "  Uploaded: bc_runs/$split/$layout"
            done
        fi
    done
else
    echo "  BC directory not found: $BC_DIR"
fi
echo ""

# Upload GAIL models
echo "=== Uploading GAIL models ==="
GAIL_DIR="$SRC_DIR/human_aware_rl/gail_runs_run${{RUN_NUM}}"
if [ -d "$GAIL_DIR" ]; then
    for layout_dir in "$GAIL_DIR"/*/; do
        layout=$(basename "$layout_dir")
        dbxcli mkdir "$DROPBOX_RUN_DIR/models/gail_runs/$layout" 2>/dev/null || true
        for f in "$layout_dir"/*; do
            [ -f "$f" ] && dbxcli put "$f" "$DROPBOX_RUN_DIR/models/gail_runs/$layout/$(basename "$f")" 2>/dev/null
        done
        echo "  Uploaded: gail_runs/$layout"
    done
else
    echo "  GAIL directory not found: $GAIL_DIR"
fi
echo ""

# Upload PPO_SP checkpoints (final checkpoint + config)
echo "=== Uploading PPO_SP checkpoints ==="
PPO_SP_DIR="$SRC_DIR/results/ppo_sp_run${{RUN_NUM}}"
if [ -d "$PPO_SP_DIR" ]; then
    for run_dir in "$PPO_SP_DIR"/*/; do
        run_name=$(basename "$run_dir")
        latest_ckpt=$(ls -d "$run_dir"/checkpoint_* 2>/dev/null | sort -t_ -k2 -n | tail -1)
        if [ -n "$latest_ckpt" ]; then
            ckpt_name=$(basename "$latest_ckpt")
            dbxcli mkdir "$DROPBOX_RUN_DIR/models/ppo_sp/$run_name" 2>/dev/null || true
            dbxcli mkdir "$DROPBOX_RUN_DIR/models/ppo_sp/$run_name/$ckpt_name" 2>/dev/null || true
            [ -f "$run_dir/config.json" ] && dbxcli put "$run_dir/config.json" "$DROPBOX_RUN_DIR/models/ppo_sp/$run_name/config.json" 2>/dev/null
            for f in "$latest_ckpt"/*; do
                [ -f "$f" ] && dbxcli put "$f" "$DROPBOX_RUN_DIR/models/ppo_sp/$run_name/$ckpt_name/$(basename "$f")" 2>/dev/null
            done
            echo "  Uploaded: ppo_sp/$run_name/$ckpt_name"
        fi
    done
else
    echo "  PPO_SP directory not found: $PPO_SP_DIR"
fi
echo ""

# Upload PPO_BC checkpoints (final checkpoint + config)
echo "=== Uploading PPO_BC checkpoints ==="
PPO_BC_DIR="$SRC_DIR/results/ppo_bc_run${{RUN_NUM}}"
if [ -d "$PPO_BC_DIR" ]; then
    for run_dir in "$PPO_BC_DIR"/*/; do
        run_name=$(basename "$run_dir")
        latest_ckpt=$(ls -d "$run_dir"/checkpoint_* 2>/dev/null | sort -t_ -k2 -n | tail -1)
        if [ -n "$latest_ckpt" ]; then
            ckpt_name=$(basename "$latest_ckpt")
            dbxcli mkdir "$DROPBOX_RUN_DIR/models/ppo_bc/$run_name" 2>/dev/null || true
            dbxcli mkdir "$DROPBOX_RUN_DIR/models/ppo_bc/$run_name/$ckpt_name" 2>/dev/null || true
            [ -f "$run_dir/config.json" ] && dbxcli put "$run_dir/config.json" "$DROPBOX_RUN_DIR/models/ppo_bc/$run_name/config.json" 2>/dev/null
            for f in "$latest_ckpt"/*; do
                [ -f "$f" ] && dbxcli put "$f" "$DROPBOX_RUN_DIR/models/ppo_bc/$run_name/$ckpt_name/$(basename "$f")" 2>/dev/null
            done
            echo "  Uploaded: ppo_bc/$run_name/$ckpt_name"
        fi
    done
else
    echo "  PPO_BC directory not found: $PPO_BC_DIR"
fi
echo ""

# Upload PPO_GAIL checkpoints (final checkpoint + config)
echo "=== Uploading PPO_GAIL checkpoints ==="
PPO_GAIL_DIR="$SRC_DIR/results/ppo_gail_run${{RUN_NUM}}"
if [ -d "$PPO_GAIL_DIR" ]; then
    for run_dir in "$PPO_GAIL_DIR"/*/; do
        run_name=$(basename "$run_dir")
        latest_ckpt=$(ls -d "$run_dir"/checkpoint_* 2>/dev/null | sort -t_ -k2 -n | tail -1)
        if [ -n "$latest_ckpt" ]; then
            ckpt_name=$(basename "$latest_ckpt")
            dbxcli mkdir "$DROPBOX_RUN_DIR/models/ppo_gail/$run_name" 2>/dev/null || true
            dbxcli mkdir "$DROPBOX_RUN_DIR/models/ppo_gail/$run_name/$ckpt_name" 2>/dev/null || true
            [ -f "$run_dir/config.json" ] && dbxcli put "$run_dir/config.json" "$DROPBOX_RUN_DIR/models/ppo_gail/$run_name/config.json" 2>/dev/null
            for f in "$latest_ckpt"/*; do
                [ -f "$f" ] && dbxcli put "$f" "$DROPBOX_RUN_DIR/models/ppo_gail/$run_name/$ckpt_name/$(basename "$f")" 2>/dev/null
            done
            echo "  Uploaded: ppo_gail/$run_name/$ckpt_name"
        fi
    done
else
    echo "  PPO_GAIL directory not found: $PPO_GAIL_DIR"
fi
echo ""

echo "=========================================="
echo "Upload complete!"
echo "View at: $DROPBOX_RUN_DIR"
echo "=========================================="
'''


def create_run_readme(run: int, prev_run: int | None) -> str:
    """Create README.md for a run directory."""
    date = datetime.now().strftime("%B %d, %Y")
    
    diff_section = ""
    if prev_run:
        diff_section = f"""
## Differences from Run {prev_run}

*Document what changed in this run compared to Run {prev_run}:*

- [ ] Hyperparameter changes
- [ ] Code changes
- [ ] Data changes
- [ ] Other modifications

"""
    
    return f'''# Run {run}

**Date:** {date}  
**Status:** Pending

## Overview

This directory contains all scripts and logs for Run {run} of the Overcooked-AI paper reproduction experiments.

## Contents

- `scripts/` - SLURM batch scripts for all training jobs
- `logs/` - Training output and error logs
- `upload_to_dropbox.sh` - Script to upload results to Dropbox
- `Run_{run}_Results.md` - Detailed results and metrics (created after training)

## Training Jobs

| Type | Count | Description |
|------|-------|-------------|
| BC | 5 | Behavioral Cloning (1 per layout) |
| GAIL | 5 | GAIL training (1 per layout) |
| PPO_SP | 25 | PPO Self-Play (5 layouts x 5 seeds) |
| PPO_BC | 25 | PPO with BC partner (5 layouts x 5 seeds) |
| PPO_GAIL | 25 | PPO with GAIL partner (5 layouts x 5 seeds) |
| **Total** | **85** | |
{diff_section}
## Model Output Locations

Models are saved to versioned directories in `src/`:

```
src/human_aware_rl/
├── bc_runs_run{run}/
│   ├── train/        # BC models for PPO_BC training
│   └── test/         # Human Proxy models for evaluation
└── gail_runs_run{run}/

src/results/
├── ppo_sp_run{run}/
├── ppo_bc_run{run}/
└── ppo_gail_run{run}/
```

## Usage

```bash
cd scripts/

# Step 1: Train BC and GAIL models (prerequisites)
./submit_all_bc.sh
./submit_all_gail.sh

# Step 2: After BC/GAIL complete, train PPO models
./submit_all_ppo_sp.sh
./submit_all_ppo_bc.sh
./submit_all_ppo_gail.sh
```

## Uploading Results to Dropbox

After training completes, upload results to Dropbox:

```bash
# Activate Dropbox CLI environment
source /om2/user/mabdel03/anaconda/etc/profile.d/conda.sh
conda activate /om2/user/mabdel03/conda_envs/dropbox_cli

# Run upload script
./upload_to_dropbox.sh
```

Results will be uploaded to: `All files/Mahmoud Abdelmoneum/6.S890/Test_Runs/Run_{run}/`

## Results

See [Run_{run}_Results.md](Run_{run}_Results.md) for detailed results after training completes.
'''


def write_script(path: str, content: str):
    """Write script file and make it executable."""
    with open(path, 'w') as f:
        f.write(content)
    # Make executable
    st = os.stat(path)
    os.chmod(path, st.st_mode | stat.S_IEXEC)


def write_file(path: str, content: str):
    """Write a regular file."""
    with open(path, 'w') as f:
        f.write(content)


def main():
    """Generate all HPC scripts for a specific run."""
    parser = argparse.ArgumentParser(description="Generate HPC SLURM scripts for training")
    parser.add_argument("--run", type=int, required=True, help="Run number (e.g., 4 for run_4/)")
    args = parser.parse_args()
    
    run = args.run
    prev_run = run - 1 if run > 1 else None
    
    # Create run directory structure
    run_dir = os.path.join(SCRIPT_DIR, f"run_{run}")
    scripts_dir = os.path.join(run_dir, "scripts")
    logs_dir = os.path.join(run_dir, "logs")
    
    os.makedirs(scripts_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)
    
    # Track scripts for submit_all
    bc_scripts = []
    gail_scripts = []
    ppo_sp_scripts = []
    ppo_bc_scripts = []
    ppo_gail_scripts = []
    
    print(f"Generating HPC scripts for Run {run}...")
    print(f"Output directory: {run_dir}")
    print()
    
    # BC scripts (one per layout)
    print("BC scripts (1 per layout, trains train+test):")
    for layout in LAYOUTS:
        filename = f"bc_{layout}.sh"
        write_script(os.path.join(scripts_dir, filename), create_bc_script(layout, run))
        bc_scripts.append(filename)
        print(f"  {filename}")
    print()
    
    # GAIL scripts (one per layout)
    print("GAIL scripts (1 per layout):")
    for layout in LAYOUTS:
        filename = f"gail_{layout}.sh"
        write_script(os.path.join(scripts_dir, filename), create_gail_script(layout, run))
        gail_scripts.append(filename)
        print(f"  {filename}")
    print()
    
    # PPO_SP scripts (per layout, per seed)
    print("PPO_SP scripts (5 layouts x 5 seeds = 25):")
    for layout in LAYOUTS:
        for seed in SEEDS:
            filename = f"ppo_sp_{layout}_seed{seed}.sh"
            write_script(os.path.join(scripts_dir, filename), create_ppo_sp_script(layout, seed, run))
            ppo_sp_scripts.append(filename)
    print(f"  Created {len(ppo_sp_scripts)} scripts")
    print()
    
    # PPO_BC scripts (per layout, per seed)
    print("PPO_BC scripts (5 layouts x 5 seeds = 25):")
    for layout in LAYOUTS:
        for seed in SEEDS:
            filename = f"ppo_bc_{layout}_seed{seed}.sh"
            write_script(os.path.join(scripts_dir, filename), create_ppo_bc_script(layout, seed, run))
            ppo_bc_scripts.append(filename)
    print(f"  Created {len(ppo_bc_scripts)} scripts")
    print()
    
    # PPO_GAIL scripts (per layout, per seed)
    print("PPO_GAIL scripts (5 layouts x 5 seeds = 25):")
    for layout in LAYOUTS:
        for seed in SEEDS:
            filename = f"ppo_gail_{layout}_seed{seed}.sh"
            write_script(os.path.join(scripts_dir, filename), create_ppo_gail_script(layout, seed, run))
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
        write_script(os.path.join(scripts_dir, filename), create_submit_all_script(model_type, scripts, run))
        print(f"  {filename} ({len(scripts)} jobs)")
    print()
    
    # Create run README
    readme_path = os.path.join(run_dir, "README.md")
    write_file(readme_path, create_run_readme(run, prev_run))
    print(f"Created: {readme_path}")
    
    # Create upload script
    upload_path = os.path.join(run_dir, "upload_to_dropbox.sh")
    write_script(upload_path, create_upload_script(run))
    print(f"Created: {upload_path}")
    print()
    
    total_scripts = len(bc_scripts) + len(gail_scripts) + len(ppo_sp_scripts) + len(ppo_bc_scripts) + len(ppo_gail_scripts)
    print(f"Total: {total_scripts} training scripts + 5 submit scripts + 1 upload script")
    print()
    print(f"Run {run} directory structure:")
    print(f"  {run_dir}/")
    print(f"  ├── README.md")
    print(f"  ├── upload_to_dropbox.sh")
    print(f"  ├── scripts/  ({total_scripts + 5} files)")
    print(f"  └── logs/     (populated during training)")
    print()
    print("Training order:")
    print(f"  cd run_{run}/scripts")
    print("  1. BC (needed by PPO_BC): ./submit_all_bc.sh")
    print("  2. GAIL (needed by PPO_GAIL): ./submit_all_gail.sh")
    print("  3. After BC/GAIL complete:")
    print("     - PPO_SP: ./submit_all_ppo_sp.sh")
    print("     - PPO_BC: ./submit_all_ppo_bc.sh")
    print("     - PPO_GAIL: ./submit_all_ppo_gail.sh")


if __name__ == "__main__":
    main()
