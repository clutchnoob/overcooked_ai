# Run 4

**Date:** December 08, 2025  
**Status:** Pending

## Overview

This directory contains all scripts and logs for Run 4 of the Overcooked-AI paper reproduction experiments.

## Contents

- `scripts/` - SLURM batch scripts for all training jobs
- `logs/` - Training output and error logs
- `Run_4_Results.md` - Detailed results and metrics (created after training)

## Training Jobs

| Type | Count | Description |
|------|-------|-------------|
| BC | 5 | Behavioral Cloning (1 per layout) |
| GAIL | 5 | GAIL training (1 per layout) |
| PPO_SP | 25 | PPO Self-Play (5 layouts x 5 seeds) |
| PPO_BC | 25 | PPO with BC partner (5 layouts x 5 seeds) |
| PPO_GAIL | 25 | PPO with GAIL partner (5 layouts x 5 seeds) |
| **Total** | **85** | |

## Differences from Run 3

Run 3 had critical bugs that caused performance regression on coordination_ring, forced_coordination, and counter_circuit layouts. Run 4 fixes these issues:

### Bug Fixes

1. **Fixed JAX seed initialization** (`src/human_aware_rl/jaxmarl/ppo.py`)
   - Bug: `self.key = random.PRNGKey(0)` always used seed 0
   - Fix: `self.key = random.PRNGKey(config.seed)` uses the configured seed
   - Impact: Different seeds now produce different training runs (Run 3 had identical results across seeds)

2. **Fixed versioned output directories**
   - Bug: Run 3 scripts didn't specify output directories, causing models to overwrite previous runs
   - Fix: All scripts now use versioned directories (e.g., `bc_runs_run4/`, `gail_runs_run4/`)

3. **Fixed parameter name mismatches in `generate_scripts.py`**
   - Bug: Script generator used wrong CLI argument names (e.g., `--output-dir` instead of `--results_dir`)
   - Fix: Parameter names now match actual training script arguments

4. **Added `--results_dir` to GAIL training** (`src/human_aware_rl/imitation/gail.py`)
   - Bug: GAIL had no CLI option for custom output directory
   - Fix: Added `--results_dir` argument

5. **Added `--output_base_dir` to BC training** (`src/human_aware_rl/imitation/train_bc_models.py`)
   - Bug: BC training only supported output directory for single layouts
   - Fix: Added `--output_base_dir` for versioned multi-layout training

6. **Added `--gail_model_base_dir` to PPO_GAIL training** (`src/human_aware_rl/ppo/train_ppo_gail.py`)
   - Bug: PPO_GAIL couldn't specify versioned GAIL model directory
   - Fix: Added `--gail_model_base_dir` argument

### Expected Improvements

- Different seeds should now produce different results with variance
- GAIL, PPO_SP, and PPO_BC should recover performance on hard layouts
- Models are isolated per run, preventing cross-contamination


## Model Output Locations

Models are saved to versioned directories in `src/`:

```
src/human_aware_rl/
├── bc_runs_run4/
│   ├── train/        # BC models for PPO_BC training
│   └── test/         # Human Proxy models for evaluation
└── gail_runs_run4/

src/results/
├── ppo_sp_run4/
├── ppo_bc_run4/
└── ppo_gail_run4/
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

## Results

See [Run_4_Results.md](Run_4_Results.md) for detailed results after training completes.
