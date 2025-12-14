# Local Runs - Parallelized ProbMods Training

Scripts for running all 6 probabilistic models and inverse planning pipeline **locally** on a laptop or personal workstation with parallel job management.

## Quick Start

```bash
cd Overcooked_ProbMods/local_runs

# Run everything (all models, all layouts, inverse planning)
./run_all.sh

# Preview what will run without executing
./run_all.sh --dry-run
```

## System Requirements

- **Python 3.10** with PyTorch, Pyro, and project dependencies
- **4+ CPU cores** recommended for parallel execution
- **16GB+ RAM** recommended
- **Mac with Apple Silicon** (MPS acceleration) or NVIDIA GPU (CUDA)

## Conda Environment

The scripts use a pre-configured conda environment at:
```
/Users/mahmoudabdelmoneum/Desktop/MIT/Software/Research_Software/conda_envs/CogSciFinalProj
```

This environment includes all necessary packages:
- PyTorch 2.2.2 (with MPS support)
- Pyro-PPL 1.9.1
- NumPy, SciPy, Pandas, Matplotlib, Seaborn
- overcooked_ai and overcooked-probmods (editable installs)

## Scripts Overview

| Script | Purpose |
|--------|---------|
| `run_all.sh` | Main orchestrator - runs entire pipeline |
| `run_stage.sh` | Run specific stage (imitation, rl, inverse_planning) |
| `train_model.sh` | Train a single model on a single layout |
| `run_inverse_planning_local.sh` | Run complete inverse planning pipeline |
| `config.sh` | Configuration (paths, parallelism, layouts) |
| `job_queue.sh` | Background job management utilities |

## Usage Examples

### Run Everything
```bash
./run_all.sh
```

### Run Only Imitation Models
```bash
./run_stage.sh imitation
```

### Run Only RL Models  
```bash
./run_stage.sh rl
```

### Run Only Inverse Planning
```bash
./run_stage.sh inverse_planning
# Or with trajectory collection:
./run_inverse_planning_local.sh --collect
```

### Run Single Model
```bash
./train_model.sh bayesian_bc cramped_room
./train_model.sh rational_agent asymmetric_advantages
./train_model.sh bayesian_ppo_gail coordination_ring
```

### Filter by Layout
```bash
./run_all.sh --layout cramped_room
```

### Filter by Model
```bash
./run_stage.sh imitation --model bayesian_bc
```

### Skip Certain Stages
```bash
# Skip RL models (faster, no env rollouts needed)
./run_all.sh --skip-rl

# Skip inverse planning
./run_all.sh --skip-inverse
```

### Dry Run (Preview)
```bash
./run_all.sh --dry-run
./run_stage.sh imitation --dry-run
```

## Configuration

Edit `config.sh` to customize:

```bash
# Parallelization
MAX_JOBS=4              # Number of concurrent jobs

# Training parameters
IMITATION_EPOCHS=500    # Epochs for imitation models
RL_TIMESTEPS=200000     # Timesteps for RL models

# Layouts to train on
LAYOUTS=(
    "cramped_room"
    "asymmetric_advantages"
    "coordination_ring"
)
```

### Environment Configuration

The conda environment path is configured in `config.sh`:

```bash
CONDA_ENV_PATH="/Users/mahmoudabdelmoneum/Desktop/MIT/Software/Research_Software/conda_envs/CogSciFinalProj"
```

To use a different environment, update this path in `config.sh`.

## Models

### Imitation Models (supervised, fast)
| Model | Description |
|-------|-------------|
| `bayesian_bc` | Bayesian Behavioral Cloning |
| `rational_agent` | Softmax-rational agent with learnable β |
| `hierarchical_bc` | Hierarchical goal-conditioned BC |

### RL Models (require environment rollouts)
| Model | Description |
|-------|-------------|
| `bayesian_gail` | Bayesian GAIL (adversarial IL) |
| `bayesian_ppo_bc` | Bayesian PPO with BC anchor |
| `bayesian_ppo_gail` | Bayesian PPO + GAIL |

## Job Distribution

| Stage | Jobs | Description |
|-------|------|-------------|
| Imitation | 9 | 3 models × 3 layouts |
| RL | 9 | 3 models × 3 layouts |
| Inverse Planning | 9 | 3 layouts × 3 sources |
| **Total** | **27** | With 4 parallel jobs: ~7 batches |

## Device Support

The scripts auto-detect the best available device:

| Priority | Device | Platform |
|----------|--------|----------|
| 1 | CUDA | Linux/Windows with NVIDIA GPU |
| 2 | MPS | Mac with Apple Silicon |
| 3 | CPU | Fallback |

Override with environment variable:
```bash
DEVICE=cpu ./run_all.sh
```

## Output Structure

```
Overcooked_ProbMods/
├── local_runs/
│   ├── logs/                         # Training logs
│   │   ├── bayesian_bc_cramped_room_20241214_103045.log
│   │   └── ...
│   └── run_summary.json              # Overall run summary
└── results/
    ├── bayesian_bc/
    │   ├── cramped_room/
    │   ├── asymmetric_advantages/
    │   └── coordination_ring/
    ├── rational_agent/
    ├── hierarchical_bc/
    ├── bayesian_gail/
    ├── bayesian_ppo_bc/
    ├── bayesian_ppo_gail/
    └── inverse_planning/
        ├── cramped_room/
        │   ├── human_demo/
        │   ├── ppo_bc/
        │   └── ppo_gail/
        ├── analysis_summary.json
        └── plots/
```

## Monitoring Progress

```bash
# Watch all logs
tail -f logs/*.log

# Watch specific model
tail -f logs/bayesian_bc_*.log

# Count running jobs
ps aux | grep train_model | grep -v grep | wc -l
```

## Comparison with HPC Scripts

| Feature | `hpc/` (SLURM) | `local_runs/` |
|---------|----------------|---------------|
| Execution | Cluster scheduler | Background jobs |
| Parallelism | Array jobs | Job queue (4 max) |
| GPU | CUDA (cluster GPUs) | MPS/CUDA/CPU |
| Dependencies | `--dependency=afterok` | Sequential stages |
| Monitoring | `squeue`, `sacct` | `ps`, log files |

## Troubleshooting

### Python not found
Ensure the conda environment path in `config.sh` is correct:
```bash
CONDA_ENV_PATH="/Users/mahmoudabdelmoneum/Desktop/MIT/Software/Research_Software/conda_envs/CogSciFinalProj"
```

### Out of memory
Reduce `MAX_JOBS` in `config.sh`:
```bash
MAX_JOBS=2
```

### MPS errors on Mac
Some operations fall back to CPU. This is handled automatically via:
```bash
export PYTORCH_ENABLE_MPS_FALLBACK=1
```

### Model import errors
Ensure both projects are installed in the conda environment:
```bash
conda activate /Users/mahmoudabdelmoneum/Desktop/MIT/Software/Research_Software/conda_envs/CogSciFinalProj
cd /path/to/overcooked_ai
pip install -e .
cd Overcooked_ProbMods
pip install -e .
```

