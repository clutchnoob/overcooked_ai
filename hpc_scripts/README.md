# HPC Training Scripts

This folder contains SLURM batch scripts for training Overcooked-AI models on an HPC cluster.

## Directory Structure

Each run has its own subdirectory containing all related files:

```
hpc_scripts/
├── generate_scripts.py      # Script generator (use --run N)
├── README.md                 # This file
├── run_2/                    # Completed run
│   ├── README.md             # Run overview and diff from previous
│   ├── Run_2_Results.md      # Detailed results
│   ├── scripts/              # SLURM scripts used
│   └── logs/                 # Training logs
├── run_3/                    # Completed run
│   ├── README.md
│   ├── Run_3_Results.md
│   ├── scripts/
│   └── logs/
└── run_N/                    # Future runs...
```

Model outputs are versioned in `src/`:

```
src/human_aware_rl/
├── bc_runs_run2/
├── bc_runs_run3/
├── gail_runs_run2/
├── gail_runs_run3/
└── ppo_gail_runs_run3/

src/results/
├── ppo_sp_run3/
├── ppo_bc_run3/
└── ppo_overcooked/
```

## Run History

| Run | Date | Methods | Key Finding |
|-----|------|---------|-------------|
| [Run 3](run_3/) | Dec 7, 2025 | BC, GAIL, PPO_SP, PPO_BC, PPO_GAIL | PPO_GAIL dominates all layouts |
| [Run 2](run_2/) | Dec 6, 2025 | BC, GAIL, PPO_SP, PPO_BC | PPO_BC > PPO_SP on hard layouts |

## Starting a New Run

```bash
# Generate scripts for a new run (e.g., Run 4)
python generate_scripts.py --run 4

# This creates:
#   run_4/
#   ├── README.md
#   ├── scripts/    (85 training scripts + 5 submit scripts)
#   └── logs/       (populated during training)
```

## Training Order

From within a run's `scripts/` directory:

```bash
cd run_N/scripts

# Step 1: Train BC and GAIL models (prerequisites)
./submit_all_bc.sh
./submit_all_gail.sh

# Step 2: After BC/GAIL complete, train PPO models
./submit_all_ppo_sp.sh     # Self-play PPO
./submit_all_ppo_bc.sh     # PPO with BC partner
./submit_all_ppo_gail.sh   # PPO with GAIL partner
```

## Script Types

| Type | Count | Description |
|------|-------|-------------|
| `bc_*.sh` | 5 | Behavioral Cloning (100 epochs) |
| `gail_*.sh` | 5 | GAIL training (200 epochs) |
| `ppo_sp_*.sh` | 25 | PPO Self-Play (5 layouts x 5 seeds) |
| `ppo_bc_*.sh` | 25 | PPO with BC partner |
| `ppo_gail_*.sh` | 25 | PPO with GAIL partner |
| `submit_all_*.sh` | 5 | Job submission scripts |
| **Total** | **90** | |

## Paper Hyperparameters

From "On the Utility of Learning about Humans for Human-AI Coordination":

| Layout | Training Iterations |
|--------|---------------------|
| cramped_room | 550 |
| asymmetric_advantages | 650 |
| coordination_ring | 650 |
| forced_coordination | 650 |
| counter_circuit | 650 |

Each iteration = 12,000 timesteps. Seeds: 0, 10, 20, 30, 40.

## Dropbox Backups

Results are backed up to Dropbox after each run:
```
All files/Mahmoud Abdelmoneum/6.S890/Test_Runs/
├── Run_2/
└── Run_3/
```
