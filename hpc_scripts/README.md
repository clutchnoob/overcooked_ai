# HPC Training Scripts

This folder contains SLURM batch scripts for training models on an HPC cluster.

## Run History

### Run 3 (December 7, 2024) - Current
Results stored in Dropbox: `All files/Mahmoud Abdelmoneum/6.S890/Test_Runs/Run_3/`

| Layout | BC | GAIL | PPO_SP | PPO_BC | PPO_GAIL |
|--------|-----|------|--------|--------|----------|
| cramped_room | 76.0 | 0.0 | 186.8 | 195.2 | **296.0** |
| asymmetric_advantages | 76.0 | 0.4 | 194.6 | 218.4 | **442.8** |
| coordination_ring | 48.0 | 0.0 | 0.0 | 0.0 | **246.5** |
| forced_coordination | 16.0 | 0.0 | 0.0 | 0.0 | **10.4** |
| counter_circuit | 52.0 | 0.0 | 0.0 | 0.0 | **76.5** |

**Key finding:** PPO_GAIL significantly outperforms other methods across all layouts.

### Run 2 (December 6, 2024) - Archived
Results stored in Dropbox: `All files/Mahmoud Abdelmoneum/6.S890/Test_Runs/Run_2/`
Local archives: `logs_run2/`, `paper_run2/`, `bc_runs_run2/`, `gail_runs_run2/`

---

## Generate All Scripts

To generate all training scripts:

```bash
python generate_scripts.py
```

This creates:
- **BC**: 5 scripts (1 per layout, trains both train and test data)
- **GAIL**: 5 scripts (1 per layout)
- **PPO_SP**: 25 scripts (5 layouts × 5 seeds)
- **PPO_BC**: 25 scripts (5 layouts × 5 seeds)
- **PPO_GAIL**: 25 scripts (5 layouts × 5 seeds)

## Training Order

1. **First**: Train BC and GAIL models (these are needed as partners)
   ```bash
   ./submit_all_bc.sh
   ./submit_all_gail.sh
   ```

2. **After BC/GAIL complete**: Train PPO models
   ```bash
   ./submit_all_ppo_sp.sh    # Self-play PPO
   ./submit_all_ppo_bc.sh    # PPO with BC partner
   ./submit_all_ppo_gail.sh  # PPO with GAIL partner
   ```

## Paper Hyperparameters

The scripts use paper hyperparameters from "On the Utility of Learning about Humans for Human-AI Coordination":

| Layout | Training Iterations |
|--------|---------------------|
| cramped_room | 550 |
| asymmetric_advantages | 650 |
| coordination_ring | 650 |
| forced_coordination | 650 |
| counter_circuit | 650 |

Each iteration = 12,000 timesteps (train_batch_size from paper).

## Script Types

- `bc_*.sh`: Behavior Cloning (100 epochs)
- `gail_*.sh`: GAIL training (200 epochs)
- `ppo_sp_*_seed*.sh`: PPO Self-Play (paper iterations, no early stopping)
- `ppo_bc_*_seed*.sh`: PPO with BC partner (paper iterations, no early stopping)
- `ppo_gail_*_seed*.sh`: PPO with GAIL partner (paper iterations, no early stopping)

## Fast Training Mode

For quick experiments, add `--fast` flag to the Python commands which:
- Uses 1M timesteps instead of full training
- Enables early stopping
- Reduces training time from ~24h to ~4h

## Seeds

Paper uses seeds: 0, 10, 20, 30, 40 for reproducibility.

## Directory Structure

```
hpc_scripts/
├── logs/              # Current run logs
├── logs_run2/         # Archived Run 2 logs
├── paper_run2/        # Archived Run 2 paper scripts
├── generate_scripts.py
├── bc_*.sh
├── gail_*.sh
├── ppo_sp_*.sh
├── ppo_bc_*.sh
├── ppo_gail_*.sh
└── submit_all_*.sh

src/human_aware_rl/
├── bc_runs/           # Current BC models
├── bc_runs_run2/      # Archived Run 2 BC models
├── gail_runs/         # Current GAIL models
├── gail_runs_run2/    # Archived Run 2 GAIL models
└── ppo_gail_runs/     # PPO_GAIL checkpoints

src/results/
├── ppo_sp/            # PPO Self-Play results
├── ppo_bc/            # PPO_BC results
└── ppo_overcooked/    # PPO_GAIL results
```
