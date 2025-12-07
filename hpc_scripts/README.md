# HPC Training Scripts

This folder contains SLURM batch scripts for training models on an HPC cluster.

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

