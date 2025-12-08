# Run 2

**Date:** December 6, 2025  
**Status:** Completed

## Overview

Run 2 was the first systematic paper reproduction attempt, training BC, GAIL, PPO_SP, and PPO_BC models across all 5 Overcooked layouts.

## Contents

- `scripts/` - SLURM batch scripts used for training
- `logs/` - Training output and error logs
- [Run_2_Results.md](Run_2_Results.md) - Detailed results and metrics

## Training Jobs

| Type | Count | Description |
|------|-------|-------------|
| BC | 5 | Behavioral Cloning (1 per layout) |
| GAIL | 5 | GAIL training (1 per layout) |
| PPO_SP | 25 | PPO Self-Play (5 layouts x 5 seeds) |
| PPO_BC | 25 | PPO with BC partner (5 layouts x 5 seeds) |
| **Total** | **60** | |

## Differences from Run 1

Run 2 was the first organized run. Run 1 was initial testing/debugging.

## Key Results

| Layout | BC | GAIL | PPO_SP | PPO_BC |
|--------|-----|------|--------|--------|
| cramped_room | 76.0 | 58.4 | **188.2** | 184.8 |
| asymmetric_advantages | 120.0 | 42.4 | 213.5 | **232.8** |
| coordination_ring | 56.0 | 19.6 | 3.9 | **8.8** |
| forced_coordination | 28.0 | 5.2 | 2.2 | **25.2** |
| counter_circuit | 24.0 | 9.2 | 0.0 | **6.2** |

## Model Output Locations

Models saved to:

```
src/human_aware_rl/
├── bc_runs_run2/
└── gail_runs_run2/

src/results/
├── ppo_sp_run2/
└── ppo_bc_run2/
```

## Notes

- PPO_GAIL was NOT included in Run 2
- GAIL models showed moderate performance
- PPO_BC outperformed PPO_SP on harder layouts

## Results

See [Run_2_Results.md](Run_2_Results.md) for detailed results.
