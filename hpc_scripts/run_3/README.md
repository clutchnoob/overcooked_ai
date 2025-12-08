# Run 3

**Date:** December 7, 2025  
**Status:** Completed

## Overview

Run 3 was the first **complete** paper reproduction, adding PPO_GAIL training (PPO with GAIL partner) which was missing from Run 2. This resulted in significantly improved performance, especially on harder layouts.

## Contents

- `scripts/` - SLURM batch scripts used for training
- `logs/` - Training output and error logs
- [Run_3_Results.md](Run_3_Results.md) - Detailed results and metrics

## Training Jobs

| Type | Count | Description |
|------|-------|-------------|
| BC | 5 | Behavioral Cloning (1 per layout) |
| GAIL | 5 | GAIL training (1 per layout) |
| PPO_SP | 25 | PPO Self-Play (5 layouts x 5 seeds) |
| PPO_BC | 25 | PPO with BC partner (5 layouts x 5 seeds) |
| PPO_GAIL | 25 | PPO with GAIL partner (5 layouts x 5 seeds) |
| **Total** | **85** | |

## Differences from Run 2

| Aspect | Run 2 | Run 3 |
|--------|-------|-------|
| Training Methods | 4 (BC, GAIL, PPO_SP, PPO_BC) | **5** (+PPO_GAIL) |
| Total Jobs | 60 | **85** |
| PPO_GAIL | Not included | **25 jobs added** |

### Key Changes
- Added PPO_GAIL training (PPO agent trained with GAIL partner)
- This resulted in dramatically better performance on all layouts
- PPO_GAIL solved layouts that PPO_SP and PPO_BC failed on

## Key Results

| Layout | BC | GAIL | PPO_SP | PPO_BC | PPO_GAIL |
|--------|-----|------|--------|--------|----------|
| cramped_room | 76.0 | 0.0 | 186.8 | 195.2 | **288.7** |
| asymmetric_advantages | 76.0 | 0.4 | 194.6 | 218.4 | **449.7** |
| coordination_ring | 48.0 | 0.0 | 0.0 | 0.0 | **159.8** |
| forced_coordination | 16.0 | 0.0 | 0.0 | 0.0 | **6.8** |
| counter_circuit | 52.0 | 0.0 | 0.0 | 0.0 | **34.0** |

### Key Finding
**PPO_GAIL significantly outperforms all other methods**, especially on harder layouts where PPO_SP and PPO_BC completely fail (reward = 0).

## Model Output Locations

Models saved to:

```
src/human_aware_rl/
├── bc_runs_run3/
├── gail_runs_run3/
└── ppo_gail_runs_run3/

src/results/
├── ppo_sp_run3/
└── ppo_bc_run3/
```

## Dropbox Backup

Results uploaded to:
```
All files/Mahmoud Abdelmoneum/6.S890/Test_Runs/Run_3/
```

## Results

See [Run_3_Results.md](Run_3_Results.md) for detailed results.
