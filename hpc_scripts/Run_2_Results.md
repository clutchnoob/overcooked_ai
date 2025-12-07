# Run 2 Results - Paper Reproduction

**Date:** December 6, 2024  
**Cluster:** OpenMind (MIT)  
**Environment:** MAL_env (Python 3.12)

## Overview

This document contains the training results for reproducing the Overcooked-AI paper experiments across 5 layouts with multiple training methods.

### Layouts
- `cramped_room`
- `asymmetric_advantages`
- `coordination_ring`
- `forced_coordination`
- `counter_circuit`

### Training Methods
- **BC** - Behavioral Cloning (imitation learning from human data)
- **GAIL** - Generative Adversarial Imitation Learning
- **PPO_SP** - PPO Self-Play (agent plays with copy of itself)
- **PPO_BC** - PPO with BC partner (agent plays with BC-trained partner)

---

## Summary Table

| Layout | BC | HP | GAIL | PPO_SP (mean) | PPO_BC (mean) |
|--------|-----|-----|------|---------------|---------------|
| cramped_room | 76.0 | 56.0 | 58.4 | **188.2** | 184.8 |
| asymmetric_advantages | 120.0 | 88.0 | 42.4 | 213.5 | **232.8** |
| coordination_ring | 56.0 | 24.0 | 19.6 | 3.9 | **8.8** |
| forced_coordination | 28.0 | 12.0 | 5.2 | 2.2 | **25.2** |
| counter_circuit | 24.0 | 16.0 | 9.2 | 0.0 | **6.2** |

---

## Detailed Results

### 1. Behavioral Cloning (BC)

Training BC and Human Proxy (HP) models from human demonstration data.

| Layout | BC Reward | HP Reward |
|--------|-----------|-----------|
| cramped_room | 76.0 | 56.0 |
| asymmetric_advantages | 120.0 | 88.0 |
| coordination_ring | 56.0 | 24.0 |
| forced_coordination | 28.0 | 12.0 |
| counter_circuit | 24.0 | 16.0 |

---

### 2. GAIL (Generative Adversarial Imitation Learning)

Training GAIL models from human demonstration data (1250 iterations).

| Layout | Final Avg Reward (last 50) |
|--------|---------------------------|
| cramped_room | 58.4 |
| asymmetric_advantages | 42.4 |
| coordination_ring | 19.6 |
| forced_coordination | 5.2 |
| counter_circuit | 9.2 |

---

### 3. PPO Self-Play (PPO_SP)

Agent trained via self-play (plays with a copy of itself).

| Layout | Seed 0 | Seed 10 | Seed 20 | Seed 30 | Seed 40 | Mean | Std |
|--------|--------|---------|---------|---------|---------|------|-----|
| cramped_room | 188.2 | 188.2 | 188.2 | 188.2 | 188.2 | **188.2** | 0.0 |
| asymmetric_advantages | 205.8 | 205.8 | 224.0 | 224.0 | 224.0 | **216.7** | 9.4 |
| coordination_ring | 3.9 | 3.9 | 3.9 | 3.9 | 3.9 | **3.9** | 0.0 |
| forced_coordination | 2.2 | 2.2 | 2.2 | 2.2 | 2.2 | **2.2** | 0.0 |
| counter_circuit | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | **0.0** | 0.0 |

---

### 4. PPO with BC Partner (PPO_BC)

Agent trained to play with a BC-trained partner.

| Layout | Seed 0 | Seed 10 | Seed 20 | Seed 30 | Seed 40 | Mean | Std |
|--------|--------|---------|---------|---------|---------|------|-----|
| cramped_room | 184.0 | 170.6 | 193.8 | 193.6 | 182.0 | **184.8** | 9.0 |
| asymmetric_advantages | 231.6 | 283.0 | 205.6 | 227.5 | 216.3 | **232.8** | 28.9 |
| coordination_ring | 9.5 | 8.1 | 10.4 | 8.8 | 7.3 | **8.8** | 1.2 |
| forced_coordination | 31.1 | 29.1 | 10.5 | 16.6 | 38.9 | **25.2** | 11.4 |
| counter_circuit | 1.3 | 0.4 | 14.2 | 9.2 | 6.1 | **6.2** | 5.5 |

---

## Key Observations

### 1. PPO_BC outperforms PPO_SP on most layouts
- **asymmetric_advantages**: PPO_BC (232.8) > PPO_SP (216.7)
- **forced_coordination**: PPO_BC (25.2) >> PPO_SP (2.2) - 11x improvement!
- **counter_circuit**: PPO_BC (6.2) >> PPO_SP (0.0)
- **coordination_ring**: PPO_BC (8.8) > PPO_SP (3.9)

### 2. PPO_SP excels on cramped_room
- cramped_room: PPO_SP (188.2) slightly > PPO_BC (184.8)

### 3. Harder layouts show lower scores across all methods
- coordination_ring, forced_coordination, counter_circuit are more challenging
- Self-play struggles significantly on these layouts

### 4. BC provides reasonable baselines
- BC models achieve decent performance but RL methods improve significantly on easier layouts

### 5. Variance in PPO_BC
- PPO_BC shows higher variance across seeds, especially on harder layouts
- forced_coordination: std=11.4, counter_circuit: std=5.5

---

## Training Configuration

### BC Training
- Human demonstration data from 2019 trials
- Both train and test splits

### GAIL Training
- 1250 iterations
- KL regularization (c=10.0)
- Discriminator accuracy ~0.6-0.9

### PPO Training
- **cramped_room**: 550 iterations
- **Other layouts**: 650 iterations
- 5 seeds per layout: [0, 10, 20, 30, 40]
- Memory: 32GB, CPUs: 8

---

## Files Location

### Dropbox
```
All files/Mahmoud Abdelmoneum/6.S890/Test_Runs/Run_2/
├── logs/           # Training progress logs
└── models/
    ├── bc_runs/    # BC model weights
    ├── gail_runs/  # GAIL model weights
    ├── ppo_sp/     # PPO Self-Play checkpoints
    └── ppo_bc/     # PPO_BC checkpoints
```

### Local (OpenMind)
```
/om/scratch/Mon/mabdel03/6.S890/overcooked_ai/
├── hpc_scripts/logs/     # Training logs
└── src/
    ├── human_aware_rl/
    │   ├── bc_runs/      # BC models
    │   └── gail_runs/    # GAIL models
    └── results/
        ├── ppo_sp/       # PPO_SP results
        └── ppo_bc/       # PPO_BC results
```

---

## Reproduction

To reproduce these results:

```bash
cd hpc_scripts/paper
bash submit_all_paper.sh
```

This submits 55 jobs:
- 5 BC training jobs
- 25 PPO Self-Play jobs (5 layouts × 5 seeds)
- 25 PPO_BC jobs (5 layouts × 5 seeds)
