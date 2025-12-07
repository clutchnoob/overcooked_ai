# Test 1 - Experiment Results

**Date:** December 6, 2025  
**Environment:** MIT OpenMind HPC Cluster

---

## Summary

This test ran training experiments across 5 Overcooked layouts using three training paradigms:
1. **BC (Behavior Cloning)** - Imitation learning from human demonstrations
2. **PPO SP (Self-Play)** - Reinforcement learning with self-play
3. **PPO BC** - Reinforcement learning initialized with BC policy

---

## BC (Behavior Cloning) Results

| Layout | BC Model Reward | HP Model Reward |
|--------|-----------------|-----------------|
| cramped_room | 72.00 | 48.00 |
| asymmetric_advantages | 72.00 | 32.00 |
| coordination_ring | 72.00 | 16.00 |
| counter_circuit | 20.00 | 16.00 |
| forced_coordination | 20.00 | 12.00 |

**Notes:**
- All BC models trained for up to 100 epochs with early stopping
- Training time: ~2-5 minutes per layout

---

## PPO Self-Play Results

| Layout | Final Reward | Best Reward | Training Time | Episodes |
|--------|--------------|-------------|---------------|----------|
| asymmetric_advantages | 274.49 | 299.34 | 3626.9s (~60 min) | 2496 |
| cramped_room | 34.63 | 34.63 | 3290.0s (~55 min) | 2496 |
| forced_coordination | 5.72 | 8.12 | 3248.8s (~54 min) | 2496 |
| coordination_ring | 0.00 | 4.27 | 3450.6s (~58 min) | 2496 |
| counter_circuit | 0.00 | 0.81 | 3603.9s (~60 min) | 2496 |

**Notes:**
- Self-play agents learn to coordinate with themselves
- `asymmetric_advantages` achieved highest rewards
- `coordination_ring` and `counter_circuit` struggled to learn effective policies

---

## PPO BC (BC-Initialized) Results

| Layout | Final Reward | Best Reward | Training Time | Episodes |
|--------|--------------|-------------|---------------|----------|
| asymmetric_advantages | 216.86 | 225.03 | 3747.9s (~62 min) | 2496 |
| forced_coordination | 38.58 | 41.70 | 3492.4s (~58 min) | 2496 |
| cramped_room | 21.99 | 27.13 | 3530.0s (~59 min) | 2496 |
| coordination_ring | 4.60 | 5.83 | 3435.2s (~57 min) | 2496 |
| counter_circuit | 2.79 | 3.40 | 3388.2s (~56 min) | 2496 |

**Notes:**
- PPO BC agents are initialized with behavior cloning weights and trained to coordinate with a frozen BC partner
- Generally shows improved coordination compared to pure self-play on harder layouts

---

## Comparison: PPO SP vs PPO BC

| Layout | PPO SP (Final) | PPO BC (Final) | Better |
|--------|----------------|----------------|--------|
| asymmetric_advantages | 274.49 | 216.86 | PPO SP |
| cramped_room | 34.63 | 21.99 | PPO SP |
| forced_coordination | 5.72 | 38.58 | **PPO BC** |
| coordination_ring | 0.00 | 4.60 | **PPO BC** |
| counter_circuit | 0.00 | 2.79 | **PPO BC** |

**Key Observations:**
- PPO Self-Play excels on layouts where symmetric strategies work well (asymmetric_advantages, cramped_room)
- PPO BC outperforms on coordination-heavy layouts (forced_coordination, coordination_ring, counter_circuit)
- BC initialization helps bootstrap learning on difficult coordination tasks

---

## Job Configuration

- **CPUs:** 16 cores per job
- **Memory:** 16GB (BC), 32GB (PPO)
- **Time Limit:** 2 hours (BC), 12 hours (PPO)
- **Training Mode:** Fast mode enabled for PPO experiments

---

## Log Files

All logs stored in: `hpc_scripts/logs/`

- BC logs: `bc_<layout>_<job_id>.out/.err`
- PPO SP logs: `ppo_sp_<layout>_<job_id>.out/.err`
- PPO BC logs: `ppo_bc_<layout>_<job_id>.out/.err`

