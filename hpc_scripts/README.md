# HPC Training Scripts

Parallelized SLURM batch scripts for training all Overcooked AI models.

## Quick Start

```bash
# Submit all training jobs (105 total)
./submit_all.sh

# Dry run (see what would be submitted)
./submit_all.sh --dry-run

# Submit only BC models first
./submit_all.sh --bc-only

# Submit only PPO models (assumes BC already trained)
./submit_all.sh --ppo-only
```

## Directory Structure

```
hpc_scripts/
├── config.sh           # Shared configuration (conda, paths)
├── submit_all.sh       # Master submission script
├── README.md           # This file
├── bc/                 # Behavior Cloning (5 scripts)
│   ├── submit_bc.sh
│   ├── cramped_room.sh
│   ├── asymmetric_advantages.sh
│   ├── coordination_ring.sh
│   ├── forced_coordination.sh
│   └── counter_circuit.sh
├── ppo_sp/             # PPO Self-Play (25 scripts)
│   ├── submit_ppo_sp.sh
│   └── {layout}_seed{0,10,20,30,40}.sh
├── ppo_bc/             # PPO with BC partner (25 scripts)
│   ├── submit_ppo_bc.sh
│   └── {layout}_seed{0,10,20,30,40}.sh
├── ppo_gail/           # PPO with GAIL partner (25 scripts)
│   ├── submit_ppo_gail.sh
│   └── {layout}_seed{0,10,20,30,40}.sh
├── ppo_airl/           # PPO with AIRL partner (25 scripts)
│   ├── submit_ppo_airl.sh
│   └── {layout}_seed{0,10,20,30,40}.sh
└── logs/               # SLURM output logs
```

## Training Configuration

| Parameter | Value |
|-----------|-------|
| **Layouts** | cramped_room, asymmetric_advantages, coordination_ring, forced_coordination, counter_circuit |
| **Seeds** | 0, 10, 20, 30, 40 |
| **Time limit** | 48 hours (PPO), 4 hours (BC) |
| **Memory** | 32GB (PPO), 16GB (BC) |
| **CPUs** | 16 (PPO), 8 (BC) |

## Job Dependencies

```
BC (5 jobs) ─────┬──> PPO_BC (25 jobs)
                 ├──> PPO_GAIL (25 jobs)
                 └──> PPO_AIRL (25 jobs)

PPO_SP (25 jobs) ───> (independent, no dependencies)
```

## Output Locations

| Model | Output Directory |
|-------|------------------|
| BC | `src/human_aware_rl/bc_runs/{train\|test}/{layout}/` |
| PPO_SP | `src/human_aware_rl/results/ppo_sp/ppo_sp_{layout}_seed{seed}/` |
| PPO_BC | `src/human_aware_rl/results/ppo_bc/ppo_bc_{layout}_seed{seed}/` |
| PPO_GAIL | `src/human_aware_rl/results/ppo_gail/ppo_gail_{layout}_seed{seed}/` |
| PPO_AIRL | `src/human_aware_rl/results/ppo_airl/ppo_airl_{layout}_seed{seed}/` |

## Monitoring Jobs

```bash
# View your running jobs
squeue -u $USER

# View job details
scontrol show job <job_id>

# Cancel a job
scancel <job_id>

# Cancel all your jobs
scancel -u $USER
```

## Submitting Individual Jobs

```bash
# Submit a single BC job
sbatch hpc_scripts/bc/cramped_room.sh

# Submit a single PPO_SP job
sbatch hpc_scripts/ppo_sp/cramped_room_seed0.sh

# Submit PPO_BC with dependency on BC completion
sbatch --dependency=afterok:<bc_job_id> hpc_scripts/ppo_bc/cramped_room_seed0.sh
```

## Environment

- **Conda**: `/om/scratch/Mon/mabdel03/conda_envs/MAL_env`
- **Project root**: `/om/scratch/Mon/mabdel03/6.S890/overcooked_ai`
