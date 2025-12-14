"""Train Bayesian PPO with BC initialization/anchor."""

from __future__ import annotations

import argparse

from probmods.models.bayesian_ppo_bc import train_bayesian_ppo_bc


def main():
    parser = argparse.ArgumentParser(description="Train Bayesian PPO+BC")
    parser.add_argument("--layout", default="cramped_room")
    parser.add_argument("--timesteps", type=int, default=200_000)
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--bc_anchor_path", type=str, default=None)
    args = parser.parse_args()

    train_bayesian_ppo_bc(
        args.layout,
        total_timesteps=args.timesteps,
        results_dir=args.results_dir,
        bc_anchor_path=args.bc_anchor_path,
    )


if __name__ == "__main__":
    main()
