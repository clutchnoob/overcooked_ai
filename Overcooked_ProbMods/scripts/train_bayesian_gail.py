"""Train Bayesian GAIL."""

from __future__ import annotations

import argparse

from probmods.models.bayesian_gail import train_bayesian_gail


def main():
    parser = argparse.ArgumentParser(description="Train Bayesian GAIL")
    parser.add_argument("--layout", default="cramped_room")
    parser.add_argument("--timesteps", type=int, default=200_000)
    parser.add_argument("--results_dir", type=str, default="results")
    args = parser.parse_args()

    train_bayesian_gail(
        args.layout,
        total_timesteps=args.timesteps,
        results_dir=args.results_dir,
    )


if __name__ == "__main__":
    main()
