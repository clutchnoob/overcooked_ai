"""Train Bayesian Behavior Cloning (BC)."""

from __future__ import annotations

import argparse

from probmods.models.bayesian_bc import train_bayesian_bc


def main():
    parser = argparse.ArgumentParser(description="Train Bayesian BC")
    parser.add_argument("--layout", default="cramped_room")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--prior_scale", type=float, default=1.0)
    parser.add_argument("--results_dir", type=str, default="results")
    args = parser.parse_args()

    train_bayesian_bc(
        args.layout,
        num_epochs=args.epochs,
        prior_scale=args.prior_scale,
        results_dir=args.results_dir,
    )


if __name__ == "__main__":
    main()
