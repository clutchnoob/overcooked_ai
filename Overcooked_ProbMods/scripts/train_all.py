"""Unified training script for all probabilistic models."""

from __future__ import annotations

import argparse

from probmods.models.bayesian_bc import train_bayesian_bc
from probmods.models.rational_agent import train_rational_agent
from probmods.models.hierarchical_bc import train_hierarchical_bc
from probmods.models.bayesian_gail import train_bayesian_gail
from probmods.models.bayesian_ppo_bc import train_bayesian_ppo_bc
from probmods.models.bayesian_ppo_gail import train_bayesian_ppo_gail


LAYOUTS = [
    "cramped_room",
    "asymmetric_advantages",
    "coordination_ring",
    "forced_coordination",
    "counter_circuit",
]


def train_all_for_layout(layout: str, results_dir: str):
    print(f"\n=== Training all models for {layout} ===")
    train_bayesian_bc(layout, results_dir=results_dir)
    train_rational_agent(layout, results_dir=results_dir)
    train_hierarchical_bc(layout, results_dir=results_dir)
    train_bayesian_gail(layout, results_dir=results_dir)
    train_bayesian_ppo_bc(layout, results_dir=results_dir)
    train_bayesian_ppo_gail(layout, results_dir=results_dir)


def main():
    parser = argparse.ArgumentParser(description="Train all probabilistic models")
    parser.add_argument("--layout", default="cramped_room")
    parser.add_argument("--all_layouts", action="store_true")
    parser.add_argument("--results_dir", default="results")
    args = parser.parse_args()

    if args.all_layouts:
        for layout in LAYOUTS:
            train_all_for_layout(layout, args.results_dir)
    else:
        train_all_for_layout(args.layout, args.results_dir)


if __name__ == "__main__":
    main()
