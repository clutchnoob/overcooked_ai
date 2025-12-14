#!/usr/bin/env python
"""
Run inverse planning SVI for one or more layouts/tags and save posterior params.

Supports:
- Training on human demonstrations (default)
- Training on collected policy trajectories (--use-trajectories)

Usage:
    # Train on human demos
    python scripts/run_inverse_planning.py --layouts cramped_room --tags human_demo

    # Train on collected policy trajectories
    python scripts/run_inverse_planning.py --layouts cramped_room --tags ppo_bc --use-trajectories
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch

from probmods.models.inverse_planning import InversePlanningConfig, InversePlanningTrainer


def load_trajectories(layout: str, source: str, results_dir: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Load previously saved trajectories from collect_policy_trajectories.py."""
    load_dir = results_dir / "trajectories" / layout / source
    if not load_dir.exists():
        raise FileNotFoundError(f"Trajectories not found at {load_dir}")
    
    states = np.load(load_dir / "states.npy")
    actions = np.load(load_dir / "actions.npy")
    print(f"  Loaded {len(states)} transitions from {load_dir}")
    return states, actions


def run_one(
    layout: str,
    tag: str,
    results_dir: Path,
    device: str,
    epochs: int,
    use_trajectories: bool = False,
):
    """Run inverse planning for one layout/tag combination."""
    
    # Determine data source
    states_actions = None
    if use_trajectories or tag not in ["human_demo", "default"]:
        # Try to load from collected trajectories
        source = tag if tag != "default" else "human_demo"
        try:
            states, actions = load_trajectories(layout, source, results_dir)
            states_actions = (states, actions)
            print(f"  Using collected trajectories from {source}")
        except FileNotFoundError as e:
            print(f"  WARNING: Trajectories not found for {source}, falling back to human demo data")
            print(f"  (Run collect_policy_trajectories.py first to collect from trained policies)")
            states_actions = None  # Will load from human data
    
    cfg = InversePlanningConfig(
        layout_name=layout,
        results_dir=str(results_dir),
        tag=tag,
        num_epochs=epochs,
        verbose=True,
    )
    
    trainer = InversePlanningTrainer(cfg, states_actions=states_actions)
    trainer.train()


def main():
    parser = argparse.ArgumentParser(description="Run Bayesian inverse planning SVI")
    parser.add_argument("--layouts", nargs="+", default=["cramped_room"],
                        help="Layouts to process")
    parser.add_argument("--tags", nargs="+", default=["human_demo"],
                        help="Tags/sources: human_demo, bc, ppo_bc, ppo_gail")
    parser.add_argument("--results-dir", type=str, default=None,
                        help="Results directory")
    parser.add_argument("--device", type=str, default=None,
                        help="Device (cuda/cpu)")
    parser.add_argument("--epochs", type=int, default=500,
                        help="Number of training epochs")
    parser.add_argument("--use-trajectories", action="store_true",
                        help="Load from collected trajectories instead of human data")
    
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    results_dir = Path(args.results_dir) if args.results_dir else Path(__file__).resolve().parents[1] / "results"

    print("=" * 60)
    print("BAYESIAN INVERSE PLANNING")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Layouts: {args.layouts}")
    print(f"Tags: {args.tags}")
    print(f"Epochs: {args.epochs}")
    print(f"Use trajectories: {args.use_trajectories}")
    print("=" * 60)

    for layout in args.layouts:
        for tag in args.tags:
            print(f"\n{'='*60}")
            print(f"Layout: {layout}, Tag: {tag}")
            print("=" * 60)
            
            try:
                run_one(layout, tag, results_dir, device, args.epochs, args.use_trajectories)
            except Exception as e:
                print(f"ERROR: {e}")
                continue


if __name__ == "__main__":
    main()
