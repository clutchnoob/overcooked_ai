"""
Training script for AIRL (Adversarial Inverse Reinforcement Learning) models.

This script trains AIRL models on human demonstration data for all layouts,
producing more robust human proxy agents compared to Behavior Cloning.

Usage:
    # Train all layouts
    python -m human_aware_rl.imitation.train_airl --all_layouts
    
    # Train specific layout
    python -m human_aware_rl.imitation.train_airl --layout cramped_room
    
    # Fast training (reduced timesteps)
    python -m human_aware_rl.imitation.train_airl --all_layouts --fast
"""

import argparse
import os
import sys
from typing import Dict, List, Optional

import numpy as np

from human_aware_rl.data_dir import DATA_DIR
from human_aware_rl.imitation.airl import (
    AIRL_SAVE_DIR,
    AIRLConfig,
    AIRLTrainer,
    save_airl_model,
)
from human_aware_rl.static import (
    CLEAN_2019_HUMAN_DATA_TRAIN,
    CLEAN_2019_HUMAN_DATA_TEST,
)


# Layout mapping: paper names to data layout names
PAPER_LAYOUTS = [
    "cramped_room",
    "asymmetric_advantages", 
    "coordination_ring",
    "forced_coordination",
    "counter_circuit",
]

# Mapping from paper layout names to data layout names
LAYOUT_TO_DATA = {
    "cramped_room": "cramped_room",
    "asymmetric_advantages": "asymmetric_advantages",
    "coordination_ring": "coordination_ring",
    "forced_coordination": "random0",
    "counter_circuit": "random3",
}

# Mapping from paper layout names to environment layout names
LAYOUT_TO_ENV = {
    "cramped_room": "cramped_room",
    "asymmetric_advantages": "asymmetric_advantages",
    "coordination_ring": "coordination_ring",
    "forced_coordination": "forced_coordination",
    "counter_circuit": "counter_circuit_o_1order",
}


# AIRL hyperparameters (based on Fu et al. 2018 Appendix D)
DEFAULT_AIRL_PARAMS = {
    # Discriminator
    "disc_hidden_dim": 64,
    "disc_num_layers": 2,
    "disc_g_linear": True,  # Linear g(s) for reward disentanglement
    
    # Policy
    "policy_hidden_dim": 64,
    "policy_num_layers": 2,
    
    # Training
    "discriminator_lr": 3e-4,
    "policy_lr": 3e-4,
    "gamma": 0.99,
    "batch_size": 256,
    "disc_updates_per_iter": 5,
    "policy_epochs": 8,
    
    # PPO
    "clip_eps": 0.2,
    "vf_coef": 0.5,
    "ent_coef": 0.1,
    "gae_lambda": 0.95,
    
    # Sample mixing
    "sample_buffer_size": 20,
    
    # Length
    "total_timesteps": 5_000_000,  # 5M timesteps
    "steps_per_iter": 10000,
    
    # Early stopping
    "early_stop_patience": 50,
}


# Fast training parameters
FAST_AIRL_PARAMS = {
    **DEFAULT_AIRL_PARAMS,
    "total_timesteps": 500_000,  # 500K timesteps
    "steps_per_iter": 5000,
    "early_stop_patience": 30,
    "sample_buffer_size": 10,
}


def train_airl_for_layout(
    layout: str,
    data_split: str = "train",
    output_dir: Optional[str] = None,
    verbose: bool = True,
    fast: bool = False,
    seed: int = 0,
    **overrides
) -> Dict:
    """
    Train an AIRL model for a specific layout.
    
    Args:
        layout: Paper layout name (e.g., 'cramped_room')
        data_split: 'train' for AIRL model, 'test' for Human Proxy-style model
        output_dir: Directory to save model (default: AIRL_SAVE_DIR/<split>/<layout>)
        verbose: Whether to print progress
        fast: Use faster training settings
        seed: Random seed
        **overrides: Additional config overrides
        
    Returns:
        Dictionary with training results
    """
    # Get layout names
    data_layout = LAYOUT_TO_DATA.get(layout, layout)
    env_layout = LAYOUT_TO_ENV.get(layout, layout)
    
    # Determine data path
    if data_split == "train":
        data_path = CLEAN_2019_HUMAN_DATA_TRAIN
    elif data_split == "test":
        data_path = CLEAN_2019_HUMAN_DATA_TEST
    else:
        raise ValueError(f"Invalid data_split: {data_split}. Use 'train' or 'test'.")
    
    # Set output directory
    if output_dir is None:
        output_dir = os.path.join(AIRL_SAVE_DIR, data_split, layout)
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"Training AIRL model for layout: {layout}")
        print(f"Data split: {data_split}")
        print(f"Data layout (for loading): {data_layout}")
        print(f"Environment layout: {env_layout}")
        print(f"Output directory: {output_dir}")
        print(f"Fast mode: {fast}")
        print(f"Seed: {seed}")
        print(f"{'='*60}\n")
    
    # Get parameters
    params = FAST_AIRL_PARAMS if fast else DEFAULT_AIRL_PARAMS
    params = {**params, **overrides}
    
    # Set seed
    np.random.seed(seed)
    
    # Create config
    # Note: AIRL internally loads data using layout_name for environment
    # and data_path for demonstrations. We need to modify the data loading
    # to use data_layout for loading demonstrations
    config = AIRLConfig(
        layout_name=env_layout,  # For environment
        horizon=400,
        old_dynamics=True,
        data_path=data_path,
        featurize_states=True,
        disc_hidden_dim=params["disc_hidden_dim"],
        disc_num_layers=params["disc_num_layers"],
        disc_g_linear=params["disc_g_linear"],
        policy_hidden_dim=params["policy_hidden_dim"],
        policy_num_layers=params["policy_num_layers"],
        discriminator_lr=params["discriminator_lr"],
        policy_lr=params["policy_lr"],
        gamma=params["gamma"],
        batch_size=params["batch_size"],
        disc_updates_per_iter=params["disc_updates_per_iter"],
        policy_epochs=params["policy_epochs"],
        clip_eps=params["clip_eps"],
        vf_coef=params["vf_coef"],
        ent_coef=params["ent_coef"],
        gae_lambda=params["gae_lambda"],
        sample_buffer_size=params["sample_buffer_size"],
        total_timesteps=params["total_timesteps"],
        steps_per_iter=params["steps_per_iter"],
        early_stop_patience=params["early_stop_patience"],
        verbose=verbose,
        results_dir=output_dir,
        experiment_name=f"airl_{layout}_{data_split}",
        seed=seed,
    )
    
    # We need to override the data loading to use data_layout
    # This is done by modifying the config's data loading behavior
    # Since AIRLConfig doesn't have a separate data_layout field,
    # we need to patch the trainer's data loading
    
    # Create trainer
    trainer = AIRLTrainer(config)
    
    # If data_layout differs from env_layout, reload data with correct layout
    if data_layout != env_layout:
        from human_aware_rl.human.process_dataframes import get_human_human_trajectories
        import torch
        
        data_params = {
            "layouts": [data_layout],  # Use data layout for loading
            "check_trajectories": False,
            "featurize_states": True,
            "data_path": data_path,
        }
        
        processed_trajs = get_human_human_trajectories(**data_params, silent=not verbose)
        
        # Rebuild expert data
        expert_states = []
        expert_actions = []
        expert_next_states = []
        expert_dones = []
        
        ep_states = processed_trajs["ep_states"]
        ep_actions = processed_trajs["ep_actions"]
        
        for ep_idx in range(len(ep_states)):
            states = ep_states[ep_idx]
            actions = ep_actions[ep_idx]
            
            for t in range(len(states) - 1):
                expert_states.append(states[t].flatten())
                expert_actions.append(int(actions[t]))
                expert_next_states.append(states[t + 1].flatten())
                expert_dones.append(0.0)
            
            if len(states) > 0:
                expert_states.append(states[-1].flatten())
                expert_actions.append(int(actions[-1]) if len(actions) > 0 else 0)
                expert_next_states.append(states[-1].flatten())
                expert_dones.append(1.0)
        
        trainer.expert_states = torch.tensor(
            np.array(expert_states), dtype=torch.float32, device=trainer.device
        )
        trainer.expert_actions = torch.tensor(
            expert_actions, dtype=torch.long, device=trainer.device
        )
        trainer.expert_next_states = torch.tensor(
            np.array(expert_next_states), dtype=torch.float32, device=trainer.device
        )
        trainer.expert_dones = torch.tensor(
            expert_dones, dtype=torch.float32, device=trainer.device
        )
        
        if verbose:
            print(f"Reloaded {len(trainer.expert_states)} expert transitions from {data_layout}")
    
    # Train
    results = trainer.train()
    
    # Save final model to output directory
    final_model_dir = os.path.join(output_dir, f"airl_{layout}_{data_split}", "final")
    save_airl_model(
        final_model_dir,
        trainer.policy,
        trainer.discriminator,
        config,
        verbose=verbose,
    )
    
    return {
        "layout": layout,
        "data_split": data_split,
        "output_dir": output_dir,
        "final_model_dir": final_model_dir,
        "best_reward": results.get("best_reward", 0),
        "total_timesteps": results.get("total_timesteps", 0),
    }


def train_all_layouts(
    data_split: str = "train",
    layouts: Optional[List[str]] = None,
    verbose: bool = True,
    fast: bool = False,
    seed: int = 0,
) -> Dict[str, Dict]:
    """
    Train AIRL models for all layouts.
    
    Args:
        data_split: 'train' for AIRL models, 'test' for Human Proxy models
        layouts: List of layouts to train (default: all paper layouts)
        verbose: Whether to print progress
        fast: Use faster training settings
        seed: Random seed
        
    Returns:
        Dictionary mapping layout names to training results
    """
    if layouts is None:
        layouts = PAPER_LAYOUTS
    
    results = {}
    
    for layout in layouts:
        try:
            result = train_airl_for_layout(
                layout=layout,
                data_split=data_split,
                verbose=verbose,
                fast=fast,
                seed=seed,
            )
            results[layout] = result
        except Exception as e:
            import traceback
            print(f"Error training {layout}: {e}")
            traceback.print_exc()
            results[layout] = {"error": str(e)}
    
    return results


def train_all_models(
    layouts: Optional[List[str]] = None,
    verbose: bool = True,
    fast: bool = False,
    seed: int = 0,
) -> Dict[str, Dict[str, Dict]]:
    """
    Train both AIRL and Human Proxy models for all layouts.
    
    Args:
        layouts: List of layouts to train (default: all paper layouts)
        verbose: Whether to print progress
        fast: Use faster training settings
        seed: Random seed
        
    Returns:
        Dictionary with 'airl' and 'hp' keys, each mapping layouts to results
    """
    results = {
        "airl": {},
        "hp": {},
    }
    
    if verbose:
        print("\n" + "="*60)
        print("TRAINING AIRL MODELS (training data)")
        print("="*60)
    
    results["airl"] = train_all_layouts(
        data_split="train",
        layouts=layouts,
        verbose=verbose,
        fast=fast,
        seed=seed,
    )
    
    if verbose:
        print("\n" + "="*60)
        print("TRAINING AIRL HUMAN PROXY MODELS (test data)")
        print("="*60)
    
    results["hp"] = train_all_layouts(
        data_split="test",
        layouts=layouts,
        verbose=verbose,
        fast=fast,
        seed=seed,
    )
    
    return results


def print_summary(results: Dict[str, Dict[str, Dict]]):
    """Print a summary of training results."""
    print("\n" + "="*60)
    print("AIRL TRAINING SUMMARY")
    print("="*60)
    
    for model_type, model_results in results.items():
        print(f"\n{model_type.upper()} Models:")
        print("-"*40)
        
        for layout, result in model_results.items():
            if "error" in result:
                print(f"  {layout}: ERROR - {result['error']}")
            elif "best_reward" in result:
                print(f"  {layout}: best_reward = {result['best_reward']:.2f}, "
                      f"timesteps = {result['total_timesteps']:,}")
            else:
                print(f"  {layout}: trained")


def main():
    parser = argparse.ArgumentParser(
        description="Train AIRL models for Overcooked AI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    parser.add_argument(
        "--layout",
        type=str,
        default=None,
        choices=PAPER_LAYOUTS,
        help="Train a single layout"
    )
    
    parser.add_argument(
        "--all_layouts",
        action="store_true",
        help="Train all 5 paper layouts"
    )
    
    parser.add_argument(
        "--airl_only",
        action="store_true",
        help="Train only AIRL models (training data)"
    )
    
    parser.add_argument(
        "--hp_only",
        action="store_true",
        help="Train only Human Proxy models (test data)"
    )
    
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Use faster training settings (500K timesteps)"
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed"
    )
    
    parser.add_argument(
        "--timesteps",
        type=int,
        default=None,
        help="Override total timesteps"
    )
    
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce output verbosity"
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Custom output directory (only for single layout)"
    )
    
    args = parser.parse_args()
    
    verbose = not args.quiet
    
    # Build overrides
    overrides = {}
    if args.timesteps:
        overrides["total_timesteps"] = args.timesteps
    
    if args.layout:
        # Train single layout
        if args.hp_only:
            results = {"hp": {args.layout: train_airl_for_layout(
                args.layout, "test", args.output_dir, verbose, args.fast, args.seed, **overrides
            )}}
        elif args.airl_only:
            results = {"airl": {args.layout: train_airl_for_layout(
                args.layout, "train", args.output_dir, verbose, args.fast, args.seed, **overrides
            )}}
        else:
            results = {
                "airl": {args.layout: train_airl_for_layout(
                    args.layout, "train", args.output_dir, verbose, args.fast, args.seed, **overrides
                )},
                "hp": {args.layout: train_airl_for_layout(
                    args.layout, "test", args.output_dir, verbose, args.fast, args.seed, **overrides
                )},
            }
    elif args.all_layouts:
        # Train all layouts
        if args.hp_only:
            results = {"hp": train_all_layouts("test", None, verbose, args.fast, args.seed)}
        elif args.airl_only:
            results = {"airl": train_all_layouts("train", None, verbose, args.fast, args.seed)}
        else:
            results = train_all_models(None, verbose, args.fast, args.seed)
    else:
        parser.print_help()
        print("\nError: Must specify --layout or --all_layouts")
        sys.exit(1)
    
    print_summary(results)
    
    return results


if __name__ == "__main__":
    main()

