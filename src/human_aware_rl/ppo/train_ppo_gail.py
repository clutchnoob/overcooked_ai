"""
Train PPO agents with GAIL partner (PPO_GAIL).

Similar to PPO_BC but uses GAIL-trained policy as the partner instead of BC.
The GAIL agent may provide a better human proxy for training.
"""

import os
import argparse
import json
from typing import Optional, List
import numpy as np

from human_aware_rl.data_dir import DATA_DIR
from human_aware_rl.ppo.configs.paper_configs import (
    PAPER_PPO_BC_CONFIGS,
    PAPER_COMMON_PARAMS,
)

# Output directories
PPO_GAIL_SAVE_DIR = os.path.join(DATA_DIR, "ppo_gail_runs")

# Layouts
LAYOUTS = [
    "cramped_room",
    "asymmetric_advantages",
    "coordination_ring",
    "forced_coordination",
    "counter_circuit",
]

# Layout name mapping (paper name -> environment name)
LAYOUT_TO_ENV = {
    "cramped_room": "cramped_room",
    "asymmetric_advantages": "asymmetric_advantages",
    "coordination_ring": "coordination_ring",
    "forced_coordination": "forced_coordination",
    "counter_circuit": "counter_circuit_o_1order",
}


def train_ppo_gail(
    layout: str,
    seed: int = 0,
    total_timesteps: Optional[int] = None,
    early_stop_patience: int = 100,
    gail_model_dir: Optional[str] = None,
    verbose: bool = True,
):
    """
    Train PPO with GAIL partner for a specific layout.
    
    Args:
        layout: Layout name (paper name, will be mapped to env name)
        seed: Random seed
        total_timesteps: Override total timesteps (None uses paper config)
        early_stop_patience: Updates without improvement before stopping
        gail_model_dir: Path to GAIL model directory
        verbose: Print training progress
    """
    import torch
    from human_aware_rl.jaxmarl.ppo import PPOConfig, PPOTrainer
    from human_aware_rl.imitation.gail import GAIL_SAVE_DIR, GAILPolicy
    from human_aware_rl.imitation.bc_agent import BCAgent
    from human_aware_rl.imitation.behavior_cloning import load_bc_model
    from overcooked_ai_py.agents.benchmarking import AgentEvaluator
    from overcooked_ai_py.mdp.overcooked_env import DEFAULT_ENV_PARAMS
    from overcooked_ai_py.mdp.actions import Action
    
    # Get environment layout name
    env_layout = LAYOUT_TO_ENV.get(layout, layout)
    
    # Get paper config for this layout (use same as PPO_BC)
    layout_config = PAPER_PPO_BC_CONFIGS.get(layout, {})
    
    # Merge with common params
    config_dict = {**PAPER_COMMON_PARAMS, **layout_config}
    
    # Override total_timesteps if provided
    if total_timesteps is not None:
        # Calculate num_training_iters from timesteps
        steps_per_iter = config_dict.get("num_envs", 32) * config_dict.get("num_steps", 400)
        config_dict["num_training_iters"] = total_timesteps // steps_per_iter
    
    # Set early stopping
    config_dict["early_stop_patience"] = early_stop_patience
    
    # Set seed
    config_dict["seed"] = seed
    
    # Load GAIL model as partner
    if gail_model_dir is None:
        gail_model_dir = os.path.join(GAIL_SAVE_DIR, layout)
    
    gail_checkpoint_path = os.path.join(gail_model_dir, "model.pt")
    
    if not os.path.exists(gail_checkpoint_path):
        print(f"ERROR: No GAIL model found at {gail_checkpoint_path}")
        print(f"Run: python -m human_aware_rl.imitation.gail --layout {layout}")
        return None
    
    # Setup environment to get dimensions
    ae = AgentEvaluator.from_layout_name(
        mdp_params={"layout_name": env_layout, "old_dynamics": True},
        env_params=DEFAULT_ENV_PARAMS,
    )
    
    def featurize_fn(state):
        return ae.env.featurize_state_mdp(state)
    
    # Get state/action dims
    dummy_state = ae.env.mdp.get_standard_start_state()
    obs_shape = featurize_fn(dummy_state)[0].shape
    state_dim = int(np.prod(obs_shape))
    action_dim = len(Action.ALL_ACTIONS)
    
    # Load GAIL policy
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    gail_policy = GAILPolicy(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=64,
    ).to(device)
    
    checkpoint = torch.load(gail_checkpoint_path, map_location=device)
    gail_policy.load_state_dict(checkpoint["policy_state_dict"])
    gail_policy.eval()
    
    # Create GAIL agent wrapper (similar to BCAgent but for GAIL)
    class GAILAgentWrapper:
        """Wrapper for GAIL policy to work with PPO training."""
        
        def __init__(self, policy, featurize_fn, stochastic=True):
            self.policy = policy
            self.featurize_fn = featurize_fn
            self.stochastic = stochastic
            self.agent_index = 1  # GAIL partner is usually player 1
            
        def action(self, state):
            """Get action for state."""
            obs = self.featurize_fn(state)[self.agent_index]
            obs_flat = obs.flatten()
            obs_tensor = torch.FloatTensor(obs_flat).unsqueeze(0).to(device)
            
            with torch.no_grad():
                action_probs = self.policy(obs_tensor)
                
                if self.stochastic:
                    action_idx = torch.multinomial(action_probs, 1).item()
                else:
                    action_idx = action_probs.argmax(dim=-1).item()
            
            return Action.INDEX_TO_ACTION[action_idx]
        
        def set_agent_index(self, index):
            self.agent_index = index
            
        def reset(self):
            pass
    
    gail_agent = GAILAgentWrapper(gail_policy, featurize_fn, stochastic=True)
    
    # Create output directory
    run_dir = os.path.join(PPO_GAIL_SAVE_DIR, layout, f"seed_{seed}")
    os.makedirs(run_dir, exist_ok=True)
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"Training PPO_GAIL: {layout} (seed={seed})")
        print(f"{'='*60}")
        print(f"Environment: {env_layout}")
        print(f"GAIL model: {gail_model_dir}")
        print(f"Output: {run_dir}")
        print(f"Config: {config_dict}")
        print()
    
    # Create PPO config
    config = PPOConfig(
        layout_name=env_layout,
        **config_dict
    )
    
    # Create trainer with GAIL partner
    trainer = PPOTrainer(config, bc_agent=gail_agent)
    
    # Train
    metrics = trainer.train()
    
    # Save model and config
    trainer.save(run_dir)
    
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(config_dict, f, indent=2)
    
    with open(os.path.join(run_dir, "metrics.json"), "w") as f:
        json.dump({k: float(v) if isinstance(v, (int, float, np.floating)) else v 
                   for k, v in metrics.items()}, f, indent=2)
    
    if verbose:
        print(f"\nTraining complete!")
        print(f"Final reward: {metrics.get('final_mean_reward', 'N/A')}")
        print(f"Saved to: {run_dir}")
    
    return metrics


def train_all_layouts(
    seeds: List[int] = [0],
    total_timesteps: Optional[int] = None,
    early_stop_patience: int = 100,
    verbose: bool = True,
):
    """Train PPO_GAIL for all layouts."""
    results = {}
    
    for layout in LAYOUTS:
        for seed in seeds:
            try:
                metrics = train_ppo_gail(
                    layout=layout,
                    seed=seed,
                    total_timesteps=total_timesteps,
                    early_stop_patience=early_stop_patience,
                    verbose=verbose,
                )
                results[f"{layout}_seed{seed}"] = metrics
            except Exception as e:
                print(f"ERROR training {layout} seed {seed}: {e}")
                import traceback
                traceback.print_exc()
                results[f"{layout}_seed{seed}"] = {"error": str(e)}
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Train PPO with GAIL partner")
    parser.add_argument("--layout", type=str, help="Layout name")
    parser.add_argument("--all_layouts", action="store_true", help="Train all layouts")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--seeds", type=str, default="0", 
                        help="Comma-separated list of seeds")
    parser.add_argument("--fast", action="store_true",
                        help="Fast training mode (1M timesteps, early stopping)")
    parser.add_argument("--timesteps", type=int, default=None,
                        help="Override total timesteps")
    parser.add_argument("--patience", type=int, default=100,
                        help="Early stopping patience")
    
    args = parser.parse_args()
    
    # Parse seeds
    seeds = [int(s) for s in args.seeds.split(",")]
    
    # Set timesteps
    if args.fast:
        total_timesteps = 1_000_000
        early_stop_patience = 40
    else:
        total_timesteps = args.timesteps
        early_stop_patience = args.patience
    
    if args.all_layouts:
        train_all_layouts(
            seeds=seeds,
            total_timesteps=total_timesteps,
            early_stop_patience=early_stop_patience,
        )
    elif args.layout:
        train_ppo_gail(
            layout=args.layout,
            seed=args.seed,
            total_timesteps=total_timesteps,
            early_stop_patience=early_stop_patience,
        )
    else:
        print("Please specify --layout or --all_layouts")
        parser.print_help()


if __name__ == "__main__":
    main()

