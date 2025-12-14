#!/usr/bin/env python
"""
Collect state-action trajectories from trained policies for inverse planning.

Supports:
- Human demonstrations (from existing data)
- BC policies (behavioral cloning)
- PPO-BC policies
- PPO-GAIL policies

Outputs trajectories as (states, actions) numpy arrays for use with
inverse planning models.

Usage:
    python scripts/collect_policy_trajectories.py --layout cramped_room --source human_demo
    python scripts/collect_policy_trajectories.py --layout cramped_room --source ppo_bc --num-episodes 100
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

# Add paths
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
OVERCOOKED_ROOT = PROJECT_DIR.parent
sys.path.insert(0, str(OVERCOOKED_ROOT / "src"))
sys.path.insert(0, str(PROJECT_DIR))

from overcooked_ai_py.agents.agent import Agent, AgentPair
from overcooked_ai_py.agents.benchmarking import AgentEvaluator
from overcooked_ai_py.mdp.actions import Action
from overcooked_ai_py.mdp.overcooked_env import DEFAULT_ENV_PARAMS

# Data loading
from probmods.data.overcooked_data import DataConfig, load_human_data

# Layouts
LAYOUTS = ["cramped_room", "asymmetric_advantages", "coordination_ring"]

LAYOUT_TO_ENV = {
    "cramped_room": "cramped_room",
    "asymmetric_advantages": "asymmetric_advantages",
    "coordination_ring": "coordination_ring",
    "forced_coordination": "forced_coordination",
    "counter_circuit": "counter_circuit_o_1order",
}

# Model paths - adjust as needed
PPO_BC_DIRS = [
    OVERCOOKED_ROOT / "src" / "human_aware_rl" / "ppo_bc_runs_run4",
    OVERCOOKED_ROOT / "src" / "human_aware_rl" / "ppo_bc_runs_run3",
]

PPO_GAIL_DIRS = [
    OVERCOOKED_ROOT / "src" / "human_aware_rl" / "ppo_gail_runs_run4",
    OVERCOOKED_ROOT / "src" / "human_aware_rl" / "ppo_gail_runs_run3",
]

BC_DIRS = [
    OVERCOOKED_ROOT / "src" / "human_aware_rl" / "bc_runs_run4" / "train",
    OVERCOOKED_ROOT / "src" / "human_aware_rl" / "bc_runs_run3" / "train",
]


class BCAgentWrapper(Agent):
    """Wrapper for BC model to collect trajectories."""

    def __init__(self, model, featurize_fn, stochastic: bool = True):
        super().__init__()
        self.model = model
        self.featurize_fn = featurize_fn
        self.stochastic = stochastic

    def action(self, state):
        obs = self.featurize_fn(state)[self.agent_index]
        obs_flat = obs.flatten()
        obs_tensor = torch.FloatTensor(obs_flat).unsqueeze(0)

        with torch.no_grad():
            logits = self.model(obs_tensor)
            probs = F.softmax(logits, dim=-1)

            if self.stochastic:
                action_idx = torch.multinomial(probs, 1).item()
            else:
                action_idx = probs.argmax(dim=-1).item()

        return Action.INDEX_TO_ACTION[action_idx], {"action_probs": probs.numpy()}

    def reset(self):
        pass


class PPOAgentWrapper(Agent):
    """Wrapper for PPO model (PyTorch version)."""

    def __init__(self, model, featurize_fn, stochastic: bool = True):
        super().__init__()
        self.model = model
        self.featurize_fn = featurize_fn
        self.stochastic = stochastic

    def action(self, state):
        obs = self.featurize_fn(state)[self.agent_index]
        obs_flat = obs.flatten()
        obs_tensor = torch.FloatTensor(obs_flat).unsqueeze(0)

        with torch.no_grad():
            # PPO policy returns (logits, value) or just logits
            output = self.model(obs_tensor)
            if isinstance(output, tuple):
                logits = output[0]
            else:
                logits = output
            probs = F.softmax(logits, dim=-1)

            if self.stochastic:
                action_idx = torch.multinomial(probs, 1).item()
            else:
                action_idx = probs.argmax(dim=-1).item()

        return Action.INDEX_TO_ACTION[action_idx], {"action_probs": probs.numpy()}

    def reset(self):
        pass


def load_bc_model(model_dir: str):
    """Load BC model from directory."""
    from human_aware_rl.imitation.behavior_cloning import load_bc_model as _load
    return _load(model_dir, verbose=False)


def find_model_dir(base_dirs: List[Path], layout: str) -> Optional[Path]:
    """Find model directory for a layout."""
    for base_dir in base_dirs:
        model_path = base_dir / layout
        if model_path.exists():
            return model_path
    return None


def collect_human_trajectories(layout: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load human demonstration trajectories."""
    config = DataConfig(layout_name=layout, dataset="train")
    states, actions = load_human_data(config)
    print(f"  Loaded {len(states)} human demo transitions")
    return states, actions


def collect_policy_trajectories(
    agent: Agent,
    layout: str,
    num_episodes: int = 100,
    horizon: int = 400,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Collect trajectories by running the agent in the environment.
    Uses a copy of the agent as partner for self-play.
    """
    env_layout = LAYOUT_TO_ENV.get(layout, layout)
    
    ae = AgentEvaluator.from_layout_name(
        mdp_params={"layout_name": env_layout, "old_dynamics": True},
        env_params={"horizon": horizon},
    )
    
    def featurize_fn(state):
        return ae.env.featurize_state_mdp(state)
    
    # Set featurize function if needed
    if hasattr(agent, "featurize_fn") and agent.featurize_fn is None:
        agent.featurize_fn = featurize_fn
    
    all_states = []
    all_actions = []
    
    for ep in range(num_episodes):
        # Reset environment
        state = ae.env.mdp.get_standard_start_state()
        ae.env.state = state
        
        agent.set_agent_index(0)
        
        for step in range(horizon):
            # Get observation for agent 0
            obs = featurize_fn(state)
            obs_flat = obs[0].flatten()
            
            # Get action from agent
            action, _ = agent.action(state)
            action_idx = Action.ACTION_TO_INDEX[action]
            
            # Record state-action pair
            all_states.append(obs_flat)
            all_actions.append(action_idx)
            
            # Take random action for partner (agent 1)
            partner_action = Action.ALL_ACTIONS[np.random.randint(len(Action.ALL_ACTIONS))]
            
            # Step environment
            joint_action = (action, partner_action)
            state, _, done, _ = ae.env.step(joint_action)
            
            if done:
                break
        
        if (ep + 1) % 10 == 0:
            print(f"  Collected episode {ep + 1}/{num_episodes}")
    
    states = np.array(all_states)
    actions = np.array(all_actions)
    print(f"  Total transitions collected: {len(states)}")
    
    return states, actions


def collect_from_bc(layout: str, num_episodes: int = 100) -> Tuple[np.ndarray, np.ndarray]:
    """Collect trajectories from BC policy."""
    model_dir = find_model_dir(BC_DIRS, layout)
    if model_dir is None:
        raise FileNotFoundError(f"BC model not found for {layout}")
    
    print(f"  Loading BC model from {model_dir}")
    model, _ = load_bc_model(str(model_dir))
    
    # Setup environment for featurization
    env_layout = LAYOUT_TO_ENV.get(layout, layout)
    ae = AgentEvaluator.from_layout_name(
        mdp_params={"layout_name": env_layout, "old_dynamics": True},
        env_params=DEFAULT_ENV_PARAMS,
    )
    
    def featurize_fn(state):
        return ae.env.featurize_state_mdp(state)
    
    agent = BCAgentWrapper(model, featurize_fn, stochastic=True)
    return collect_policy_trajectories(agent, layout, num_episodes)


def collect_from_ppo_bc(layout: str, num_episodes: int = 100) -> Tuple[np.ndarray, np.ndarray]:
    """Collect trajectories from PPO-BC policy."""
    model_dir = find_model_dir(PPO_BC_DIRS, layout)
    if model_dir is None:
        raise FileNotFoundError(f"PPO-BC model not found for {layout}")
    
    print(f"  Loading PPO-BC model from {model_dir}")
    
    # Try to load PyTorch checkpoint
    checkpoint_path = model_dir / "policy.pt"
    if not checkpoint_path.exists():
        checkpoint_path = model_dir / "checkpoint.pt"
    
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        
        # Create policy network
        from human_aware_rl.ppo.ppo_torch import ActorCritic
        
        # Get state dim from environment
        env_layout = LAYOUT_TO_ENV.get(layout, layout)
        ae = AgentEvaluator.from_layout_name(
            mdp_params={"layout_name": env_layout, "old_dynamics": True},
            env_params=DEFAULT_ENV_PARAMS,
        )
        dummy_state = ae.env.mdp.get_standard_start_state()
        obs_shape = ae.env.featurize_state_mdp(dummy_state)[0].shape
        state_dim = int(np.prod(obs_shape))
        action_dim = len(Action.ALL_ACTIONS)
        
        model = ActorCritic(state_dim, action_dim, hidden_dim=64)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        
        def featurize_fn(state):
            return ae.env.featurize_state_mdp(state)
        
        agent = PPOAgentWrapper(model, featurize_fn, stochastic=True)
        return collect_policy_trajectories(agent, layout, num_episodes)
    else:
        raise FileNotFoundError(f"No checkpoint found in {model_dir}")


def collect_from_ppo_gail(layout: str, num_episodes: int = 100) -> Tuple[np.ndarray, np.ndarray]:
    """Collect trajectories from PPO-GAIL policy."""
    model_dir = find_model_dir(PPO_GAIL_DIRS, layout)
    if model_dir is None:
        raise FileNotFoundError(f"PPO-GAIL model not found for {layout}")
    
    print(f"  Loading PPO-GAIL model from {model_dir}")
    
    # Similar to PPO-BC loading
    checkpoint_path = model_dir / "policy.pt"
    if not checkpoint_path.exists():
        checkpoint_path = model_dir / "checkpoint.pt"
    
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        
        from human_aware_rl.ppo.ppo_torch import ActorCritic
        
        env_layout = LAYOUT_TO_ENV.get(layout, layout)
        ae = AgentEvaluator.from_layout_name(
            mdp_params={"layout_name": env_layout, "old_dynamics": True},
            env_params=DEFAULT_ENV_PARAMS,
        )
        dummy_state = ae.env.mdp.get_standard_start_state()
        obs_shape = ae.env.featurize_state_mdp(dummy_state)[0].shape
        state_dim = int(np.prod(obs_shape))
        action_dim = len(Action.ALL_ACTIONS)
        
        model = ActorCritic(state_dim, action_dim, hidden_dim=64)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        
        def featurize_fn(state):
            return ae.env.featurize_state_mdp(state)
        
        agent = PPOAgentWrapper(model, featurize_fn, stochastic=True)
        return collect_policy_trajectories(agent, layout, num_episodes)
    else:
        raise FileNotFoundError(f"No checkpoint found in {model_dir}")


def save_trajectories(
    states: np.ndarray,
    actions: np.ndarray,
    layout: str,
    source: str,
    output_dir: Path,
):
    """Save collected trajectories to disk."""
    save_dir = output_dir / "trajectories" / layout / source
    save_dir.mkdir(parents=True, exist_ok=True)
    
    np.save(save_dir / "states.npy", states)
    np.save(save_dir / "actions.npy", actions)
    
    # Also save metadata
    metadata = {
        "layout": layout,
        "source": source,
        "num_transitions": len(states),
        "state_dim": states.shape[1] if len(states) > 0 else 0,
    }
    with open(save_dir / "metadata.pkl", "wb") as f:
        pickle.dump(metadata, f)
    
    print(f"  Saved to {save_dir}")


def load_trajectories(layout: str, source: str, results_dir: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Load previously saved trajectories."""
    load_dir = results_dir / "trajectories" / layout / source
    if not load_dir.exists():
        raise FileNotFoundError(f"Trajectories not found at {load_dir}")
    
    states = np.load(load_dir / "states.npy")
    actions = np.load(load_dir / "actions.npy")
    return states, actions


def main():
    parser = argparse.ArgumentParser(description="Collect policy trajectories for inverse planning")
    parser.add_argument("--layout", type=str, default="cramped_room", 
                        choices=LAYOUTS, help="Layout to collect from")
    parser.add_argument("--source", type=str, default="human_demo",
                        choices=["human_demo", "bc", "ppo_bc", "ppo_gail"],
                        help="Source of trajectories")
    parser.add_argument("--num-episodes", type=int, default=100,
                        help="Number of episodes to collect (for policy sources)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory for trajectories")
    parser.add_argument("--all-layouts", action="store_true",
                        help="Collect for all layouts")
    parser.add_argument("--all-sources", action="store_true",
                        help="Collect from all sources")
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir) if args.output_dir else PROJECT_DIR / "results"
    layouts = LAYOUTS if args.all_layouts else [args.layout]
    sources = ["human_demo", "bc", "ppo_bc", "ppo_gail"] if args.all_sources else [args.source]
    
    print("=" * 60)
    print("COLLECTING POLICY TRAJECTORIES")
    print("=" * 60)
    
    for layout in layouts:
        for source in sources:
            print(f"\nLayout: {layout}, Source: {source}")
            print("-" * 40)
            
            try:
                if source == "human_demo":
                    states, actions = collect_human_trajectories(layout)
                elif source == "bc":
                    states, actions = collect_from_bc(layout, args.num_episodes)
                elif source == "ppo_bc":
                    states, actions = collect_from_ppo_bc(layout, args.num_episodes)
                elif source == "ppo_gail":
                    states, actions = collect_from_ppo_gail(layout, args.num_episodes)
                else:
                    raise ValueError(f"Unknown source: {source}")
                
                save_trajectories(states, actions, layout, source, output_dir)
                
            except Exception as e:
                print(f"  ERROR: {e}")
                continue
    
    print("\n" + "=" * 60)
    print("COLLECTION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
