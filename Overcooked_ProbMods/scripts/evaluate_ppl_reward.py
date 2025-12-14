"""
Evaluate PPL models by running games with Human Proxy.

This script evaluates the trained PPL models (Bayesian BC, Rational Agent, 
Hierarchical BC) by pairing them with a Human Proxy model and measuring 
game rewards - the same evaluation used for Run 3/4 RL models.
"""

import os
import sys
import json
import pickle
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F

# Add paths
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
OVERCOOKED_ROOT = PROJECT_DIR.parent
sys.path.insert(0, str(OVERCOOKED_ROOT / "src"))
sys.path.insert(0, str(PROJECT_DIR))

from overcooked_ai_py.agents.benchmarking import AgentEvaluator
from overcooked_ai_py.agents.agent import Agent, AgentPair
from overcooked_ai_py.mdp.actions import Action
from overcooked_ai_py.mdp.overcooked_env import DEFAULT_ENV_PARAMS

# PPL model imports
from probmods.models.bayesian_bc import BayesianBCModel, BayesianBCAgent, load_bayesian_bc
from probmods.models.rational_agent import RationalAgentModel, RationalAgent, QNetwork
from probmods.models.hierarchical_bc import HierarchicalBCModel, HierarchicalBCAgent


# Layouts to evaluate
LAYOUTS = [
    "cramped_room",
    "asymmetric_advantages",
    "coordination_ring",
]

LAYOUT_TO_ENV = {
    "cramped_room": "cramped_room",
    "asymmetric_advantages": "asymmetric_advantages",
    "coordination_ring": "coordination_ring",
    "forced_coordination": "forced_coordination",
    "counter_circuit": "counter_circuit_o_1order",
}

# Paths
PPL_RESULTS_DIR = PROJECT_DIR / "results"
# Try multiple possible locations for BC test models
BC_RUNS_DIRS = [
    OVERCOOKED_ROOT / "src" / "human_aware_rl" / "bc_runs_run4" / "test",
    OVERCOOKED_ROOT / "src" / "human_aware_rl" / "bc_runs_run3" / "test",
    OVERCOOKED_ROOT / "src" / "human_aware_rl" / "Run_3" / "models" / "bc_runs" / "test",
]


class BCAgentWrapper(Agent):
    """Wrapper for BC model (human proxy)."""

    def __init__(self, model, featurize_fn, stochastic=True):
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


def load_bc_model(model_dir: str):
    """Load BC model."""
    from human_aware_rl.imitation.behavior_cloning import load_bc_model as _load
    return _load(model_dir, verbose=False)


def load_rational_agent(model_dir: str, device: str = "cpu") -> Tuple[RationalAgent, Dict]:
    """Load trained Rational Agent model."""
    import pyro
    
    config_path = os.path.join(model_dir, "config.pkl")
    with open(config_path, "rb") as f:
        config = pickle.load(f)
    
    # Create model
    model = RationalAgentModel(
        state_dim=config["state_dim"],
        action_dim=config["action_dim"],
        q_hidden_dims=config["q_hidden_dims"],
        learn_beta=config["learn_beta"],
    ).to(device)
    
    # Load Q-network weights (use weights_only=False for Pyro compatibility)
    q_network_path = os.path.join(model_dir, "q_network.pt")
    if os.path.exists(q_network_path):
        model.q_network.load_state_dict(
            torch.load(q_network_path, map_location=device, weights_only=False)
        )
    
    # Load Pyro param store for beta (must use weights_only=False for Pyro)
    params_path = os.path.join(model_dir, "params.pt")
    if os.path.exists(params_path):
        pyro.clear_param_store()
        # Pyro's param store uses pickle internally, so we need weights_only=False
        state = torch.load(params_path, map_location=device, weights_only=False)
        pyro.get_param_store().set_state(state)
    
    return model, config


def load_hierarchical_bc(model_dir: str, device: str = "cpu") -> Tuple[HierarchicalBCModel, Dict]:
    """Load trained Hierarchical BC model."""
    config_path = os.path.join(model_dir, "config.pkl")
    with open(config_path, "rb") as f:
        config = pickle.load(f)
    
    model = HierarchicalBCModel(
        state_dim=config["state_dim"],
        num_goals=config["num_goals"],
        action_dim=config["action_dim"],
        goal_hidden_dims=config["goal_hidden_dims"],
        policy_hidden_dims=config["policy_hidden_dims"],
    ).to(device)
    
    model_path = os.path.join(model_dir, "model.pt")
    model.load_state_dict(torch.load(model_path, map_location=device))
    
    return model, config


def evaluate_pair(
    agent1: Agent,
    agent2: Agent,
    layout: str,
    num_games: int = 10,
    swapped: bool = False,
) -> List[float]:
    """Evaluate an agent pair and return rewards."""
    env_layout = LAYOUT_TO_ENV.get(layout, layout)
    
    ae = AgentEvaluator.from_layout_name(
        mdp_params={"layout_name": env_layout, "old_dynamics": True},
        env_params={"horizon": 400},
    )
    
    if swapped:
        agent2.set_agent_index(0)
        agent1.set_agent_index(1)
        pair = AgentPair(agent2, agent1)
    else:
        agent1.set_agent_index(0)
        agent2.set_agent_index(1)
        pair = AgentPair(agent1, agent2)
    
    results = ae.evaluate_agent_pair(pair, num_games=num_games, display=False)
    return results["ep_returns"]


def evaluate_ppl_model(
    model_type: str,
    layout: str,
    hp_model,
    featurize_fn,
    num_games: int = 10,
    device: str = "cpu",
    verbose: bool = True,
) -> Dict[str, Any]:
    """Evaluate a PPL model against human proxy."""
    
    model_dir = PPL_RESULTS_DIR / model_type / layout
    
    if not model_dir.exists():
        if verbose:
            print(f"  ✗ No {model_type} model found for {layout}")
        return {"mean": 0, "std": 0, "se": 0, "n": 0, "error": "Model not found"}
    
    try:
        # Load PPL model based on type
        if model_type == "rational_agent":
            model, config = load_rational_agent(str(model_dir), device)
            agent = RationalAgent(
                model=model,
                featurize_fn=featurize_fn,
                agent_index=0,
                beta=1.0,  # Use moderate rationality for evaluation
                stochastic=True,
                device=device,
            )
        elif model_type == "bayesian_bc":
            model, guide, config = load_bayesian_bc(str(model_dir), device)
            agent = BayesianBCAgent(
                model=model,
                guide=guide,
                featurize_fn=featurize_fn,
                agent_index=0,
                stochastic=True,
                num_posterior_samples=5,  # Fewer samples for faster eval
                device=device,
            )
        elif model_type == "hierarchical_bc":
            model, config = load_hierarchical_bc(str(model_dir), device)
            agent = HierarchicalBCAgent(
                model=model,
                featurize_fn=featurize_fn,
                agent_index=0,
                stochastic=True,
                device=device,
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        if verbose:
            print(f"  ✓ Loaded {model_type}")
        
        # Create human proxy agent
        hp_agent = BCAgentWrapper(hp_model, featurize_fn, stochastic=True)
        
        # Evaluate both positions
        rewards_normal = evaluate_pair(agent, hp_agent, layout, num_games, swapped=False)
        rewards_swapped = evaluate_pair(agent, hp_agent, layout, num_games, swapped=True)
        
        all_rewards = rewards_normal + rewards_swapped
        
        return {
            "mean": float(np.mean(all_rewards)),
            "std": float(np.std(all_rewards)),
            "se": float(np.std(all_rewards) / np.sqrt(len(all_rewards))),
            "n": len(all_rewards),
            "normal": {
                "mean": float(np.mean(rewards_normal)),
                "std": float(np.std(rewards_normal)),
                "rewards": rewards_normal,
            },
            "swapped": {
                "mean": float(np.mean(rewards_swapped)),
                "std": float(np.std(rewards_swapped)),
                "rewards": rewards_swapped,
            },
        }
        
    except Exception as e:
        if verbose:
            print(f"  ✗ Error evaluating {model_type}: {e}")
        return {"mean": 0, "std": 0, "se": 0, "n": 0, "error": str(e)}


def run_full_evaluation(
    num_games: int = 10,
    layouts: Optional[List[str]] = None,
    model_types: Optional[List[str]] = None,
    verbose: bool = True,
) -> Dict:
    """Run full evaluation of all PPL models."""
    
    if layouts is None:
        layouts = LAYOUTS
    if model_types is None:
        model_types = ["rational_agent", "bayesian_bc", "hierarchical_bc"]
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if verbose:
        print(f"Using device: {device}")
    
    results = {}
    
    for layout in layouts:
        if verbose:
            print(f"\n{'='*60}")
            print(f"Layout: {layout}")
            print(f"{'='*60}")
        
        env_layout = LAYOUT_TO_ENV.get(layout, layout)
        layout_results = {}
        
        # Setup environment for featurization
        ae = AgentEvaluator.from_layout_name(
            mdp_params={"layout_name": env_layout, "old_dynamics": True},
            env_params=DEFAULT_ENV_PARAMS,
        )
        
        def featurize_fn(state):
            return ae.env.featurize_state_mdp(state)
        
        # Load human proxy (BC test model) - try multiple locations
        hp_model = None
        hp_dir = None
        load_errors = []
        for bc_dir in BC_RUNS_DIRS:
            hp_path = bc_dir / layout
            if verbose:
                print(f"  Trying HP path: {hp_path} (exists: {hp_path.exists()})")
            if hp_path.exists():
                try:
                    hp_model, _ = load_bc_model(str(hp_path))
                    hp_dir = hp_path
                    break
                except Exception as e:
                    load_errors.append(f"{hp_path}: {e}")
                    continue
        
        if hp_model is None:
            print(f"  ✗ Could not load Human Proxy for {layout}")
            for err in load_errors:
                print(f"    Error: {err}")
            continue
        
        try:
            if verbose:
                print(f"  ✓ Loaded Human Proxy from {hp_dir}")
        except Exception as e:
            print(f"  ✗ Failed to load Human Proxy: {e}")
            continue
        
        # Evaluate each PPL model
        for model_type in model_types:
            if verbose:
                print(f"\n  Evaluating {model_type}...")
            
            result = evaluate_ppl_model(
                model_type=model_type,
                layout=layout,
                hp_model=hp_model,
                featurize_fn=featurize_fn,
                num_games=num_games,
                device=device,
                verbose=verbose,
            )
            
            layout_results[f"{model_type}_hp"] = result
            
            if verbose and "error" not in result:
                print(f"    {model_type}+HP: {result['mean']:.1f} ± {result['se']:.1f}")
        
        results[layout] = layout_results
    
    return results


def print_results_table(results: Dict):
    """Print results in a nice table format."""
    print("\n" + "="*100)
    print("PPL MODEL EVALUATION RESULTS (Reward when paired with Human Proxy)")
    print("="*100)
    
    # Header
    model_types = ["rational_agent_hp", "bayesian_bc_hp", "hierarchical_bc_hp"]
    header = f"{'Layout':<25}"
    for mt in model_types:
        name = mt.replace("_hp", "").replace("_", " ").title()
        header += f" {name:<20}"
    print(header)
    print("-"*100)
    
    # Data rows
    for layout in results:
        row = f"{layout:<25}"
        for mt in model_types:
            if mt in results[layout]:
                r = results[layout][mt]
                if "error" in r:
                    row += f" {'N/A':<20}"
                else:
                    row += f" {r['mean']:.1f} ± {r['se']:.1f}".ljust(21)
            else:
                row += f" {'N/A':<20}"
        print(row)
    
    print("="*100)


def compare_with_baselines(results: Dict, run4_results_path: Optional[str] = None):
    """Compare PPL results with Run 4 baseline results."""
    print("\n" + "="*100)
    print("COMPARISON WITH RUN 4 BASELINES")
    print("="*100)
    
    # Try to load Run 4 results
    run4_path = run4_results_path or str(OVERCOOKED_ROOT / "eval_results" / "run4" / "run4_results_20251211_005738.json")
    
    if os.path.exists(run4_path):
        with open(run4_path) as f:
            run4_results = json.load(f)
        
        print(f"\n{'Layout':<22} {'BC+HP':<12} {'PPO_BC+HP':<12} {'Rational':<12} {'Bayesian BC':<12} {'Hierarchical':<12}")
        print("-"*100)
        
        for layout in results:
            if layout in run4_results:
                bc = run4_results[layout].get("bc_hp", {}).get("mean", 0)
                ppo_bc = run4_results[layout].get("ppo_bc_hp", {}).get("mean", 0)
            else:
                bc = ppo_bc = 0
            
            rational = results[layout].get("rational_agent_hp", {}).get("mean", 0)
            bayesian = results[layout].get("bayesian_bc_hp", {}).get("mean", 0)
            hierarchical = results[layout].get("hierarchical_bc_hp", {}).get("mean", 0)
            
            print(f"{layout:<22} {bc:>10.1f}  {ppo_bc:>10.1f}  {rational:>10.1f}  {bayesian:>10.1f}  {hierarchical:>10.1f}")
        
        print("="*100)
    else:
        print(f"Run 4 results not found at {run4_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate PPL models with Human Proxy")
    parser.add_argument("--num_games", type=int, default=10,
                        help="Number of games per evaluation")
    parser.add_argument("--layouts", nargs="+", default=None,
                        help="Layouts to evaluate (default: all)")
    parser.add_argument("--models", nargs="+", default=None,
                        help="Model types to evaluate (default: all)")
    parser.add_argument("--save_results", type=str, default=None,
                        help="Path to save results JSON")
    parser.add_argument("--compare_baselines", action="store_true",
                        help="Compare with Run 4 baselines")
    parser.add_argument("--verbose", action="store_true", default=True,
                        help="Verbose output")
    
    args = parser.parse_args()
    
    print("="*60)
    print("PPL Model Evaluation (Reward-based)")
    print("="*60)
    print(f"PPL Results Dir: {PPL_RESULTS_DIR}")
    print(f"BC Runs Dirs: {[str(d) for d in BC_RUNS_DIRS]}")
    print(f"Games per evaluation: {args.num_games}")
    
    # Run evaluation
    results = run_full_evaluation(
        num_games=args.num_games,
        layouts=args.layouts,
        model_types=args.models,
        verbose=args.verbose,
    )
    
    # Print results
    print_results_table(results)
    
    # Compare with baselines
    if args.compare_baselines:
        compare_with_baselines(results)
    
    # Save results
    if args.save_results:
        save_path = args.save_results
    else:
        save_path = str(PROJECT_DIR / "results" / f"ppl_eval_rewards_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {save_path}")


if __name__ == "__main__":
    main()
