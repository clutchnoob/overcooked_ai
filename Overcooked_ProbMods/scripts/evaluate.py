#!/usr/bin/env python
"""
Evaluate trained probabilistic models on held-out data.

Usage:
    python scripts/evaluate.py --model bayesian_bc --layout cramped_room
    python scripts/evaluate.py --model all --layout cramped_room
"""

import argparse
import json
import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import numpy as np

from probmods.data import load_human_data, to_torch, DataConfig
from probmods.analysis import compute_metrics, summarize_uncertainty


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_DIR = ROOT_DIR / "results"


AVAILABLE_MODELS = [
    "bayesian_bc",
    "rational_agent",
    "hierarchical_bc",
    "bayesian_gail",
    "bayesian_ppo_bc",
    "bayesian_ppo_gail",
]


def load_model(model_name: str, layout: str, results_dir: Path, device: str):
    """Load a trained model."""
    model_dir = results_dir / model_name / layout
    
    if not model_dir.exists():
        raise FileNotFoundError(f"Model not found: {model_dir}")
    
    if model_name == "bayesian_bc":
        from probmods.models.bayesian_bc import BayesianBCModel, BayesianBCConfig
        from pyro.infer.autoguide import AutoDiagonalNormal
        
        # Load config
        config_path = model_dir / "config.json"
        if config_path.exists():
            with open(config_path) as f:
                config_dict = json.load(f)
            config = BayesianBCConfig(**config_dict)
        else:
            config = BayesianBCConfig(layout_name=layout)
        
        # Create model and guide
        model = BayesianBCModel(config.state_dim, config.action_dim, config.hidden_dim)
        guide = AutoDiagonalNormal(model)
        
        # Load state
        state = torch.load(model_dir / "model.pt", map_location=device)
        guide.load_state_dict(state["guide_state_dict"])
        
        return model, guide, config
    
    elif model_name == "rational_agent":
        from probmods.models.rational_agent import RationalAgentModel, RationalAgentConfig
        from pyro.infer.autoguide import AutoDiagonalNormal
        
        config_path = model_dir / "config.json"
        if config_path.exists():
            with open(config_path) as f:
                config_dict = json.load(f)
            config = RationalAgentConfig(**config_dict)
        else:
            config = RationalAgentConfig(layout_name=layout)
        
        model = RationalAgentModel(config.state_dim, config.action_dim, config.hidden_dim)
        guide = AutoDiagonalNormal(model)
        
        state = torch.load(model_dir / "model.pt", map_location=device)
        guide.load_state_dict(state["guide_state_dict"])
        
        return model, guide, config
    
    elif model_name == "hierarchical_bc":
        from probmods.models.hierarchical_bc import HierarchicalBCModel, HierarchicalBCConfig
        from pyro.infer.autoguide import AutoDiagonalNormal
        
        config_path = model_dir / "config.json"
        if config_path.exists():
            with open(config_path) as f:
                config_dict = json.load(f)
            config = HierarchicalBCConfig(**config_dict)
        else:
            config = HierarchicalBCConfig(layout_name=layout)
        
        model = HierarchicalBCModel(
            config.state_dim, config.action_dim, 
            config.hidden_dim, config.num_goals
        )
        guide = AutoDiagonalNormal(model)
        
        state = torch.load(model_dir / "model.pt", map_location=device)
        guide.load_state_dict(state["guide_state_dict"])
        
        return model, guide, config
    
    else:
        raise ValueError(f"Unknown model: {model_name}")


def evaluate_model(
    model_name: str,
    layout: str,
    results_dir: Path,
    device: str,
    num_samples: int = 50,
) -> dict:
    """Evaluate a single model."""
    print(f"\nEvaluating {model_name} on {layout}...")
    
    # Load test data
    data_config = DataConfig(layout_name=layout, dataset="test")
    states, actions = load_human_data(data_config)
    states_t, actions_t = to_torch(states, actions, device)
    
    # Load model
    model, guide, config = load_model(model_name, layout, results_dir, device)
    model.to(device)
    
    # Get predictions with uncertainty
    from pyro.infer import Predictive
    
    predictive = Predictive(model, guide=guide, num_samples=num_samples)
    
    with torch.no_grad():
        samples = predictive(states_t)
        # Get action logits from samples
        if "action_logits" in samples:
            logits_samples = samples["action_logits"]
        else:
            # Handle different model outputs
            logits_samples = samples.get("obs", samples.get("_RETURN", None))
        
        if logits_samples is not None:
            probs_samples = torch.softmax(logits_samples, dim=-1).cpu().numpy()
            mean_probs = probs_samples.mean(axis=0)
        else:
            # Fallback: run model directly
            mean_probs = torch.softmax(model(states_t), dim=-1).cpu().numpy()
            probs_samples = mean_probs[np.newaxis, ...]
    
    # Compute metrics
    metrics = compute_metrics(mean_probs, actions)
    
    # Add uncertainty metrics
    uncertainty = summarize_uncertainty(probs_samples)
    metrics.update(uncertainty)
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate probabilistic models")
    parser.add_argument(
        "--model", 
        type=str, 
        default="all",
        help=f"Model to evaluate: {AVAILABLE_MODELS} or 'all'"
    )
    parser.add_argument(
        "--layout",
        type=str,
        default="cramped_room",
        help="Layout name"
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default=str(DEFAULT_RESULTS_DIR),
        help="Directory with trained models"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device (cuda/cpu)"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=50,
        help="Number of posterior samples"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON file for results"
    )
    
    args = parser.parse_args()
    
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    results_dir = Path(args.results_dir)
    
    # Determine which models to evaluate
    if args.model == "all":
        models = AVAILABLE_MODELS
    else:
        models = [args.model]
    
    all_results = {}
    
    for model_name in models:
        try:
            metrics = evaluate_model(
                model_name,
                args.layout,
                results_dir,
                device,
                args.num_samples,
            )
            all_results[model_name] = metrics
            
            print(f"\n{model_name} results:")
            for key, value in metrics.items():
                print(f"  {key}: {value:.4f}")
                
        except FileNotFoundError as e:
            print(f"Skipping {model_name}: {e}")
        except Exception as e:
            print(f"Error evaluating {model_name}: {e}")
    
    # Save results
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nResults saved to {output_path}")
    
    return all_results


if __name__ == "__main__":
    main()
