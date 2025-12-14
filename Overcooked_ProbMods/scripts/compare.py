#!/usr/bin/env python
"""
Compare probabilistic models with each other and baseline models.

Usage:
    python scripts/compare.py --layouts cramped_room asymmetric_advantages
    python scripts/compare.py --layouts cramped_room --include-baselines
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from probmods.data import load_human_data, to_torch, DataConfig
from probmods.analysis import compute_metrics, summarize_uncertainty, kl_between_policies


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_DIR = ROOT_DIR / "results"
BASELINE_DIR = ROOT_DIR.parent / "src" / "human_aware_rl"


PROB_MODELS = [
    "bayesian_bc",
    "rational_agent", 
    "hierarchical_bc",
    "bayesian_gail",
    "bayesian_ppo_bc",
    "bayesian_ppo_gail",
]


def get_model_predictions(
    model_name: str,
    layout: str,
    states: torch.Tensor,
    results_dir: Path,
    device: str,
    num_samples: int = 50,
) -> np.ndarray:
    """Get action probability predictions from a model."""
    from scripts.evaluate import load_model
    from pyro.infer import Predictive
    
    model, guide, config = load_model(model_name, layout, results_dir, device)
    model.to(device)
    
    predictive = Predictive(model, guide=guide, num_samples=num_samples)
    
    with torch.no_grad():
        samples = predictive(states)
        if "action_logits" in samples:
            logits = samples["action_logits"]
        else:
            logits = samples.get("obs", samples.get("_RETURN", model(states).unsqueeze(0)))
        
        probs = torch.softmax(logits, dim=-1).mean(dim=0).cpu().numpy()
    
    return probs


def compare_models(
    layouts: List[str],
    results_dir: Path,
    device: str,
    include_baselines: bool = False,
    num_samples: int = 50,
) -> Dict:
    """Compare all models across layouts."""
    
    results = {}
    
    for layout in layouts:
        print(f"\n{'='*60}")
        print(f"Layout: {layout}")
        print("="*60)
        
        data_config = DataConfig(layout_name=layout, dataset="test")
        states, actions = load_human_data(data_config)
        states_t, _ = to_torch(states, actions, device)
        
        layout_results = {"metrics": {}, "pairwise_kl": {}}
        model_probs = {}
        
        for model_name in PROB_MODELS:
            try:
                probs = get_model_predictions(
                    model_name, layout, states_t, results_dir, device, num_samples
                )
                model_probs[model_name] = probs
                
                metrics = compute_metrics(probs, actions)
                layout_results["metrics"][model_name] = metrics
                
                print(f"\n{model_name}:")
                print(f"  Cross-entropy: {metrics['cross_entropy']:.4f}")
                print(f"  Accuracy: {metrics['accuracy']:.4f}")
                print(f"  Perplexity: {metrics['perplexity']:.4f}")
                
            except Exception as e:
                print(f"Skipping {model_name}: {e}")
        
        # Compute pairwise KL divergences
        print("\nPairwise KL divergences:")
        model_names = list(model_probs.keys())
        for i, name_i in enumerate(model_names):
            for name_j in model_names[i+1:]:
                kl = kl_between_policies(model_probs[name_i], model_probs[name_j])
                layout_results["pairwise_kl"][f"{name_i}_vs_{name_j}"] = kl
                print(f"  {name_i} vs {name_j}: {kl:.4f}")
        
        results[layout] = layout_results
    
    return results


def generate_summary_table(results: Dict) -> str:
    """Generate a markdown summary table."""
    lines = ["# Model Comparison Results\n"]
    
    for layout, layout_results in results.items():
        lines.append(f"\n## {layout}\n")
        
        metrics = layout_results["metrics"]
        if metrics:
            lines.append("| Model | Cross-Entropy | Accuracy | Perplexity |")
            lines.append("|-------|---------------|----------|------------|")
            
            for model_name, model_metrics in metrics.items():
                ce = model_metrics.get("cross_entropy", float("nan"))
                acc = model_metrics.get("accuracy", float("nan"))
                ppl = model_metrics.get("perplexity", float("nan"))
                lines.append(f"| {model_name} | {ce:.4f} | {acc:.4f} | {ppl:.4f} |")
        
        lines.append("")
    
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Compare probabilistic models")
    parser.add_argument("--layouts", nargs="+", default=["cramped_room"])
    parser.add_argument("--results-dir", type=str, default=str(DEFAULT_RESULTS_DIR))
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--include-baselines", action="store_true")
    parser.add_argument("--num-samples", type=int, default=50)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--output-md", type=str, default=None)
    
    args = parser.parse_args()
    
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    results_dir = Path(args.results_dir)
    
    results = compare_models(
        args.layouts, results_dir, device, args.include_baselines, args.num_samples
    )
    
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {output_path}")
    
    if args.output_md:
        md_path = Path(args.output_md)
        md_path.parent.mkdir(parents=True, exist_ok=True)
        with open(md_path, "w") as f:
            f.write(generate_summary_table(results))
        print(f"Summary saved to {md_path}")
    
    return results


if __name__ == "__main__":
    main()
