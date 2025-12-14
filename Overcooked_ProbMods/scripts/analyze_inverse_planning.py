#!/usr/bin/env python
"""
End-to-end analysis for Bayesian inverse planning on Overcooked policies.

Steps:
1) Load trained inverse planning posteriors (or run inference via trainer)
2) Compute posterior stats (theta, beta)
3) Plot feature weights and beta comparisons
4) Compute cosine similarity across algorithms
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch


def convert_to_serializable(obj: Any) -> Any:
    """Convert numpy arrays and other non-serializable objects to JSON-safe types."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(v) for v in obj]
    elif isinstance(obj, tuple):
        return [convert_to_serializable(v) for v in obj]
    return obj

from probmods.analysis.feature_mapping import FEATURE_INDEX_TO_NAME
from probmods.analysis.interpretability import posterior_stats_from_guide
from probmods.analysis.visualization import (
    plot_algorithm_comparison,
    plot_beta_comparison,
    plot_feature_weights,
)
from probmods.analysis.compare_models import compare_cognitive_parameters
from probmods.models.inverse_planning import (
    InversePlanningConfig,
    InversePlanningTrainer,
    load_inverse_planning,
)


def run_inference_if_needed(layout: str, tag: str, results_dir: Path, device: str):
    model_dir = results_dir / "inverse_planning" / layout / tag
    if model_dir.exists():
        model, guide, _ = load_inverse_planning(str(model_dir), device=device)
        trainer = None
    else:
        cfg = InversePlanningConfig(layout_name=layout, results_dir=str(results_dir), tag=tag, verbose=True)
        trainer = InversePlanningTrainer(cfg)
        trainer.train()
        model, guide, _ = load_inverse_planning(str(model_dir), device=device)
    return model, guide, trainer


def main():
    parser = argparse.ArgumentParser(description="Analyze Bayesian inverse planning results")
    parser.add_argument("--layouts", nargs="+", default=["cramped_room"])
    parser.add_argument("--tags", nargs="+", default=["default"])
    parser.add_argument("--results-dir", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--save-plots", action="store_true")
    parser.add_argument("--output-json", type=str, default=None)
    parser.add_argument("--top-k", type=int, default=20)
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    results_dir = Path(args.results_dir) if args.results_dir else Path(__file__).resolve().parents[1] / "results"

    all_stats: Dict[str, Dict] = {}

    for layout in args.layouts:
        for tag in args.tags:
            key = f"{layout}:{tag}"
            model, guide, _ = run_inference_if_needed(layout, tag, results_dir, device)
            stats = posterior_stats_from_guide(model, guide, num_samples=1000, return_samples=False)
            all_stats[key] = stats

            plot_feature_weights(
                stats,
                feature_names=FEATURE_INDEX_TO_NAME,
                top_k=args.top_k,
                title=f"{key} feature weights",
                save_path=str(results_dir / "plots" / f"{key}_features.png") if args.save_plots else None,
            )

    if len(all_stats) >= 2:
        plot_algorithm_comparison(
            all_stats,
            feature_names=FEATURE_INDEX_TO_NAME,
            top_k=args.top_k,
            title="Algorithm comparison",
            save_path=str(results_dir / "plots" / "alg_comparison.png") if args.save_plots else None,
        )
        plot_beta_comparison(
            all_stats,
            title="Beta comparison",
            save_path=str(results_dir / "plots" / "beta_comparison.png") if args.save_plots else None,
        )

    cognitive = compare_cognitive_parameters(all_stats)

    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            output = convert_to_serializable({"posteriors": all_stats, "cognitive": cognitive})
            json.dump(output, f, indent=2)
        print(f"\nSaved analysis to {out_path}")


if __name__ == "__main__":
    main()
