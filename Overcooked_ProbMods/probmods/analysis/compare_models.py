"""
Model comparison utilities for Overcooked ProbMods.

Metrics:
- Cross-entropy on held-out human data
- Accuracy
- Entropy / uncertainty
- Perplexity

Usage:
    from probmods.analysis.compare_models import evaluate_model_probs
    metrics = evaluate_model_probs(model_fn, layout="cramped_room", dataset="test")

`model_fn` should accept a torch Tensor of states and return action probabilities.
"""

from __future__ import annotations

from typing import Dict, Any

import numpy as np
import torch

from probmods.data.overcooked_data import load_human_data, to_torch, DataConfig


def compute_metrics(action_probs: np.ndarray, true_actions: np.ndarray) -> Dict[str, float]:
    N = len(true_actions)
    true_probs = action_probs[np.arange(N), true_actions]
    cross_entropy = -np.mean(np.log(true_probs + 1e-8))
    accuracy = np.mean(np.argmax(action_probs, axis=1) == true_actions)
    entropy = -np.sum(action_probs * np.log(action_probs + 1e-8), axis=1).mean()
    perplexity = np.exp(cross_entropy)
    return {
        "cross_entropy": float(cross_entropy),
        "accuracy": float(accuracy),
        "mean_entropy": float(entropy),
        "perplexity": float(perplexity),
    }


def evaluate_model_probs(model_fn, layout: str, device: str | None = None, dataset: str = "test") -> Dict[str, float]:
    states_np, actions_np = load_human_data(DataConfig(layout_name=layout, dataset=dataset))
    states, actions = to_torch(states_np, actions_np, device)
    with torch.no_grad():
        action_probs = model_fn(states).cpu().numpy()
    return compute_metrics(action_probs, actions_np)
