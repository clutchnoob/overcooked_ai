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

from typing import Dict, Any, Tuple, Mapping

import numpy as np
import torch

from probmods.data.overcooked_data import load_human_data, to_torch, DataConfig
from probmods.analysis.interpretability import kl_between_policies


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


def cosine_similarity_theta(theta1: np.ndarray, theta2: np.ndarray) -> float:
    """Cosine similarity between flattened theta arrays."""
    v1 = theta1.reshape(-1)
    v2 = theta2.reshape(-1)
    denom = np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-12
    return float(np.dot(v1, v2) / denom)


def policy_kl_divergence(
    probs_p: np.ndarray,
    probs_q: np.ndarray,
) -> float:
    """KL divergence between two policy distributions over the same states."""
    return kl_between_policies(probs_p, probs_q)


def compare_cognitive_parameters(
    model_posteriors: Mapping[str, Mapping[str, np.ndarray]],
) -> Dict[str, Any]:
    """
    Compare algorithms on cognitive parameters:
      - Feature weights (theta) cosine similarity
      - Beta comparison
    Assumes posterior stats contain theta_mean and beta_mean.
    """
    algs = list(model_posteriors.keys())
    results: Dict[str, Any] = {"cosine_similarity": {}, "beta": {}}

    # Pairwise cosine similarity over theta
    for i, a_i in enumerate(algs):
        for a_j in algs[i + 1 :]:
            theta_i = model_posteriors[a_i]["theta_mean"]
            theta_j = model_posteriors[a_j]["theta_mean"]
            cos = cosine_similarity_theta(theta_i, theta_j)
            results["cosine_similarity"][f"{a_i}_vs_{a_j}"] = cos

    # Collect beta means
    for a in algs:
        results["beta"][a] = model_posteriors[a].get("beta_mean", np.nan)

    return results
