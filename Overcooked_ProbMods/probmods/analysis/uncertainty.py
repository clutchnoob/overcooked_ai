"""
Uncertainty utilities for Overcooked ProbMods.

Summaries of predictive uncertainty for Bayesian models.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np


def summarize_uncertainty(probs_samples: np.ndarray) -> Dict[str, float]:
    """
    Args:
        probs_samples: (S, N, A) posterior samples of action probabilities
    Returns:
        mean_entropy: entropy of mean probs
        epistemic_variance: average variance across samples
    """
    mean_probs = probs_samples.mean(axis=0)
    mean_entropy = float(-np.sum(mean_probs * np.log(mean_probs + 1e-8), axis=-1).mean())
    epistemic_var = float(probs_samples.var(axis=0).mean())
    return {
        "mean_entropy": mean_entropy,
        "epistemic_variance": epistemic_var,
    }


def entropy_of_mean(probs: np.ndarray) -> float:
    return float(-np.sum(probs * np.log(probs + 1e-8), axis=-1).mean())


def variance_stats(probs_samples: np.ndarray) -> Tuple[float, float]:
    var = probs_samples.var(axis=0)
    return float(var.mean()), float(var.max())
