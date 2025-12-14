"""
Visualization helpers for inverse planning results.

Provides:
    - plot_feature_weights: horizontal bar plot with 95% CIs
    - plot_algorithm_comparison: side-by-side comparison across algorithms
    - plot_beta_comparison: rationality comparison across algorithms
"""

from __future__ import annotations

from typing import Dict, List, Mapping, Optional

import matplotlib.pyplot as plt
import numpy as np

from probmods.analysis.feature_mapping import FEATURE_INDEX_TO_NAME


def _aggregate_theta(posterior_stats: Mapping[str, np.ndarray]) -> np.ndarray:
    """
    Aggregate theta to a single feature vector.
    If theta is (A, F), we average over actions; if (F,), return directly.
    """
    theta_mean = posterior_stats["theta_mean"]
    if theta_mean.ndim == 2:
        return theta_mean.mean(axis=0)
    return theta_mean


def _aggregate_ci(posterior_stats: Mapping[str, np.ndarray]) -> np.ndarray:
    """Aggregate CIs over actions if needed."""
    lower = posterior_stats["theta_ci_lower"]
    upper = posterior_stats["theta_ci_upper"]
    if lower.ndim == 2:
        return lower.mean(axis=0), upper.mean(axis=0)
    return lower, upper


def plot_feature_weights(
    posterior_stats: Mapping[str, np.ndarray],
    feature_names: Optional[Mapping[int, str]] = None,
    top_k: Optional[int] = None,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
):
    """
    Plot aggregated feature weights with 95% credible intervals.
    Aggregation: average theta over actions to one weight per feature.
    """
    names = feature_names or FEATURE_INDEX_TO_NAME
    weights = _aggregate_theta(posterior_stats)
    ci_lower, ci_upper = _aggregate_ci(posterior_stats)

    # Optional top-k by absolute weight
    idxs = np.arange(len(weights))
    if top_k is not None and top_k < len(weights):
        top_order = np.argsort(np.abs(weights))[::-1][:top_k]
        idxs = top_order

    labels = [names[i] if isinstance(names, dict) else names[i] for i in idxs]
    means = weights[idxs]
    lower_err = means - ci_lower[idxs]
    upper_err = ci_upper[idxs] - means

    y_pos = np.arange(len(idxs))
    plt.figure(figsize=(10, max(6, 0.4 * len(idxs))))
    plt.barh(y_pos, means, xerr=[lower_err, upper_err], color="#4C78A8", alpha=0.8)
    plt.axvline(0, color="black", linewidth=1)
    plt.yticks(y_pos, labels)
    plt.xlabel("Weight")
    plt.title(title or "Feature weights (95% CI)")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    else:
        plt.show()
    plt.close()


def plot_algorithm_comparison(
    results: Dict[str, Mapping[str, np.ndarray]],
    feature_names: Optional[Mapping[int, str]] = None,
    top_k: int = 15,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
):
    """
    Compare feature weights across algorithms.
    Selects top_k features by max abs weight across algorithms.
    """
    names = feature_names or FEATURE_INDEX_TO_NAME

    # Aggregate weights and CIs
    alg_weights: Dict[str, np.ndarray] = {alg: _aggregate_theta(stats) for alg, stats in results.items()}
    alg_ci: Dict[str, tuple[np.ndarray, np.ndarray]] = {
        alg: _aggregate_ci(stats) for alg, stats in results.items()
    }

    # Determine top-k features by max abs weight across algs
    stacked = np.stack(list(alg_weights.values()), axis=0)  # (A, F)
    max_abs = np.max(np.abs(stacked), axis=0)
    top_features = np.argsort(max_abs)[::-1][:top_k]

    label_list = [names[i] if isinstance(names, dict) else names[i] for i in top_features]
    y_pos = np.arange(len(top_features))
    bar_width = 0.8 / max(1, len(results))

    plt.figure(figsize=(12, max(6, 0.5 * len(top_features))))
    for j, (alg, w) in enumerate(alg_weights.items()):
        lower, upper = alg_ci[alg]
        means = w[top_features]
        lerr = means - lower[top_features]
        uerr = upper[top_features] - means
        offsets = y_pos + (j - (len(results) - 1) / 2) * bar_width
        plt.barh(
            offsets,
            means,
            height=bar_width,
            xerr=[lerr, uerr],
            label=alg,
            alpha=0.8,
        )

    plt.axvline(0, color="black", linewidth=1)
    plt.yticks(y_pos, label_list)
    plt.xlabel("Weight")
    plt.title(title or "Algorithm comparison (feature weights, 95% CI)")
    plt.legend()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    else:
        plt.show()
    plt.close()


def plot_beta_comparison(
    results: Dict[str, Mapping[str, np.ndarray]],
    title: Optional[str] = None,
    save_path: Optional[str] = None,
):
    """
    Compare beta (rationality) across algorithms with 95% CI.
    """
    algs = list(results.keys())
    means = [results[a]["beta_mean"] for a in algs]
    ci = [results[a]["beta_ci"] for a in algs]
    lower = [m - c[0] for m, c in zip(means, ci)]
    upper = [c[1] - m for m, c in zip(means, ci)]

    x = np.arange(len(algs))
    plt.figure(figsize=(8, 4))
    plt.bar(x, means, yerr=[lower, upper], color="#F58518", alpha=0.8)
    plt.xticks(x, algs, rotation=15)
    plt.ylabel("Beta")
    plt.title(title or "Beta comparison (95% CI)")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    else:
        plt.show()
    plt.close()


__all__ = [
    "plot_feature_weights",
    "plot_algorithm_comparison",
    "plot_beta_comparison",
]
