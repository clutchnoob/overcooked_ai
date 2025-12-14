"""
Analysis tools for probabilistic models.
"""

from .compare_models import compute_metrics, evaluate_model_probs
from .uncertainty import summarize_uncertainty, entropy_of_mean, variance_stats
from .interpretability import goal_distribution, rationality_beta, kl_between_policies

__all__ = [
    # Comparison metrics
    "compute_metrics",
    "evaluate_model_probs",
    # Uncertainty quantification
    "summarize_uncertainty",
    "entropy_of_mean",
    "variance_stats",
    # Interpretability
    "goal_distribution",
    "rationality_beta",
    "kl_between_policies",
]
