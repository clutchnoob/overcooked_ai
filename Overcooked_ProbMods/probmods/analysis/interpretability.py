"""
Interpretability helpers for Overcooked ProbMods.

- Extract goal distributions (hierarchical BC)
- Extract rationality β estimates (rational agent)
- KL divergence between policies
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F


def goal_distribution(agent, states: torch.Tensor) -> np.ndarray:
    """Return mean goal distribution for HierarchicalBCAgent/model."""
    with torch.no_grad():
        goal_probs, _ = agent.model(states) if hasattr(agent, "model") else agent(states)
    return goal_probs.mean(dim=0).cpu().numpy()


def rationality_beta(trainer_or_model) -> float:
    """Extract learned β if present; else 1.0."""
    if hasattr(trainer_or_model, "model") and hasattr(trainer_or_model.model, "beta"):
        beta = trainer_or_model.model.beta
        return float(beta.item() if hasattr(beta, "item") else beta)
    if hasattr(trainer_or_model, "beta"):
        return float(trainer_or_model.beta)
    return 1.0


def kl_between_policies(probs_p: np.ndarray, probs_q: np.ndarray) -> float:
    """Compute KL(P || Q) given action probability arrays."""
    return float(np.sum(probs_p * (np.log(probs_p + 1e-8) - np.log(probs_q + 1e-8)), axis=-1).mean())
