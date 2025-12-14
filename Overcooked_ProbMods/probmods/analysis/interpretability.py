"""
Interpretability helpers for Overcooked ProbMods.

- Extract goal distributions (hierarchical BC)
- Extract rationality β estimates (rational agent)
- KL divergence between policies
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from pyro.infer import Predictive


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


def posterior_stats_from_guide(
    model,
    guide,
    num_samples: int = 1000,
    return_samples: bool = False,
) -> Dict[str, np.ndarray]:
    """
    Extract posterior means and 95% credible intervals for theta and beta.
    Assumes guide/model expose `theta` (action_dim x feature_dim) and `beta` (scalar).
    
    Uses Pyro's param store to get variational parameters and samples from them.
    """
    import pyro
    import pyro.distributions as dist
    
    # Get variational parameters from the param store
    param_store = pyro.get_param_store()
    
    # Find loc and scale parameters for the variational distribution
    locs = {}
    scales = {}
    for name in param_store.keys():
        param = param_store[name]
        if "loc" in name:
            locs[name] = param.detach().cpu()
        elif "scale" in name:
            scales[name] = param.detach().cpu()
    
    # For AutoDiagonalNormal, there's typically one loc and one scale
    # covering all latent variables concatenated
    if len(locs) == 1 and len(scales) == 1:
        loc = list(locs.values())[0]
        scale = list(scales.values())[0]
        
        # Sample from the variational posterior
        with torch.no_grad():
            samples = dist.Normal(loc, scale).sample((num_samples,))
        
        # We need to figure out how the samples map to theta and beta
        # theta has shape (action_dim, state_dim) and beta is a scalar
        # Total params = action_dim * state_dim + 1
        
        theta_size = model.action_dim * model.state_dim
        
        theta_flat = samples[:, :theta_size].numpy()
        theta_s = theta_flat.reshape(num_samples, model.action_dim, model.state_dim)
        
        # Beta is stored in log space for LogNormal, so exponentiate
        log_beta_s = samples[:, theta_size:theta_size + 1].numpy().flatten()
        beta_s = np.exp(log_beta_s)
    else:
        # Fallback: sample by running guide multiple times
        theta_samples = []
        beta_samples = []
        
        with torch.no_grad():
            for _ in range(num_samples):
                guide()  # This sets model.theta and model.beta via PyroSample
                theta_samples.append(model.theta.cpu().numpy())
                beta_val = model.beta
                if hasattr(beta_val, 'cpu'):
                    beta_samples.append(float(beta_val.cpu().numpy()))
                else:
                    beta_samples.append(float(beta_val))
        
        theta_s = np.stack(theta_samples, axis=0)
        beta_s = np.array(beta_samples)

    stats = {
        "theta_mean": theta_s.mean(axis=0),
        "theta_std": theta_s.std(axis=0),
        "theta_ci_lower": np.percentile(theta_s, 2.5, axis=0),
        "theta_ci_upper": np.percentile(theta_s, 97.5, axis=0),
        "beta_mean": float(beta_s.mean()),
        "beta_std": float(beta_s.std()),
        "beta_ci": np.percentile(beta_s, [2.5, 97.5]),
    }

    if return_samples:
        stats["theta_samples"] = theta_s
        stats["beta_samples"] = beta_s
    return stats
