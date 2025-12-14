"""
Inference utilities for probabilistic models.

This module provides common inference patterns and utilities
for working with Pyro models.
"""

import torch
from typing import Callable, Dict, Any, Optional

try:
    import pyro
    from pyro.infer import SVI, Trace_ELBO, TraceEnum_ELBO, Predictive
    from pyro.infer.autoguide import AutoDiagonalNormal, AutoMultivariateNormal
    from pyro.optim import Adam, ClippedAdam
    PYRO_AVAILABLE = True
except ImportError:
    PYRO_AVAILABLE = False


def check_pyro():
    """Check if Pyro is available."""
    if not PYRO_AVAILABLE:
        raise ImportError(
            "Pyro is required for inference. Install with: pip install pyro-ppl"
        )


def setup_svi(
    model: Callable,
    guide: Optional[Callable] = None,
    lr: float = 0.001,
    elbo_type: str = "trace",
    num_particles: int = 1,
) -> "SVI":
    """
    Set up Stochastic Variational Inference.
    
    Args:
        model: Pyro model
        guide: Pyro guide (if None, uses AutoDiagonalNormal)
        lr: Learning rate
        elbo_type: "trace" or "trace_enum"
        num_particles: Number of particles for ELBO estimation
        
    Returns:
        Configured SVI object
    """
    check_pyro()
    
    if guide is None:
        guide = AutoDiagonalNormal(model)
    
    optimizer = ClippedAdam({"lr": lr})
    
    if elbo_type == "trace_enum":
        loss = TraceEnum_ELBO(num_particles=num_particles)
    else:
        loss = Trace_ELBO(num_particles=num_particles)
    
    return SVI(model, guide, optimizer, loss=loss)


def sample_posterior(
    model: Callable,
    guide: Callable,
    num_samples: int,
    *args,
    **kwargs
) -> Dict[str, torch.Tensor]:
    """
    Sample from the posterior distribution.
    
    Args:
        model: Pyro model
        guide: Trained guide
        num_samples: Number of posterior samples
        *args, **kwargs: Arguments to pass to the model
        
    Returns:
        Dictionary of posterior samples
    """
    check_pyro()
    
    predictive = Predictive(model, guide=guide, num_samples=num_samples)
    return predictive(*args, **kwargs)


def posterior_predictive(
    model: Callable,
    guide: Callable,
    inputs: torch.Tensor,
    num_samples: int = 100,
    return_sites: Optional[list] = None,
) -> Dict[str, torch.Tensor]:
    """
    Get posterior predictive samples.
    
    Args:
        model: Pyro model
        guide: Trained guide
        inputs: Input tensor
        num_samples: Number of posterior samples
        return_sites: Specific sites to return
        
    Returns:
        Dictionary of posterior predictive samples
    """
    check_pyro()
    
    predictive = Predictive(
        model, 
        guide=guide, 
        num_samples=num_samples,
        return_sites=return_sites,
    )
    return predictive(inputs)


__all__ = [
    "PYRO_AVAILABLE",
    "check_pyro",
    "setup_svi",
    "sample_posterior",
    "posterior_predictive",
]
