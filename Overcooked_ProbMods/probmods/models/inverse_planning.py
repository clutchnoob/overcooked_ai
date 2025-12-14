"""
Bayesian inverse planning with linear reward features.

Model:
    R(s, a) = theta_a · phi(s)
    P(a | s) ∝ exp(beta * R(s, a))

We place a Normal prior on each theta_a component and a LogNormal prior on beta.
Inference uses Pyro SVI with either diagonal or low-rank guides.
"""

from __future__ import annotations

import os
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

import numpy as np
import pyro
import pyro.distributions as dist
import torch
import torch.nn.functional as F
from pyro.infer import Predictive, SVI, Trace_ELBO
from pyro.infer.autoguide import AutoDiagonalNormal, AutoLowRankMultivariateNormal
from pyro.nn import PyroModule, PyroSample
from pyro.optim import ClippedAdam

from probmods.data.overcooked_data import DataConfig, load_human_data, to_torch

# Default results directory scoped to Overcooked_ProbMods
ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS_DIR = ROOT_DIR / "results"


@dataclass
class InversePlanningConfig:
    """Configuration for Bayesian inverse planning."""

    layout_name: str = "cramped_room"
    dataset: str = "train"  # "train" or "test"
    data_path: Optional[str] = None  # override if desired

    # Priors
    theta_prior_scale: float = 1.0
    beta_prior_mean: float = 1.0  # lognormal mean parameter (in log-space)
    beta_prior_scale: float = 1.0  # lognormal scale (std in log-space)

    # Inference
    guide_type: str = "diagonal"  # "diagonal" or "lowrank"
    learning_rate: float = 1e-3
    batch_size: int = 256
    num_epochs: int = 200
    num_particles: int = 1
    seed: int = 0

    # Output
    results_dir: str = str(DEFAULT_RESULTS_DIR)
    tag: str = "default"  # subdir name under results/inverse_planning/{layout}/{tag}
    verbose: bool = True


class LinearInversePlanningModel(PyroModule):
    """Linear reward model with action-specific weights and global beta."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        theta_prior_scale: float = 1.0,
        beta_prior_mean: float = 1.0,
        beta_prior_scale: float = 1.0,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.theta_prior_scale = theta_prior_scale
        self.beta_prior_mean = beta_prior_mean
        self.beta_prior_scale = beta_prior_scale

        self.theta = PyroSample(
            dist.Normal(
                torch.zeros(action_dim, state_dim),
                torch.full((action_dim, state_dim), theta_prior_scale),
            ).to_event(2)
        )
        self.beta = PyroSample(
            dist.LogNormal(
                torch.tensor(beta_prior_mean),
                torch.tensor(beta_prior_scale),
            )
        )

    def forward(self, states: torch.Tensor, actions: Optional[torch.Tensor] = None):
        """
        Args:
            states: (B, state_dim)
            actions: (B,) optional ground-truth actions
        """
        theta = self.theta  # (A, F)
        beta = self.beta  # scalar

        # logits: (B, A) = beta * states @ theta^T
        logits = beta * (states @ theta.T)

        if actions is not None:
            with pyro.plate("data", states.shape[0]):
                pyro.sample("obs", dist.Categorical(logits=logits), obs=actions)
        return logits

    def action_probs(self, states: torch.Tensor, beta_override: Optional[float] = None):
        """Return softmax probabilities for given states."""
        theta = self.theta
        beta = beta_override if beta_override is not None else self.beta
        logits = beta * (states @ theta.T)
        return F.softmax(logits, dim=-1)


class InversePlanningTrainer:
    """Trainer for LinearInversePlanningModel using SVI."""

    def __init__(
        self,
        config: InversePlanningConfig,
        states_actions: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    ):
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        pyro.set_rng_seed(config.seed)
        torch.manual_seed(config.seed)

        # Load data (human data by default) if not provided
        if states_actions is None:
            data_cfg = DataConfig(
                layout_name=config.layout_name,
                dataset=config.dataset,
                data_path=config.data_path if config.data_path else None,
            )
            states_np, actions_np = load_human_data(data_cfg)
        else:
            states_np, actions_np = states_actions

        self.states, self.actions = to_torch(states_np, actions_np, device=self.device)
        self.state_dim = self.states.shape[1]
        # Action dim inferred from data (assume all actions appear); default Overcooked has 6 actions
        self.action_dim = int(actions_np.max()) + 1 if len(actions_np) > 0 else 6

        self._setup_model_and_guide()
        self.optimizer = ClippedAdam({"lr": config.learning_rate})
        self.svi = SVI(
            self.model,
            self.guide,
            self.optimizer,
            loss=Trace_ELBO(num_particles=config.num_particles),
        )

        if config.verbose:
            print("Initialized InversePlanningTrainer")
            print(f"  Device: {self.device}")
            print(f"  State dim: {self.state_dim}, Action dim: {self.action_dim}")
            print(f"  Dataset size: {len(self.states)}")

    def _setup_model_and_guide(self):
        self.model = LinearInversePlanningModel(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            theta_prior_scale=self.config.theta_prior_scale,
            beta_prior_mean=self.config.beta_prior_mean,
            beta_prior_scale=self.config.beta_prior_scale,
        ).to(self.device)

        if self.config.guide_type == "diagonal":
            self.guide = AutoDiagonalNormal(self.model)
        elif self.config.guide_type == "lowrank":
            self.guide = AutoLowRankMultivariateNormal(self.model, rank=20)
        else:
            raise ValueError(f"Unknown guide type: {self.config.guide_type}")

        # Warm-start guide to create parameters on device
        with torch.no_grad():
            sample_states = self.states[:2]
            sample_actions = self.actions[:2]
            self.guide(sample_states, sample_actions)

    def train(self) -> Dict[str, Any]:
        num_samples = len(self.states)
        batch_size = self.config.batch_size
        num_batches = (num_samples + batch_size - 1) // batch_size
        losses = []

        for epoch in range(self.config.num_epochs):
            perm = torch.randperm(num_samples, device=self.device)
            epoch_loss = 0.0
            for b in range(num_batches):
                start = b * batch_size
                end = min(start + batch_size, num_samples)
                idx = perm[start:end]
                batch_states = self.states[idx]
                batch_actions = self.actions[idx]
                loss = self.svi.step(batch_states, batch_actions)
                epoch_loss += loss
            avg_loss = epoch_loss / num_batches
            losses.append(avg_loss)

            if self.config.verbose and (epoch + 1) % 10 == 0:
                acc = self._compute_accuracy()
                print(
                    f"Epoch {epoch + 1}/{self.config.num_epochs} "
                    f"| Loss: {avg_loss:.4f} | Acc: {acc:.3f}"
                )

        self._save()
        return {"losses": losses}

    def _compute_accuracy(self) -> float:
        with torch.no_grad():
            self.guide()
            logits = self.model(self.states)
            preds = torch.argmax(logits, dim=-1)
            return (preds == self.actions).float().mean().item()

    def _save(self):
        save_dir = Path(self.config.results_dir) / "inverse_planning" / self.config.layout_name / self.config.tag
        save_dir.mkdir(parents=True, exist_ok=True)
        pyro.get_param_store().save(save_dir / "params.pt")
        config_dict = {
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "theta_prior_scale": self.config.theta_prior_scale,
            "beta_prior_mean": self.config.beta_prior_mean,
            "beta_prior_scale": self.config.beta_prior_scale,
            "guide_type": self.config.guide_type,
            "layout_name": self.config.layout_name,
        }
        with open(save_dir / "config.pkl", "wb") as f:
            pickle.dump(config_dict, f)
        if self.config.verbose:
            print(f"Saved inverse planning model to {save_dir}")

    def get_posterior_samples(self, num_samples: int = 1000) -> Dict[str, np.ndarray]:
        predictive = Predictive(self.model, guide=self.guide, num_samples=num_samples, return_sites=("theta", "beta"))
        with torch.no_grad():
            samples = predictive(self.states)
        return {k: v.cpu().numpy() for k, v in samples.items()}


def load_inverse_planning(model_dir: str, device: Optional[str] = None) -> Tuple[LinearInversePlanningModel, Any, Dict]:
    """Load model + guide + config from disk."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    config_path = os.path.join(model_dir, "config.pkl")
    with open(config_path, "rb") as f:
        config = pickle.load(f)

    model = LinearInversePlanningModel(
        state_dim=config["state_dim"],
        action_dim=config["action_dim"],
        theta_prior_scale=config["theta_prior_scale"],
        beta_prior_mean=config["beta_prior_mean"],
        beta_prior_scale=config["beta_prior_scale"],
    ).to(device)

    if config.get("guide_type", "diagonal") == "diagonal":
        guide = AutoDiagonalNormal(model)
    else:
        guide = AutoLowRankMultivariateNormal(model, rank=20)

    pyro.clear_param_store()
    params_path = os.path.join(model_dir, "params.pt")
    state = torch.load(params_path, map_location=device, weights_only=False)
    pyro.get_param_store().set_state(state)
    return model, guide, config


__all__ = [
    "InversePlanningConfig",
    "LinearInversePlanningModel",
    "InversePlanningTrainer",
    "load_inverse_planning",
]
