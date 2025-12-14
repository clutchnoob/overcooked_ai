"""
Bayesian Behavior Cloning using Pyro.

This module implements a Bayesian neural network for behavior cloning.
It maintains uncertainty over weights to quantify confidence, detect
out-of-distribution states, and regularize learning.

Key References:
- Blundell et al., 2015. Weight Uncertainty in Neural Networks.
- Graves, 2011. Practical Variational Inference for Neural Networks.
"""

from __future__ import annotations

import os
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import pyro
import pyro.distributions as dist
from pyro.infer import Predictive, SVI, Trace_ELBO
from pyro.infer.autoguide import AutoDiagonalNormal, AutoLowRankMultivariateNormal
from pyro.nn import PyroModule, PyroSample
from pyro.optim import ClippedAdam

from overcooked_ai_py.agents.agent import Agent
from overcooked_ai_py.agents.benchmarking import AgentEvaluator
from overcooked_ai_py.mdp.actions import Action
from overcooked_ai_py.mdp.overcooked_env import DEFAULT_ENV_PARAMS

from human_aware_rl.data_dir import DATA_DIR
from human_aware_rl.human.process_dataframes import get_human_human_trajectories
from human_aware_rl.static import CLEAN_2019_HUMAN_DATA_TRAIN

# Default results directory scoped to Overcooked_ProbMods
ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS_DIR = ROOT_DIR / "results"


@dataclass
class BayesianBCConfig:
    """Configuration for Bayesian BC training."""

    # Environment
    layout_name: str = "cramped_room"

    # Data
    data_path: str = CLEAN_2019_HUMAN_DATA_TRAIN

    # Model architecture
    hidden_dims: Tuple[int, ...] = (64, 64)

    # Prior specification (controls regularization)
    prior_scale: float = 1.0  # Scale of weight prior (larger = more uncertainty)

    # Training
    num_epochs: int = 100
    batch_size: int = 64
    learning_rate: float = 1e-3
    num_particles: int = 1  # Number of samples for ELBO estimation

    # Inference
    guide_type: str = "diagonal"  # "diagonal" or "lowrank"

    # Prediction
    num_posterior_samples: int = 100  # Samples for uncertainty estimation

    # Output
    results_dir: str = str(DEFAULT_RESULTS_DIR)
    seed: int = 0
    verbose: bool = True


class BayesianBCModel(PyroModule):
    """Bayesian neural network for Behavior Cloning."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: Tuple[int, ...] = (64, 64),
        prior_scale: float = 1.0,
    ):
        super().__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dims = hidden_dims
        self.prior_scale = prior_scale

        # Store layer dimensions for forward pass
        self.dims = [state_dim] + list(hidden_dims) + [action_dim]
        self.num_layers = len(self.dims) - 1
        
        # Create standard linear layers (not PyroModule) - we'll sample weights manually
        self.layers = nn.ModuleList()
        for i in range(self.num_layers):
            self.layers.append(nn.Linear(self.dims[i], self.dims[i + 1]))

    def forward(self, x: torch.Tensor, y: torch.Tensor | None = None) -> torch.Tensor:
        device = x.device
        
        # Sample weights from priors and apply layers
        for i, layer in enumerate(self.layers[:-1]):
            # Sample weight and bias for this layer
            w = pyro.sample(
                f"layer_{i}_weight",
                dist.Normal(
                    torch.zeros(self.dims[i + 1], self.dims[i], device=device),
                    torch.ones(self.dims[i + 1], self.dims[i], device=device) * self.prior_scale
                ).to_event(2)
            )
            b = pyro.sample(
                f"layer_{i}_bias",
                dist.Normal(
                    torch.zeros(self.dims[i + 1], device=device),
                    torch.ones(self.dims[i + 1], device=device) * self.prior_scale
                ).to_event(1)
            )
            x = F.relu(F.linear(x, w, b))
        
        # Output layer
        i = self.num_layers - 1
        w = pyro.sample(
            f"layer_{i}_weight",
            dist.Normal(
                torch.zeros(self.dims[i + 1], self.dims[i], device=device),
                torch.ones(self.dims[i + 1], self.dims[i], device=device) * self.prior_scale
            ).to_event(2)
        )
        b = pyro.sample(
            f"layer_{i}_bias",
            dist.Normal(
                torch.zeros(self.dims[i + 1], device=device),
                torch.ones(self.dims[i + 1], device=device) * self.prior_scale
            ).to_event(1)
        )
        logits = F.linear(x, w, b)

        if y is not None:
            with pyro.plate("data", x.shape[0]):
                pyro.sample("obs", dist.Categorical(logits=logits), obs=y)
        return logits

    def predict_with_uncertainty(
        self,
        x: torch.Tensor,
        guide,
        num_samples: int = 100,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        predictive = Predictive(self, guide=guide, num_samples=num_samples)
        with torch.no_grad():
            _ = predictive(x)

        probs_samples = []
        for _ in range(num_samples):
            guide()
            logits = self(x)
            probs = F.softmax(logits, dim=-1)
            probs_samples.append(probs.detach().cpu().numpy())

        probs_samples = np.stack(probs_samples, axis=0)
        mean_probs = np.mean(probs_samples, axis=0)
        std_probs = np.std(probs_samples, axis=0)
        entropy = -np.sum(mean_probs * np.log(mean_probs + 1e-8), axis=-1)
        return mean_probs, std_probs, entropy


class BayesianBCTrainer:
    """Trainer for Bayesian BC models using SVI."""

    def __init__(self, config: BayesianBCConfig):
        self.config = config
        self.device = self._get_device()

        pyro.set_rng_seed(config.seed)
        torch.manual_seed(config.seed)

        self._setup_environment()
        self._load_data()
        self._setup_model()

        if config.verbose:
            print("Bayesian BC Trainer initialized")
            print(f"  Device: {self.device}")
            print(f"  State dim: {self.state_dim}")
            print(f"  Action dim: {self.action_dim}")
            print(f"  Training samples: {len(self.train_states)}")

    @staticmethod
    def _get_device() -> str:
        """Get best available device (CUDA > MPS > CPU)."""
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def _setup_environment(self):
        mdp_params = {"layout_name": self.config.layout_name, "old_dynamics": True}
        self.agent_evaluator = AgentEvaluator.from_layout_name(
            mdp_params=mdp_params,
            env_params=DEFAULT_ENV_PARAMS,
        )
        self.base_env = self.agent_evaluator.env
        self.mdp = self.base_env.mdp
        dummy_state = self.mdp.get_standard_start_state()
        obs_shape = self.base_env.featurize_state_mdp(dummy_state)[0].shape
        self.state_dim = int(np.prod(obs_shape))
        self.action_dim = len(Action.ALL_ACTIONS)

    def _load_data(self):
        data_params = {
            "layouts": [self.config.layout_name],
            "check_trajectories": False,
            "featurize_states": True,
            "data_path": self.config.data_path,
        }
        if self.config.verbose:
            print(f"Loading data for {self.config.layout_name}...")

        processed_trajs = get_human_human_trajectories(
            **data_params, silent=not self.config.verbose
        )

        states, actions = [], []
        for ep_states, ep_actions in zip(
            processed_trajs["ep_states"], processed_trajs["ep_actions"]
        ):
            for s, a in zip(ep_states, ep_actions):
                states.append(s.flatten())
                actions.append(int(a))

        self.train_states = torch.tensor(
            np.array(states), dtype=torch.float32, device=self.device
        )
        self.train_actions = torch.tensor(actions, dtype=torch.long, device=self.device)

        if self.config.verbose:
            print(f"Loaded {len(self.train_states)} transitions")

    def _setup_model(self):
        pyro.clear_param_store()
        self.model = BayesianBCModel(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            hidden_dims=self.config.hidden_dims,
            prior_scale=self.config.prior_scale,
        ).to(self.device)

        if self.config.guide_type == "diagonal":
            self.guide = AutoDiagonalNormal(self.model)
        elif self.config.guide_type == "lowrank":
            self.guide = AutoLowRankMultivariateNormal(self.model, rank=10)
        else:
            raise ValueError(f"Unknown guide type: {self.config.guide_type}")

        # Initialize guide by running a forward pass (this will create params on correct device)
        with torch.no_grad():
            sample_x = self.train_states[:2]
            sample_y = self.train_actions[:2]
            self.guide(sample_x, sample_y)

        self.optimizer = ClippedAdam({"lr": self.config.learning_rate})
        self.svi = SVI(
            self.model,
            self.guide,
            self.optimizer,
            loss=Trace_ELBO(num_particles=self.config.num_particles),
        )

    def train(self) -> Dict[str, Any]:
        if self.config.verbose:
            print(f"\n{'='*60}")
            print("Training Bayesian BC")
            print(f"{'='*60}")

        num_samples = len(self.train_states)
        batch_size = self.config.batch_size
        num_batches = (num_samples + batch_size - 1) // batch_size

        losses = []
        for epoch in range(self.config.num_epochs):
            epoch_loss = 0.0
            perm = torch.randperm(num_samples)
            for batch_idx in range(num_batches):
                start = batch_idx * batch_size
                end = min(start + batch_size, num_samples)
                idx = perm[start:end]
                batch_states = self.train_states[idx]
                batch_actions = self.train_actions[idx]
                loss = self.svi.step(batch_states, batch_actions)
                epoch_loss += loss

            avg_loss = epoch_loss / num_batches
            losses.append(avg_loss)
            if self.config.verbose and (epoch + 1) % 10 == 0:
                acc = self._compute_accuracy()
                print(
                    f"Epoch {epoch + 1}/{self.config.num_epochs} | Loss: {avg_loss:.4f} | Acc: {acc:.3f}"
                )

        self._save()
        return {"losses": losses}

    def _compute_accuracy(self) -> float:
        self.model.eval()
        with torch.no_grad():
            self.guide()
            logits = self.model(self.train_states)
            preds = torch.argmax(logits, dim=-1)
            acc = (preds == self.train_actions).float().mean().item()
        return acc

    def _save(self):
        save_dir = Path(self.config.results_dir) / "bayesian_bc" / self.config.layout_name
        save_dir.mkdir(parents=True, exist_ok=True)
        pyro.get_param_store().save(save_dir / "params.pt")
        config_dict = {
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "hidden_dims": self.config.hidden_dims,
            "prior_scale": self.config.prior_scale,
            "guide_type": self.config.guide_type,
            "layout_name": self.config.layout_name,
        }
        with open(save_dir / "config.pkl", "wb") as f:
            pickle.dump(config_dict, f)
        if self.config.verbose:
            print(f"Saved model to {save_dir}")


class BayesianBCAgent(Agent):
    """Agent that uses a trained Bayesian BC model and exposes uncertainty."""

    def __init__(
        self,
        model: BayesianBCModel,
        guide,
        featurize_fn,
        agent_index: int = 0,
        stochastic: bool = True,
        num_posterior_samples: int = 10,
        device: str | None = None,
    ):
        super().__init__()
        self.model = model
        self.guide = guide
        self.featurize_fn = featurize_fn
        self.agent_index = agent_index
        self.stochastic = stochastic
        self.num_posterior_samples = num_posterior_samples
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        self.model.eval()

    def action(self, state) -> Tuple[Any, Dict]:
        obs = self.featurize_fn(state)
        my_obs = obs[self.agent_index]
        obs_tensor = torch.tensor(my_obs.flatten(), dtype=torch.float32, device=self.device).unsqueeze(0)

        probs_samples = []
        with torch.no_grad():
            for _ in range(self.num_posterior_samples):
                self.guide()
                logits = self.model(obs_tensor)
                probs = F.softmax(logits, dim=-1)
                probs_samples.append(probs.cpu().numpy())

        probs_samples = np.stack(probs_samples, axis=0)
        mean_probs = np.mean(probs_samples, axis=0).squeeze()
        std_probs = np.std(probs_samples, axis=0).squeeze()
        entropy = -np.sum(mean_probs * np.log(mean_probs + 1e-8))

        if self.stochastic:
            action_idx = np.random.choice(len(mean_probs), p=mean_probs)
        else:
            action_idx = int(np.argmax(mean_probs))

        action = Action.INDEX_TO_ACTION[action_idx]
        return action, {
            "action_probs": mean_probs,
            "action_std": std_probs,
            "entropy": entropy,
        }

    def reset(self):
        pass

    @classmethod
    def from_saved(
        cls,
        model_dir: str,
        featurize_fn,
        agent_index: int = 0,
        **kwargs,
    ) -> "BayesianBCAgent":
        with open(os.path.join(model_dir, "config.pkl"), "rb") as f:
            config = pickle.load(f)

        model = BayesianBCModel(
            state_dim=config["state_dim"],
            action_dim=config["action_dim"],
            hidden_dims=config["hidden_dims"],
            prior_scale=config["prior_scale"],
        )

        if config["guide_type"] == "diagonal":
            guide = AutoDiagonalNormal(model)
        else:
            guide = AutoLowRankMultivariateNormal(model, rank=10)

        pyro.clear_param_store()
        pyro.get_param_store().load(os.path.join(model_dir, "params.pt"))
        return cls(model, guide, featurize_fn, agent_index, **kwargs)


def train_bayesian_bc(layout: str, verbose: bool = True, **kwargs) -> Dict[str, Any]:
    config = BayesianBCConfig(layout_name=layout, verbose=verbose, **kwargs)
    trainer = BayesianBCTrainer(config)
    return trainer.train()


def load_bayesian_bc(model_dir: str, device: str | None = None):
    with open(os.path.join(model_dir, "config.pkl"), "rb") as f:
        config = pickle.load(f)

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = BayesianBCModel(
        state_dim=config["state_dim"],
        action_dim=config["action_dim"],
        hidden_dims=config["hidden_dims"],
        prior_scale=config["prior_scale"],
    ).to(device)

    if config["guide_type"] == "diagonal":
        guide = AutoDiagonalNormal(model)
    else:
        guide = AutoLowRankMultivariateNormal(model, rank=10)

    pyro.clear_param_store()
    # Use weights_only=False for Pyro compatibility with PyTorch 2.6+
    params_path = os.path.join(model_dir, "params.pt")
    state = torch.load(params_path, map_location=device, weights_only=False)
    pyro.get_param_store().set_state(state)
    return model, guide, config


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train Bayesian BC models")
    parser.add_argument("--layout", default="cramped_room")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--prior_scale", type=float, default=1.0)
    parser.add_argument("--all_layouts", action="store_true")
    parser.add_argument("--results_dir", type=str, default=str(DEFAULT_RESULTS_DIR))
    args = parser.parse_args()

    if args.all_layouts:
        layouts = [
            "cramped_room",
            "asymmetric_advantages",
            "coordination_ring",
            "forced_coordination",
            "counter_circuit",
        ]
        for layout in layouts:
            print(f"\n{'='*60}")
            print(f"Training Bayesian BC for {layout}")
            train_bayesian_bc(
                layout,
                num_epochs=args.epochs,
                prior_scale=args.prior_scale,
                results_dir=args.results_dir,
            )
    else:
        train_bayesian_bc(
            args.layout,
            num_epochs=args.epochs,
            prior_scale=args.prior_scale,
            results_dir=args.results_dir,
        )
