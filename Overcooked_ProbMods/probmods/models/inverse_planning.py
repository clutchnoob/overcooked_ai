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
from typing import Any, Dict, Iterable, Optional, Tuple, TYPE_CHECKING

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
        self.device = self._get_device()
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

    @staticmethod
    def _get_device() -> str:
        """Get best available device (CUDA > MPS > CPU)."""
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

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


class InversePlanningAgent:
    """
    Agent that uses a trained Inverse Planning model to select actions.
    
    Uses the inferred reward weights (θ) and rationality (β) to compute
    action probabilities via a softmax-rational decision rule:
        P(a | s) ∝ exp(β × θ_a · φ(s))
    
    This allows the inverse planning model to be evaluated in gameplay
    scenarios (e.g., paired with a Human Proxy).
    """

    def __init__(
        self,
        model: LinearInversePlanningModel,
        guide,
        featurize_fn,
        agent_index: int = 0,
        stochastic: bool = True,
        use_posterior_mean: bool = True,
        num_posterior_samples: int = 10,
        beta_override: Optional[float] = None,
        device: Optional[str] = None,
    ):
        # Initialize agent_index for compatibility with AgentPair
        self.agent_index = agent_index
        """
        Initialize the Inverse Planning Agent.
        
        Args:
            model: Trained LinearInversePlanningModel
            guide: Pyro guide containing posterior parameters
            featurize_fn: Function to convert game state to feature vector
            agent_index: Which player this agent controls (0 or 1)
            stochastic: Whether to sample actions or take argmax
            use_posterior_mean: If True, use posterior mean of θ and β.
                               If False, sample from posterior each step.
            num_posterior_samples: Number of samples if not using posterior mean
            beta_override: If set, use this β instead of inferred value
            device: Device for computation (cuda/cpu)
        """
        self.model = model
        self.guide = guide
        self.featurize_fn = featurize_fn
        self.agent_index = agent_index
        self.stochastic = stochastic
        self.use_posterior_mean = use_posterior_mean
        self.num_posterior_samples = num_posterior_samples
        self.beta_override = beta_override
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        self.model = self.model.to(self.device)
        
        # Extract posterior mean parameters
        self._extract_posterior_params()
    
    def _extract_posterior_params(self):
        """Extract θ and β posterior means from the guide."""
        # Create dummy input to sample from guide
        # The guide needs states input to properly initialize/sample
        dummy_states = torch.zeros(1, self.model.state_dim, device=self.device)
        
        # Sample from guide to populate model parameters
        with torch.no_grad():
            # Call guide with dummy states (actions=None since we're not conditioning)
            self.guide(dummy_states, None)
            self.theta_mean = self.model.theta.detach().clone()  # (A, F)
            self.beta_mean = self.model.beta.detach().clone()    # scalar
    
    def action(self, state) -> Tuple[Any, Dict[str, Any]]:
        """
        Select an action for the current state.
        
        Args:
            state: Overcooked game state
            
        Returns:
            Tuple of (action, info_dict)
        """
        from overcooked_ai_py.mdp.actions import Action
        
        # Get observation for this agent
        obs = self.featurize_fn(state)
        my_obs = obs[self.agent_index]
        obs_flat = my_obs.flatten()
        obs_tensor = torch.tensor(obs_flat, dtype=torch.float32, device=self.device).unsqueeze(0)
        
        with torch.no_grad():
            if self.use_posterior_mean:
                # Use posterior mean
                theta = self.theta_mean  # (A, F)
                beta = self.beta_override if self.beta_override is not None else self.beta_mean
            else:
                # Sample from posterior
                self.guide(obs_tensor, None)
                theta = self.model.theta  # (A, F)
                beta = self.beta_override if self.beta_override is not None else self.model.beta
            
            # Compute action logits: β × (θ @ s^T)
            # obs_tensor: (1, F), theta: (A, F)
            # logits: (1, A) = β × (obs_tensor @ theta^T)
            logits = beta * (obs_tensor @ theta.T)  # (1, A)
            probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()  # (A,)
        
        # Select action
        if self.stochastic:
            action_idx = np.random.choice(len(probs), p=probs)
        else:
            action_idx = int(np.argmax(probs))
        
        action = Action.INDEX_TO_ACTION[action_idx]
        
        info = {
            "action_probs": probs,
            "beta": float(beta) if isinstance(beta, (int, float)) else float(beta.item()),
            "logits": logits.squeeze(0).cpu().numpy(),
        }
        
        return action, info
    
    def reset(self):
        """Reset agent state (no-op for this agent)."""
        pass
    
    def set_mdp(self, mdp):
        """Set the MDP (required by AgentEvaluator)."""
        self.mdp = mdp
    
    def set_agent_index(self, index: int):
        """Set which player this agent controls."""
        self.agent_index = index
    
    @classmethod
    def from_saved(
        cls,
        model_dir: str,
        featurize_fn,
        agent_index: int = 0,
        device: Optional[str] = None,
        **kwargs,
    ) -> "InversePlanningAgent":
        """
        Load an InversePlanningAgent from a saved model directory.
        
        Args:
            model_dir: Path to saved model (contains params.pt and config.pkl)
            featurize_fn: Function to convert game state to features
            agent_index: Which player this agent controls
            device: Device for computation
            **kwargs: Additional arguments passed to agent constructor
            
        Returns:
            Initialized InversePlanningAgent
        """
        model, guide, config = load_inverse_planning(model_dir, device)
        return cls(
            model=model,
            guide=guide,
            featurize_fn=featurize_fn,
            agent_index=agent_index,
            device=device,
            **kwargs,
        )


def create_inverse_planning_agent(
    layout: str,
    tag: str = "human_demo",
    results_dir: Optional[str] = None,
    agent_index: int = 0,
    stochastic: bool = True,
    beta_override: Optional[float] = None,
    device: Optional[str] = None,
) -> InversePlanningAgent:
    """
    Convenience function to create an InversePlanningAgent for a given layout.
    
    Args:
        layout: Layout name (e.g., "cramped_room")
        tag: Model tag (e.g., "human_demo", "ppo_bc", "ppo_gail")
        results_dir: Directory containing trained models
        agent_index: Which player this agent controls
        stochastic: Whether to sample actions
        beta_override: Override the inferred β value
        device: Device for computation
        
    Returns:
        Initialized InversePlanningAgent ready to play
    """
    from overcooked_ai_py.agents.benchmarking import AgentEvaluator
    from overcooked_ai_py.mdp.overcooked_env import DEFAULT_ENV_PARAMS
    
    results_dir = results_dir or str(DEFAULT_RESULTS_DIR)
    model_dir = os.path.join(results_dir, "inverse_planning", layout, tag)
    
    if not os.path.exists(model_dir):
        raise FileNotFoundError(f"Model not found at {model_dir}")
    
    # Create featurize function for this layout
    ae = AgentEvaluator.from_layout_name(
        mdp_params={"layout_name": layout, "old_dynamics": True},
        env_params=DEFAULT_ENV_PARAMS,
    )
    
    def featurize_fn(state):
        return ae.env.featurize_state_mdp(state)
    
    return InversePlanningAgent.from_saved(
        model_dir=model_dir,
        featurize_fn=featurize_fn,
        agent_index=agent_index,
        stochastic=stochastic,
        beta_override=beta_override,
        device=device,
    )


__all__ = [
    "InversePlanningConfig",
    "LinearInversePlanningModel",
    "InversePlanningTrainer",
    "load_inverse_planning",
    "InversePlanningAgent",
    "create_inverse_planning_agent",
]
