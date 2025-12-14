"""
Hierarchical Behavior Cloning with Goal Inference.

Two-level model:
1) Infer latent goal/intention given state.
2) Select action given state and inferred goal.

Provides interpretable subgoal distributions and goal-conditioned policies.
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
from pyro.infer import SVI, TraceEnum_ELBO, config_enumerate
from pyro.optim import ClippedAdam
from pyro.nn import PyroModule

from overcooked_ai_py.agents.agent import Agent
from overcooked_ai_py.agents.benchmarking import AgentEvaluator
from overcooked_ai_py.mdp.actions import Action
from overcooked_ai_py.mdp.overcooked_env import DEFAULT_ENV_PARAMS

from human_aware_rl.human.process_dataframes import get_human_human_trajectories
from human_aware_rl.static import CLEAN_2019_HUMAN_DATA_TRAIN

ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS_DIR = ROOT_DIR / "results"


@dataclass
class HierarchicalBCConfig:
    layout_name: str = "cramped_room"
    data_path: str = CLEAN_2019_HUMAN_DATA_TRAIN
    num_goals: int = 8
    goal_hidden_dims: Tuple[int, ...] = (64,)
    policy_hidden_dims: Tuple[int, ...] = (64,)
    goal_prior_alpha: float = 1.0
    num_epochs: int = 100
    batch_size: int = 64
    learning_rate: float = 1e-3
    results_dir: str = str(DEFAULT_RESULTS_DIR)
    seed: int = 0
    verbose: bool = True


class GoalInferenceNetwork(nn.Module):
    def __init__(self, state_dim: int, num_goals: int, hidden_dims: Tuple[int, ...]):
        super().__init__()
        dims = [state_dim] + list(hidden_dims) + [num_goals]
        layers: list[nn.Module] = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.ReLU())
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class GoalConditionedPolicy(nn.Module):
    def __init__(self, state_dim: int, num_goals: int, action_dim: int, hidden_dims: Tuple[int, ...]):
        super().__init__()
        self.goal_embedding = nn.Embedding(num_goals, hidden_dims[0] if hidden_dims else 32)
        input_dim = state_dim + (hidden_dims[0] if hidden_dims else 32)
        dims = [input_dim] + list(hidden_dims) + [action_dim]
        layers: list[nn.Module] = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.ReLU())
        self.network = nn.Sequential(*layers)

    def forward(self, state: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        goal_emb = self.goal_embedding(goal)
        x = torch.cat([state, goal_emb], dim=-1)
        return self.network(x)


class HierarchicalBCModel(PyroModule):
    def __init__(
        self,
        state_dim: int,
        num_goals: int,
        action_dim: int,
        goal_hidden_dims: Tuple[int, ...] = (64,),
        policy_hidden_dims: Tuple[int, ...] = (64,),
        goal_prior_alpha: float = 1.0,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.num_goals = num_goals
        self.action_dim = action_dim
        self.goal_prior_alpha = goal_prior_alpha
        self.goal_inference = GoalInferenceNetwork(state_dim, num_goals, goal_hidden_dims)
        self.policy = GoalConditionedPolicy(state_dim, num_goals, action_dim, policy_hidden_dims)

    @config_enumerate
    def model(self, states: torch.Tensor, actions: torch.Tensor | None = None):
        batch_size = states.shape[0]
        goal_prior = torch.ones(self.num_goals, device=states.device) * self.goal_prior_alpha
        with pyro.plate("data", batch_size):
            goal = pyro.sample("goal", dist.Categorical(logits=goal_prior.log()))
            action_logits = self.policy(states, goal)
            if actions is not None:
                pyro.sample("action", dist.Categorical(logits=action_logits), obs=actions)
        return goal

    def guide(self, states: torch.Tensor, actions: torch.Tensor | None = None):
        batch_size = states.shape[0]
        goal_logits = self.goal_inference(states)
        with pyro.plate("data", batch_size):
            pyro.sample("goal", dist.Categorical(logits=goal_logits))

    def forward(self, states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        goal_logits = self.goal_inference(states)
        goal_probs = F.softmax(goal_logits, dim=-1)
        batch_size = states.shape[0]
        action_probs = torch.zeros(batch_size, self.action_dim, device=states.device)
        for g in range(self.num_goals):
            goal_tensor = torch.full((batch_size,), g, dtype=torch.long, device=states.device)
            action_logits_g = self.policy(states, goal_tensor)
            action_probs_g = F.softmax(action_logits_g, dim=-1)
            action_probs += goal_probs[:, g:g+1] * action_probs_g
        return goal_probs, action_probs


class HierarchicalBCTrainer:
    def __init__(self, config: HierarchicalBCConfig):
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        pyro.set_rng_seed(config.seed)
        torch.manual_seed(config.seed)
        self._setup_environment()
        self._load_data()
        self._setup_model()
        if config.verbose:
            print("Hierarchical BC Trainer initialized")
            print(f"  Device: {self.device}")
            print(f"  Num goals: {config.num_goals}")

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
        processed_trajs = get_human_human_trajectories(**data_params, silent=True)
        states, actions = [], []
        for ep_states, ep_actions in zip(processed_trajs["ep_states"], processed_trajs["ep_actions"]):
            for s, a in zip(ep_states, ep_actions):
                states.append(s.flatten())
                actions.append(int(a))
        self.train_states = torch.tensor(np.array(states), dtype=torch.float32, device=self.device)
        self.train_actions = torch.tensor(actions, dtype=torch.long, device=self.device)
        if self.config.verbose:
            print(f"Loaded {len(self.train_states)} transitions")

    def _setup_model(self):
        pyro.clear_param_store()
        self.model = HierarchicalBCModel(
            state_dim=self.state_dim,
            num_goals=self.config.num_goals,
            action_dim=self.action_dim,
            goal_hidden_dims=self.config.goal_hidden_dims,
            policy_hidden_dims=self.config.policy_hidden_dims,
            goal_prior_alpha=self.config.goal_prior_alpha,
        ).to(self.device)
        self.optimizer = ClippedAdam({"lr": self.config.learning_rate})
        self.svi = SVI(
            self.model.model,
            self.model.guide,
            self.optimizer,
            loss=TraceEnum_ELBO(max_plate_nesting=1),
        )

    def train(self) -> Dict[str, Any]:
        if self.config.verbose:
            print(f"\n{'='*60}")
            print("Training Hierarchical BC")
            print(f"{'='*60}")
        num_samples = len(self.train_states)
        batch_size = self.config.batch_size
        num_batches = (num_samples + batch_size - 1) // batch_size
        losses: list[float] = []
        for epoch in range(self.config.num_epochs):
            epoch_loss = 0.0
            perm = torch.randperm(num_samples)
            for batch_idx in range(num_batches):
                start = batch_idx * batch_size
                end = min(start + batch_size, num_samples)
                idx = perm[start:end]
                loss = self.svi.step(self.train_states[idx], self.train_actions[idx])
                epoch_loss += loss
            avg_loss = epoch_loss / num_batches
            losses.append(avg_loss)
            if self.config.verbose and (epoch + 1) % 10 == 0:
                acc = self._compute_accuracy()
                goal_dist = self._get_goal_distribution()
                print(
                    f"Epoch {epoch + 1}/{self.config.num_epochs} | Loss: {avg_loss:.4f} | Acc: {acc:.3f}"
                )
                print(f"  Goal distribution: {goal_dist}")
        self._save()
        return {"losses": losses}

    def _compute_accuracy(self) -> float:
        with torch.no_grad():
            _, action_probs = self.model(self.train_states)
            preds = torch.argmax(action_probs, dim=-1)
            return (preds == self.train_actions).float().mean().item()

    def _get_goal_distribution(self) -> str:
        with torch.no_grad():
            goal_probs, _ = self.model(self.train_states)
            avg_probs = goal_probs.mean(dim=0).cpu().numpy()
            return "[" + ", ".join([f"{p:.2f}" for p in avg_probs]) + "]"

    def _save(self):
        save_dir = Path(self.config.results_dir) / "hierarchical_bc" / self.config.layout_name
        save_dir.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), save_dir / "model.pt")
        config_dict = {
            "state_dim": self.state_dim,
            "num_goals": self.config.num_goals,
            "action_dim": self.action_dim,
            "goal_hidden_dims": self.config.goal_hidden_dims,
            "policy_hidden_dims": self.config.policy_hidden_dims,
            "layout_name": self.config.layout_name,
        }
        with open(save_dir / "config.pkl", "wb") as f:
            pickle.dump(config_dict, f)
        if self.config.verbose:
            print(f"Saved to {save_dir}")


class HierarchicalBCAgent(Agent):
    def __init__(
        self,
        model: HierarchicalBCModel,
        featurize_fn,
        agent_index: int = 0,
        stochastic: bool = True,
        device: str | None = None,
    ):
        super().__init__()
        self.model = model
        self.featurize_fn = featurize_fn
        self.agent_index = agent_index
        self.stochastic = stochastic
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        self.model.eval()

    def action(self, state) -> Tuple[Any, Dict]:
        obs = self.featurize_fn(state)
        my_obs = obs[self.agent_index]
        obs_tensor = torch.tensor(my_obs.flatten(), dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            goal_probs, action_probs = self.model(obs_tensor)
            goal_probs = goal_probs.squeeze().cpu().numpy()
            action_probs = action_probs.squeeze().cpu().numpy()
        inferred_goal = int(np.argmax(goal_probs))
        if self.stochastic:
            action_idx = np.random.choice(len(action_probs), p=action_probs)
        else:
            action_idx = int(np.argmax(action_probs))
        action = Action.INDEX_TO_ACTION[action_idx]
        return action, {
            "action_probs": action_probs,
            "goal_probs": goal_probs,
            "inferred_goal": inferred_goal,
        }

    def reset(self):
        pass

    @classmethod
    def from_saved(cls, model_dir: str, featurize_fn, **kwargs) -> "HierarchicalBCAgent":
        with open(os.path.join(model_dir, "config.pkl"), "rb") as f:
            config = pickle.load(f)
        device = kwargs.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        model = HierarchicalBCModel(
            state_dim=config["state_dim"],
            num_goals=config["num_goals"],
            action_dim=config["action_dim"],
            goal_hidden_dims=config["goal_hidden_dims"],
            policy_hidden_dims=config["policy_hidden_dims"],
        ).to(device)
        model.load_state_dict(torch.load(os.path.join(model_dir, "model.pt"), map_location=device))
        return cls(model, featurize_fn, **kwargs)


def train_hierarchical_bc(layout: str, verbose: bool = True, **kwargs) -> Dict[str, Any]:
    config = HierarchicalBCConfig(layout_name=layout, verbose=verbose, **kwargs)
    trainer = HierarchicalBCTrainer(config)
    return trainer.train()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--layout", default="cramped_room")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--num_goals", type=int, default=8)
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
            print(f"Training Hierarchical BC for {layout}")
            train_hierarchical_bc(
                layout,
                num_epochs=args.epochs,
                num_goals=args.num_goals,
                results_dir=args.results_dir,
            )
    else:
        train_hierarchical_bc(
            args.layout,
            num_epochs=args.epochs,
            num_goals=args.num_goals,
            results_dir=args.results_dir,
        )
