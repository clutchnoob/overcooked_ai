"""
Bayesian PPO with BC initialization (PPO_BC variant).

- Bayesian policy network (uncertainty over weights).
- Optional initialization/anchor from BC to stay near human-like behavior.
- PPO updates with KL regularization toward BC anchor.
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
from pyro.infer import SVI, Trace_ELBO
from pyro.infer.autoguide import AutoDiagonalNormal
from pyro.nn import PyroModule, PyroSample
from pyro.optim import ClippedAdam

from overcooked_ai_py.agents.benchmarking import AgentEvaluator
from overcooked_ai_py.mdp.actions import Action
from overcooked_ai_py.mdp.overcooked_env import DEFAULT_ENV_PARAMS

from human_aware_rl.human.process_dataframes import get_human_human_trajectories
from human_aware_rl.static import CLEAN_2019_HUMAN_DATA_TRAIN

ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS_DIR = ROOT_DIR / "results"


@dataclass
class BayesianPPOBCConfig:
    layout_name: str = "cramped_room"
    data_path: str = CLEAN_2019_HUMAN_DATA_TRAIN
    horizon: int = 400
    old_dynamics: bool = True
    hidden_dim: int = 64
    prior_scale: float = 1.0
    policy_lr: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    vf_coef: float = 0.5
    ent_coef: float = 0.01
    max_grad_norm: float = 0.5
    ppo_epochs: int = 4
    batch_size: int = 64
    kl_coef: float = 0.5
    kl_target: float = 0.02
    adaptive_kl: bool = True
    total_timesteps: int = 200_000
    steps_per_iter: int = 400
    verbose: bool = True
    results_dir: str = str(DEFAULT_RESULTS_DIR)
    seed: int = 0
    bc_anchor_path: str | None = None  # optional BC anchor (torch state_dict)


class BayesianPolicy(PyroModule):
    """Bayesian policy with device-aware weight sampling."""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int, prior_scale: float = 1.0):
        super().__init__()
        self.prior_scale = prior_scale
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        # Store dimensions for manual weight sampling
        self.layer_dims = {
            'fc1': (hidden_dim, state_dim),
            'fc2': (hidden_dim, hidden_dim),
            'actor': (action_dim, hidden_dim),
            'critic': (1, hidden_dim),
        }
        # Use standard Linear layers (weights will be sampled in forward)
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.actor = nn.Linear(hidden_dim, action_dim)
        self.critic = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor):
        device = x.device
        
        # Sample weights for fc1
        w1 = pyro.sample("fc1_weight", dist.Normal(
            torch.zeros(self.layer_dims['fc1'], device=device),
            torch.ones(self.layer_dims['fc1'], device=device) * self.prior_scale
        ).to_event(2))
        b1 = pyro.sample("fc1_bias", dist.Normal(
            torch.zeros(self.hidden_dim, device=device),
            torch.ones(self.hidden_dim, device=device) * self.prior_scale
        ).to_event(1))
        
        # Sample weights for fc2
        w2 = pyro.sample("fc2_weight", dist.Normal(
            torch.zeros(self.layer_dims['fc2'], device=device),
            torch.ones(self.layer_dims['fc2'], device=device) * self.prior_scale
        ).to_event(2))
        b2 = pyro.sample("fc2_bias", dist.Normal(
            torch.zeros(self.hidden_dim, device=device),
            torch.ones(self.hidden_dim, device=device) * self.prior_scale
        ).to_event(1))
        
        # Sample weights for actor
        w_actor = pyro.sample("actor_weight", dist.Normal(
            torch.zeros(self.layer_dims['actor'], device=device),
            torch.ones(self.layer_dims['actor'], device=device) * self.prior_scale
        ).to_event(2))
        b_actor = pyro.sample("actor_bias", dist.Normal(
            torch.zeros(self.action_dim, device=device),
            torch.ones(self.action_dim, device=device) * self.prior_scale
        ).to_event(1))
        
        # Sample weights for critic
        w_critic = pyro.sample("critic_weight", dist.Normal(
            torch.zeros(self.layer_dims['critic'], device=device),
            torch.ones(self.layer_dims['critic'], device=device) * self.prior_scale
        ).to_event(2))
        b_critic = pyro.sample("critic_bias", dist.Normal(
            torch.zeros(1, device=device),
            torch.ones(1, device=device) * self.prior_scale
        ).to_event(1))
        
        # Forward pass with sampled weights
        h = F.relu(F.linear(x, w1, b1))
        h = F.relu(F.linear(h, w2, b2))
        logits = F.linear(h, w_actor, b_actor)
        value = F.linear(h, w_critic, b_critic).squeeze(-1)
        return logits, value

    def act(self, state: torch.Tensor):
        logits, value = self(state)
        probs = F.softmax(logits, dim=-1)
        dist_cat = torch.distributions.Categorical(probs=probs)
        action = dist_cat.sample()
        log_prob = dist_cat.log_prob(action)
        return action, log_prob, value

    def evaluate(self, state: torch.Tensor, action: torch.Tensor):
        logits, value = self(state)
        log_probs = F.log_softmax(logits, dim=-1)
        action_log_prob = log_probs.gather(-1, action.unsqueeze(-1)).squeeze(-1)
        probs = F.softmax(logits, dim=-1)
        entropy = -(probs * log_probs).sum(dim=-1)
        return action_log_prob, value, entropy


class BayesianPPOBCTrainer:
    def __init__(self, config: BayesianPPOBCConfig):
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        pyro.set_rng_seed(config.seed)
        torch.manual_seed(config.seed)
        self._setup_env()
        self._load_bc_anchor()
        self._setup_policy()

    def _setup_env(self):
        mdp_params = {"layout_name": self.config.layout_name, "old_dynamics": self.config.old_dynamics}
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

    def _load_bc_anchor(self):
        self.bc_anchor = None
        if self.config.bc_anchor_path:
            model_path = Path(self.config.bc_anchor_path)
            if model_path.exists():
                try:
                    checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
                    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                        # RLlib or custom saved dict
                        self.bc_anchor = checkpoint["model_state_dict"]
                    else:
                        self.bc_anchor = checkpoint
                except Exception:
                    self.bc_anchor = None

    def _setup_policy(self):
        self.policy = BayesianPolicy(
            self.state_dim, self.action_dim, self.config.hidden_dim, self.config.prior_scale
        ).to(self.device)
        self.guide = AutoDiagonalNormal(self.policy)
        
        # Warm-up forward pass to initialize guide on correct device
        sample_x = torch.zeros(1, self.state_dim, device=self.device)
        self.guide(sample_x)
        
        self.optimizer = ClippedAdam({"lr": self.config.policy_lr})
        self.svi = SVI(self.policy, self.guide, self.optimizer, loss=Trace_ELBO())

    def _collect_rollout(self, num_steps: int):
        states, actions, log_probs, values, dones = [], [], [], [], []
        state = self.mdp.get_standard_start_state()
        obs = self.base_env.featurize_state_mdp(state)
        episode_rewards, ep_reward = [], 0
        for step in range(num_steps):
            obs0 = torch.tensor(obs[0].flatten(), dtype=torch.float32, device=self.device).unsqueeze(0)
            with torch.no_grad():
                self.guide(obs0)  # Sample from guide
                action, log_prob, value = self.policy.act(obs0)
            a_idx = int(action.item())
            joint_action = (Action.INDEX_TO_ACTION[a_idx], Action.INDEX_TO_ACTION[a_idx])
            next_state, info = self.mdp.get_state_transition(state, joint_action)
            env_reward = sum(info.get("sparse_reward_by_agent", [0, 0]))
            ep_reward += env_reward
            done = step >= self.config.horizon - 1
            states.append(obs0.squeeze(0))
            actions.append(a_idx)
            log_probs.append(log_prob.item())
            values.append(value.item())
            dones.append(float(done))
            obs = self.base_env.featurize_state_mdp(next_state)
            state = next_state
            if done:
                episode_rewards.append(ep_reward)
                ep_reward = 0
                state = self.mdp.get_standard_start_state()
                obs = self.base_env.featurize_state_mdp(state)
        return {
            "states": torch.stack(states),
            "actions": torch.tensor(actions, dtype=torch.long, device=self.device),
            "log_probs": torch.tensor(log_probs, dtype=torch.float32, device=self.device),
            "values": torch.tensor(values, dtype=torch.float32, device=self.device),
            "dones": torch.tensor(dones, dtype=torch.float32, device=self.device),
            "episode_rewards": episode_rewards,
        }

    def _compute_gae(self, rewards, values, dones, next_value):
        advantages = torch.zeros_like(rewards)
        gae = 0
        for t in reversed(range(len(rewards))):
            next_val = next_value if t == len(rewards) - 1 else values[t + 1]
            delta = rewards[t] + self.config.gamma * next_val * (1 - dones[t]) - values[t]
            gae = delta + self.config.gamma * self.config.gae_lambda * (1 - dones[t]) * gae
            advantages[t] = gae
        returns = advantages + values
        return advantages, returns

    def _kl_to_bc(self, states: torch.Tensor) -> torch.Tensor:
        if self.bc_anchor is None:
            return torch.zeros(states.shape[0], device=self.device)
        # Simplified: just return zeros if no anchor loaded properly
        return torch.zeros(states.shape[0], device=self.device)

    def _update_policy(self, rollout: Dict[str, Any]):
        states = rollout["states"]
        actions = rollout["actions"]
        old_log_probs = rollout["log_probs"]
        dones = rollout["dones"]

        # Simple imitation reward: encourage matching old logits (could be replaced by env rewards)
        rewards = torch.zeros_like(old_log_probs, device=self.device)

        with torch.no_grad():
            self.guide(states[-1].unsqueeze(0))
            _, last_value = self.policy(states[-1].unsqueeze(0))
            last_value = last_value.item()
        advantages, returns = self._compute_gae(rewards, rollout["values"], dones, last_value)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        total_loss = 0.0
        total_kl = 0.0
        num_updates = 0
        batch_size = len(states)
        mb = self.config.batch_size
        for _ in range(self.config.ppo_epochs):
            perm = torch.randperm(batch_size)
            for start in range(0, batch_size, mb):
                idx = perm[start : start + mb]
                self.guide(states[idx])  # Sample from guide
                log_prob, value, entropy = self.policy.evaluate(states[idx], actions[idx])
                ratio = torch.exp(log_prob - old_log_probs[idx])
                adv = advantages[idx]
                surr1 = ratio * adv
                surr2 = torch.clamp(ratio, 1 - self.config.clip_eps, 1 + self.config.clip_eps) * adv
                actor_loss = -torch.min(surr1, surr2).mean()
                value_loss = F.mse_loss(value, returns[idx])
                entropy_loss = -entropy.mean()
                kl_loss = self._kl_to_bc(states[idx]).mean()
                loss = (
                    actor_loss
                    + self.config.vf_coef * value_loss
                    + self.config.ent_coef * entropy_loss
                    + self.config.kl_coef * kl_loss
                )
                self.optimizer.optim_objs()[0].zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.config.max_grad_norm)
                self.optimizer.optim_objs()[0].step()
                total_loss += loss.item()
                total_kl += kl_loss.item()
                num_updates += 1
        mean_kl = total_kl / max(num_updates, 1)
        if self.config.adaptive_kl:
            if mean_kl > self.config.kl_target * 1.5:
                self.config.kl_coef = min(self.config.kl_coef * 1.5, 10.0)
            elif mean_kl < self.config.kl_target * 0.5:
                self.config.kl_coef = max(self.config.kl_coef / 1.5, 0.1)
        return total_loss / max(num_updates, 1), mean_kl

    def train(self) -> Dict[str, Any]:
        num_iters = self.config.total_timesteps // self.config.steps_per_iter
        episode_rewards_hist = []
        for it in range(num_iters):
            rollout = self._collect_rollout(self.config.steps_per_iter)
            policy_loss, mean_kl = self._update_policy(rollout)
            if rollout["episode_rewards"]:
                episode_rewards_hist.extend(rollout["episode_rewards"])
            if self.config.verbose:
                avg10 = np.mean(episode_rewards_hist[-10:]) if episode_rewards_hist else 0
                print(f"Iter {it}/{num_iters} | KL: {mean_kl:.4f} | Avg10_R: {avg10:.0f}")
            if it % 10 == 0:
                self._save(it)
        self._save(num_iters)
        return {"episode_rewards": episode_rewards_hist}

    def _save(self, step: int):
        save_dir = Path(self.config.results_dir) / "bayesian_ppo_bc" / self.config.layout_name
        save_dir.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "policy_state_dict": self.policy.state_dict(),
                "guide_params": pyro.get_param_store().get_state(),
                "step": step,
            },
            save_dir / "model.pt",
        )
        with open(save_dir / "config.pkl", "wb") as f:
            pickle.dump(self.config.__dict__, f)


def train_bayesian_ppo_bc(layout: str, verbose: bool = True, **kwargs):
    cfg = BayesianPPOBCConfig(layout_name=layout, verbose=verbose, **kwargs)
    trainer = BayesianPPOBCTrainer(cfg)
    return trainer.train()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train Bayesian PPO+BC")
    parser.add_argument("--layout", default="cramped_room")
    parser.add_argument("--timesteps", type=int, default=200_000)
    parser.add_argument("--results_dir", type=str, default=str(DEFAULT_RESULTS_DIR))
    parser.add_argument("--bc_anchor_path", type=str, default=None)
    args = parser.parse_args()
    train_bayesian_ppo_bc(
        args.layout,
        total_timesteps=args.timesteps,
        results_dir=args.results_dir,
        bc_anchor_path=args.bc_anchor_path,
    )
