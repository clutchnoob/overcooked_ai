"""
Adversarial Inverse Reinforcement Learning (AIRL) for Overcooked AI.

This module implements AIRL from the paper:
"Learning Robust Rewards with Adversarial Inverse Reinforcement Learning"
(Fu et al., 2018)

AIRL learns a disentangled reward function from demonstrations that can
produce more robust human proxy agents compared to Behavior Cloning.

Architecture:
    D(s, a, s') = exp(f(s, a, s')) / (exp(f(s, a, s')) + π(a|s))
    f(s, a, s') = g(s) + γ·h(s') - h(s)
    
Where:
    - g(s): Learned reward function (state-only for disentanglement)
    - h(s): Learned shaping potential (approximates value function)
    - π(a|s): Current policy
"""

import os
import pickle
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from human_aware_rl.data_dir import DATA_DIR
from human_aware_rl.human.process_dataframes import get_human_human_trajectories
from human_aware_rl.static import CLEAN_2019_HUMAN_DATA_TRAIN, CLEAN_2019_HUMAN_DATA_TEST
from overcooked_ai_py.mdp.actions import Action
from overcooked_ai_py.mdp.overcooked_env import DEFAULT_ENV_PARAMS
from overcooked_ai_py.agents.benchmarking import AgentEvaluator


#################
# Configuration #
#################

AIRL_SAVE_DIR = os.path.join(DATA_DIR, "airl_runs")

@dataclass
class AIRLConfig:
    """Configuration for AIRL training."""
    
    # Environment
    layout_name: str = "cramped_room"
    horizon: int = 400
    old_dynamics: bool = True
    
    # Data
    data_path: str = CLEAN_2019_HUMAN_DATA_TRAIN
    featurize_states: bool = True
    
    # Discriminator architecture
    disc_hidden_dim: int = 64
    disc_num_layers: int = 2
    disc_g_linear: bool = True  # Use linear g(s) for reward (paper recommendation)
    
    # Policy architecture (same as BC)
    policy_hidden_dim: int = 64
    policy_num_layers: int = 2
    use_lstm: bool = False
    cell_size: int = 256
    
    # Training hyperparameters (from AIRL paper Appendix D)
    discriminator_lr: float = 3e-4
    policy_lr: float = 3e-4
    gamma: float = 0.99
    batch_size: int = 256
    disc_updates_per_iter: int = 5  # Discriminator updates per policy update
    policy_epochs: int = 8  # PPO epochs for policy
    
    # PPO hyperparameters
    clip_eps: float = 0.2
    vf_coef: float = 0.5
    ent_coef: float = 0.1  # Higher entropy for exploration
    gae_lambda: float = 0.95
    max_grad_norm: float = 0.5
    
    # Sample mixing (prevents reward forgetting)
    sample_buffer_size: int = 20  # Keep samples from last N iterations
    
    # Training length
    total_timesteps: int = 5_000_000  # 5M timesteps
    steps_per_iter: int = 10000  # Steps before each update cycle
    
    # Logging and saving
    log_interval: int = 1
    save_interval: int = 10
    verbose: bool = True
    
    # Early stopping
    early_stop_patience: int = 50
    
    # Output
    results_dir: str = "results/airl"
    experiment_name: str = "airl_overcooked"
    seed: int = 0


##############
# Discriminator #
##############


class RewardNetwork(nn.Module):
    """
    Reward function g(s) - learns the disentangled reward from state only.
    Can be linear (as recommended in paper) or MLP.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        use_linear: bool = True,
    ):
        super().__init__()
        
        self.use_linear = use_linear
        
        if use_linear:
            # Linear g(s) as recommended in paper for disentanglement
            self.network = nn.Linear(input_dim, 1)
        else:
            # MLP for more expressive reward
            layers = []
            prev_dim = input_dim
            for _ in range(num_layers):
                layers.append(nn.Linear(prev_dim, hidden_dim))
                layers.append(nn.ReLU())
                prev_dim = hidden_dim
            layers.append(nn.Linear(prev_dim, 1))
            self.network = nn.Sequential(*layers)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Compute reward for states.
        
        Args:
            state: State tensor of shape (batch_size, state_dim)
            
        Returns:
            Reward tensor of shape (batch_size,)
        """
        return self.network(state).squeeze(-1)


class ShapingNetwork(nn.Module):
    """
    Shaping potential h(s) - approximates the value function.
    Used to compute the shaping term γ·h(s') - h(s).
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
    ):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        for _ in range(num_layers):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, 1))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Compute shaping potential for states.
        
        Args:
            state: State tensor of shape (batch_size, state_dim)
            
        Returns:
            Potential tensor of shape (batch_size,)
        """
        return self.network(state).squeeze(-1)


class AIRLDiscriminator(nn.Module):
    """
    AIRL Discriminator with disentangled reward structure.
    
    D(s, a, s') = exp(f(s, a, s')) / (exp(f(s, a, s')) + π(a|s))
    f(s, a, s') = g(s) + γ·h(s') - h(s)
    """
    
    def __init__(
        self,
        state_dim: int,
        gamma: float = 0.99,
        hidden_dim: int = 64,
        num_layers: int = 2,
        g_linear: bool = True,
    ):
        super().__init__()
        
        self.gamma = gamma
        
        # Reward function g(s)
        self.g_network = RewardNetwork(
            input_dim=state_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            use_linear=g_linear,
        )
        
        # Shaping potential h(s)
        self.h_network = ShapingNetwork(
            input_dim=state_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
        )
    
    def compute_f(
        self,
        state: torch.Tensor,
        next_state: torch.Tensor,
        done: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute f(s, a, s') = g(s) + γ·h(s') - h(s).
        
        Args:
            state: Current state tensor
            next_state: Next state tensor
            done: Done mask (1 if terminal, 0 otherwise)
            
        Returns:
            f values of shape (batch_size,)
        """
        g = self.g_network(state)
        h_s = self.h_network(state)
        h_s_prime = self.h_network(next_state)
        
        # Shaping: γ·h(s') - h(s), with h(s')=0 if terminal
        shaping = self.gamma * h_s_prime * (1 - done) - h_s
        
        return g + shaping
    
    def forward(
        self,
        state: torch.Tensor,
        next_state: torch.Tensor,
        done: torch.Tensor,
        log_pi: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute discriminator output D(s, a, s').
        
        Args:
            state: Current state tensor
            next_state: Next state tensor
            done: Done mask
            log_pi: Log probability of action under policy π(a|s)
            
        Returns:
            Discriminator output D in [0, 1]
        """
        f = self.compute_f(state, next_state, done)
        
        # D = exp(f) / (exp(f) + π(a|s))
        # Using log-sum-exp trick for numerical stability:
        # D = sigmoid(f - log_pi)
        return torch.sigmoid(f - log_pi)
    
    def get_reward(
        self,
        state: torch.Tensor,
        next_state: torch.Tensor,
        done: torch.Tensor,
        log_pi: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute AIRL reward for policy training.
        
        reward = log(D) - log(1-D) = f - log_pi
        
        This is equivalent to the advantage of the optimal policy.
        """
        f = self.compute_f(state, next_state, done)
        return f - log_pi
    
    def get_learned_reward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Get the learned reward g(s) for a state.
        This is the disentangled reward that transfers across dynamics.
        """
        return self.g_network(state)


##############
# Policy Network #
##############


class AIRLPolicy(nn.Module):
    """
    Policy network for AIRL.
    Maps states to action probabilities.
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
    ):
        super().__init__()
        
        self.action_dim = action_dim
        
        # Shared feature extractor
        layers = []
        prev_dim = state_dim
        for _ in range(num_layers):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        
        self.feature_extractor = nn.Sequential(*layers)
        
        # Actor head
        self.actor = nn.Linear(hidden_dim, action_dim)
        
        # Critic head (for PPO updates)
        self.critic = nn.Linear(hidden_dim, 1)
    
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through policy.
        
        Args:
            state: State tensor
            
        Returns:
            Tuple of (action_logits, value)
        """
        features = self.feature_extractor(state)
        logits = self.actor(features)
        value = self.critic(features).squeeze(-1)
        return logits, value
    
    def get_action(
        self,
        state: torch.Tensor,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample action from policy.
        
        Args:
            state: State tensor
            deterministic: If True, return argmax action
            
        Returns:
            Tuple of (action, log_prob, value)
        """
        logits, value = self.forward(state)
        probs = F.softmax(logits, dim=-1)
        
        if deterministic:
            action = torch.argmax(logits, dim=-1)
            log_prob = torch.log(probs[torch.arange(len(action)), action] + 1e-8)
        else:
            dist = torch.distributions.Categorical(probs=probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)
        
        return action, log_prob, value
    
    def evaluate_actions(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate actions under the policy.
        
        Args:
            state: State tensor
            action: Action tensor
            
        Returns:
            Tuple of (log_prob, value, entropy)
        """
        logits, value = self.forward(state)
        probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)
        
        # Log probability of taken action
        action_log_prob = log_probs[torch.arange(len(action)), action]
        
        # Entropy
        entropy = -(probs * log_probs).sum(dim=-1)
        
        return action_log_prob, value, entropy


class AIRLPolicyLSTM(nn.Module):
    """
    LSTM-based policy network for AIRL.
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        cell_size: int = 256,
    ):
        super().__init__()
        
        self.action_dim = action_dim
        self.cell_size = cell_size
        
        # Feature extractor before LSTM
        layers = []
        prev_dim = state_dim
        for _ in range(num_layers):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        
        self.feature_extractor = nn.Sequential(*layers)
        
        # LSTM
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=cell_size,
            batch_first=True,
        )
        
        # Actor and critic heads
        self.actor = nn.Linear(cell_size, action_dim)
        self.critic = nn.Linear(cell_size, 1)
    
    def forward(
        self,
        state: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass through LSTM policy.
        """
        # Feature extraction
        features = self.feature_extractor(state)
        
        # Add sequence dimension if needed
        if features.dim() == 2:
            features = features.unsqueeze(1)
        
        # LSTM
        if hidden is None:
            batch_size = features.size(0)
            hidden = self.get_initial_state(batch_size, features.device)
        
        lstm_out, new_hidden = self.lstm(features, hidden)
        lstm_out = lstm_out.squeeze(1)
        
        # Heads
        logits = self.actor(lstm_out)
        value = self.critic(lstm_out).squeeze(-1)
        
        return logits, value, new_hidden
    
    def get_initial_state(
        self,
        batch_size: int,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get initial LSTM hidden state."""
        return (
            torch.zeros(1, batch_size, self.cell_size, device=device),
            torch.zeros(1, batch_size, self.cell_size, device=device),
        )
    
    def get_action(
        self,
        state: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Sample action from LSTM policy."""
        logits, value, new_hidden = self.forward(state, hidden)
        probs = F.softmax(logits, dim=-1)
        
        if deterministic:
            action = torch.argmax(logits, dim=-1)
            log_prob = torch.log(probs[torch.arange(len(action)), action] + 1e-8)
        else:
            dist = torch.distributions.Categorical(probs=probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)
        
        return action, log_prob, value, new_hidden


##############
# AIRL Trainer #
##############


class AIRLTrainer:
    """
    AIRL Training loop.
    
    Alternates between:
    1. Discriminator update (distinguish expert vs policy)
    2. Policy update (PPO with AIRL reward)
    """
    
    def __init__(
        self,
        config: AIRLConfig,
        device: Optional[str] = None,
    ):
        self.config = config
        
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        # Setup environment
        self._setup_environment()
        
        # Load expert demonstrations
        self._load_expert_data()
        
        # Create networks
        self._create_networks()
        
        # Sample buffer for mixing
        self.policy_sample_buffer = deque(maxlen=config.sample_buffer_size)
        
        # Logging
        self.train_info = {
            "disc_losses": [],
            "policy_losses": [],
            "episode_rewards": [],
            "disc_accuracy": [],
        }
        
        # Create output directory
        os.makedirs(config.results_dir, exist_ok=True)
    
    def _setup_environment(self):
        """Setup the Overcooked environment."""
        mdp_params = {
            "layout_name": self.config.layout_name,
            "old_dynamics": self.config.old_dynamics,
        }
        
        self.agent_evaluator = AgentEvaluator.from_layout_name(
            mdp_params=mdp_params,
            env_params=DEFAULT_ENV_PARAMS,
        )
        self.base_env = self.agent_evaluator.env
        self.mdp = self.base_env.mdp
        
        # Get observation shape
        dummy_state = self.mdp.get_standard_start_state()
        self.obs_shape = self.base_env.featurize_state_mdp(dummy_state)[0].shape
        self.state_dim = int(np.prod(self.obs_shape))
        self.action_dim = len(Action.ALL_ACTIONS)
    
    def _load_expert_data(self):
        """Load and preprocess expert demonstrations."""
        data_params = {
            "layouts": [self.config.layout_name],
            "check_trajectories": False,
            "featurize_states": self.config.featurize_states,
            "data_path": self.config.data_path,
        }
        
        processed_trajs = get_human_human_trajectories(**data_params, silent=not self.config.verbose)
        
        # Flatten all episodes into (s, a, s', done) tuples
        expert_states = []
        expert_actions = []
        expert_next_states = []
        expert_dones = []
        
        ep_states = processed_trajs["ep_states"]
        ep_actions = processed_trajs["ep_actions"]
        
        for ep_idx in range(len(ep_states)):
            states = ep_states[ep_idx]
            actions = ep_actions[ep_idx]
            
            for t in range(len(states) - 1):
                expert_states.append(states[t].flatten())
                expert_actions.append(int(actions[t]))
                expert_next_states.append(states[t + 1].flatten())
                expert_dones.append(0.0)
            
            # Last transition (done=1)
            if len(states) > 0:
                expert_states.append(states[-1].flatten())
                expert_actions.append(int(actions[-1]) if len(actions) > 0 else 0)
                expert_next_states.append(states[-1].flatten())  # Terminal
                expert_dones.append(1.0)
        
        self.expert_states = torch.tensor(
            np.array(expert_states), dtype=torch.float32, device=self.device
        )
        self.expert_actions = torch.tensor(
            expert_actions, dtype=torch.long, device=self.device
        )
        self.expert_next_states = torch.tensor(
            np.array(expert_next_states), dtype=torch.float32, device=self.device
        )
        self.expert_dones = torch.tensor(
            expert_dones, dtype=torch.float32, device=self.device
        )
        
        if self.config.verbose:
            print(f"Loaded {len(self.expert_states)} expert transitions")
    
    def _create_networks(self):
        """Create discriminator and policy networks."""
        # Discriminator
        self.discriminator = AIRLDiscriminator(
            state_dim=self.state_dim,
            gamma=self.config.gamma,
            hidden_dim=self.config.disc_hidden_dim,
            num_layers=self.config.disc_num_layers,
            g_linear=self.config.disc_g_linear,
        ).to(self.device)
        
        # Policy
        if self.config.use_lstm:
            self.policy = AIRLPolicyLSTM(
                state_dim=self.state_dim,
                action_dim=self.action_dim,
                hidden_dim=self.config.policy_hidden_dim,
                num_layers=self.config.policy_num_layers,
                cell_size=self.config.cell_size,
            ).to(self.device)
        else:
            self.policy = AIRLPolicy(
                state_dim=self.state_dim,
                action_dim=self.action_dim,
                hidden_dim=self.config.policy_hidden_dim,
                num_layers=self.config.policy_num_layers,
            ).to(self.device)
        
        # Optimizers
        self.disc_optimizer = torch.optim.Adam(
            self.discriminator.parameters(),
            lr=self.config.discriminator_lr,
        )
        self.policy_optimizer = torch.optim.Adam(
            self.policy.parameters(),
            lr=self.config.policy_lr,
        )
    
    def _collect_rollout(self, num_steps: int) -> Dict[str, torch.Tensor]:
        """Collect a rollout from the environment using current policy."""
        states_list = []
        actions_list = []
        next_states_list = []
        dones_list = []
        log_probs_list = []
        values_list = []
        rewards_list = []  # Environment rewards (for monitoring)
        
        # Reset environment
        state = self.mdp.get_standard_start_state()
        obs = self.base_env.featurize_state_mdp(state)
        
        episode_reward = 0
        episode_rewards = []
        
        hidden = None
        if self.config.use_lstm:
            hidden = self.policy.get_initial_state(1, torch.device(self.device))
        
        for step in range(num_steps):
            # Get observations for both agents
            obs_0 = torch.tensor(obs[0].flatten(), dtype=torch.float32, device=self.device).unsqueeze(0)
            obs_1 = torch.tensor(obs[1].flatten(), dtype=torch.float32, device=self.device).unsqueeze(0)
            
            # Get actions from policy (we train agent 0, agent 1 uses same policy for self-play)
            with torch.no_grad():
                if self.config.use_lstm:
                    action_0, log_prob_0, value_0, hidden = self.policy.get_action(obs_0, hidden)
                    action_1, _, _, _ = self.policy.get_action(obs_1, hidden)
                else:
                    action_0, log_prob_0, value_0 = self.policy.get_action(obs_0)
                    action_1, _, _ = self.policy.get_action(obs_1)
            
            action_0 = action_0.item()
            action_1 = action_1.item()
            
            # Step environment
            joint_action = (Action.INDEX_TO_ACTION[action_0], Action.INDEX_TO_ACTION[action_1])
            next_state, info = self.base_env.mdp.get_state_transition(state, joint_action)
            
            # Get reward (sparse environment reward)
            env_reward = sum(info.get("sparse_reward_by_agent", [0, 0]))
            episode_reward += env_reward
            
            # Check if done
            done = step >= self.config.horizon - 1
            
            # Get next observation
            next_obs = self.base_env.featurize_state_mdp(next_state)
            next_obs_0 = torch.tensor(next_obs[0].flatten(), dtype=torch.float32, device=self.device)
            
            # Store transition
            states_list.append(obs_0.squeeze(0))
            actions_list.append(action_0)
            next_states_list.append(next_obs_0)
            dones_list.append(float(done))
            log_probs_list.append(log_prob_0.item())
            values_list.append(value_0.item())
            rewards_list.append(env_reward)
            
            # Update state
            state = next_state
            obs = next_obs
            
            if done:
                episode_rewards.append(episode_reward)
                episode_reward = 0
                
                # Reset
                state = self.mdp.get_standard_start_state()
                obs = self.base_env.featurize_state_mdp(state)
                
                if self.config.use_lstm:
                    hidden = self.policy.get_initial_state(1, torch.device(self.device))
        
        # Record final partial episode
        if episode_reward > 0:
            episode_rewards.append(episode_reward)
        
        rollout = {
            "states": torch.stack(states_list),
            "actions": torch.tensor(actions_list, dtype=torch.long, device=self.device),
            "next_states": torch.stack(next_states_list),
            "dones": torch.tensor(dones_list, dtype=torch.float32, device=self.device),
            "log_probs": torch.tensor(log_probs_list, dtype=torch.float32, device=self.device),
            "values": torch.tensor(values_list, dtype=torch.float32, device=self.device),
            "env_rewards": torch.tensor(rewards_list, dtype=torch.float32, device=self.device),
            "episode_rewards": episode_rewards,
        }
        
        return rollout
    
    def _update_discriminator(
        self,
        expert_batch: Dict[str, torch.Tensor],
        policy_batch: Dict[str, torch.Tensor],
    ) -> float:
        """
        Update discriminator to distinguish expert from policy samples.
        
        Returns:
            Discriminator loss
        """
        # Get log probs under current policy for both batches
        with torch.no_grad():
            # Expert actions
            expert_logits, _ = self.policy(expert_batch["states"])
            expert_log_probs = F.log_softmax(expert_logits, dim=-1)
            expert_action_log_probs = expert_log_probs[
                torch.arange(len(expert_batch["actions"])), expert_batch["actions"]
            ]
            
            # Policy actions
            policy_logits, _ = self.policy(policy_batch["states"])
            policy_log_probs = F.log_softmax(policy_logits, dim=-1)
            policy_action_log_probs = policy_log_probs[
                torch.arange(len(policy_batch["actions"])), policy_batch["actions"]
            ]
        
        # Discriminator forward pass
        expert_d = self.discriminator(
            expert_batch["states"],
            expert_batch["next_states"],
            expert_batch["dones"],
            expert_action_log_probs,
        )
        
        policy_d = self.discriminator(
            policy_batch["states"],
            policy_batch["next_states"],
            policy_batch["dones"],
            policy_action_log_probs,
        )
        
        # Binary cross-entropy loss
        # Expert should be classified as 1, policy as 0
        expert_loss = -torch.log(expert_d + 1e-8).mean()
        policy_loss = -torch.log(1 - policy_d + 1e-8).mean()
        disc_loss = expert_loss + policy_loss
        
        # Gradient step
        self.disc_optimizer.zero_grad()
        disc_loss.backward()
        nn.utils.clip_grad_norm_(self.discriminator.parameters(), self.config.max_grad_norm)
        self.disc_optimizer.step()
        
        # Compute accuracy for monitoring
        with torch.no_grad():
            accuracy = (
                (expert_d > 0.5).float().mean() + (policy_d < 0.5).float().mean()
            ) / 2
        
        return disc_loss.item(), accuracy.item()
    
    def _compute_airl_rewards(self, rollout: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute AIRL rewards for policy update."""
        with torch.no_grad():
            # Get action log probs
            logits, _ = self.policy(rollout["states"])
            log_probs = F.log_softmax(logits, dim=-1)
            action_log_probs = log_probs[
                torch.arange(len(rollout["actions"])), rollout["actions"]
            ]
            
            # AIRL reward
            rewards = self.discriminator.get_reward(
                rollout["states"],
                rollout["next_states"],
                rollout["dones"],
                action_log_probs,
            )
        
        return rewards
    
    def _compute_gae(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        dones: torch.Tensor,
        next_value: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute Generalized Advantage Estimation."""
        advantages = torch.zeros_like(rewards)
        gae = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_val = next_value
            else:
                next_val = values[t + 1]
            
            delta = rewards[t] + self.config.gamma * next_val * (1 - dones[t]) - values[t]
            gae = delta + self.config.gamma * self.config.gae_lambda * (1 - dones[t]) * gae
            advantages[t] = gae
        
        returns = advantages + values
        return advantages, returns
    
    def _update_policy(self, rollout: Dict[str, torch.Tensor]) -> float:
        """
        Update policy with PPO using AIRL rewards.
        
        Returns:
            Policy loss
        """
        # Compute AIRL rewards
        airl_rewards = self._compute_airl_rewards(rollout)
        
        # Compute advantages
        with torch.no_grad():
            # Get final value estimate
            last_obs = rollout["states"][-1].unsqueeze(0)
            _, last_value = self.policy(last_obs)
            last_value = last_value.item()
        
        advantages, returns = self._compute_gae(
            airl_rewards,
            rollout["values"],
            rollout["dones"],
            last_value,
        )
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Old log probs
        old_log_probs = rollout["log_probs"]
        
        # PPO update
        batch_size = len(rollout["states"])
        minibatch_size = self.config.batch_size
        total_loss = 0
        num_updates = 0
        
        for epoch in range(self.config.policy_epochs):
            perm = torch.randperm(batch_size)
            
            for start in range(0, batch_size, minibatch_size):
                idx = perm[start:start + minibatch_size]
                
                # Get current policy outputs
                log_probs, values, entropy = self.policy.evaluate_actions(
                    rollout["states"][idx],
                    rollout["actions"][idx],
                )
                
                # Ratio
                ratio = torch.exp(log_probs - old_log_probs[idx])
                
                # Clipped surrogate loss
                adv = advantages[idx]
                surr1 = ratio * adv
                surr2 = torch.clamp(ratio, 1 - self.config.clip_eps, 1 + self.config.clip_eps) * adv
                actor_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss
                value_loss = F.mse_loss(values, returns[idx])
                
                # Entropy bonus
                entropy_loss = -entropy.mean()
                
                # Total loss
                loss = (
                    actor_loss
                    + self.config.vf_coef * value_loss
                    + self.config.ent_coef * entropy_loss
                )
                
                # Gradient step
                self.policy_optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.config.max_grad_norm)
                self.policy_optimizer.step()
                
                total_loss += loss.item()
                num_updates += 1
        
        return total_loss / max(num_updates, 1)
    
    def train(self) -> Dict[str, Any]:
        """
        Run AIRL training.
        
        Returns:
            Training results dictionary
        """
        total_timesteps = 0
        num_iters = self.config.total_timesteps // self.config.steps_per_iter
        
        # Early stopping tracking
        best_reward = float('-inf')
        no_improvement_count = 0
        
        if self.config.verbose:
            print(f"\nStarting AIRL training")
            print(f"Layout: {self.config.layout_name}")
            print(f"Total timesteps: {self.config.total_timesteps:,}")
            print(f"Steps per iteration: {self.config.steps_per_iter:,}")
            print(f"Device: {self.device}")
            print()
        
        start_time = time.time()
        
        for iteration in range(num_iters):
            # Collect rollout
            rollout = self._collect_rollout(self.config.steps_per_iter)
            total_timesteps += self.config.steps_per_iter
            
            # Add to sample buffer
            self.policy_sample_buffer.append(rollout)
            
            # Combine samples from buffer for discriminator training
            combined_states = torch.cat([r["states"] for r in self.policy_sample_buffer])
            combined_actions = torch.cat([r["actions"] for r in self.policy_sample_buffer])
            combined_next_states = torch.cat([r["next_states"] for r in self.policy_sample_buffer])
            combined_dones = torch.cat([r["dones"] for r in self.policy_sample_buffer])
            
            policy_batch = {
                "states": combined_states,
                "actions": combined_actions,
                "next_states": combined_next_states,
                "dones": combined_dones,
            }
            
            # Discriminator updates
            disc_losses = []
            disc_accs = []
            
            for _ in range(self.config.disc_updates_per_iter):
                # Sample expert batch
                expert_idx = torch.randint(0, len(self.expert_states), (self.config.batch_size,))
                expert_batch = {
                    "states": self.expert_states[expert_idx],
                    "actions": self.expert_actions[expert_idx],
                    "next_states": self.expert_next_states[expert_idx],
                    "dones": self.expert_dones[expert_idx],
                }
                
                # Sample policy batch
                policy_idx = torch.randint(0, len(combined_states), (self.config.batch_size,))
                policy_mini_batch = {k: v[policy_idx] for k, v in policy_batch.items()}
                
                disc_loss, disc_acc = self._update_discriminator(expert_batch, policy_mini_batch)
                disc_losses.append(disc_loss)
                disc_accs.append(disc_acc)
            
            # Policy update
            policy_loss = self._update_policy(rollout)
            
            # Track episode rewards
            episode_rewards = rollout["episode_rewards"]
            if episode_rewards:
                mean_reward = np.mean(episode_rewards)
                self.train_info["episode_rewards"].extend(episode_rewards)
            else:
                mean_reward = 0
            
            # Early stopping
            if episode_rewards:
                if mean_reward > best_reward:
                    best_reward = mean_reward
                    no_improvement_count = 0
                else:
                    no_improvement_count += 1
                
                if no_improvement_count >= self.config.early_stop_patience:
                    if self.config.verbose:
                        print(f"\nEarly stopping at iteration {iteration}")
                    break
            
            # Logging
            self.train_info["disc_losses"].append(np.mean(disc_losses))
            self.train_info["policy_losses"].append(policy_loss)
            self.train_info["disc_accuracy"].append(np.mean(disc_accs))
            
            if iteration % self.config.log_interval == 0 and self.config.verbose:
                elapsed = time.time() - start_time
                fps = total_timesteps / elapsed if elapsed > 0 else 0
                
                print(f"Iter {iteration}/{num_iters} | "
                      f"Steps: {total_timesteps:,} | "
                      f"FPS: {fps:.0f} | "
                      f"D_loss: {np.mean(disc_losses):.4f} | "
                      f"D_acc: {np.mean(disc_accs):.2f} | "
                      f"P_loss: {policy_loss:.4f} | "
                      f"Reward: {mean_reward:.1f}")
            
            # Save checkpoint
            if iteration % self.config.save_interval == 0:
                self.save_checkpoint(iteration)
        
        # Final save
        self.save_checkpoint(num_iters)
        
        if self.config.verbose:
            total_time = time.time() - start_time
            print(f"\nTraining completed in {total_time:.1f}s")
            print(f"Best reward: {best_reward:.2f}")
        
        return {
            "total_timesteps": total_timesteps,
            "train_info": self.train_info,
            "best_reward": best_reward,
        }
    
    def save_checkpoint(self, step: int):
        """Save a training checkpoint."""
        checkpoint_dir = os.path.join(
            self.config.results_dir,
            self.config.experiment_name,
            f"checkpoint_{step:06d}"
        )
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Save policy
        torch.save({
            "policy_state_dict": self.policy.state_dict(),
            "discriminator_state_dict": self.discriminator.state_dict(),
        }, os.path.join(checkpoint_dir, "model.pt"))
        
        # Save config
        with open(os.path.join(checkpoint_dir, "config.pkl"), "wb") as f:
            pickle.dump(self.config, f)
        
        if self.config.verbose:
            print(f"Saved checkpoint to {checkpoint_dir}")
    
    def load_checkpoint(self, checkpoint_dir: str):
        """Load a training checkpoint."""
        checkpoint = torch.load(
            os.path.join(checkpoint_dir, "model.pt"),
            map_location=self.device,
        )
        
        self.policy.load_state_dict(checkpoint["policy_state_dict"])
        self.discriminator.load_state_dict(checkpoint["discriminator_state_dict"])
        
        if self.config.verbose:
            print(f"Loaded checkpoint from {checkpoint_dir}")
    
    def get_policy(self) -> nn.Module:
        """Return the trained policy."""
        return self.policy


##############
# Helper Functions #
##############


def save_airl_model(
    save_dir: str,
    policy: nn.Module,
    discriminator: nn.Module,
    config: AIRLConfig,
    verbose: bool = False,
):
    """Save AIRL model to disk."""
    os.makedirs(save_dir, exist_ok=True)
    
    # Save model
    torch.save({
        "policy_state_dict": policy.state_dict(),
        "discriminator_state_dict": discriminator.state_dict(),
    }, os.path.join(save_dir, "model.pt"))
    
    # Save config
    with open(os.path.join(save_dir, "config.pkl"), "wb") as f:
        pickle.dump(config, f)
    
    if verbose:
        print(f"Saved AIRL model to {save_dir}")


def load_airl_model(
    model_dir: str,
    device: Optional[str] = None,
    verbose: bool = False,
) -> Tuple[nn.Module, nn.Module, AIRLConfig]:
    """
    Load AIRL model from disk.
    
    Returns:
        Tuple of (policy, discriminator, config)
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load config
    with open(os.path.join(model_dir, "config.pkl"), "rb") as f:
        config = pickle.load(f)
    
    # Get state dim from config
    mdp_params = {"layout_name": config.layout_name, "old_dynamics": config.old_dynamics}
    agent_evaluator = AgentEvaluator.from_layout_name(
        mdp_params=mdp_params,
        env_params=DEFAULT_ENV_PARAMS,
    )
    base_env = agent_evaluator.env
    dummy_state = base_env.mdp.get_standard_start_state()
    obs_shape = base_env.featurize_state_mdp(dummy_state)[0].shape
    state_dim = int(np.prod(obs_shape))
    action_dim = len(Action.ALL_ACTIONS)
    
    # Create networks
    if config.use_lstm:
        policy = AIRLPolicyLSTM(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=config.policy_hidden_dim,
            num_layers=config.policy_num_layers,
            cell_size=config.cell_size,
        )
    else:
        policy = AIRLPolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=config.policy_hidden_dim,
            num_layers=config.policy_num_layers,
        )
    
    discriminator = AIRLDiscriminator(
        state_dim=state_dim,
        gamma=config.gamma,
        hidden_dim=config.disc_hidden_dim,
        num_layers=config.disc_num_layers,
        g_linear=config.disc_g_linear,
    )
    
    # Load weights
    checkpoint = torch.load(
        os.path.join(model_dir, "model.pt"),
        map_location=device,
    )
    policy.load_state_dict(checkpoint["policy_state_dict"])
    discriminator.load_state_dict(checkpoint["discriminator_state_dict"])
    
    policy = policy.to(device)
    discriminator = discriminator.to(device)
    policy.eval()
    discriminator.eval()
    
    if verbose:
        print(f"Loaded AIRL model from {model_dir}")
    
    return policy, discriminator, config


if __name__ == "__main__":
    # Simple test
    config = AIRLConfig(
        layout_name="cramped_room",
        total_timesteps=50000,
        steps_per_iter=1000,
        verbose=True,
    )
    
    trainer = AIRLTrainer(config)
    results = trainer.train()
    print(f"Training complete: {results}")

