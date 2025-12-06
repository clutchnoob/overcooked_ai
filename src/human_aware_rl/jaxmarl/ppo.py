"""
PPO Training for Overcooked using JAX/Flax.

This module provides a JAX-based PPO implementation for training
agents in the Overcooked environment. It supports self-play and
BC-schedule training modes.
"""

import os
import pickle
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, NamedTuple

import numpy as np

try:
    import jax
    import jax.numpy as jnp
    from jax import random
    import flax
    from flax import linen as nn
    from flax.training.train_state import TrainState
    import optax
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False
    jax = None
    jnp = None
    nn = None
    optax = None
    TrainState = None

from human_aware_rl.jaxmarl.overcooked_env import (
    OvercookedJaxEnv,
    OvercookedJaxEnvConfig,
    VectorizedOvercookedEnv,
)


@dataclass
class PPOConfig:
    """Configuration for PPO training."""
    
    # Environment
    layout_name: str = "cramped_room"
    horizon: int = 400
    num_envs: int = 32  # Increased from 8 for better sample efficiency
    old_dynamics: bool = True  # Paper uses old dynamics
    
    # Training
    total_timesteps: int = 1_000_000
    learning_rate: float = 3e-4
    num_steps: int = 400  # Full episode per env before update (matches horizon)
    num_minibatches: int = 10  # Paper uses 10 minibatches
    num_epochs: int = 8  # num_sgd_iter in paper
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    kl_coeff: float = 0.2  # KL divergence coefficient
    
    # Entropy annealing (paper uses this)
    entropy_coeff_start: float = 0.2
    entropy_coeff_end: float = 0.1
    entropy_coeff_horizon: float = 3e5
    use_entropy_annealing: bool = True
    
    # Network architecture
    num_hidden_layers: int = 3
    hidden_dim: int = 64
    num_filters: int = 25
    num_conv_layers: int = 3
    use_lstm: bool = False
    cell_size: int = 256
    
    # Reward shaping
    reward_shaping_factor: float = 1.0
    reward_shaping_horizon: float = float('inf')
    use_phi: bool = False  # Paper doesn't use potential-based shaping
    
    # BC schedule for training with BC agents
    # List of (timestep, bc_factor) tuples
    bc_schedule: List[Tuple[int, float]] = field(default_factory=lambda: [(0, 0.0), (float('inf'), 0.0)])
    bc_model_dir: Optional[str] = None
    
    # Logging and saving
    log_interval: int = 1  # Log every update for reward tracking
    save_interval: int = 50  # Save more frequently
    eval_interval: int = 25  # Evaluate more often
    eval_num_games: int = 5
    verbose: bool = True
    
    # Early stopping
    early_stop_patience: int = 20  # Stop if no improvement for this many updates
    early_stop_min_reward: float = float('inf')  # Minimum reward threshold (disabled by default)
    
    # Output
    results_dir: str = "results"
    experiment_name: str = "ppo_overcooked"
    seed: int = 0
    
    # Training batch settings (paper values)
    train_batch_size: int = 12000
    num_workers: int = 30


class Transition(NamedTuple):
    """A single transition from environment interaction."""
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray


if JAX_AVAILABLE:
    
    class ActorCritic(nn.Module):
        """Actor-Critic network for PPO."""
        
        action_dim: int
        hidden_dim: int = 64
        num_hidden_layers: int = 3
        num_filters: int = 25
        num_conv_layers: int = 3
        
        @nn.compact
        def __call__(self, x):
            # Handle both flat and image observations
            if len(x.shape) == 4:  # Image observation (batch, H, W, C)
                # Conv layers
                for i in range(self.num_conv_layers):
                    kernel_size = (5, 5) if i == 0 else (3, 3)
                    x = nn.Conv(
                        features=self.num_filters,
                        kernel_size=kernel_size,
                        padding='SAME' if i < self.num_conv_layers - 1 else 'VALID'
                    )(x)
                    x = nn.leaky_relu(x)
                
                # Flatten
                x = x.reshape((x.shape[0], -1))
            elif len(x.shape) == 2:  # Flat observation (batch, obs_dim)
                pass  # Already flat
            else:
                raise ValueError(f"Unexpected observation shape: {x.shape}")
            
            # Hidden layers
            for i in range(self.num_hidden_layers):
                x = nn.Dense(self.hidden_dim)(x)
                x = nn.leaky_relu(x)
            
            # Actor head
            actor_logits = nn.Dense(self.action_dim)(x)
            
            # Critic head
            critic = nn.Dense(1)(x)
            
            return actor_logits, jnp.squeeze(critic, axis=-1)
    
    
    class ActorCriticLSTM(nn.Module):
        """Actor-Critic network with LSTM for PPO."""
        
        action_dim: int
        hidden_dim: int = 64
        num_hidden_layers: int = 3
        cell_size: int = 256
        
        @nn.compact
        def __call__(self, x, hidden_state):
            batch_size = x.shape[0]
            
            # Hidden layers before LSTM
            for i in range(self.num_hidden_layers):
                x = nn.Dense(self.hidden_dim)(x)
                x = nn.leaky_relu(x)
            
            # LSTM
            lstm_cell = nn.LSTMCell(features=self.cell_size)
            carry, x = lstm_cell(hidden_state, x)
            
            # Actor head
            actor_logits = nn.Dense(self.action_dim)(x)
            
            # Critic head
            critic = nn.Dense(1)(x)
            
            return actor_logits, jnp.squeeze(critic, axis=-1), carry
        
        def initialize_carry(self, batch_size: int):
            """Initialize LSTM carry state."""
            return (
                jnp.zeros((batch_size, self.cell_size)),
                jnp.zeros((batch_size, self.cell_size))
            )


class PPOTrainer:
    """
    PPO Trainer for Overcooked environment.
    
    Supports self-play and BC-schedule training modes.
    """
    
    def __init__(self, config: PPOConfig):
        """
        Initialize the PPO trainer.
        
        Args:
            config: PPO training configuration
        """
        if not JAX_AVAILABLE:
            raise ImportError(
                "JAX is required for PPO training. "
                "Install with: pip install jax jaxlib flax optax"
            )
        
        self.config = config
        
        # Create environment config
        env_config = OvercookedJaxEnvConfig(
            layout_name=config.layout_name,
            horizon=config.horizon,
            reward_shaping_factor=config.reward_shaping_factor,
            reward_shaping_horizon=config.reward_shaping_horizon,
            use_phi=config.use_phi,
        )
        
        # Create vectorized environment
        self.envs = VectorizedOvercookedEnv(
            num_envs=config.num_envs,
            config=env_config
        )
        
        # Get observation and action space info
        self.obs_shape = self.envs.obs_shape
        self.num_actions = self.envs.num_actions
        
        # Initialize random key
        self.key = random.PRNGKey(0)
        
        # Create networks
        self._init_networks()
        
        # BC agent for BC-schedule training
        self.bc_agent = None
        if config.bc_model_dir:
            self._load_bc_agent()
        
        # Logging
        self.train_info = {
            "timesteps": [],
            "episode_returns": [],
            "episode_lengths": [],
            "losses": [],
        }
        
        # Create output directory
        os.makedirs(config.results_dir, exist_ok=True)

    def _init_networks(self):
        """Initialize actor-critic networks."""
        self.key, subkey = random.split(self.key)
        
        # Determine observation shape for network initialization
        dummy_obs = jnp.zeros((1, *self.obs_shape))
        
        if self.config.use_lstm:
            self.network = ActorCriticLSTM(
                action_dim=self.num_actions,
                hidden_dim=self.config.hidden_dim,
                num_hidden_layers=self.config.num_hidden_layers,
                cell_size=self.config.cell_size,
            )
            dummy_hidden = self.network.initialize_carry(1)
            params = self.network.init(subkey, dummy_obs, dummy_hidden)
        else:
            self.network = ActorCritic(
                action_dim=self.num_actions,
                hidden_dim=self.config.hidden_dim,
                num_hidden_layers=self.config.num_hidden_layers,
                num_filters=self.config.num_filters,
                num_conv_layers=self.config.num_conv_layers,
            )
            params = self.network.init(subkey, dummy_obs)
        
        # Create optimizer
        tx = optax.chain(
            optax.clip_by_global_norm(self.config.max_grad_norm),
            optax.adam(self.config.learning_rate, eps=1e-5)
        )
        
        # Create train state
        self.train_state = TrainState.create(
            apply_fn=self.network.apply,
            params=params,
            tx=tx,
        )

    def _load_bc_agent(self):
        """Load BC agent for BC-schedule training."""
        from human_aware_rl.imitation.behavior_cloning import load_bc_model
        from human_aware_rl.imitation.bc_agent import BCAgent
        from overcooked_ai_py.mdp.overcooked_mdp import OvercookedState as OCState
        
        model, bc_params = load_bc_model(self.config.bc_model_dir)
        
        # Create featurization function using the same environment
        def featurize_fn(state):
            return self.envs.envs[0].base_env.featurize_state_mdp(state)
        
        self.bc_agent = BCAgent(
            model=model,
            bc_params=bc_params,
            featurize_fn=featurize_fn,
            agent_index=1,  # BC agent plays as agent 1
            stochastic=True
        )
        
        # Store reference to reconstruct raw states
        self._bc_featurize_fn = featurize_fn

    def _get_bc_factor(self, timesteps: int) -> float:
        """Get BC factor based on schedule."""
        schedule = self.config.bc_schedule
        
        # Find the two points to interpolate between
        p0 = schedule[0]
        p1 = schedule[1]
        i = 2
        
        while timesteps > p1[0] and i < len(schedule):
            p0 = p1
            p1 = schedule[i]
            i += 1
        
        t0, v0 = p0
        t1, v1 = p1
        
        if t1 == t0:
            return v0
        
        # Linear interpolation
        alpha = (timesteps - t0) / (t1 - t0)
        alpha = min(max(alpha, 0.0), 1.0)
        
        return v0 + alpha * (v1 - v0)

    @staticmethod
    def _select_action(key, logits):
        """Sample action from categorical distribution."""
        return random.categorical(key, logits)


    def _compute_gae(self, transitions, last_value):
        """Compute Generalized Advantage Estimation."""
        advantages = []
        gae = 0
        
        for t in reversed(range(len(transitions))):
            if t == len(transitions) - 1:
                next_value = last_value
            else:
                next_value = transitions[t + 1].value
            
            delta = transitions[t].reward + self.config.gamma * next_value * (1 - transitions[t].done) - transitions[t].value
            gae = delta + self.config.gamma * self.config.gae_lambda * (1 - transitions[t].done) * gae
            advantages.insert(0, gae)
        
        advantages = jnp.stack(advantages)
        returns = advantages + jnp.stack([t.value for t in transitions])
        
        return advantages, returns

    def _get_entropy_coef(self, timesteps: int) -> float:
        """Get entropy coefficient with annealing."""
        if not self.config.use_entropy_annealing:
            return self.config.ent_coef
        
        # Linear annealing from start to end over horizon
        alpha = min(1.0, timesteps / self.config.entropy_coeff_horizon)
        return self.config.entropy_coeff_start + alpha * (
            self.config.entropy_coeff_end - self.config.entropy_coeff_start
        )

    def _update(self, train_state, batch):
        """Perform PPO update."""
        # Get annealed entropy coefficient
        ent_coef = self._get_entropy_coef(self.total_timesteps)
        vf_coef = self.config.vf_coef
        clip_eps = self.config.clip_eps
        use_lstm = self.config.use_lstm
        
        def loss_fn(params, obs, actions, old_log_probs, advantages, returns):
            if use_lstm:
                logits, values, _ = train_state.apply_fn(
                    params, obs, self.network.initialize_carry(obs.shape[0])
                )
            else:
                logits, values = train_state.apply_fn(params, obs)
            
            # Actor loss with clipping
            log_probs = jax.nn.log_softmax(logits)[jnp.arange(len(actions)), actions]
            ratio = jnp.exp(log_probs - old_log_probs)
            
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            
            actor_loss1 = -advantages * ratio
            actor_loss2 = -advantages * jnp.clip(ratio, 1 - clip_eps, 1 + clip_eps)
            actor_loss = jnp.maximum(actor_loss1, actor_loss2).mean()
            
            # Critic loss
            critic_loss = ((values - returns) ** 2).mean()
            
            # Entropy bonus
            probs = jax.nn.softmax(logits)
            entropy = -(probs * jax.nn.log_softmax(logits)).sum(axis=-1).mean()
            
            total_loss = actor_loss + vf_coef * critic_loss - ent_coef * entropy
            
            return total_loss, {
                "actor_loss": actor_loss,
                "critic_loss": critic_loss,
                "entropy": entropy,
                "entropy_coef": ent_coef,
            }
        
        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        
        obs = batch["obs"]
        actions = batch["actions"]
        old_log_probs = batch["log_probs"]
        advantages = batch["advantages"]
        returns = batch["returns"]
        
        (loss, metrics), grads = grad_fn(
            train_state.params, obs, actions, old_log_probs, advantages, returns
        )
        
        train_state = train_state.apply_gradients(grads=grads)
        
        return train_state, loss, metrics

    def train(self) -> Dict[str, Any]:
        """
        Run PPO training with reward tracking and early stopping.
        
        Returns:
            Dictionary of training results
        """
        self.total_timesteps = 0
        num_updates = self.config.total_timesteps // (self.config.num_envs * self.config.num_steps)
        
        # Reset environments
        states, obs = self.envs.reset()
        
        # Episode reward tracking
        episode_rewards = []  # Per-episode rewards
        current_episode_rewards = np.zeros(self.config.num_envs)  # Running rewards for each env
        recent_rewards = []  # For moving average
        reward_window = 100  # Number of episodes for moving average
        
        # Early stopping tracking
        best_mean_reward = float('-inf')
        no_improvement_count = 0
        
        if self.config.verbose:
            print(f"Starting PPO training for {self.config.total_timesteps} timesteps")
            print(f"Layout: {self.config.layout_name}")
            print(f"Num envs: {self.config.num_envs}, Num steps: {self.config.num_steps}")
            print(f"Batch size: {self.config.num_envs * self.config.num_steps}")
            print(f"Early stop patience: {self.config.early_stop_patience} updates")
        
        start_time = time.time()
        
        for update in range(num_updates):
            # Update reward shaping
            self.envs.anneal_reward_shaping(self.total_timesteps)
            
            # Collect rollout with reward tracking
            transitions, states, obs, ep_rewards = self._collect_rollout_with_rewards(
                self.train_state, states, obs, current_episode_rewards
            )
            
            # Update episode rewards
            if ep_rewards:
                episode_rewards.extend(ep_rewards)
                recent_rewards.extend(ep_rewards)
                # Keep only the last 'reward_window' episodes
                if len(recent_rewards) > reward_window:
                    recent_rewards = recent_rewards[-reward_window:]
            
            # Compute advantages
            if self.config.use_lstm:
                _, last_value, _ = self.train_state.apply_fn(
                    self.train_state.params,
                    obs["agent_0"],
                    self.network.initialize_carry(self.config.num_envs)
                )
            else:
                _, last_value = self.train_state.apply_fn(self.train_state.params, obs["agent_0"])
            
            advantages, returns = self._compute_gae(transitions, last_value)
            
            # Flatten batch
            batch = {
                "obs": jnp.concatenate([t.obs for t in transitions]),
                "actions": jnp.concatenate([t.action for t in transitions]),
                "log_probs": jnp.concatenate([t.log_prob for t in transitions]),
                "advantages": advantages.reshape(-1),
                "returns": returns.reshape(-1),
            }
            
            # PPO update epochs
            batch_size = len(batch["obs"])
            minibatch_size = max(1, batch_size // self.config.num_minibatches)
            
            for epoch in range(self.config.num_epochs):
                self.key, perm_key = random.split(self.key)
                perm = random.permutation(perm_key, batch_size)
                
                for start in range(0, batch_size, minibatch_size):
                    idx = perm[start:start + minibatch_size]
                    minibatch = {k: v[idx] for k, v in batch.items()}
                    
                    self.train_state, loss, metrics = self._update(self.train_state, minibatch)
            
            # Compute reward statistics
            mean_reward = np.mean(recent_rewards) if recent_rewards else 0.0
            std_reward = np.std(recent_rewards) if len(recent_rewards) > 1 else 0.0
            
            # Early stopping check
            if recent_rewards and len(recent_rewards) >= 20:  # Need at least 20 episodes
                if mean_reward > best_mean_reward:
                    best_mean_reward = mean_reward
                    no_improvement_count = 0
                else:
                    no_improvement_count += 1
                
                if no_improvement_count >= self.config.early_stop_patience:
                    if self.config.verbose:
                        print(f"\nEarly stopping: No improvement for {self.config.early_stop_patience} updates")
                        print(f"Best mean reward: {best_mean_reward:.2f}")
                    break
            
            # Logging
            if update % self.config.log_interval == 0 and self.config.verbose:
                elapsed = time.time() - start_time
                fps = self.total_timesteps / elapsed if elapsed > 0 else 0
                ent_coef = self._get_entropy_coef(self.total_timesteps)
                
                # Format reward info
                if recent_rewards:
                    reward_str = f"Reward: {mean_reward:.1f}±{std_reward:.1f}"
                else:
                    reward_str = "Reward: N/A"
                
                print(f"Update {update}/{num_updates} | "
                      f"Steps: {self.total_timesteps:,} | "
                      f"FPS: {fps:.0f} | "
                      f"{reward_str} | "
                      f"Loss: {loss:.4f} | "
                      f"Ent: {ent_coef:.3f}")
            
            # Save checkpoint
            if update % self.config.save_interval == 0:
                self.save_checkpoint(update)
        
        # Final save
        self.save_checkpoint(num_updates)
        
        if self.config.verbose:
            total_time = time.time() - start_time
            final_reward = np.mean(recent_rewards) if recent_rewards else 0.0
            print(f"\nTraining completed in {total_time:.1f}s")
            print(f"Final mean reward: {final_reward:.2f}")
            print(f"Best mean reward: {best_mean_reward:.2f}")
            print(f"Total episodes: {len(episode_rewards)}")
        
        return {
            "total_timesteps": self.total_timesteps,
            "train_info": self.train_info,
            "episode_rewards": episode_rewards,
            "final_mean_reward": np.mean(recent_rewards) if recent_rewards else 0.0,
            "best_mean_reward": best_mean_reward,
        }

    def _collect_rollout_with_rewards(self, train_state, states, obs, current_episode_rewards):
        """Collect a rollout from all environments with reward tracking."""
        transitions = []
        completed_episode_rewards = []
        
        for step in range(self.config.num_steps):
            self.key, action_key = random.split(self.key)
            
            # Get actions for agent 0 from policy
            obs_0 = obs["agent_0"]
            
            if self.config.use_lstm:
                logits, values, _ = train_state.apply_fn(
                    train_state.params,
                    obs_0,
                    self.network.initialize_carry(self.config.num_envs)
                )
            else:
                logits, values = train_state.apply_fn(train_state.params, obs_0)
            
            # Sample actions
            actions_0 = self._select_action(action_key, logits)
            log_probs = jax.nn.log_softmax(logits)[jnp.arange(len(actions_0)), actions_0]
            
            # For agent 1, either use self-play or BC
            bc_factor = self._get_bc_factor(self.total_timesteps)
            
            if bc_factor > 0 and self.bc_agent is not None and np.random.random() < bc_factor:
                # Use BC agent for agent 1
                from overcooked_ai_py.mdp.actions import Action
                bc_actions = []
                for i in range(self.config.num_envs):
                    raw_state = self.envs.envs[i].base_env.state
                    action, _ = self.bc_agent.action(raw_state)
                    action_idx = Action.ACTION_TO_INDEX[action]
                    bc_actions.append(action_idx)
                actions_1 = jnp.array(bc_actions)
            else:
                # Self-play: use same policy for agent 1
                self.key, action_key_1 = random.split(self.key)
                obs_1 = obs["agent_1"]
                
                if self.config.use_lstm:
                    logits_1, _, _ = train_state.apply_fn(
                        train_state.params,
                        obs_1,
                        self.network.initialize_carry(self.config.num_envs)
                    )
                else:
                    logits_1, _ = train_state.apply_fn(train_state.params, obs_1)
                
                actions_1 = self._select_action(action_key_1, logits_1)
            
            # Step environments
            actions = {
                "agent_0": np.array(actions_0),
                "agent_1": np.array(actions_1),
            }
            
            states, next_obs, rewards, dones, infos = self.envs.step(states, actions)
            
            # Track rewards (sparse rewards from deliveries)
            sparse_rewards = np.array(rewards["agent_0"])
            current_episode_rewards += sparse_rewards
            
            # Store transition (for agent 0)
            transition = Transition(
                done=dones["agent_0"],
                action=actions_0,
                value=values,
                reward=rewards["agent_0"],
                log_prob=log_probs,
                obs=obs_0,
            )
            transitions.append(transition)
            
            obs = next_obs
            self.total_timesteps += self.config.num_envs
            
            # Handle episode ends
            for i, done in enumerate(np.array(dones["__all__"])):
                if done:
                    # Record completed episode reward
                    completed_episode_rewards.append(current_episode_rewards[i])
                    current_episode_rewards[i] = 0.0
                    
                    # Reset environment
                    states[i], new_obs = self.envs.envs[i].reset()
                    obs["agent_0"] = obs["agent_0"].at[i].set(new_obs["agent_0"])
                    obs["agent_1"] = obs["agent_1"].at[i].set(new_obs["agent_1"])
        
        return transitions, states, obs, completed_episode_rewards

    def save_checkpoint(self, step: int):
        """Save a training checkpoint."""
        checkpoint_dir = os.path.join(
            self.config.results_dir,
            self.config.experiment_name,
            f"checkpoint_{step:06d}"
        )
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Save model parameters
        with open(os.path.join(checkpoint_dir, "params.pkl"), "wb") as f:
            pickle.dump(self.train_state.params, f)
        
        # Save config
        with open(os.path.join(checkpoint_dir, "config.pkl"), "wb") as f:
            pickle.dump(self.config, f)
        
        if self.config.verbose:
            print(f"Saved checkpoint to {checkpoint_dir}")

    def load_checkpoint(self, checkpoint_dir: str):
        """Load a training checkpoint."""
        with open(os.path.join(checkpoint_dir, "params.pkl"), "rb") as f:
            params = pickle.load(f)
        
        self.train_state = self.train_state.replace(params=params)
        
        if self.config.verbose:
            print(f"Loaded checkpoint from {checkpoint_dir}")

    def get_policy(self):
        """Return the trained policy for evaluation."""
        return self.train_state.params


def train_ppo(config: PPOConfig) -> Dict[str, Any]:
    """
    Train a PPO agent in the Overcooked environment.
    
    Args:
        config: PPO training configuration
        
    Returns:
        Dictionary of training results
    """
    trainer = PPOTrainer(config)
    return trainer.train()

