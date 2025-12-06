"""
JAX-based Overcooked environment wrapper.

This module provides a JAX-compatible wrapper around the Overcooked environment
for efficient training with JaxMARL or similar JAX-based RL libraries.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple, Callable
import functools

import numpy as np

try:
    import jax
    import jax.numpy as jnp
    from flax import struct
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False
    jax = None
    jnp = None
    struct = None

from overcooked_ai_py.mdp.actions import Action
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv


@dataclass
class OvercookedJaxEnvConfig:
    """Configuration for the Overcooked JAX environment."""
    
    # Layout configuration
    layout_name: str = "cramped_room"
    old_dynamics: bool = False
    
    # Environment parameters
    horizon: int = 400
    
    # Reward shaping
    rew_shaping_params: Dict[str, float] = field(default_factory=lambda: {
        "PLACEMENT_IN_POT_REW": 3,
        "DISH_PICKUP_REWARD": 3,
        "SOUP_PICKUP_REWARD": 5,
        "DISH_DISP_DISTANCE_REW": 0,
        "POT_DISTANCE_REW": 0,
        "SOUP_DISTANCE_REW": 0,
    })
    reward_shaping_factor: float = 1.0
    reward_shaping_horizon: int = 0
    use_phi: bool = True
    
    # Observation type
    use_lossless_encoding: bool = True  # If False, use featurized state


if JAX_AVAILABLE:
    @struct.dataclass
    class OvercookedState:
        """JAX-compatible state representation for Overcooked."""
        
        # Encoded observations for each agent
        obs_0: jnp.ndarray
        obs_1: jnp.ndarray
        
        # Environment state info (serialized)
        state_dict: Dict[str, Any]
        
        # Step counter
        step: int
        
        # Done flag
        done: bool
        
        # Additional info
        info: Dict[str, Any] = struct.field(default_factory=dict)


class OvercookedJaxEnv:
    """
    JAX-compatible wrapper for the Overcooked environment.
    
    This wrapper enables vectorized environment stepping using JAX,
    making it compatible with JaxMARL-style training loops.
    
    Note: The actual game logic still runs in Python/NumPy, but the
    interface is JAX-compatible for integration with JAX-based RL.
    """
    
    def __init__(self, config: Optional[OvercookedJaxEnvConfig] = None):
        """
        Initialize the Overcooked JAX environment.
        
        Args:
            config: Environment configuration. Uses defaults if None.
        """
        if not JAX_AVAILABLE:
            raise ImportError(
                "JAX is required for OvercookedJaxEnv. "
                "Install with: pip install jax jaxlib flax"
            )
        
        self.config = config or OvercookedJaxEnvConfig()
        
        # Create the underlying MDP and environment
        self.mdp = OvercookedGridworld.from_layout_name(
            self.config.layout_name,
            rew_shaping_params=self.config.rew_shaping_params,
            old_dynamics=self.config.old_dynamics
        )
        
        self.base_env = OvercookedEnv.from_mdp(
            self.mdp,
            horizon=self.config.horizon,
            info_level=0
        )
        
        # Get observation and action space info
        dummy_state = self.mdp.get_standard_start_state()
        
        if self.config.use_lossless_encoding:
            dummy_obs = self.base_env.lossless_state_encoding_mdp(dummy_state)
        else:
            dummy_obs = self.base_env.featurize_state_mdp(dummy_state)
        
        self.obs_shape = dummy_obs[0].shape
        self.num_actions = len(Action.ALL_ACTIONS)
        self.num_agents = 2
        
        # Reward shaping
        self._initial_reward_shaping_factor = self.config.reward_shaping_factor
        self.reward_shaping_factor = self.config.reward_shaping_factor
        self.reward_shaping_horizon = self.config.reward_shaping_horizon
        
        # Track timesteps for annealing
        self.total_timesteps = 0

    @property
    def observation_space(self) -> Dict[str, Any]:
        """Return observation space specification."""
        return {
            "shape": self.obs_shape,
            "dtype": np.float32,
        }

    @property
    def action_space(self) -> Dict[str, Any]:
        """Return action space specification."""
        return {
            "n": self.num_actions,
            "dtype": np.int32,
        }

    def _encode_state(self, state) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Encode state to observations for both agents."""
        if self.config.use_lossless_encoding:
            obs = self.base_env.lossless_state_encoding_mdp(state)
        else:
            obs = self.base_env.featurize_state_mdp(state)
        
        return jnp.array(obs[0], dtype=jnp.float32), jnp.array(obs[1], dtype=jnp.float32)

    def reset(self, key: Optional[Any] = None) -> Tuple[Any, Dict[str, jnp.ndarray]]:
        """
        Reset the environment.
        
        Args:
            key: JAX random key (optional, for API compatibility)
            
        Returns:
            Tuple of (state, observations_dict)
        """
        self.base_env.reset()
        
        obs_0, obs_1 = self._encode_state(self.base_env.state)
        
        state = OvercookedState(
            obs_0=obs_0,
            obs_1=obs_1,
            state_dict=self.base_env.state.to_dict(),
            step=0,
            done=False,
            info={}
        )
        
        obs = {
            "agent_0": obs_0,
            "agent_1": obs_1,
        }
        
        return state, obs

    def step(
        self,
        state: Any,
        actions: Dict[str, int],
        key: Optional[Any] = None
    ) -> Tuple[Any, Dict[str, jnp.ndarray], Dict[str, float], Dict[str, bool], Dict[str, Any]]:
        """
        Take a step in the environment.
        
        Args:
            state: Current environment state
            actions: Dictionary mapping agent names to action indices
            key: JAX random key (optional, for API compatibility)
            
        Returns:
            Tuple of (next_state, observations, rewards, dones, infos)
        """
        # Convert actions to joint action
        action_0 = int(actions.get("agent_0", 0))
        action_1 = int(actions.get("agent_1", 0))
        
        joint_action = (
            Action.INDEX_TO_ACTION[action_0],
            Action.INDEX_TO_ACTION[action_1]
        )
        
        # Step the underlying environment
        if self.config.use_phi:
            next_state, sparse_reward, done, info = self.base_env.step(
                joint_action, display_phi=True
            )
            potential = info.get("phi_s_prime", 0) - info.get("phi_s", 0)
            dense_reward = (potential, potential)
        else:
            next_state, sparse_reward, done, info = self.base_env.step(
                joint_action, display_phi=False
            )
            dense_reward = info.get("shaped_r_by_agent", (0, 0))
        
        # Compute shaped rewards
        shaped_reward_0 = sparse_reward + self.reward_shaping_factor * dense_reward[0]
        shaped_reward_1 = sparse_reward + self.reward_shaping_factor * dense_reward[1]
        
        # Encode observations
        obs_0, obs_1 = self._encode_state(next_state)
        
        # Create new state
        new_state = OvercookedState(
            obs_0=obs_0,
            obs_1=obs_1,
            state_dict=next_state.to_dict(),
            step=state.step + 1,
            done=done,
            info=info
        )
        
        obs = {
            "agent_0": obs_0,
            "agent_1": obs_1,
        }
        
        rewards = {
            "agent_0": shaped_reward_0,
            "agent_1": shaped_reward_1,
        }
        
        dones = {
            "agent_0": done,
            "agent_1": done,
            "__all__": done,
        }
        
        infos = {
            "agent_0": info,
            "agent_1": info,
            "sparse_reward": sparse_reward,
        }
        
        return new_state, obs, rewards, dones, infos

    def anneal_reward_shaping(self, timesteps: int) -> None:
        """
        Anneal the reward shaping factor based on current timesteps.
        
        Args:
            timesteps: Current total timesteps
        """
        if self.reward_shaping_horizon == 0:
            return
        
        fraction = max(1.0 - float(timesteps) / self.reward_shaping_horizon, 0.0)
        self.reward_shaping_factor = fraction * self._initial_reward_shaping_factor

    def get_obs(self, state: Any) -> Dict[str, jnp.ndarray]:
        """Get observations from state."""
        return {
            "agent_0": state.obs_0,
            "agent_1": state.obs_1,
        }


class VectorizedOvercookedEnv:
    """
    Vectorized wrapper for running multiple Overcooked environments in parallel.
    
    This is useful for efficient data collection during training.
    """
    
    def __init__(
        self,
        num_envs: int,
        config: Optional[OvercookedJaxEnvConfig] = None
    ):
        """
        Initialize vectorized environments.
        
        Args:
            num_envs: Number of parallel environments
            config: Environment configuration
        """
        self.num_envs = num_envs
        self.config = config or OvercookedJaxEnvConfig()
        
        # Create individual environments
        self.envs = [OvercookedJaxEnv(config) for _ in range(num_envs)]
        
        # Get space info from first env
        self.obs_shape = self.envs[0].obs_shape
        self.num_actions = self.envs[0].num_actions
        self.num_agents = self.envs[0].num_agents

    def reset(self, key: Optional[Any] = None) -> Tuple[Any, Dict[str, np.ndarray]]:
        """Reset all environments."""
        states = []
        obs_0_list = []
        obs_1_list = []
        
        for env in self.envs:
            state, obs = env.reset()
            states.append(state)
            obs_0_list.append(obs["agent_0"])
            obs_1_list.append(obs["agent_1"])
        
        batched_obs = {
            "agent_0": jnp.stack(obs_0_list),
            "agent_1": jnp.stack(obs_1_list),
        }
        
        return states, batched_obs

    def step(
        self,
        states: Any,
        actions: Dict[str, np.ndarray],
        key: Optional[Any] = None
    ) -> Tuple[Any, Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, Any]]:
        """
        Step all environments.
        
        Args:
            states: List of environment states
            actions: Dictionary with batched actions for each agent
            key: Random key (optional)
            
        Returns:
            Tuple of (next_states, observations, rewards, dones, infos)
        """
        next_states = []
        obs_0_list = []
        obs_1_list = []
        rewards_0_list = []
        rewards_1_list = []
        dones_list = []
        infos_list = []
        
        for i, (env, state) in enumerate(zip(self.envs, states)):
            env_actions = {
                "agent_0": int(actions["agent_0"][i]),
                "agent_1": int(actions["agent_1"][i]),
            }
            
            next_state, obs, rewards, dones, info = env.step(state, env_actions)
            
            next_states.append(next_state)
            obs_0_list.append(obs["agent_0"])
            obs_1_list.append(obs["agent_1"])
            rewards_0_list.append(rewards["agent_0"])
            rewards_1_list.append(rewards["agent_1"])
            dones_list.append(dones["__all__"])
            infos_list.append(info)
        
        batched_obs = {
            "agent_0": jnp.stack(obs_0_list),
            "agent_1": jnp.stack(obs_1_list),
        }
        
        batched_rewards = {
            "agent_0": jnp.array(rewards_0_list),
            "agent_1": jnp.array(rewards_1_list),
        }
        
        batched_dones = {
            "agent_0": jnp.array(dones_list),
            "agent_1": jnp.array(dones_list),
            "__all__": jnp.array(dones_list),
        }
        
        return next_states, batched_obs, batched_rewards, batched_dones, infos_list

    def anneal_reward_shaping(self, timesteps: int) -> None:
        """Anneal reward shaping for all environments."""
        for env in self.envs:
            env.anneal_reward_shaping(timesteps)


# For compatibility, provide a factory function
def make_overcooked_env(
    layout_name: str = "cramped_room",
    horizon: int = 400,
    use_lossless_encoding: bool = True,
    **kwargs
) -> OvercookedJaxEnv:
    """
    Factory function to create an Overcooked JAX environment.
    
    Args:
        layout_name: Name of the layout
        horizon: Episode length
        use_lossless_encoding: Whether to use lossless state encoding
        **kwargs: Additional config parameters
        
    Returns:
        OvercookedJaxEnv instance
    """
    config = OvercookedJaxEnvConfig(
        layout_name=layout_name,
        horizon=horizon,
        use_lossless_encoding=use_lossless_encoding,
        **{k: v for k, v in kwargs.items() if hasattr(OvercookedJaxEnvConfig, k)}
    )
    return OvercookedJaxEnv(config)

