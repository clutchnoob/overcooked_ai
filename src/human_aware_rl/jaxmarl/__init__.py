"""
JaxMARL integration for Overcooked AI.

This module provides JAX-based multi-agent RL training infrastructure
for the Overcooked environment.
"""

from human_aware_rl.jaxmarl.overcooked_env import (
    OvercookedJaxEnv,
    OvercookedJaxEnvConfig,
)
from human_aware_rl.jaxmarl.ppo import (
    PPOConfig,
    PPOTrainer,
)

__all__ = [
    "OvercookedJaxEnv",
    "OvercookedJaxEnvConfig",
    "PPOConfig",
    "PPOTrainer",
]

