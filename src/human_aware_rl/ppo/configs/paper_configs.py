"""
Paper hyperparameter configurations for PPO training.

These configurations match the hyperparameters used in:
"On the Utility of Learning about Humans for Human-AI Coordination"
by Micah Carroll et al.

Tables 3 and 4 from the paper appendix are encoded here.
"""

from typing import Dict, Any, List, Tuple

# Layout mapping from paper names to environment names
PAPER_LAYOUTS = [
    "cramped_room",
    "asymmetric_advantages",
    "coordination_ring", 
    "forced_coordination",
    "counter_circuit",
]

LAYOUT_TO_ENV = {
    "cramped_room": "cramped_room",
    "asymmetric_advantages": "asymmetric_advantages",
    "coordination_ring": "coordination_ring",
    "forced_coordination": "forced_coordination",
    "counter_circuit": "counter_circuit_o_1order",
}

# Common parameters across all experiments (from paper)
PAPER_COMMON_PARAMS = {
    # Network architecture
    "num_hidden_layers": 3,
    "hidden_dim": 64,
    "num_filters": 25,
    "num_conv_layers": 3,
    "use_lstm": False,
    "cell_size": 256,
    
    # Training batch settings
    "train_batch_size": 12000,
    "num_minibatches": 10,  # minibatch_size = train_batch_size / num_minibatches = 2000
    "rollout_fragment_length": 400,
    "num_sgd_iter": 8,
    
    # Entropy annealing (from paper)
    "entropy_coeff_start": 0.2,
    "entropy_coeff_end": 0.1,
    "entropy_coeff_horizon": 3e5,
    
    # Episode settings
    "horizon": 400,
    "old_dynamics": True,  # Paper uses old dynamics (cooking starts automatically)
    
    # Reward shaping (paper uses shaped rewards)
    "use_phi": False,  # Paper trains without potential-based shaping
    "reward_shaping_factor": 1.0,
    
    # Number of parallel workers
    "num_workers": 30,
    
    # Evaluation
    "evaluation_interval": 50,
    "evaluation_num_games": 1,
}


# Table 3: PPO Self-Play Hyperparameters (per-layout)
# Each entry contains the tuned hyperparameters for that layout
PAPER_PPO_SP_CONFIGS: Dict[str, Dict[str, Any]] = {
    "cramped_room": {
        "learning_rate": 1.63e-4,
        "gamma": 0.964,
        "clip_eps": 0.132,
        "max_grad_norm": 0.247,
        "gae_lambda": 0.6,
        "vf_coef": 9.95e-3,
        "kl_coeff": 0.197,
        "reward_shaping_horizon": 4.5e6,
        "num_training_iters": 550,  # Paper value
    },
    "asymmetric_advantages": {
        "learning_rate": 2.1e-4,
        "gamma": 0.964,
        "clip_eps": 0.229,
        "max_grad_norm": 0.256,
        "gae_lambda": 0.5,
        "vf_coef": 0.022,
        "kl_coeff": 0.185,
        "reward_shaping_horizon": 5e6,
        "num_training_iters": 650,  # Paper value
    },
    "coordination_ring": {
        "learning_rate": 1.6e-4,
        "gamma": 0.975,
        "clip_eps": 0.069,
        "max_grad_norm": 0.359,
        "gae_lambda": 0.5,
        "vf_coef": 9.33e-3,
        "kl_coeff": 0.156,
        "reward_shaping_horizon": 5e6,
        "num_training_iters": 650,  # Paper value
    },
    "forced_coordination": {
        "learning_rate": 2.77e-4,
        "gamma": 0.972,
        "clip_eps": 0.258,
        "max_grad_norm": 0.295,
        "gae_lambda": 0.6,
        "vf_coef": 0.016,
        "kl_coeff": 0.31,
        "reward_shaping_horizon": 4e6,
        "num_training_iters": 650,  # Paper value
    },
    "counter_circuit": {
        "learning_rate": 2.29e-4,
        "gamma": 0.978,
        "clip_eps": 0.146,
        "max_grad_norm": 0.229,
        "gae_lambda": 0.6,
        "vf_coef": 9.92e-3,
        "kl_coeff": 0.299,
        "reward_shaping_horizon": 5e6,
        "num_training_iters": 650,  # Paper value
    },
}


# Table 4: PBT Hyperparameters (per-layout)
# PBT-specific settings differ from self-play
PAPER_PBT_CONFIGS: Dict[str, Dict[str, Any]] = {
    "cramped_room": {
        "learning_rate": 2e-3,
        "reward_shaping_horizon": 3e6,
        "total_env_steps": 8e6,
    },
    "asymmetric_advantages": {
        "learning_rate": 8e-4,
        "reward_shaping_horizon": 5e6,
        "total_env_steps": 1.1e7,
    },
    "coordination_ring": {
        "learning_rate": 8e-4,
        "reward_shaping_horizon": 4e6,
        "total_env_steps": 5e6,
    },
    "forced_coordination": {
        "learning_rate": 3e-3,
        "reward_shaping_horizon": 7e6,
        "total_env_steps": 8e6,
    },
    "counter_circuit": {
        "learning_rate": 1e-3,
        "reward_shaping_horizon": 4e6,
        "total_env_steps": 6e6,
    },
}

# PBT common parameters
PBT_COMMON_PARAMS = {
    "population_size": 8,
    "ppo_iteration_timesteps": 40000,
    "num_minibatches": 10,
    "minibatch_size": 2000,
    
    # Mutation parameters
    "mutation_prob": 0.33,  # 33% chance of mutation
    "mutation_factor_low": 0.75,
    "mutation_factor_high": 1.25,
    
    # Parameters that can be mutated
    "mutable_params": ["learning_rate", "entropy_coeff", "vf_coef", "gae_lambda"],
    
    # Initial ranges for mutable parameters
    "initial_entropy_coeff": 0.5,
    "initial_vf_coef": 0.1,
}


# PPO_BC configurations (PPO trained with BC partner)
# Uses same hyperparameters as self-play but with BC schedule
PAPER_PPO_BC_CONFIGS: Dict[str, Dict[str, Any]] = {}
for layout in PAPER_LAYOUTS:
    PAPER_PPO_BC_CONFIGS[layout] = {
        **PAPER_PPO_SP_CONFIGS[layout],
        # BC schedule: start with 100% BC, anneal to 0%
        # Format: [(timestep, bc_factor), ...]
        "bc_schedule": [
            (0, 1.0),           # Start with 100% BC partner
            (8e6, 0.0),         # Anneal to 0% over 8M timesteps
            (float('inf'), 0.0) # Stay at 0%
        ],
    }


def get_ppo_sp_config(layout: str, seed: int = 0, **overrides) -> Dict[str, Any]:
    """
    Get PPO self-play configuration for a layout.
    
    Args:
        layout: Layout name (paper name, e.g., 'cramped_room')
        seed: Random seed
        **overrides: Additional parameter overrides
        
    Returns:
        Configuration dictionary
    """
    if layout not in PAPER_PPO_SP_CONFIGS:
        raise ValueError(f"Unknown layout: {layout}. Available: {list(PAPER_PPO_SP_CONFIGS.keys())}")
    
    env_layout = LAYOUT_TO_ENV.get(layout, layout)
    
    config = {
        **PAPER_COMMON_PARAMS,
        **PAPER_PPO_SP_CONFIGS[layout],
        "layout_name": env_layout,
        "seed": seed,
        "experiment_name": f"ppo_sp_{layout}_seed{seed}",
        "bc_schedule": [(0, 0.0), (float('inf'), 0.0)],  # No BC partner
    }
    
    # Convert num_training_iters to total_timesteps
    # Each iteration = train_batch_size timesteps
    config["total_timesteps"] = int(
        config["num_training_iters"] * config["train_batch_size"]
    )
    
    config.update(overrides)
    return config


def get_pbt_config(layout: str, **overrides) -> Dict[str, Any]:
    """
    Get PBT configuration for a layout.
    
    Args:
        layout: Layout name (paper name)
        **overrides: Additional parameter overrides
        
    Returns:
        Configuration dictionary
    """
    if layout not in PAPER_PBT_CONFIGS:
        raise ValueError(f"Unknown layout: {layout}. Available: {list(PAPER_PBT_CONFIGS.keys())}")
    
    env_layout = LAYOUT_TO_ENV.get(layout, layout)
    
    config = {
        **PAPER_COMMON_PARAMS,
        **PBT_COMMON_PARAMS,
        **PAPER_PBT_CONFIGS[layout],
        "layout_name": env_layout,
        "experiment_name": f"pbt_{layout}",
    }
    
    config.update(overrides)
    return config


def get_ppo_bc_config(layout: str, seed: int = 0, bc_model_dir: str = None, **overrides) -> Dict[str, Any]:
    """
    Get PPO_BC configuration for a layout.
    
    Args:
        layout: Layout name (paper name)
        seed: Random seed
        bc_model_dir: Path to BC model directory
        **overrides: Additional parameter overrides
        
    Returns:
        Configuration dictionary
    """
    if layout not in PAPER_PPO_BC_CONFIGS:
        raise ValueError(f"Unknown layout: {layout}. Available: {list(PAPER_PPO_BC_CONFIGS.keys())}")
    
    env_layout = LAYOUT_TO_ENV.get(layout, layout)
    
    config = {
        **PAPER_COMMON_PARAMS,
        **PAPER_PPO_BC_CONFIGS[layout],
        "layout_name": env_layout,
        "seed": seed,
        "experiment_name": f"ppo_bc_{layout}_seed{seed}",
        "bc_model_dir": bc_model_dir,
    }
    
    # Convert num_training_iters to total_timesteps
    config["total_timesteps"] = int(
        config["num_training_iters"] * config["train_batch_size"]
    )
    
    config.update(overrides)
    return config


def print_config_summary():
    """Print a summary of all paper configurations."""
    print("="*80)
    print("Paper PPO Self-Play Configurations")
    print("="*80)
    
    headers = ["Layout", "LR", "Gamma", "Clip", "GradClip", "Lambda", "VF", "RewHorizon", "Iters"]
    row_format = "{:<20}" + "{:<10}" * (len(headers) - 1)
    
    print(row_format.format(*headers))
    print("-"*80)
    
    for layout in PAPER_LAYOUTS:
        cfg = PAPER_PPO_SP_CONFIGS[layout]
        print(row_format.format(
            layout,
            f"{cfg['learning_rate']:.2e}",
            f"{cfg['gamma']:.3f}",
            f"{cfg['clip_eps']:.3f}",
            f"{cfg['max_grad_norm']:.3f}",
            f"{cfg['gae_lambda']:.1f}",
            f"{cfg['vf_coef']:.2e}",
            f"{cfg['reward_shaping_horizon']:.0e}",
            str(cfg['num_training_iters']),
        ))
    
    print("\n")
    print("="*80)
    print("Paper PBT Configurations")
    print("="*80)
    
    headers = ["Layout", "LR", "RewHorizon", "TotalSteps"]
    row_format = "{:<20}" + "{:<15}" * (len(headers) - 1)
    
    print(row_format.format(*headers))
    print("-"*80)
    
    for layout in PAPER_LAYOUTS:
        cfg = PAPER_PBT_CONFIGS[layout]
        print(row_format.format(
            layout,
            f"{cfg['learning_rate']:.2e}",
            f"{cfg['reward_shaping_horizon']:.0e}",
            f"{cfg['total_env_steps']:.0e}",
        ))


if __name__ == "__main__":
    print_config_summary()

