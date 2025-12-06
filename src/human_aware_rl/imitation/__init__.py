"""
Imitation learning module for Overcooked AI.

This module provides behavior cloning (BC) functionality for training
agents from human demonstration data.

Main components:
- behavior_cloning: PyTorch-based BC training (recommended)
- bc_agent: Agent wrapper for BC models
- behavior_cloning_tf2: Legacy TensorFlow implementation (deprecated)
"""

# Import PyTorch BC components (recommended)
try:
    from human_aware_rl.imitation.behavior_cloning import (
        BC_SAVE_DIR,
        BCModel,
        BCLSTMModel,
        build_bc_model,
        get_bc_params,
        load_bc_model,
        load_data,
        save_bc_model,
        train_bc_model,
        evaluate_bc_model,
    )
    from human_aware_rl.imitation.bc_agent import (
        BCAgent,
        BehaviorCloningPolicy,
    )
    PYTORCH_BC_AVAILABLE = True
except ImportError:
    PYTORCH_BC_AVAILABLE = False

__all__ = [
    "BC_SAVE_DIR",
    "BCModel",
    "BCLSTMModel",
    "BCAgent",
    "BehaviorCloningPolicy",
    "build_bc_model",
    "get_bc_params",
    "load_bc_model",
    "load_data",
    "save_bc_model",
    "train_bc_model",
    "evaluate_bc_model",
    "PYTORCH_BC_AVAILABLE",
]

