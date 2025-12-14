"""
Data utilities for Overcooked probabilistic models.

- Load human demonstrations (train/test) with featurization
- Convert to torch tensors on the desired device
- Simple batching helper
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Iterator

import numpy as np
import torch

from human_aware_rl.human.process_dataframes import get_human_human_trajectories
from human_aware_rl.static import (
    CLEAN_2019_HUMAN_DATA_TRAIN,
    CLEAN_2019_HUMAN_DATA_TEST,
)


@dataclass
class DataConfig:
    layout_name: str = "cramped_room"
    data_path: str = CLEAN_2019_HUMAN_DATA_TRAIN
    check_trajectories: bool = False
    featurize_states: bool = True
    dataset: str = "train"  # "train" or "test"


def load_human_data(config: DataConfig) -> Tuple[np.ndarray, np.ndarray]:
    """Load human demonstrations for a layout."""
    data_path = config.data_path
    if config.dataset == "test":
        data_path = CLEAN_2019_HUMAN_DATA_TEST

    params = {
        "layouts": [config.layout_name],
        "check_trajectories": config.check_trajectories,
        "featurize_states": config.featurize_states,
        "data_path": data_path,
    }
    processed = get_human_human_trajectories(**params, silent=True)
    states, actions = [], []
    for ep_states, ep_actions in zip(processed["ep_states"], processed["ep_actions"]):
        for s, a in zip(ep_states, ep_actions):
            states.append(s.flatten())
            actions.append(int(a))
    return np.array(states), np.array(actions)


def _get_device() -> str:
    """Get best available device (CUDA > MPS > CPU)."""
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def to_torch(states: np.ndarray, actions: np.ndarray, device: str | None = None):
    device = device or _get_device()
    states_t = torch.tensor(states, dtype=torch.float32, device=device)
    actions_t = torch.tensor(actions, dtype=torch.long, device=device)
    return states_t, actions_t


def batchify(states: np.ndarray, actions: np.ndarray, batch_size: int) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    indices = np.arange(len(states))
    np.random.shuffle(indices)
    for start in range(0, len(states), batch_size):
        idx = indices[start : start + batch_size]
        yield states[idx], actions[idx]
