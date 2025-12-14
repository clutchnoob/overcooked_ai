"""
Utility functions for Overcooked ProbMods.
"""

import torch
import numpy as np
import random
from pathlib import Path
from typing import Optional, Union
import logging

logger = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(device: Optional[str] = None) -> torch.device:
    """Get torch device, defaulting to CUDA > MPS > CPU."""
    if device is not None:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_device_str(device: Optional[str] = None) -> str:
    """Get device string (for inline use in configs)."""
    if device is not None:
        return device
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def ensure_dir(path: Union[str, Path]) -> Path:
    """Ensure directory exists, creating if necessary."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_checkpoint(
    state: dict,
    path: Union[str, Path],
    filename: str = "checkpoint.pt"
) -> Path:
    """Save training checkpoint."""
    path = ensure_dir(path)
    filepath = path / filename
    torch.save(state, filepath)
    logger.info(f"Saved checkpoint to {filepath}")
    return filepath


def load_checkpoint(
    path: Union[str, Path],
    filename: str = "checkpoint.pt",
    device: Optional[str] = None
) -> dict:
    """Load training checkpoint."""
    filepath = Path(path) / filename
    device = get_device(device)
    state = torch.load(filepath, map_location=device)
    logger.info(f"Loaded checkpoint from {filepath}")
    return state


def count_parameters(model: torch.nn.Module) -> int:
    """Count trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def setup_logging(
    level: str = "INFO",
    log_file: Optional[Union[str, Path]] = None
) -> None:
    """Set up logging configuration."""
    handlers = [logging.StreamHandler()]
    
    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=handlers,
    )


__all__ = [
    "set_seed",
    "get_device",
    "get_device_str",
    "ensure_dir",
    "save_checkpoint",
    "load_checkpoint",
    "count_parameters",
    "setup_logging",
]
