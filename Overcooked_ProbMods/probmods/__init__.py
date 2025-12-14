"""
Overcooked ProbMods - Probabilistic Programming for Overcooked AI

This package provides PPL-based versions of imitation learning and
reinforcement learning algorithms for the Overcooked environment.
"""

__version__ = "0.1.0"

# Lazy imports to avoid requiring all dependencies at import time
def __getattr__(name):
    if name == "models":
        from . import models
        return models
    elif name == "data":
        from . import data
        return data
    elif name == "analysis":
        from . import analysis
        return analysis
    elif name == "inference":
        from . import inference
        return inference
    elif name == "utils":
        from . import utils
        return utils
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "__version__",
    "models",
    "data",
    "analysis",
    "inference",
    "utils",
]
