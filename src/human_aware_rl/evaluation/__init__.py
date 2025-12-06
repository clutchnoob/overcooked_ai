"""
Evaluation utilities for Overcooked AI agents.
"""

from human_aware_rl.evaluation.evaluate_all import (
    evaluate_bc_self_play,
    evaluate_ppo_self_play,
    evaluate_bc_with_ppo,
    evaluate_agent_with_human_proxy,
    run_all_evaluations,
)

from human_aware_rl.evaluation.evaluate_paper import (
    evaluate_paper_experiments,
    EVALUATION_CONFIGS,
)

__all__ = [
    "evaluate_bc_self_play",
    "evaluate_ppo_self_play",
    "evaluate_bc_with_ppo",
    "evaluate_agent_with_human_proxy",
    "run_all_evaluations",
    "evaluate_paper_experiments",
    "EVALUATION_CONFIGS",
]

