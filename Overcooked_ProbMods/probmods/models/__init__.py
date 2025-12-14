"""
Probabilistic models for Overcooked AI.

Available models:
- BayesianBCModel, BayesianBCTrainer: Bayesian Behavioral Cloning
- RationalAgentModel, RationalAgentTrainer: Softmax-rational agent
- HierarchicalBCModel, HierarchicalBCTrainer: Goal-conditioned BC
- BayesianGAILTrainer: Bayesian GAIL
- BayesianPPOBCTrainer: Bayesian PPO with BC initialization
- BayesianPPOGAILTrainer: Bayesian PPO + GAIL
"""

# Lazy imports to handle missing pyro gracefully
_model_imports = {
    "BayesianBCModel": "bayesian_bc",
    "BayesianBCTrainer": "bayesian_bc",
    "BayesianBCConfig": "bayesian_bc",
    "train_bayesian_bc": "bayesian_bc",
    "RationalAgentModel": "rational_agent",
    "RationalAgentTrainer": "rational_agent",
    "RationalAgentConfig": "rational_agent",
    "train_rational_agent": "rational_agent",
    "HierarchicalBCModel": "hierarchical_bc",
    "HierarchicalBCTrainer": "hierarchical_bc",
    "HierarchicalBCConfig": "hierarchical_bc",
    "train_hierarchical_bc": "hierarchical_bc",
    "BayesianGAILTrainer": "bayesian_gail",
    "BayesianGAILConfig": "bayesian_gail",
    "train_bayesian_gail": "bayesian_gail",
    "BayesianPPOBCTrainer": "bayesian_ppo_bc",
    "BayesianPPOBCConfig": "bayesian_ppo_bc",
    "train_bayesian_ppo_bc": "bayesian_ppo_bc",
    "BayesianPPOGAILTrainer": "bayesian_ppo_gail",
    "BayesianPPOGAILConfig": "bayesian_ppo_gail",
    "train_bayesian_ppo_gail": "bayesian_ppo_gail",
}


def __getattr__(name):
    if name in _model_imports:
        module_name = _model_imports[name]
        import importlib
        module = importlib.import_module(f".{module_name}", __name__)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = list(_model_imports.keys())
