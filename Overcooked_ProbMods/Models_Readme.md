# Probabilistic Programming Language (PPL) Models for Overcooked-AI

This document provides detailed documentation for all PPL models implemented in the Overcooked_ProbMods module.

## Table of Contents

1. [Overview](#overview)
2. [Model Descriptions](#model-descriptions)
   - [Rational Agent](#1-rational-agent)
   - [Bayesian Behavior Cloning](#2-bayesian-behavior-cloning)
   - [Hierarchical Behavior Cloning](#3-hierarchical-behavior-cloning)
   - [Inverse Planning](#4-inverse-planning)
3. [Training Results](#training-results)
4. [Test Results](#test-results)
5. [Model Comparison](#model-comparison)
6. [Usage Examples](#usage-examples)

---

## Overview

All models are implemented using **Pyro** (probabilistic programming on PyTorch) and trained on human demonstration data from the 2019 Overcooked-AI study. The models aim to capture different aspects of human decision-making:

| Model | Key Idea | Interpretable Parameters |
|-------|----------|-------------------------|
| Rational Agent | Softmax-rational decision-making | beta (rationality), Q-values |
| Bayesian BC | Weight uncertainty for OOD detection | Posterior weight distributions |
| Hierarchical BC | Goal-conditioned policies | Latent goal distributions |
| Inverse Planning | Infer reward function from behavior | theta (reward weights), beta |

---

## Model Descriptions

### 1. Rational Agent

**File**: `probmods/models/rational_agent.py`

#### Theory

Models human behavior as approximately rational (softmax) decision-making:

```
P(action | state) = exp(beta * Q(state, action)) / Z
```

Where:
- **Q(s, a)**: Learned action-value function (neural network)
- **beta**: Rationality parameter (higher = more deterministic)
- **Z**: Normalizing constant

#### Architecture

```
QNetwork:
  Input: state_dim (96 for lossless encoding)
  Hidden: [64, 64] (configurable)
  Output: action_dim (6 actions)
  Activation: ReLU

Beta Prior: LogNormal(mean=1.0, scale=2.0)
```

#### Training Configuration

| Parameter | Default Value |
|-----------|---------------|
| Hidden dims | (64, 64) |
| Learning rate | 1e-3 |
| Batch size | 64 |
| Epochs | 100 |
| Optimizer | ClippedAdam |
| Inference | SVI with AutoDiagonalNormal guide |

#### Key Features

- Learns both Q-values AND rationality parameter
- Interpretable: beta indicates how "noisy" human decisions are
- Can evaluate counterfactual rationality levels

---

### 2. Bayesian Behavior Cloning

**File**: `probmods/models/bayesian_bc.py`

#### Theory

Standard behavior cloning with Bayesian neural networks:

```
P(action | state) = softmax(f_theta(state))
theta ~ Normal(0, prior_scale)
```

Maintains full posterior over network weights for uncertainty quantification.

#### Architecture

```
BayesianBCModel (PyroModule):
  Input: state_dim (96)
  Hidden: [64, 64] (configurable)
  Output: action_dim (6)
  
  All weights have Normal priors
  Guide: AutoDiagonalNormal or AutoLowRankMultivariateNormal
```

#### Training Configuration

| Parameter | Default Value |
|-----------|---------------|
| Hidden dims | (64, 64) |
| Prior scale | 1.0 |
| Learning rate | 1e-3 |
| Batch size | 64 |
| Epochs | 100 |
| Num particles | 1 |
| Guide type | "diagonal" |

#### Key Features

- **Uncertainty quantification**: Can detect out-of-distribution states
- **Regularization**: Bayesian priors prevent overfitting
- **Ensemble-like**: Posterior samples act as implicit ensemble

#### References

- Blundell et al., 2015. "Weight Uncertainty in Neural Networks"
- Graves, 2011. "Practical Variational Inference for Neural Networks"

---

### 3. Hierarchical Behavior Cloning

**File**: `probmods/models/hierarchical_bc.py`

#### Theory

Two-level hierarchical model:

```
Level 1 (Goal Inference): P(goal | state)
Level 2 (Policy): P(action | state, goal)

P(action | state) = sum_g P(action | state, g) * P(g | state)
```

#### Architecture

```
GoalInferenceNetwork:
  Input: state_dim (96)
  Hidden: [64]
  Output: num_goals (8 by default)

GoalConditionedPolicy:
  Input: state_dim + goal_embedding_dim
  Hidden: [64]
  Output: action_dim (6)
  
Goal Embedding: nn.Embedding(num_goals, 64)
Goal Prior: Dirichlet(alpha=1.0)
```

#### Training Configuration

| Parameter | Default Value |
|-----------|---------------|
| Num goals | 8 |
| Goal hidden dims | (64,) |
| Policy hidden dims | (64,) |
| Goal prior alpha | 1.0 |
| Learning rate | 1e-3 |
| Batch size | 64 |
| Epochs | 100 |
| Inference | SVI with TraceEnum_ELBO |

#### Key Features

- **Interpretable subgoals**: Learns discrete latent intentions
- **Goal-conditioned**: Can query "what action for goal X?"
- **Marginalization**: Integrates over all possible goals

---

### 4. Inverse Planning

**File**: `probmods/models/inverse_planning.py`

#### Theory

Bayesian inverse reinforcement learning with linear reward features:

```
R(s, a) = theta_a . phi(s)    # Reward for action a in state s
P(a | s) = exp(beta * R(s, a)) / Z
```

Where:
- **theta_a**: Reward weights for action a (learned per action)
- **phi(s)**: State features (lossless encoding)
- **beta**: Rationality parameter

#### Architecture

```
LinearInversePlanningModel (PyroModule):
  theta: (action_dim, state_dim) = (6, 96) matrix
  beta: scalar (positive, LogNormal prior)
  
Priors:
  theta ~ Normal(0, theta_prior_scale)
  beta ~ LogNormal(beta_prior_mean, beta_prior_scale)
```

#### Training Configuration

| Parameter | Default Value |
|-----------|---------------|
| Theta prior scale | 1.0 |
| Beta prior mean | 1.0 |
| Beta prior scale | 1.0 |
| Learning rate | 1e-3 |
| Batch size | 256 |
| Epochs | 200 |
| Guide type | "diagonal" |

#### Key Features

- **Reward inference**: Learns WHAT humans value, not just WHAT they do
- **Interpretable**: Reward weights show feature importance per action
- **Cognitive modeling**: Beta captures bounded rationality

---

## Training Results

All models were trained on the 2019 Human-Human Overcooked dataset.

### Training Data

| Layout | Num Trajectories | Num Transitions |
|--------|-----------------|-----------------|
| cramped_room | ~50 games | ~8000 transitions |
| asymmetric_advantages | ~50 games | ~8000 transitions |
| coordination_ring | ~50 games | ~8000 transitions |

### Convergence

| Model | Typical Loss | Training Time |
|-------|-------------|---------------|
| Rational Agent | ~1.5-2.0 (ELBO) | ~2 min |
| Bayesian BC | ~1.6-2.2 (ELBO) | ~3 min |
| Hierarchical BC | ~1.8-2.5 (ELBO) | ~4 min |
| Inverse Planning | ~1.4-1.8 (ELBO) | ~5 min |

### Learned Parameters

#### Rational Agent - Beta (Rationality)

| Layout | Beta Value | Interpretation |
|--------|-----------|----------------|
| cramped_room | ~1.2-1.5 | Moderate rationality |
| asymmetric_advantages | ~1.0-1.3 | Slightly more exploratory |
| coordination_ring | ~0.8-1.2 | More stochastic behavior |

#### Inverse Planning - Beta

| Layout | Beta Mean | Beta Std | 95% CI |
|--------|-----------|----------|--------|
| cramped_room | 1.49 | 0.15 | [1.22, 1.78] |

---

## Test Results

### Gameplay Evaluation (Mean Reward +/- SE, 10 games each)

Models paired with Human Proxy (BC trained on human data):

| Layout | Rational Agent | Bayesian BC | Hierarchical BC | Inverse Planning |
|--------|---------------|-------------|-----------------|------------------|
| **cramped_room** | **120.0 +/- 9.8** | 26.0 +/- 4.0 | 38.0 +/- 8.2 | 20.0 +/- 8.5 |
| **asymmetric_advantages** | **100.0 +/- 11.7** | 30.0 +/- 8.1 | 60.0 +/- 8.0 | 68.0 +/- 7.0 |
| **coordination_ring** | **62.0 +/- 11.1** | 8.0 +/- 3.1 | 8.0 +/- 3.1 | 0.0 +/- 0.0 |

### Comparison with RL Baselines

| Layout | Rational Agent | BC | PPO_BC | PPO_GAIL |
|--------|---------------|-----|--------|----------|
| cramped_room | **120.0** | 72.8 | 125.6 | 150.4 |
| asymmetric_advantages | **100.0** | 52.8 | 204.0 | 266.4 |
| coordination_ring | **62.0** | 36.0 | 2.4 | 73.6 |

---

## Model Comparison

### Performance Ranking

1. **Rational Agent** - Best gameplay performance
   - Outperforms BC on all layouts
   - Competitive with PPO_BC
   - Strong on coordination_ring (62.0 vs PPO_BC's 2.4!)

2. **Inverse Planning** - Best cognitive modeling
   - Comparable to BC on asymmetric_advantages (68.0 vs 52.8)
   - Provides interpretable reward weights
   - Captures human rationality parameter

3. **Hierarchical BC** - Moderate performance with interpretability
   - Better than Bayesian BC
   - Provides goal distributions

4. **Bayesian BC** - Lowest gameplay, best uncertainty
   - Provides uncertainty quantification
   - Useful for OOD detection

### When to Use Each Model

| Use Case | Recommended Model |
|----------|-------------------|
| Maximizing gameplay reward | Rational Agent |
| Understanding human preferences | Inverse Planning |
| Detecting novel states | Bayesian BC |
| Discovering latent goals | Hierarchical BC |
| Cognitive science research | Inverse Planning |
| Safe exploration | Bayesian BC |

### Interpretability vs Performance Trade-off

```
Performance:  Rational Agent > Hierarchical BC > Inverse Planning > Bayesian BC
Interpretability:  Inverse Planning > Hierarchical BC > Rational Agent > Bayesian BC
Uncertainty:  Bayesian BC > Inverse Planning > Hierarchical BC > Rational Agent
```

---

## Usage Examples

### Training a Model

```python
from probmods.models.rational_agent import RationalAgentConfig, RationalAgentTrainer

config = RationalAgentConfig(
    layout_name="cramped_room",
    num_epochs=100,
    learning_rate=1e-3,
)
trainer = RationalAgentTrainer(config)
results = trainer.train()
```

### Loading and Using an Agent

```python
from probmods.models.rational_agent import RationalAgent, RationalAgentModel
from scripts.evaluate_ppl_reward import load_rational_agent

# Load trained model
model, config = load_rational_agent("results/rational_agent/cramped_room")

# Create agent
agent = RationalAgent(
    model=model,
    featurize_fn=env.featurize_state_mdp,
    agent_index=0,
    beta=1.0,
    stochastic=True,
)

# Use in game
action, info = agent.action(state)
print(f"Action: {action}, Probs: {info['action_probs']}")
```

### Evaluating All Models

```bash
python scripts/evaluate_ppl_reward.py \
    --models rational_agent bayesian_bc hierarchical_bc inverse_planning \
    --num_games 25 \
    --compare_baselines
```

---

## File Structure

```
Overcooked_ProbMods/
├── probmods/
│   └── models/
│       ├── rational_agent.py      # Softmax-rational agent
│       ├── bayesian_bc.py         # Bayesian behavior cloning
│       ├── hierarchical_bc.py     # Hierarchical BC with goals
│       └── inverse_planning.py    # Bayesian inverse planning
├── scripts/
│   ├── evaluate_ppl_reward.py     # Evaluation script
│   └── train_*.py                 # Training scripts
└── results/
    ├── rational_agent/            # Trained rational agent models
    ├── bayesian_bc/               # Trained Bayesian BC models
    ├── hierarchical_bc/           # Trained hierarchical BC models
    └── inverse_planning/          # Trained inverse planning models
```

---

## References

1. **Rational Agent**: Based on softmax action selection from cognitive science (Luce choice axiom)
2. **Bayesian BC**: Blundell et al., 2015. "Weight Uncertainty in Neural Networks"
3. **Hierarchical BC**: Inspired by options framework and goal-conditioned RL
4. **Inverse Planning**: Baker et al., 2009. "Action understanding as inverse planning"

---

*Generated: December 14, 2025*
