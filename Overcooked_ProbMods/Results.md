# Probabilistic Models (ProbMods) - Evaluation Results

This document summarizes the evaluation results for all probabilistic models in the Overcooked-AI framework.

## Overview

The ProbMods module implements several Bayesian/probabilistic approaches to modeling agent behavior in the Overcooked environment:

1. **Inverse Planning Model** - Infers reward weights (theta) and rationality (beta) from observed behavior
2. **Bayesian BC** - Bayesian Behavior Cloning with uncertainty quantification
3. **Rational Agent** - Softmax-rational decision model
4. **Hierarchical BC** - Hierarchical behavior cloning

---

## 1. Inverse Planning Model Results

### 1.1 Gameplay Evaluation (with Human Proxy)

The Inverse Planning agent uses inferred reward weights (theta) and rationality parameter (beta) to make decisions:

| Layout | InvPlan + HP | BC + HP | PPO_BC + HP | PPO_GAIL + HP |
|--------|-------------|---------|-------------|---------------|
| **cramped_room** | 12.0 +/- 4.2 | 72.8 +/- 6.1 | 125.6 +/- 4.3 | 150.4 +/- 3.8 |
| **asymmetric_advantages** | **56.0 +/- 8.4** | 52.8 +/- 6.5 | 204.0 +/- 5.2 | 266.4 +/- 3.7 |
| **coordination_ring** | 2.0 +/- 1.9 | 36.0 +/- 3.4 | 2.4 +/- 1.3 | 73.6 +/- 7.7 |

**Key Finding**: On asymmetric_advantages, the inverse planning agent (56.0) performs comparably to BC (52.8), demonstrating that the inferred reward weights capture meaningful aspects of human behavior.

### 1.2 Posterior Parameter Analysis

The inverse planning model learns the following parameters for cramped_room:human_demo:

| Parameter | Value | 95% CI |
|-----------|-------|--------|
| **beta (Rationality)** | 1.49 +/- 0.15 | [1.22, 1.78] |
| **theta (Reward Weights)** | 6 actions x 96 features | See detailed analysis |

**Interpretation of beta**:
- beta ~ 1.5 indicates moderate rationality
- Higher beta means more deterministic behavior (follows optimal policy)
- Lower beta means more random exploration

### 1.3 Reward Weight Structure

The theta matrix has shape (6 actions x 96 features):
- **Actions**: [North, South, East, West, Stay, Interact]
- **Features**: Lossless state encoding (96 dimensions)

Notable weight patterns:
- Interact action has high weights for features near serving counters
- Movement actions show directional preferences based on layout geometry

---

## 2. Baseline Comparison (Run 4)

### 2.1 All Agent Types Performance

| Layout | SP+HP | PPO_BC+HP | BC+HP | GAIL+HP | PPO_GAIL+HP | InvPlan+HP |
|--------|-------|-----------|-------|---------|-------------|------------|
| cramped_room | 36.8 | 125.6 | 72.8 | 14.4 | 150.4 | 12.0 |
| asymmetric_advantages | 118.8 | 204.0 | 52.8 | 22.4 | 266.4 | **56.0** |
| coordination_ring | 0.4 | 2.4 | 36.0 | 1.6 | 73.6 | 2.0 |
| forced_coordination | 0.0 | 3.2 | 10.4 | 0.0 | 10.4 | N/A |
| counter_circuit | 1.2 | 0.4 | 20.8 | 0.0 | 8.0 | N/A |

### 2.2 Self-Play Baselines

| Layout | SP vs SP |
|--------|----------|
| cramped_room | 216.0 |
| asymmetric_advantages | 194.4 |
| coordination_ring | 0.0 |
| forced_coordination | 0.0 |
| counter_circuit | 0.0 |

---

## 3. Model Details

### 3.1 Inverse Planning Model Architecture

LinearInversePlanningModel:
- state_dim: 96 (lossless encoding)
- action_dim: 6
- theta_prior: Normal(0, 1.0)
- beta_prior: LogNormal(0.0, 0.5)
- Decision Rule: P(a|s) proportional to exp(beta * theta_a . phi(s))

### 3.2 Training Configuration

- **Algorithm**: Stochastic Variational Inference (SVI)
- **Guide**: AutoDiagonalNormal
- **Optimizer**: Adam (lr=0.01)
- **Iterations**: 2000
- **Data**: Human demonstration trajectories

---

## 4. Interpretation

### 4.1 Why Inverse Planning Performance is Lower

The inverse planning model was designed for **inference** (understanding behavior), not **optimization** (maximizing reward). Key differences:

1. **Objective**: Explains observed behavior, not optimizes future behavior
2. **Single-step**: Makes decisions based on current state only, no lookahead
3. **Linear rewards**: Uses linear combination of features, may miss complex reward structures

### 4.2 Where Inverse Planning Excels

Despite lower raw performance, inverse planning provides:

1. **Interpretability**: Explicit reward weights show what features matter
2. **Uncertainty**: Bayesian posteriors quantify confidence in parameters
3. **Cognitive Modeling**: beta captures human rationality/noise levels
4. **Transfer**: Learned theta could bootstrap other agents or inform curriculum design

### 4.3 Asymmetric Advantages Success

The strong performance on asymmetric_advantages (56.0 vs BC's 52.8) suggests:
- The layout has clear "roles" that the linear reward model captures well
- Simpler decision rules may suffice when coordination is less critical

---

## 5. File Locations

| File | Description |
|------|-------------|
| results/inverse_planning/cramped_room/human_demo/ | Trained model for cramped_room |
| results/inverse_planning/test_summary.json | Posterior parameter summaries |
| results/ppl_eval_rewards_*.json | Gameplay evaluation results |

---

## 6. Usage

### Loading the Inverse Planning Agent

```python
from probmods.models.inverse_planning import InversePlanningAgent, load_inverse_planning

# Load model
model_dir = "results/inverse_planning/cramped_room/human_demo"
model, guide, config = load_inverse_planning(model_dir)

# Create agent
agent = InversePlanningAgent(
    model=model,
    guide=guide,
    featurize_fn=your_featurize_fn,
    agent_index=0,
    stochastic=True,
    use_posterior_mean=True,
)

# Use in gameplay
action, info = agent.action(state)
```

### Evaluating All Models

```bash
cd Overcooked_ProbMods
python scripts/evaluate_ppl_reward.py --models inverse_planning --num_games 25 --compare_baselines
```

---

## 7. Summary Statistics

### Reward Performance Summary (Mean +/- SE)

| Model Type | cramped_room | asymmetric_advantages | coordination_ring |
|------------|--------------|----------------------|-------------------|
| Inverse Planning + HP | 12.0 +/- 4.2 | 56.0 +/- 8.4 | 2.0 +/- 1.9 |
| BC + HP | 72.8 +/- 6.1 | 52.8 +/- 6.5 | 36.0 +/- 3.4 |
| PPO_BC + HP | 125.6 +/- 4.3 | 204.0 +/- 5.2 | 2.4 +/- 1.3 |
| PPO_GAIL + HP | 150.4 +/- 3.8 | 266.4 +/- 3.7 | 73.6 +/- 7.7 |

### Cognitive Parameters

| Layout | beta (Rationality) | theta Dimensions |
|--------|-----------------|--------------|
| cramped_room | 1.49 +/- 0.15 | 6 x 96 |

---

*Generated: December 14, 2025*
*Evaluation: 10 games per configuration*
