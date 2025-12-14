# Overcooked ProbMods

**Probabilistic Programming Language (PPL) models for understanding cooperative human behavior in the Overcooked AI environment.**

This module provides Bayesian and probabilistic versions of imitation learning and reinforcement learning algorithms, enabling:
- **Uncertainty quantification**: Know when the model is confident vs. uncertain
- **Interpretability**: Extract meaningful parameters like rationality and goals
- **Principled comparison**: Compare models using probabilistic metrics

---

## Table of Contents

1. [Motivation](#motivation)
2. [Theoretical Background](#theoretical-background)
3. [Models Overview](#models-overview)
4. [Detailed Model Documentation](#detailed-model-documentation)
   - [Bayesian Behavioral Cloning](#1-bayesian-behavioral-cloning)
   - [Rational Agent Model](#2-rational-agent-model)
   - [Hierarchical Behavioral Cloning](#3-hierarchical-behavioral-cloning)
   - [Bayesian GAIL](#4-bayesian-gail)
   - [Bayesian PPO+BC](#5-bayesian-ppobc)
   - [Bayesian PPO+GAIL](#6-bayesian-ppogail)
5. [Installation](#installation)
6. [Usage](#usage)
7. [Project Structure](#project-structure)
8. [Analysis Tools](#analysis-tools)
9. [References](#references)

---

## Motivation

Standard deep learning models for imitation learning (like behavioral cloning) provide point estimates—a single "best guess" for what action to take. But when modeling human behavior, especially in cooperative settings like Overcooked, we need more:

1. **Uncertainty**: Humans are variable. A model should express uncertainty when human behavior is inherently stochastic or when it hasn't seen similar states.

2. **Interpretability**: We want to understand *why* humans make certain decisions. Are they being rational? What goals are they pursuing?

3. **Robustness**: Point estimates can overfit. Bayesian approaches provide natural regularization.

4. **Model Comparison**: Probabilistic models allow principled comparison using metrics like marginal likelihood and KL divergence.

This package bridges **cognitive science** approaches (rational agents, goal inference) with **modern deep learning** (neural networks, PPO) using **probabilistic programming** (Pyro).

---

## Theoretical Background

### Probabilistic Programming with Pyro

[Pyro](https://pyro.ai/) is a probabilistic programming language built on PyTorch. It allows us to:
- Define probabilistic models with latent variables
- Perform inference using variational methods (SVI)
- Maintain uncertainty over model parameters

Key concepts used in this package:
- **PyroModule**: Neural networks with probabilistic weights
- **PyroSample**: Declares a random variable with a prior distribution
- **AutoDiagonalNormal**: Automatic variational guide with diagonal Gaussian posterior
- **SVI (Stochastic Variational Inference)**: Scalable approximate inference
- **ELBO (Evidence Lower Bound)**: The objective we maximize

### Bayesian Neural Networks

Instead of learning point estimates for weights θ*, we learn a posterior distribution p(θ|D):

```
p(θ|D) ∝ p(D|θ) · p(θ)
         ↑         ↑
     likelihood   prior
```

For prediction, we marginalize over the posterior:
```
p(a|s) = ∫ p(a|s,θ) p(θ|D) dθ
```

This integral is approximated by sampling from the posterior.

### Softmax-Rational Agents

From cognitive science and behavioral economics, humans are modeled as "noisily rational"—they tend toward optimal actions but with some randomness:

```
π(a|s) ∝ exp(β · Q(s,a))
```

Where:
- Q(s,a) is the value of action a in state s
- β is the **rationality parameter** (inverse temperature)
  - β → 0: random behavior
  - β → ∞: deterministic optimal behavior
  - β ≈ 1-5: typical human-like behavior

### Hierarchical Models and Goal Inference

Human behavior is often hierarchical—we pursue high-level goals that determine low-level actions:

```
p(a|s) = Σ_g p(a|s,g) · p(g|s)
            ↑           ↑
     goal-conditioned  goal
         policy       inference
```

This allows us to ask: "What goal is this human pursuing?"

---

## Models Overview

| Model | File | Inspiration | Key Idea |
|-------|------|-------------|----------|
| **Bayesian BC** | `bayesian_bc.py` | Bayesian deep learning | Posterior over policy weights |
| **Rational Agent** | `rational_agent.py` | Cognitive science / economics | Softmax-rational with learnable β |
| **Hierarchical BC** | `hierarchical_bc.py` | Hierarchical RL / goal inference | Latent goals + goal-conditioned policy |
| **Bayesian GAIL** | `bayesian_gail.py` | Adversarial IL + uncertainty | Bayesian discriminator |
| **Bayesian PPO+BC** | `bayesian_ppo_bc.py` | Safe RL / uncertainty-aware RL | Bayesian actor-critic with BC anchor |
| **Bayesian PPO+GAIL** | `bayesian_ppo_gail.py` | Inverse RL + uncertainty | Full Bayesian IRL pipeline |

---

## Detailed Model Documentation

### 1. Bayesian Behavioral Cloning

**File**: `probmods/models/bayesian_bc.py`

#### What It Does

Learns a policy π(a|s) by imitating human demonstrations, but maintains a **posterior distribution over the policy's neural network weights** instead of point estimates.

#### Theoretical Inspiration

- **Blundell et al. (2015)**: "Weight Uncertainty in Neural Networks" - Introduced practical variational inference for neural network weights
- **Graves (2011)**: "Practical Variational Inference for Neural Networks" - Showed how to scale Bayesian inference to large networks
- **Gal & Ghahramani (2016)**: Dropout as Bayesian approximation

#### Architecture

```
Input: State s ∈ ℝ^96 (Overcooked featurized state)
  ↓
[Linear(96 → 64)] → ReLU    # Weights sampled from posterior
  ↓
[Linear(64 → 64)] → ReLU    # Weights sampled from posterior
  ↓
[Linear(64 → 6)]            # Output logits for 6 actions
  ↓
Output: Action distribution π(a|s)
```

Each weight matrix W has a prior and learned posterior:
```python
W ~ Normal(0, prior_scale)           # Prior
W ~ Normal(μ_W, σ_W)                 # Learned posterior (via guide)
```

#### Key Features

1. **Epistemic Uncertainty**: By sampling multiple weight configurations, we get a distribution over predictions. High variance = high uncertainty.

2. **Automatic Regularization**: The prior acts as a regularizer, preventing overfitting.

3. **Calibrated Confidence**: The model "knows what it doesn't know"—uncertain in unfamiliar states.

#### Usage

```python
from probmods.models.bayesian_bc import BayesianBCConfig, BayesianBCTrainer

config = BayesianBCConfig(
    layout_name="cramped_room",
    hidden_dims=(64, 64),
    num_epochs=500,
    prior_scale=1.0,  # Width of weight prior
)

trainer = BayesianBCTrainer(config)
trainer.train()

# Get predictions with uncertainty
mean_probs, std_probs, entropy = trainer.model.predict_with_uncertainty(
    states, trainer.guide, num_samples=50
)
```

#### Outputs

- `results/bayesian_bc/{layout}/params.pt`: Pyro parameter store (posterior parameters)
- `results/bayesian_bc/{layout}/config.pkl`: Model configuration

---

### 2. Rational Agent Model

**File**: `probmods/models/rational_agent.py`

#### What It Does

Models human decision-making as **softmax-rational**: humans choose actions with probability proportional to their exponentiated value, with a learnable **rationality parameter β**.

#### Theoretical Inspiration

- **Luce (1959)**: Choice axiom - foundation of softmax choice models
- **McFadden (1973)**: Random utility theory in economics
- **Baker, Saxe, Tenenbaum (2009)**: "Action understanding as inverse planning" - Bayesian theory of mind
- **Ziebart et al. (2008)**: Maximum entropy inverse reinforcement learning

#### The Model

```
π(a|s) = exp(β · Q(s,a)) / Σ_a' exp(β · Q(s,a'))
```

Where:
- **Q(s,a)**: A neural network that learns state-action values
- **β**: Rationality parameter (learned with a prior)

#### Why β Matters

| β Value | Behavior | Interpretation |
|---------|----------|----------------|
| β ≈ 0 | Random | Agent ignores values |
| β ≈ 1 | Noisy rational | Human-like |
| β ≈ 5 | Near-optimal | Expert behavior |
| β → ∞ | Deterministic | Always picks argmax |

By learning β from data, we can quantify "how rational" the observed behavior is.

#### Architecture

```
Input: State s
  ↓
[Q-Network: MLP(96 → 64 → 64 → 6)]  # Learns Q(s,a)
  ↓
Q-values for each action
  ↓
[Softmax with temperature 1/β]
  ↓
Output: π(a|s) = softmax(β · Q(s,·))
```

The β parameter has a prior:
```python
β ~ LogNormal(μ_prior, σ_prior)  # Ensures β > 0
```

#### Usage

```python
from probmods.models.rational_agent import RationalAgentConfig, RationalAgentTrainer

config = RationalAgentConfig(
    layout_name="cramped_room",
    beta_prior_mean=1.0,   # Prior mean for β
    beta_prior_scale=2.0,  # Prior uncertainty
    learn_beta=True,       # Learn β from data
)

trainer = RationalAgentTrainer(config)
trainer.train()

# Extract learned rationality
# (Access via posterior samples)
```

#### Interpreting Results

- **High β (> 3)**: Humans in this layout are behaving very optimally
- **Low β (< 1)**: Behavior is noisy/exploratory
- **Variable β across layouts**: Different tasks elicit different rationality levels

---

### 3. Hierarchical Behavioral Cloning

**File**: `probmods/models/hierarchical_bc.py`

#### What It Does

Decomposes behavior into **high-level goals** and **goal-conditioned policies**. Discovers latent goals in an unsupervised manner.

#### Theoretical Inspiration

- **Sutton et al. (1999)**: Options framework - temporal abstraction in RL
- **Kulkarni et al. (2016)**: Hierarchical deep RL with intrinsic motivation
- **Andreas et al. (2017)**: Modular multitask RL
- **Fox et al. (2017)**: Multi-level discovery of deep options
- **Cognitive science**: Humans plan hierarchically with subgoals

#### The Model

The generative process:
```
1. Infer goal from state: g ~ p(g|s)     # Goal inference network
2. Choose action given goal: a ~ π(a|s,g) # Goal-conditioned policy
```

Marginalizing over goals:
```
p(a|s) = Σ_g π(a|s,g) · p(g|s)
```

#### Architecture

```
            State s
               ↓
    ┌──────────┴──────────┐
    ↓                     ↓
[Goal Inference]    [Policy Network]
   p(g|s)              π(a|s,g)
    ↓                     ↓
  Goal g ──────────────→ ⊕
                          ↓
                    Action logits
```

**Goal Inference Network**:
```
s → MLP → softmax → p(g|s) ∈ Δ^K  (K = num_goals)
```

**Goal-Conditioned Policy**:
```
[s; one_hot(g)] → MLP → softmax → π(a|s,g)
```

#### Why Hierarchical?

In Overcooked, plausible goals include:
- "Get an onion"
- "Go to the pot"
- "Deliver a dish"
- "Wait for partner"

By learning these latent goals, we can:
1. **Interpret behavior**: "The agent is pursuing goal 2 (go to pot)"
2. **Predict intent**: Anticipate what the human will do next
3. **Coordinate**: An AI partner can infer human goals and complement them

#### Usage

```python
from probmods.models.hierarchical_bc import HierarchicalBCConfig, HierarchicalBCTrainer

config = HierarchicalBCConfig(
    layout_name="cramped_room",
    num_goals=4,  # Number of latent goals to discover
    goal_hidden_dims=(64,),
    policy_hidden_dims=(64,),
)

trainer = HierarchicalBCTrainer(config)
trainer.train()

# Infer goals for new states
goal_probs = trainer.infer_goals(states)  # Shape: (N, num_goals)
```

#### Interpreting Goal Distributions

After training, examine what each goal corresponds to by:
1. Looking at states where p(g=k|s) is high
2. Visualizing the state-goal mapping
3. Analyzing the goal-conditioned action distributions

---

### 4. Bayesian GAIL

**File**: `probmods/models/bayesian_gail.py`

#### What It Does

Generative Adversarial Imitation Learning (GAIL) with a **Bayesian discriminator** that maintains uncertainty over its parameters.

#### Theoretical Inspiration

- **Ho & Ermon (2016)**: "Generative Adversarial Imitation Learning" - Original GAIL paper
- **Goodfellow et al. (2014)**: Generative Adversarial Networks
- **Inverse RL literature**: Recovering rewards from demonstrations
- **Bayesian GANs**: Uncertainty in discriminator for robustness

#### The GAIL Framework

GAIL frames imitation learning as a game:
- **Discriminator D(s,a)**: Distinguishes expert from policy behavior
- **Policy π(a|s)**: Tries to fool the discriminator

The policy is trained with RL (PPO) using discriminator output as reward:
```
r(s,a) = -log(1 - D(s,a))
```

#### Why Bayesian Discriminator?

Standard GAIL can be unstable because:
1. Discriminator becomes too confident → reward signal degrades
2. Mode collapse in policy

A **Bayesian discriminator** maintains uncertainty:
- Early in training: High uncertainty → moderate rewards
- Confident regions: Low uncertainty → strong signal
- Uncertain regions: Exploration is encouraged

#### Architecture

```
Expert demos (s,a)     Policy rollouts (s,a)
       ↓                       ↓
       └───────────┬───────────┘
                   ↓
        [Bayesian Discriminator]
         D(s,a) ~ posterior
                   ↓
              Reward signal
                   ↓
            [PPO Policy Update]
```

**Bayesian Discriminator**:
```python
# Weights sampled from posterior
D(s,a) = sigmoid(MLP([s,a]))  # P(expert | s,a)
```

#### Usage

```python
from probmods.models.bayesian_gail import BayesianGAILConfig, BayesianGAILTrainer

config = BayesianGAILConfig(
    layout_name="cramped_room",
    disc_hidden_dim=64,
    policy_hidden_dim=64,
    total_timesteps=200000,
)

trainer = BayesianGAILTrainer(config)
trainer.train()
```

---

### 5. Bayesian PPO+BC

**File**: `probmods/models/bayesian_ppo_bc.py`

#### What It Does

Trains a **Bayesian policy** using PPO, initialized from behavioral cloning and regularized to stay close to the BC policy.

#### Theoretical Inspiration

- **Schulman et al. (2017)**: "Proximal Policy Optimization" - PPO algorithm
- **Ross et al. (2011)**: DAgger - importance of BC initialization
- **Nair et al. (2018)**: Overcoming exploration with demonstrations
- **Wu et al. (2019)**: Behavior regularized RL

#### Why BC + PPO?

Pure behavioral cloning suffers from **distribution shift**: small errors compound.

Pure RL requires extensive **exploration**: inefficient with sparse rewards.

**BC + PPO** combines the best:
1. Initialize from BC → start with reasonable behavior
2. Fine-tune with PPO → improve beyond demos
3. KL regularization → don't drift too far from safe BC policy

#### The Objective

```
L = L_PPO + λ · KL(π || π_BC)
```

Where:
- L_PPO: Standard PPO surrogate loss
- KL term: Keeps policy close to BC anchor
- λ: Regularization strength

#### Bayesian Extension

The policy is a **Bayesian neural network**:
- Posterior over actor and critic weights
- Uncertainty-aware action selection
- Natural exploration through posterior sampling

#### Usage

```python
from probmods.models.bayesian_ppo_bc import BayesianPPOBCConfig, BayesianPPOBCTrainer

config = BayesianPPOBCConfig(
    layout_name="cramped_room",
    bc_kl_coef=0.1,  # KL regularization strength
    total_timesteps=200000,
)

trainer = BayesianPPOBCTrainer(config)
trainer.train()
```

---

### 6. Bayesian PPO+GAIL

**File**: `probmods/models/bayesian_ppo_gail.py`

#### What It Does

The full pipeline: **Bayesian policy** trained with **GAIL rewards** and **BC regularization**.

#### Theoretical Inspiration

Combines insights from:
- Bayesian deep learning (uncertainty)
- Inverse RL (reward from demos)
- Safe RL (BC regularization)

#### Architecture

```
Expert Demonstrations
         ↓
[Discriminator] ←──────┐
         ↓              │
    Reward signal       │
         ↓              │
[Bayesian Policy] ──────┤ (rollouts)
         ↓              │
   + BC KL penalty      │
         ↓              │
    PPO Update ─────────┘
```

#### Usage

```python
from probmods.models.bayesian_ppo_gail import BayesianPPOGAILConfig, BayesianPPOGAILTrainer

config = BayesianPPOGAILConfig(
    layout_name="cramped_room",
    bc_kl_coef=0.1,
    total_timesteps=200000,
)

trainer = BayesianPPOGAILTrainer(config)
trainer.train()
```

---

## Installation

### Prerequisites

- Python >= 3.8
- CUDA-capable GPU (recommended)
- Conda (recommended for environment management)

### Setup

```bash
# Create conda environment
conda create -n probmods python=3.10
conda activate probmods

# Install PyTorch with CUDA
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Install Pyro and other dependencies
pip install pyro-ppl numpyro scipy pandas matplotlib tqdm

# Install parent Overcooked package
cd /path/to/overcooked_ai
pip install -e .

# Install this package
cd Overcooked_ProbMods
pip install -e .
```

### On OpenMind Cluster

```bash
# Activate pre-configured environment
source /om2/user/mabdel03/conda_envs/activate_CompCogSci.sh

# Or activate directly
source /om2/user/mabdel03/anaconda/bin/activate /om2/user/mabdel03/conda_envs/CompCogSci
```

---

## Usage

### Training Models

```bash
# Train Bayesian BC (500 epochs)
python scripts/train_bayesian_bc.py --layout cramped_room --epochs 500

# Train with custom config
python -c "
from probmods.models.bayesian_bc import BayesianBCConfig, BayesianBCTrainer

config = BayesianBCConfig(
    layout_name='cramped_room',
    hidden_dims=(128, 128),
    num_epochs=1000,
    prior_scale=0.5,
)
trainer = BayesianBCTrainer(config)
trainer.train()
"
```

### HPC/SLURM Training

```bash
# Submit parallel training jobs (all models, all layouts)
sbatch hpc/train_array.sh

# Submit single model
sbatch hpc/train_bc_only.sh
```

### Evaluation

```python
from probmods.models.bayesian_bc import BayesianBCModel
from probmods.data import load_human_data, DataConfig
import torch
import pyro

# Load test data
config = DataConfig(layout_name='cramped_room', dataset='test')
states, actions = load_human_data(config)

# Load trained model
pyro.clear_param_store()
saved = torch.load('results/bayesian_bc/cramped_room/params.pt', weights_only=False)
pyro.get_param_store().set_state(saved)

# Create model and guide
model = BayesianBCModel(state_dim=96, action_dim=6, hidden_dims=(64, 64))
from pyro.infer.autoguide import AutoDiagonalNormal
guide = AutoDiagonalNormal(model)

# Get predictions with uncertainty
states_t = torch.tensor(states, dtype=torch.float32)
mean_probs, std_probs, entropy = model.predict_with_uncertainty(states_t, guide, num_samples=50)

# Compute accuracy
pred_actions = mean_probs.argmax(axis=1)
accuracy = (pred_actions == actions).mean()
print(f"Accuracy: {accuracy:.4f}")
print(f"Mean entropy: {entropy.mean():.4f}")
```

---

## Project Structure

```
Overcooked_ProbMods/
├── probmods/                          # Main Python package
│   ├── __init__.py
│   ├── models/                        # Probabilistic model implementations
│   │   ├── __init__.py
│   │   ├── bayesian_bc.py             # Bayesian Behavioral Cloning
│   │   ├── rational_agent.py          # Softmax-rational agent
│   │   ├── hierarchical_bc.py         # Hierarchical goal-conditioned BC
│   │   ├── bayesian_gail.py           # Bayesian GAIL
│   │   ├── bayesian_ppo_bc.py         # Bayesian PPO with BC
│   │   └── bayesian_ppo_gail.py       # Bayesian PPO + GAIL
│   ├── data/                          # Data utilities
│   │   ├── __init__.py
│   │   └── overcooked_data.py         # Human trajectory loading
│   ├── analysis/                      # Analysis tools
│   │   ├── __init__.py
│   │   ├── compare_models.py          # Model comparison metrics
│   │   ├── uncertainty.py             # Uncertainty quantification
│   │   └── interpretability.py        # Goal/rationality extraction
│   ├── inference/                     # Pyro inference utilities
│   │   └── __init__.py
│   └── utils/                         # General utilities
│       └── __init__.py
├── scripts/                           # Training & evaluation scripts
│   ├── train_all.py
│   ├── train_bayesian_bc.py
│   ├── evaluate.py
│   └── compare.py
├── hpc/                               # SLURM scripts
│   ├── train_array.sh                 # Parallel training
│   ├── train_bc_only.sh               # Single model
│   └── evaluate_all.sh
├── configs/
│   └── default.yaml                   # Default hyperparameters
├── results/                           # Training outputs (created at runtime)
│   ├── bayesian_bc/
│   ├── rational_agent/
│   └── hierarchical_bc/
├── logs/                              # SLURM logs
├── requirements.txt
├── setup.py
└── README.md
```

---

## Analysis Tools

### Uncertainty Quantification

```python
from probmods.analysis.uncertainty import summarize_uncertainty

# probs_samples: (num_samples, batch_size, num_actions)
metrics = summarize_uncertainty(probs_samples)

print(f"Mean entropy: {metrics['mean_entropy']:.4f}")
print(f"Epistemic variance: {metrics['epistemic_variance']:.4f}")
```

**Metrics**:
- **Mean entropy**: Average uncertainty across predictions
- **Epistemic variance**: Variance due to model uncertainty (not inherent stochasticity)

### Model Comparison

```python
from probmods.analysis.compare_models import compute_metrics

metrics = compute_metrics(predicted_probs, true_actions)

print(f"Cross-entropy: {metrics['cross_entropy']:.4f}")
print(f"Accuracy: {metrics['accuracy']:.4f}")
print(f"Perplexity: {metrics['perplexity']:.4f}")
```

### Interpretability

```python
from probmods.analysis.interpretability import goal_distribution, rationality_beta, kl_between_policies

# Extract goal distributions from hierarchical model
goals = goal_distribution(hierarchical_model, states)

# Compare two policies
kl = kl_between_policies(probs_model_a, probs_model_b)
```

---

## References

### Core Papers

1. **Bayesian Neural Networks**
   - Blundell, C., et al. (2015). "Weight Uncertainty in Neural Networks." ICML.
   - Graves, A. (2011). "Practical Variational Inference for Neural Networks." NeurIPS.

2. **Rational Agent Models**
   - Luce, R. D. (1959). "Individual Choice Behavior."
   - Baker, C., Saxe, R., Tenenbaum, J. (2009). "Action Understanding as Inverse Planning." Cognition.

3. **Hierarchical RL**
   - Sutton, R., et al. (1999). "Between MDPs and semi-MDPs: A Framework for Temporal Abstraction."
   - Kulkarni, T., et al. (2016). "Hierarchical Deep Reinforcement Learning."

4. **Imitation Learning**
   - Ho, J., Ermon, S. (2016). "Generative Adversarial Imitation Learning." NeurIPS.
   - Ross, S., et al. (2011). "A Reduction of Imitation Learning to No-Regret Online Learning."

5. **PPO**
   - Schulman, J., et al. (2017). "Proximal Policy Optimization Algorithms."

### Software

- [Pyro](https://pyro.ai/): Probabilistic programming on PyTorch
- [Overcooked AI](https://github.com/HumanCompatibleAI/overcooked_ai): Multi-agent coordination environment

---

## License

MIT License - See parent project for details.

---

## Citation

If you use this code, please cite:

```bibtex
@software{overcooked_probmods,
  title = {Overcooked ProbMods: Probabilistic Models for Cooperative AI},
  year = {2024},
  url = {https://github.com/HumanCompatibleAI/overcooked_ai}
}
```
