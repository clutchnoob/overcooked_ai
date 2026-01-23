#!/usr/bin/env python
"""
Paper Reproduction Training Script

This script trains PPO agents on the forced_coordination (random0) and counter_circuit (random3)
layouts using configurations that match the original 2019 paper implementation.

Key settings matching original ppo_sp_random0 config:
1. Uses legacy 20-channel observation encoding (not 26-channel)
2. Uses per-minibatch advantage normalization (not per-batch)
3. Entropy coefficient = 0.1 (ENTROPY=0.1 in original TF code)
4. Learning rate = 8e-4 (for random0/random3)
5. VF_COEF = 0.5 (not 0.1!)
6. num_envs = 60 (sim_threads in original)
7. BATCH_SIZE = 60 * 400 = 24000 per update
8. REW_SHAPING_HORIZON = 2.5e6 (anneals shaped rewards to 0 by update 104)
9. Total timesteps = 7.5e6 (312 updates - breakthrough happens at updates 100-200)
10. Shared shaped reward for both agents (not per-agent)
11. Agent index randomization on reset (learns from both starting positions)
12. Glorot uniform weight init for conv/dense layers (not orthogonal)
13. Leaky ReLU with negative_slope=0.2 (matching TensorFlow default)
14. Stochastic action sampling in evaluation (matches training behavior)

Usage:
    cd overcooked_ai-master/src
    python -m human_aware_rl.jaxmarl.train_paper_reproduction --layout random0_legacy
    python -m human_aware_rl.jaxmarl.train_paper_reproduction --layout random3_legacy
"""

import argparse
import os
import sys

# Ensure we can import from the correct location
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from human_aware_rl.jaxmarl.ppo import PPOConfig, PPOTrainer


def get_paper_reproduction_config(
    layout_name: str,
    seed: int = 0,
    total_timesteps: int = 7_500_000,  # random0 uses 7.5e6, random3 uses 5e6
    results_dir: str = "paper_reproduction_results"
) -> PPOConfig:
    """
    Get PPO config matching the original 2019 paper implementation.
    
    Args:
        layout_name: Layout to train on (use 'random0_legacy' or 'random3_legacy')
        seed: Random seed
        total_timesteps: Total training timesteps (paper uses 5M)
        results_dir: Directory to save results
        
    Returns:
        PPOConfig with paper-matching hyperparameters
    """
    return PPOConfig(
        # Environment
        layout_name=layout_name,
        horizon=400,
        num_envs=60,  # sim_threads=60 in original ppo_sp_random0 config
        old_dynamics=True,  # Paper uses old dynamics
        
        # Training - matching original ppo.py config
        total_timesteps=total_timesteps,
        learning_rate=8e-4,  # LR=8e-4 for random0, 5e-4 for random3 in original
        num_steps=400,  # horizon = 400, so steps per env = 400
        num_minibatches=6,  # MINIBATCHES in original
        num_epochs=8,  # STEPS_PER_UPDATE in original
        
        # PPO hyperparameters - matching original ppo_sp_random0 config
        gamma=0.99,  # GAMMA
        gae_lambda=0.98,  # LAM (not 0.95!)
        clip_eps=0.05,  # CLIPPING (not 0.2!)
        ent_coef=0.1,  # ENTROPY=0.1 in original - MUST match for paper reproduction!
        vf_coef=0.5,  # VF_COEF=0.5 in original ppo_sp_random0 config!
        max_grad_norm=0.1,  # MAX_GRAD_NORM (not 0.5!)
        
        # No LR/entropy annealing - paper uses constant values
        use_lr_annealing=False,
        use_entropy_annealing=False,
        entropy_coeff_start=0.1,  # Match original ENTROPY=0.1
        entropy_coeff_end=0.1,    # No annealing - constant entropy coef
        
        # Value function clipping (original baselines uses this)
        clip_vf=True,
        
        # Reward shaping - matching original paper settings
        # Original anneals shaped rewards to 0 by 2.5M timesteps
        # With proper orthogonal initialization, agents should learn before this
        reward_shaping_factor=1.0,
        reward_shaping_horizon=2_500_000,  # REW_SHAPING_HORIZON=2.5e6 in original
        use_phi=False,  # Paper doesn't use potential-based shaping
        
        # Observation encoding - CRITICAL: paper uses 20-channel legacy encoding
        use_legacy_encoding=True,
        
        # Network architecture - matching original
        num_hidden_layers=3,
        hidden_dim=64,  # SIZE_HIDDEN_LAYERS
        num_filters=25,  # NUM_FILTERS
        num_conv_layers=3,  # NUM_CONV_LAYERS
        use_lstm=False,
        
        # Logging
        log_interval=1,
        save_interval=50,
        eval_interval=25,
        eval_num_games=5,
        verbose=True,
        
        # No early stopping for paper reproduction
        use_early_stopping=False,
        
        # Output
        results_dir=results_dir,
        experiment_name=f"ppo_sp_{layout_name}_seed{seed}",
        seed=seed,
    )


def main():
    parser = argparse.ArgumentParser(description='Train PPO for paper reproduction')
    parser.add_argument('--layout', type=str, default='random0_legacy',
                        choices=['random0_legacy', 'random3_legacy', 'cramped_room', 
                                 'forced_coordination', 'counter_circuit'],
                        help='Layout to train on (use legacy layouts for paper reproduction)')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed')
    parser.add_argument('--timesteps', type=int, default=7_500_000,
                        help='Total training timesteps (paper uses 7.5M for random0, 5M for random3)')
    parser.add_argument('--results-dir', type=str, default='paper_reproduction_results',
                        help='Directory to save results')
    args = parser.parse_args()
    
    # Get paper reproduction config
    config = get_paper_reproduction_config(
        layout_name=args.layout,
        seed=args.seed,
        total_timesteps=args.timesteps,
        results_dir=args.results_dir
    )
    
    # Print config summary
    print("=" * 60)
    print("PAPER REPRODUCTION TRAINING")
    print("=" * 60)
    print(f"Layout: {config.layout_name}")
    print(f"Seed: {config.seed}")
    print(f"Total timesteps: {config.total_timesteps:,}")
    print(f"Num envs (sim_threads): {config.num_envs}")
    print(f"Steps per env: {config.num_steps}")
    print(f"Batch size: {config.num_envs * config.num_steps:,}")
    print()
    print("Key hyperparameters (matching original ppo_sp_random0):")
    print(f"  Learning rate: {config.learning_rate} (constant)")
    print(f"  Entropy coef: {config.ent_coef}")
    print(f"  VF coef: {config.vf_coef}")
    print(f"  Clip epsilon: {config.clip_eps}")
    print(f"  Max grad norm: {config.max_grad_norm}")
    print(f"  GAE lambda: {config.gae_lambda}")
    print(f"  Gamma: {config.gamma}")
    print()
    print("Reward shaping (matching original):")
    print(f"  Reward shaping factor: {config.reward_shaping_factor}")
    print(f"  Reward shaping horizon: {config.reward_shaping_horizon:,} (anneals to 0 by this step)")
    print("=" * 60)
    print()
    
    # Create trainer and train
    trainer = PPOTrainer(config)
    
    print(f"Observation shape: {trainer.obs_shape}")
    print(f"Expected: (5, 5, 20) for random0_legacy, (8, 5, 20) for random3_legacy")
    print()
    
    results = trainer.train()
    
    # Print final results
    print()
    print("=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Final mean reward: {results.get('final_mean_reward', 'N/A'):.2f}")
    print(f"Best mean reward: {results.get('best_mean_reward', 'N/A'):.2f}")
    print(f"Final eval reward: {results.get('final_eval_reward', 'N/A'):.2f}")
    print()
    print("Expected paper results (SP+SP self-play):")
    print("  forced_coordination (random0): ~120-160 during training")
    print("  counter_circuit (random3): ~120-160 during training")
    print()
    
    return results


if __name__ == "__main__":
    main()

