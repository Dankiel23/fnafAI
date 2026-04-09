"""
Hyperparameter tuning script for FNAF RL training.

Runs a grid search over key hyperparameters and reports
win rates for each configuration.
"""

import os
import itertools

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy

import fnaf_gymnasium


def make_env(night=1):
    def _init():
        return gym.make('FnafNight-v0', night=night)
    return _init


def evaluate_config(config, night=1, train_steps=50_000, eval_episodes=50):
    """Train and evaluate a single hyperparameter configuration."""
    env = DummyVecEnv([make_env(night) for _ in range(4)])
    eval_env = DummyVecEnv([make_env(night)])

    model = PPO(
        'MlpPolicy',
        env,
        learning_rate=config['lr'],
        n_steps=config['n_steps'],
        batch_size=config['batch_size'],
        n_epochs=config['n_epochs'],
        gamma=config['gamma'],
        ent_coef=config['ent_coef'],
        clip_range=config['clip_range'],
        verbose=0,
    )

    model.learn(total_timesteps=train_steps)

    # Evaluate
    mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=eval_episodes)

    # Count wins
    wins = 0
    obs = eval_env.reset()
    for _ in range(eval_episodes):
        obs = eval_env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, dones, infos = eval_env.step(action)
            done = dones[0]
        if infos[0].get('game_over_reason') == 'SURVIVED':
            wins += 1

    env.close()
    eval_env.close()

    return {
        'config': config,
        'mean_reward': mean_reward,
        'std_reward': std_reward,
        'win_rate': wins / eval_episodes,
    }


def main():
    # Hyperparameter grid
    param_grid = {
        'lr': [1e-4, 3e-4, 1e-3],
        'n_steps': [1024, 2048],
        'batch_size': [64, 128],
        'n_epochs': [5, 10],
        'gamma': [0.99],
        'ent_coef': [0.005, 0.01, 0.02],
        'clip_range': [0.1, 0.2],
    }

    # Generate all combinations
    keys = list(param_grid.keys())
    values = list(param_grid.values())
    configs = [dict(zip(keys, combo)) for combo in itertools.product(*values)]

    print(f"Testing {len(configs)} hyperparameter configurations")
    print("=" * 80)

    results = []
    for i, config in enumerate(configs):
        print(f"\n[{i+1}/{len(configs)}] Testing: lr={config['lr']}, "
              f"n_steps={config['n_steps']}, batch={config['batch_size']}, "
              f"epochs={config['n_epochs']}, ent={config['ent_coef']}, "
              f"clip={config['clip_range']}")

        result = evaluate_config(config, night=1, train_steps=50_000, eval_episodes=30)
        results.append(result)

        print(f"  -> Win rate: {result['win_rate']*100:.1f}% | "
              f"Mean reward: {result['mean_reward']:.2f} +/- {result['std_reward']:.2f}")

    # Sort by win rate
    results.sort(key=lambda x: x['win_rate'], reverse=True)

    print("\n" + "=" * 80)
    print("TOP 5 CONFIGURATIONS:")
    print("=" * 80)
    for i, r in enumerate(results[:5]):
        c = r['config']
        print(f"\n#{i+1} Win rate: {r['win_rate']*100:.1f}% | Reward: {r['mean_reward']:.2f}")
        print(f"   lr={c['lr']}, n_steps={c['n_steps']}, batch={c['batch_size']}, "
              f"epochs={c['n_epochs']}, ent={c['ent_coef']}, clip={c['clip_range']}")


if __name__ == '__main__':
    main()
