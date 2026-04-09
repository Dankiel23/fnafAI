"""
Compare different RL algorithms (PPO, DQN, A2C) on FNAF Night 1.

Trains each algorithm for the same number of steps and compares
win rates and training curves.

Usage:
    python compare_algorithms.py --steps 200000
"""

import argparse
import os
import time

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO, DQN, A2C
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy

import fnaf_gymnasium


def make_env(night=1):
    def _init():
        return gym.make('FnafNight-v0', night=night)
    return _init


def train_and_evaluate(algo_class, algo_name, night, total_steps, n_envs=4, eval_eps=50):
    """Train an algorithm and evaluate it."""
    print(f"\n{'='*50}")
    print(f"Training {algo_name} for {total_steps} steps on Night {night}")
    print(f"{'='*50}")

    train_env = DummyVecEnv([make_env(night) for _ in range(n_envs)])
    eval_env = DummyVecEnv([make_env(night)])

    kwargs = {
        'policy': 'MlpPolicy',
        'env': train_env,
        'verbose': 0,
    }

    if algo_name == 'PPO':
        kwargs.update({'learning_rate': 3e-4, 'n_steps': 2048, 'batch_size': 64,
                       'n_epochs': 10, 'ent_coef': 0.01})
    elif algo_name == 'DQN':
        kwargs.update({'learning_rate': 1e-4, 'buffer_size': 50000,
                       'learning_starts': 1000, 'batch_size': 64,
                       'exploration_fraction': 0.3})
    elif algo_name == 'A2C':
        kwargs.update({'learning_rate': 7e-4, 'n_steps': 5, 'ent_coef': 0.01})

    model = algo_class(**kwargs)

    start = time.time()
    model.learn(total_timesteps=total_steps, progress_bar=True)
    train_time = time.time() - start

    # Evaluate
    wins = 0
    rewards = []
    powers = []

    single_eval = gym.make('FnafNight-v0', night=night)
    for ep in range(eval_eps):
        obs, info = single_eval.reset(seed=ep)
        total_reward = 0
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = single_eval.step(action)
            total_reward += reward
            done = terminated or truncated
        if info.get('game_over_reason') == 'SURVIVED':
            wins += 1
        rewards.append(total_reward)
        powers.append(info['power'])

    single_eval.close()
    train_env.close()
    eval_env.close()

    result = {
        'algo': algo_name,
        'win_rate': wins / eval_eps,
        'avg_reward': np.mean(rewards),
        'std_reward': np.std(rewards),
        'avg_power': np.mean(powers),
        'train_time': train_time,
        'steps_per_sec': total_steps / train_time,
    }

    print(f"  Win rate: {result['win_rate']*100:.1f}%")
    print(f"  Avg reward: {result['avg_reward']:.2f} +/- {result['std_reward']:.2f}")
    print(f"  Train time: {result['train_time']:.1f}s ({result['steps_per_sec']:.0f} steps/s)")

    save_path = f"models/{algo_name.lower()}_night{night}"
    os.makedirs("models", exist_ok=True)
    model.save(save_path)
    print(f"  Model saved to {save_path}")

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps', type=int, default=200_000)
    parser.add_argument('--night', type=int, default=1)
    parser.add_argument('--eval-episodes', type=int, default=50)
    parser.add_argument('--n-envs', type=int, default=4)
    args = parser.parse_args()

    algorithms = [
        (PPO, 'PPO'),
        (DQN, 'DQN'),
        (A2C, 'A2C'),
    ]

    results = []
    for algo_class, algo_name in algorithms:
        result = train_and_evaluate(
            algo_class, algo_name, args.night,
            args.steps, args.n_envs, args.eval_episodes
        )
        results.append(result)

    # Summary table
    print(f"\n{'='*70}")
    print(f"Algorithm Comparison - Night {args.night} ({args.steps} steps)")
    print(f"{'='*70}")
    print(f"{'Algo':<8} {'Win%':<10} {'Avg Reward':<14} {'Avg Power':<12} {'Train Time':<12} {'Steps/s':<10}")
    print(f"{'-'*70}")

    for r in sorted(results, key=lambda x: -x['win_rate']):
        print(f"{r['algo']:<8} "
              f"{r['win_rate']*100:>6.1f}%   "
              f"{r['avg_reward']:>10.2f}    "
              f"{r['avg_power']:>8.1f}%    "
              f"{r['train_time']:>8.1f}s    "
              f"{r['steps_per_sec']:>8.0f}")


if __name__ == '__main__':
    main()
