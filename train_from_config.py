"""
Train from a JSON configuration file.

Usage:
    python train_from_config.py configs/ppo_night1.json
"""

import argparse
import json
import os

import gymnasium as gym
from stable_baselines3 import PPO, DQN, A2C
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

import fnaf_gymnasium
from fnaf_gymnasium.wrappers import FnafRewardShaping
from fnaf_gymnasium.callbacks import FnafMetricsCallback


ALGORITHMS = {'ppo': PPO, 'dqn': DQN, 'a2c': A2C}


def make_env(config):
    def _init():
        env = gym.make(
            'FnafNight-v0',
            night=config['night'],
            step_duration=config.get('step_duration', 1.0),
            full_observability=config.get('full_observability', True),
        )
        if 'reward_shaping' in config:
            env = FnafRewardShaping(env, **config['reward_shaping'])
        return env
    return _init


def train_from_config(config_path: str):
    with open(config_path) as f:
        config = json.load(f)

    algo_name = config['algorithm']
    night = config['night']
    total_steps = config['total_timesteps']
    n_envs = config.get('n_envs', 4)

    config_name = os.path.splitext(os.path.basename(config_path))[0]
    log_dir = os.path.join('training_logs', config_name)
    os.makedirs(log_dir, exist_ok=True)

    print(f"Training from config: {config_path}")
    print(f"  Algorithm: {algo_name.upper()}")
    print(f"  Night: {night}")
    print(f"  Steps: {total_steps}")
    print(f"  Envs: {n_envs}")
    print(f"  Log dir: {log_dir}")

    # Create environments
    env_fns = [make_env(config) for _ in range(n_envs)]
    if n_envs > 1 and algo_name != 'dqn':
        train_env = SubprocVecEnv(env_fns)
    else:
        train_env = DummyVecEnv(env_fns)

    # Build model kwargs
    model_kwargs = {
        'policy': 'MlpPolicy',
        'env': train_env,
        'verbose': 1,
        'tensorboard_log': os.path.join(log_dir, 'tensorboard'),
    }

    if 'hyperparameters' in config:
        model_kwargs.update(config['hyperparameters'])

    if 'policy_kwargs' in config:
        pk = config['policy_kwargs']
        # Convert net_arch from JSON format
        if 'net_arch' in pk:
            na = pk['net_arch']
            if isinstance(na, dict):
                pk['net_arch'] = dict(
                    pi=na.get('pi', [128, 128]),
                    vf=na.get('vf', [128, 128]),
                )
        model_kwargs['policy_kwargs'] = pk

    AlgoClass = ALGORITHMS[algo_name]
    model = AlgoClass(**model_kwargs)

    callback = FnafMetricsCallback(log_freq=5000, verbose=1)
    model.learn(total_timesteps=total_steps, callback=callback, progress_bar=True)

    save_path = os.path.join(log_dir, 'final_model')
    model.save(save_path)
    print(f"\nModel saved to {save_path}")

    # Save config alongside model
    with open(os.path.join(log_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)

    train_env.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, help='Path to JSON config file')
    args = parser.parse_args()
    train_from_config(args.config)


if __name__ == '__main__':
    main()
