"""
Training script for the FNAF 1 RL agent using Stable-Baselines3.

Supports curriculum learning (starting on easier nights) and
configurable hyperparameters.
"""

import argparse
import os

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO, DQN, A2C
from stable_baselines3.common.callbacks import (
    EvalCallback,
    CheckpointCallback,
    CallbackList,
)
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.monitor import Monitor

# Register the environment
import fnaf_gymnasium
from fnaf_gymnasium.callbacks import FnafMetricsCallback
from fnaf_gymnasium.wrappers import FnafRewardShaping
from fnaf_gymnasium.networks import get_policy_kwargs


ALGORITHMS = {
    'ppo': PPO,
    'dqn': DQN,
    'a2c': A2C,
}


def make_env(night: int, full_obs: bool = True, step_duration: float = 1.0,
             reward_shaping: dict = None):
    """Create a FNAF environment factory."""
    def _init():
        env = gym.make(
            'FnafNight-v0',
            night=night,
            step_duration=step_duration,
            full_observability=full_obs,
        )
        if reward_shaping:
            env = FnafRewardShaping(env, **reward_shaping)
        return env
    return _init


def train(args):
    """Run the training loop."""
    log_dir = os.path.join(args.log_dir, f"night{args.night}_{args.algo}")
    os.makedirs(log_dir, exist_ok=True)

    reward_shaping = None
    if args.reward_shaping:
        reward_shaping = {
            'defensive_bonus': args.defensive_bonus,
            'approaching_bonus': args.approaching_bonus,
            'power_penalty_scale': args.power_penalty_scale,
            'waste_penalty': args.waste_penalty,
            'open_door_penalty': args.open_door_penalty,
            'light_bonus': args.light_bonus,
        }

    print(f"Training {args.algo.upper()} on Night {args.night}")
    print(f"  Parallel envs: {args.n_envs}")
    print(f"  Total timesteps: {args.total_timesteps}")
    print(f"  Full observability: {args.full_obs}")
    print(f"  Reward shaping: {args.reward_shaping}")
    print(f"  Logging to: {log_dir}")

    # Create vectorized environments for parallel training
    env_fns = [make_env(args.night, args.full_obs, args.step_duration, reward_shaping) for _ in range(args.n_envs)]

    if args.n_envs > 1:
        train_env = SubprocVecEnv(env_fns)
    else:
        train_env = DummyVecEnv(env_fns)

    # Create evaluation environment
    eval_env = DummyVecEnv([make_env(args.night, args.full_obs, args.step_duration)])

    # Callbacks
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(log_dir, 'best_model'),
        log_path=os.path.join(log_dir, 'eval_logs'),
        eval_freq=max(args.total_timesteps // 50, 1000),
        n_eval_episodes=10,
        deterministic=True,
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=max(args.total_timesteps // 20, 5000),
        save_path=os.path.join(log_dir, 'checkpoints'),
        name_prefix='fnaf_model',
    )

    metrics_callback = FnafMetricsCallback(verbose=1)
    callbacks = CallbackList([eval_callback, checkpoint_callback, metrics_callback])

    # Create model
    AlgoClass = ALGORITHMS[args.algo]

    policy_kwargs = get_policy_kwargs(args.network) if args.network != 'mlp' else {}

    model_kwargs = {
        'policy': 'MlpPolicy',
        'env': train_env,
        'verbose': 1,
        'tensorboard_log': os.path.join(log_dir, 'tensorboard'),
        'seed': args.seed,
        'policy_kwargs': policy_kwargs,
    }

    # Algorithm-specific hyperparameters
    if args.algo == 'ppo':
        model_kwargs.update({
            'learning_rate': args.lr,
            'n_steps': args.n_steps,
            'batch_size': args.batch_size,
            'n_epochs': args.n_epochs,
            'gamma': 0.99,
            'gae_lambda': 0.95,
            'clip_range': 0.2,
            'ent_coef': args.ent_coef,
        })
    elif args.algo == 'dqn':
        model_kwargs.update({
            'learning_rate': args.lr,
            'buffer_size': 100000,
            'learning_starts': 1000,
            'batch_size': 64,
            'gamma': 0.99,
            'exploration_fraction': 0.3,
            'exploration_final_eps': 0.05,
        })
    elif args.algo == 'a2c':
        model_kwargs.update({
            'learning_rate': args.lr,
            'n_steps': 5,
            'gamma': 0.99,
            'ent_coef': 0.01,
        })

    if args.load_model:
        print(f"Loading model from {args.load_model}")
        model = AlgoClass.load(args.load_model, env=train_env, **{
            k: v for k, v in model_kwargs.items()
            if k not in ('policy', 'env', 'policy_kwargs')
        })
    else:
        model = AlgoClass(**model_kwargs)

    # Train
    model.learn(
        total_timesteps=args.total_timesteps,
        callback=callbacks,
        progress_bar=True,
    )

    # Save final model
    final_path = os.path.join(log_dir, 'final_model')
    model.save(final_path)
    print(f"Final model saved to {final_path}")

    train_env.close()
    eval_env.close()

    return model


def evaluate(args):
    """Evaluate a trained model."""
    env = gym.make(
        'FnafNight-v0',
        night=args.night,
        render_mode='human' if args.render else None,
        full_observability=args.full_obs,
    )

    model = ALGORITHMS[args.algo].load(args.model_path)

    wins = 0
    total_episodes = args.eval_episodes

    for ep in range(total_episodes):
        obs, info = env.reset()
        total_reward = 0
        done = False

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated

            if args.render:
                env.render()

        survived = info.get('game_over_reason', '') == 'SURVIVED'
        if survived:
            wins += 1
        result = "SURVIVED" if survived else f"DIED ({info.get('game_over_reason', 'unknown')})"
        print(f"Episode {ep+1}: {result} | Power: {info['power']:.1f}% | Reward: {total_reward:.2f}")

    print(f"\nWin rate: {wins}/{total_episodes} ({100*wins/total_episodes:.1f}%)")
    env.close()


def curriculum_train(args):
    """Train with curriculum learning - start easy, progress to harder nights."""
    nights = list(range(1, args.night + 1))
    steps_per_night = args.total_timesteps // len(nights)

    model = None
    for night in nights:
        print(f"\n{'='*50}")
        print(f"CURRICULUM: Training on Night {night}")
        print(f"{'='*50}")

        args.night = night
        args.total_timesteps = steps_per_night

        if model is not None:
            # Continue training from previous night's model
            log_dir = os.path.join(args.log_dir, f"night{night}_{args.algo}")
            os.makedirs(log_dir, exist_ok=True)

            env_fns = [make_env(night, args.full_obs, args.step_duration) for _ in range(args.n_envs)]
            if args.n_envs > 1:
                train_env = SubprocVecEnv(env_fns)
            else:
                train_env = DummyVecEnv(env_fns)

            model.set_env(train_env)
            model.learn(total_timesteps=steps_per_night, progress_bar=True)
            model.save(os.path.join(log_dir, 'final_model'))
            train_env.close()
        else:
            model = train(args)

    print(f"\nCurriculum training complete!")
    return model


def main():
    parser = argparse.ArgumentParser(description='Train FNAF 1 RL Agent')
    subparsers = parser.add_subparsers(dest='command', help='Command to run')

    # Train subcommand
    train_parser = subparsers.add_parser('train', help='Train a new model')
    train_parser.add_argument('--algo', type=str, default='ppo', choices=list(ALGORITHMS.keys()))
    train_parser.add_argument('--night', type=int, default=1, help='Night to train on (1-7)')
    train_parser.add_argument('--total-timesteps', type=int, default=500_000)
    train_parser.add_argument('--n-envs', type=int, default=4, help='Parallel environments')
    train_parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
    train_parser.add_argument('--full-obs', action='store_true', default=True)
    train_parser.add_argument('--partial-obs', action='store_true', help='Use partial observability')
    train_parser.add_argument('--step-duration', type=float, default=1.0)
    train_parser.add_argument('--seed', type=int, default=42)
    train_parser.add_argument('--log-dir', type=str, default='./training_logs')
    train_parser.add_argument('--load-model', type=str, default=None,
                              help='Path to existing model to continue training from')
    train_parser.add_argument('--n-steps', type=int, default=2048,
                              help='PPO steps per update (higher = more stable, slower)')
    train_parser.add_argument('--batch-size', type=int, default=64,
                              help='PPO minibatch size')
    train_parser.add_argument('--n-epochs', type=int, default=10,
                              help='PPO gradient epochs per update')
    train_parser.add_argument('--ent-coef', type=float, default=0.01,
                              help='Entropy coefficient (higher = more exploration)')
    train_parser.add_argument('--network', type=str, default='mlp',
                              choices=['mlp', 'default', 'small', 'large', 'custom'],
                              help='Network architecture (custom uses FnafFeatureExtractor)')
    train_parser.add_argument('--reward-shaping', action='store_true', default=False,
                              help='Enable reward shaping wrapper')
    train_parser.add_argument('--defensive-bonus', type=float, default=0.05,
                              help='Reward for closing door against threat in doorway')
    train_parser.add_argument('--approaching-bonus', type=float, default=0.01,
                              help='Reward for closing door against threat in hallway')
    train_parser.add_argument('--power-penalty-scale', type=float, default=0.01,
                              help='Penalty scale for power usage')
    train_parser.add_argument('--waste-penalty', type=float, default=0.02,
                              help='Penalty for closing doors with no nearby threat')
    train_parser.add_argument('--open-door-penalty', type=float, default=0.05,
                              help='Penalty for leaving door open while threat is present')
    train_parser.add_argument('--light-bonus', type=float, default=0.03,
                              help='Reward for using lights to confirm a doorway threat')

    # Curriculum subcommand
    curriculum_parser = subparsers.add_parser('curriculum', help='Curriculum learning')
    curriculum_parser.add_argument('--algo', type=str, default='ppo', choices=list(ALGORITHMS.keys()))
    curriculum_parser.add_argument('--night', type=int, default=5, help='Target night')
    curriculum_parser.add_argument('--total-timesteps', type=int, default=2_000_000)
    curriculum_parser.add_argument('--n-envs', type=int, default=4)
    curriculum_parser.add_argument('--lr', type=float, default=3e-4)
    curriculum_parser.add_argument('--full-obs', action='store_true', default=True)
    curriculum_parser.add_argument('--partial-obs', action='store_true')
    curriculum_parser.add_argument('--step-duration', type=float, default=1.0)
    curriculum_parser.add_argument('--seed', type=int, default=42)
    curriculum_parser.add_argument('--log-dir', type=str, default='./training_logs')
    curriculum_parser.add_argument('--load-model', type=str, default=None,
                                   help='Path to existing model to continue training from')

    # Evaluate subcommand
    eval_parser = subparsers.add_parser('eval', help='Evaluate a trained model')
    eval_parser.add_argument('--model-path', type=str, required=True)
    eval_parser.add_argument('--algo', type=str, default='ppo', choices=list(ALGORITHMS.keys()))
    eval_parser.add_argument('--night', type=int, default=1)
    eval_parser.add_argument('--eval-episodes', type=int, default=100)
    eval_parser.add_argument('--render', action='store_true')
    eval_parser.add_argument('--full-obs', action='store_true', default=True)

    args = parser.parse_args()

    if args.command == 'train':
        if args.partial_obs:
            args.full_obs = False
        train(args)
    elif args.command == 'curriculum':
        if args.partial_obs:
            args.full_obs = False
        curriculum_train(args)
    elif args.command == 'eval':
        evaluate(args)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
