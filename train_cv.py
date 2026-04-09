"""
Train a model specifically for deployment against the real game
via computer vision / screen capture.

The key differences from standard training:
- Uses FnafCVReady-v0: partial obs + memory + no leaked internals
- Curriculum: night 1 → 2 → 3 so agent learns core strategy first
- Noise is gradually increased to build robustness to CV errors
- Larger network to handle the more complex 87-dim observation

Usage:
    python train_cv.py --steps 2000000 --target-night 3
"""

import argparse
import os

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, CallbackList

import fnaf_gymnasium
from fnaf_gymnasium.callbacks import FnafMetricsCallback


def make_cv_env(night, obs_noise_std=0.0, cv_detection_rate=1.0):
    def _init():
        return gym.make(
            'FnafCVReady-v0',
            night=night,
            obs_noise_std=obs_noise_std,
            cv_detection_rate=cv_detection_rate,
        )
    return _init


def train_cv(args):
    os.makedirs(args.log_dir, exist_ok=True)
    nights = list(range(1, args.target_night + 1))
    steps_per_night = args.steps // len(nights)

    model = None

    for i, night in enumerate(nights):
        # Gradually increase noise as training progresses
        # Start clean (night 1), add noise for later nights
        noise = 0.0 if night == 1 else args.noise_std
        detection = 1.0 if night == 1 else args.detection_rate

        log_dir = os.path.join(args.log_dir, f'cv_night{night}')
        os.makedirs(log_dir, exist_ok=True)

        print(f"\n{'='*60}")
        print(f"CV Training: Night {night}/{args.target_night}")
        print(f"  Noise std: {noise}  |  Detection rate: {detection}")
        print(f"  Steps: {steps_per_night}")
        print(f"{'='*60}")

        env_fns = [make_cv_env(night, noise, detection) for _ in range(args.n_envs)]
        train_env = SubprocVecEnv(env_fns) if args.n_envs > 1 else DummyVecEnv(env_fns)
        eval_env = DummyVecEnv([make_cv_env(night, 0.0, 1.0)])  # Eval without noise

        if model is None:
            model = PPO(
                'MlpPolicy',
                train_env,
                learning_rate=3e-4,
                n_steps=2048,
                batch_size=64,
                n_epochs=10,
                gamma=0.99,
                gae_lambda=0.95,
                clip_range=0.2,
                ent_coef=0.02,
                policy_kwargs=dict(net_arch=dict(pi=[256, 256, 128], vf=[256, 256, 128])),
                tensorboard_log=os.path.join(log_dir, 'tensorboard'),
                verbose=1,
            )
        else:
            model.set_env(train_env)

        callbacks = CallbackList([
            FnafMetricsCallback(log_freq=5000, verbose=1),
            EvalCallback(
                eval_env,
                best_model_save_path=os.path.join(log_dir, 'best_model'),
                log_path=os.path.join(log_dir, 'eval_logs'),
                eval_freq=max(steps_per_night // 20, 2000),
                n_eval_episodes=20,
                deterministic=True,
            ),
            CheckpointCallback(
                save_freq=max(steps_per_night // 10, 5000),
                save_path=os.path.join(log_dir, 'checkpoints'),
                name_prefix='cv_model',
            ),
        ])

        model.learn(
            total_timesteps=steps_per_night,
            callback=callbacks,
            progress_bar=True,
            reset_num_timesteps=(i == 0),
        )

        save_path = os.path.join(log_dir, 'cv_model_final')
        model.save(save_path)
        print(f"Night {night} model saved to {save_path}")

        train_env.close()
        eval_env.close()

    # Save final model
    final_path = os.path.join(args.log_dir, 'cv_model_ready')
    model.save(final_path)
    print(f"\nFinal CV-ready model saved to {final_path}")
    print("This model is ready to be used with a screen capture pipeline.")


def main():
    parser = argparse.ArgumentParser(description='Train CV-ready FNAF agent')
    parser.add_argument('--steps', type=int, default=2_000_000,
                        help='Total training steps across all nights')
    parser.add_argument('--target-night', type=int, default=3,
                        help='Highest night to train on (1-7)')
    parser.add_argument('--n-envs', type=int, default=4,
                        help='Number of parallel environments')
    parser.add_argument('--noise-std', type=float, default=0.02,
                        help='Gaussian noise std for later nights (0=no noise)')
    parser.add_argument('--detection-rate', type=float, default=0.90,
                        help='CV detection rate for later nights (1.0=perfect)')
    parser.add_argument('--log-dir', type=str, default='./training_logs/cv')
    args = parser.parse_args()
    train_cv(args)


if __name__ == '__main__':
    main()
