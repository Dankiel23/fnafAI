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
from fnaf_gymnasium.wrappers import FnafCVRewardShaping


def make_cv_env(night, obs_noise_std=0.0, cv_detection_rate=1.0,
                reward_shaping=None):
    def _init():
        env = gym.make(
            'FnafCVReady-v0',
            night=night,
            obs_noise_std=obs_noise_std,
            cv_detection_rate=cv_detection_rate,
        )
        if reward_shaping:
            env = FnafCVRewardShaping(env, **reward_shaping)
        return env
    return _init


def apply_resume_overrides(model, args):
    """Apply tunable optimizer/exploration settings after loading a checkpoint."""
    model.learning_rate = args.lr
    model.ent_coef = args.ent_coef
    model.lr_schedule = lambda _: args.lr
    if hasattr(model.policy, 'optimizer'):
        for param_group in model.policy.optimizer.param_groups:
            param_group['lr'] = args.lr


def train_cv(args):
    os.makedirs(args.log_dir, exist_ok=True)
    nights = list(range(1, args.target_night + 1))
    steps_per_night = args.steps // len(nights)
    reward_shaping = None if args.disable_reward_shaping else {
        'defensive_bonus': args.defensive_bonus,
        'approaching_bonus': args.approaching_bonus,
        'power_penalty_scale': args.power_penalty_scale,
        'waste_penalty': args.waste_penalty,
        'open_door_penalty': args.open_door_penalty,
        'light_bonus': args.light_bonus,
        'info_gain_bonus': args.info_gain_bonus,
        'info_refresh_threshold': args.info_refresh_threshold,
        'stale_light_bonus': args.stale_light_bonus,
        'stale_light_threshold': args.stale_light_threshold,
        'light_scout_cooldown': args.light_scout_cooldown,
        'invalid_camera_penalty': args.invalid_camera_penalty,
        'redundant_camera_penalty': args.redundant_camera_penalty,
    }

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
        print(f"  LR: {args.lr}  |  Ent coef: {args.ent_coef}")
        print(f"  Reward shaping: {reward_shaping is not None}")
        print(f"{'='*60}")

        env_fns = [
            make_cv_env(night, noise, detection, reward_shaping)
            for _ in range(args.n_envs)
        ]
        train_env = SubprocVecEnv(env_fns) if args.n_envs > 1 else DummyVecEnv(env_fns)
        eval_env = DummyVecEnv([
            make_cv_env(night, 0.0, 1.0, reward_shaping)
        ])  # Eval without noise

        if model is None:
            if args.load_model:
                print(f"Loading CV model from {args.load_model}")
                model = PPO.load(args.load_model, env=train_env)
                apply_resume_overrides(model, args)
            else:
                model = PPO(
                    'MlpPolicy',
                    train_env,
                    learning_rate=args.lr,
                    n_steps=2048,
                    batch_size=64,
                    n_epochs=10,
                    gamma=0.99,
                    gae_lambda=0.95,
                    clip_range=0.2,
                    ent_coef=args.ent_coef,
                    policy_kwargs=dict(net_arch=dict(pi=[256, 256, 128], vf=[256, 256, 128])),
                    tensorboard_log=os.path.join(log_dir, 'tensorboard'),
                    verbose=1,
                )
        else:
            model.set_env(train_env)
            apply_resume_overrides(model, args)

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
    parser.add_argument('--lr', type=float, default=3e-4,
                        help='Learning rate')
    parser.add_argument('--ent-coef', type=float, default=0.02,
                        help='Entropy coefficient (higher = more exploration)')
    parser.add_argument('--noise-std', type=float, default=0.02,
                        help='Gaussian noise std for later nights (0=no noise)')
    parser.add_argument('--detection-rate', type=float, default=0.90,
                        help='CV detection rate for later nights (1.0=perfect)')
    parser.add_argument('--load-model', type=str, default=None,
                        help='Path to an existing CV model to continue training from')
    parser.add_argument('--disable-reward-shaping', action='store_true',
                        help='Use the base CV environment reward without extra shaping')
    parser.add_argument('--defensive-bonus', type=float, default=0.05,
                        help='Reward for closing door against a doorway threat')
    parser.add_argument('--approaching-bonus', type=float, default=0.01,
                        help='Reward for closing door against a hallway threat')
    parser.add_argument('--power-penalty-scale', type=float, default=0.01,
                        help='Penalty scale for power usage')
    parser.add_argument('--waste-penalty', type=float, default=0.01,
                        help='Penalty for closing doors with no nearby threat')
    parser.add_argument('--open-door-penalty', type=float, default=0.08,
                        help='Penalty for leaving a door open against a knowable threat')
    parser.add_argument('--light-bonus', type=float, default=0.02,
                        help='Reward for confirming a threat with the lights')
    parser.add_argument('--info-gain-bonus', type=float, default=0.005,
                        help='Reward for actually refreshing CV memory with a sighting')
    parser.add_argument('--info-refresh-threshold', type=float, default=5.0,
                        help='Minimum age in seconds before a repeated sighting counts as fresh info')
    parser.add_argument('--stale-light-bonus', type=float, default=0.01,
                        help='Reward for turning on a doorway light when that side has stale memory')
    parser.add_argument('--stale-light-threshold', type=float, default=8.0,
                        help='Memory age in seconds before a doorway light check counts as stale scouting')
    parser.add_argument('--light-scout-cooldown', type=float, default=5.0,
                        help='Minimum seconds between rewarded stale doorway checks on the same side')
    parser.add_argument('--invalid-camera-penalty', type=float, default=0.05,
                        help='Penalty for selecting a camera while the monitor is down')
    parser.add_argument('--redundant-camera-penalty', type=float, default=0.005,
                        help='Penalty for re-selecting the camera already being viewed')
    parser.add_argument('--log-dir', type=str, default='./training_logs/cv')
    args = parser.parse_args()
    train_cv(args)


if __name__ == '__main__':
    main()
