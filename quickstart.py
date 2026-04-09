"""
Quick start script - trains a PPO agent on Night 1 and evaluates it.

Usage:
    pip install -e ".[train]"
    python quickstart.py
"""

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

import fnaf_gymnasium
from fnaf_gymnasium.callbacks import FnafMetricsCallback


def main():
    print("FNAF 1 RL Quick Start")
    print("=" * 40)

    # Create training environments
    print("Creating environments...")
    train_env = DummyVecEnv([
        lambda: gym.make('FnafNight-v0', night=1)
        for _ in range(4)
    ])

    # Create model
    print("Initializing PPO agent...")
    model = PPO(
        'MlpPolicy',
        train_env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        ent_coef=0.01,
        verbose=1,
    )

    # Train
    print("\nTraining for 100k steps on Night 1...")
    metrics_callback = FnafMetricsCallback(log_freq=5000, verbose=1)
    model.learn(total_timesteps=100_000, callback=metrics_callback, progress_bar=True)

    train_env.close()

    # Evaluate
    print("\n\nEvaluating trained agent...")
    eval_env = gym.make('FnafNight-v0', night=1, render_mode='human')

    wins = 0
    episodes = 20

    for ep in range(episodes):
        obs, info = eval_env.reset(seed=ep)
        total_reward = 0
        done = False

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = eval_env.step(action)
            total_reward += reward
            done = terminated or truncated

        survived = info.get('game_over_reason') == 'SURVIVED'
        if survived:
            wins += 1

        status = "SURVIVED" if survived else f"DIED ({info['game_over_reason']})"
        print(f"  Episode {ep+1}: {status} | Power: {info['power']:.1f}% | Reward: {total_reward:.2f}")

    eval_env.close()

    print(f"\nFinal win rate: {wins}/{episodes} ({100*wins/episodes:.1f}%)")
    model.save("fnaf_quickstart_model")
    print("Model saved to fnaf_quickstart_model.zip")


if __name__ == '__main__':
    main()
