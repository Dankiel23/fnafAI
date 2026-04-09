"""
Watch a trained agent play FNAF with detailed step-by-step output.

Usage:
    python watch_agent.py --model fnaf_quickstart_model.zip --night 1
"""

import argparse
import time

import gymnasium as gym
from stable_baselines3 import PPO

import fnaf_gymnasium
from fnaf_gymnasium.envs.replay import EpisodeRecorder, ACTION_NAMES


def watch(model_path: str, night: int = 1, speed: float = 0.1, save_replay: str = None):
    env = gym.make('FnafNight-v0', night=night, render_mode='ansi')
    model = PPO.load(model_path)
    recorder = EpisodeRecorder()

    obs, info = env.reset(seed=42)
    step_num = 0
    done = False

    print(f"\nWatching trained agent on Night {night}")
    print("=" * 60)

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        action = int(action)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        recorder.record_step(step_num, action, obs, reward, info, env.unwrapped.game)

        # Print current state
        action_name = ACTION_NAMES[action] if action < len(ACTION_NAMES) else f'Unknown({action})'
        time_info = info['in_game_time']
        time_str = f"{time_info['hour']}:{time_info['minute']:02d}AM"

        print(f"[{time_str}] Step {step_num:>3d} | "
              f"Action: {action_name:<25s} | "
              f"Power: {info['power']:>5.1f}% | "
              f"Reward: {reward:>+6.3f} | "
              f"F:{info['freddy_pos']:<6s} B:{info['bonnie_pos']:<6s} "
              f"C:{info['chica_pos']:<6s} Fx:{info['foxy_pos']:<4s}({info['foxy_sub']})")

        step_num += 1
        if speed > 0:
            time.sleep(speed)

    # Print summary
    print("\n" + recorder.summary())

    if save_replay:
        recorder.save(save_replay)
        print(f"\nReplay saved to {save_replay}")

    env.close()


def main():
    parser = argparse.ArgumentParser(description='Watch FNAF agent play')
    parser.add_argument('--model', type=str, default='fnaf_quickstart_model.zip')
    parser.add_argument('--night', type=int, default=1)
    parser.add_argument('--speed', type=float, default=0.05,
                        help='Delay between steps in seconds (0 for max speed)')
    parser.add_argument('--save-replay', type=str, default=None,
                        help='Save replay to JSON file')
    args = parser.parse_args()
    watch(args.model, args.night, args.speed, args.save_replay)


if __name__ == '__main__':
    main()
