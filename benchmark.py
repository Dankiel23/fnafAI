"""
Benchmark script to measure environment performance and
test different agent strategies across multiple nights.
"""

import time
import random
import statistics

import gymnasium as gym
import numpy as np

import fnaf_gymnasium
from fnaf_gymnasium.envs.fnaf_env import (
    ACTION_NOOP, ACTION_TOGGLE_CAMERAS, ACTION_TOGGLE_LEFT_DOOR,
    ACTION_TOGGLE_RIGHT_DOOR, ACTION_TOGGLE_LEFT_LIGHT, ACTION_TOGGLE_RIGHT_LIGHT,
    NUM_ACTIONS,
)


def random_agent(obs, info):
    return random.randint(0, NUM_ACTIONS - 1)


def do_nothing_agent(obs, info):
    return ACTION_NOOP


def doors_only_agent(obs, info):
    """Always keep both doors closed."""
    left_door = obs[15] > 0.5
    right_door = obs[16] > 0.5
    if not left_door:
        return ACTION_TOGGLE_LEFT_DOOR
    if not right_door:
        return ACTION_TOGGLE_RIGHT_DOOR
    return ACTION_NOOP


def smart_agent(obs, info):
    """Heuristic agent with camera cycling and reactive door management."""
    left_doorway = obs[73] > 0.5
    right_doorway = obs[74] > 0.5
    cameras_on = obs[3] > 0.5
    left_door = obs[15] > 0.5
    right_door = obs[16] > 0.5
    power = obs[1]
    time_norm = obs[0]

    # Close doors when threats detected
    if left_doorway and not left_door:
        return ACTION_TOGGLE_LEFT_DOOR
    if right_doorway and not right_door:
        return ACTION_TOGGLE_RIGHT_DOOR

    # Open doors when safe
    if not left_doorway and left_door:
        return ACTION_TOGGLE_LEFT_DOOR
    if not right_doorway and right_door:
        return ACTION_TOGGLE_RIGHT_DOOR

    # Camera cycling to suppress Foxy
    second = int(time_norm * 535)
    if second % 8 < 3 and not cameras_on:
        return ACTION_TOGGLE_CAMERAS
    if second % 8 >= 3 and cameras_on:
        return ACTION_TOGGLE_CAMERAS

    # Low power conservation - stop using cameras
    if power < 0.15 and cameras_on:
        return ACTION_TOGGLE_CAMERAS

    return ACTION_NOOP


def run_benchmark(agent_fn, agent_name, night, num_episodes=100):
    """Run benchmark for a given agent."""
    env = gym.make('FnafNight-v0', night=night)
    wins = 0
    rewards = []
    steps_list = []
    powers = []
    start = time.time()

    for ep in range(num_episodes):
        obs, info = env.reset(seed=ep)
        total_reward = 0
        steps = 0

        while True:
            action = agent_fn(obs, info)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
            if terminated or truncated:
                break

        if info.get('game_over_reason') == 'SURVIVED':
            wins += 1
        rewards.append(total_reward)
        steps_list.append(steps)
        powers.append(info['power'])

    elapsed = time.time() - start
    env.close()

    return {
        'agent': agent_name,
        'night': night,
        'episodes': num_episodes,
        'win_rate': wins / num_episodes,
        'avg_reward': statistics.mean(rewards),
        'std_reward': statistics.stdev(rewards) if len(rewards) > 1 else 0,
        'avg_steps': statistics.mean(steps_list),
        'avg_final_power': statistics.mean(powers),
        'eps_per_sec': num_episodes / elapsed,
        'elapsed': elapsed,
    }


def main():
    agents = [
        (random_agent, 'Random'),
        (do_nothing_agent, 'Do Nothing'),
        (doors_only_agent, 'Doors Only'),
        (smart_agent, 'Smart Heuristic'),
    ]

    nights = [1, 2, 3]
    num_episodes = 200

    print(f"FNAF Gymnasium Benchmark - {num_episodes} episodes per config")
    print("=" * 90)
    print(f"{'Agent':<20} {'Night':<6} {'Win%':<8} {'Avg Reward':<12} {'Avg Steps':<10} {'Avg Power':<10} {'Eps/s':<8}")
    print("-" * 90)

    for night in nights:
        for agent_fn, agent_name in agents:
            result = run_benchmark(agent_fn, agent_name, night, num_episodes)
            print(
                f"{result['agent']:<20} "
                f"{result['night']:<6} "
                f"{result['win_rate']*100:>5.1f}%  "
                f"{result['avg_reward']:>10.2f}  "
                f"{result['avg_steps']:>8.1f}  "
                f"{result['avg_final_power']:>8.1f}%  "
                f"{result['eps_per_sec']:>6.1f}"
            )
        print("-" * 90)

    # Performance benchmark
    print("\nEnvironment throughput benchmark...")
    env = gym.make('FnafNight-v0', night=1)
    obs, _ = env.reset()
    start = time.time()
    total_steps = 0
    for _ in range(100):
        obs, _ = env.reset()
        while True:
            obs, _, terminated, truncated, _ = env.step(env.action_space.sample())
            total_steps += 1
            if terminated or truncated:
                break
    elapsed = time.time() - start
    env.close()
    print(f"  {total_steps} steps in {elapsed:.2f}s = {total_steps/elapsed:.0f} steps/sec")


if __name__ == '__main__':
    main()
