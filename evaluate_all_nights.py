"""
Evaluate a trained model across all nights (1-7) to measure
how well the agent generalizes to increasing difficulty.

Usage:
    python evaluate_all_nights.py --model fnaf_quickstart_model.zip
"""

import argparse

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO, DQN, A2C

import fnaf_gymnasium


ALGORITHMS = {'ppo': PPO, 'dqn': DQN, 'a2c': A2C}


def evaluate_night(model, night, episodes=100):
    """Evaluate model on a specific night."""
    env = gym.make('FnafNight-v0', night=night)
    wins = 0
    rewards = []
    powers = []
    death_causes = {}

    for ep in range(episodes):
        obs, info = env.reset(seed=ep)
        total_reward = 0
        done = False

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated

        reason = info.get('game_over_reason', 'UNKNOWN')
        if reason == 'SURVIVED':
            wins += 1
        else:
            death_causes[reason] = death_causes.get(reason, 0) + 1

        rewards.append(total_reward)
        powers.append(info['power'])

    env.close()

    return {
        'night': night,
        'episodes': episodes,
        'wins': wins,
        'win_rate': wins / episodes,
        'avg_reward': np.mean(rewards),
        'std_reward': np.std(rewards),
        'avg_power': np.mean(powers),
        'death_causes': death_causes,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--algo', type=str, default='ppo', choices=list(ALGORITHMS.keys()))
    parser.add_argument('--episodes', type=int, default=100)
    parser.add_argument('--nights', type=str, default='1-7',
                        help='Night range, e.g. "1-5" or "1,3,5"')
    args = parser.parse_args()

    # Parse night range
    if '-' in args.nights:
        start, end = map(int, args.nights.split('-'))
        nights = list(range(start, end + 1))
    elif ',' in args.nights:
        nights = [int(n) for n in args.nights.split(',')]
    else:
        nights = [int(args.nights)]

    model = ALGORITHMS[args.algo].load(args.model)

    print(f"Evaluating {args.model} across nights {nights}")
    print(f"{'='*80}")
    print(f"{'Night':<8} {'Win Rate':<12} {'Avg Reward':<14} {'Avg Power':<12} {'Top Death Cause':<20}")
    print(f"{'-'*80}")

    all_results = []
    for night in nights:
        result = evaluate_night(model, night, args.episodes)
        all_results.append(result)

        top_cause = max(result['death_causes'].items(), key=lambda x: x[1])[0] if result['death_causes'] else '-'

        print(f"Night {result['night']:<4} "
              f"{result['win_rate']*100:>6.1f}%     "
              f"{result['avg_reward']:>10.2f}    "
              f"{result['avg_power']:>8.1f}%    "
              f"{top_cause}")

    print(f"{'-'*80}")

    overall_wins = sum(r['wins'] for r in all_results)
    overall_eps = sum(r['episodes'] for r in all_results)
    print(f"{'Overall':<8} {overall_wins/overall_eps*100:>6.1f}%     "
          f"{np.mean([r['avg_reward'] for r in all_results]):>10.2f}")

    # Detailed death breakdown
    print(f"\nDeath Cause Breakdown:")
    all_causes = {}
    for r in all_results:
        for cause, count in r['death_causes'].items():
            all_causes[cause] = all_causes.get(cause, 0) + count

    for cause, count in sorted(all_causes.items(), key=lambda x: -x[1]):
        print(f"  {cause}: {count} ({count/overall_eps*100:.1f}%)")


if __name__ == '__main__':
    main()
