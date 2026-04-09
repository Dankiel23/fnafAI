"""
Analyze saved episode replays to understand agent behavior.

Usage:
    python analyze_replay.py replay.json
"""

import argparse
import json
import sys


def analyze(filepath: str):
    """Analyze a saved episode replay."""
    with open(filepath) as f:
        data = json.load(f)

    steps = data['steps']
    total_steps = data['total_steps']
    outcome = data['outcome']
    final_power = data['final_power']
    total_reward = data['total_reward']

    print(f"Replay Analysis: {filepath}")
    print(f"{'='*60}")
    print(f"Outcome: {outcome}")
    print(f"Total steps: {total_steps}")
    print(f"Total reward: {total_reward:.2f}")
    print(f"Final power: {final_power:.1f}%")

    # Action distribution
    action_counts = {}
    for s in steps:
        name = s['action_name']
        action_counts[name] = action_counts.get(name, 0) + 1

    print(f"\nAction Distribution:")
    for name, count in sorted(action_counts.items(), key=lambda x: -x[1]):
        pct = count / total_steps * 100
        bar = '#' * int(pct / 2)
        print(f"  {name:<25s} {count:>4d} ({pct:>5.1f}%) {bar}")

    # Power usage over time
    print(f"\nPower Timeline:")
    checkpoints = [0, 0.25, 0.5, 0.75, 1.0]
    for cp in checkpoints:
        idx = min(int(cp * (total_steps - 1)), total_steps - 1)
        s = steps[idx]
        print(f"  {s['second']:>5.0f}s ({cp*100:>3.0f}%): Power {s['power']:>5.1f}% | "
              f"Usage x{s['power_usage']}")

    # Threat response analysis
    threat_steps = [s for s in steps if s['left_doorway_threat'] or s['right_doorway_threat']]
    if threat_steps:
        correct_responses = 0
        for s in threat_steps:
            if s['left_doorway_threat'] and s['left_door']:
                correct_responses += 1
            elif s['right_doorway_threat'] and s['right_door']:
                correct_responses += 1

        print(f"\nThreat Response:")
        print(f"  Doorway threats: {len(threat_steps)} steps")
        print(f"  Correct door closures: {correct_responses}/{len(threat_steps)} "
              f"({correct_responses/len(threat_steps)*100:.1f}%)")

    # Camera usage patterns
    cam_on_steps = sum(1 for s in steps if s['cameras_on'])
    print(f"\nCamera Usage:")
    print(f"  Cameras on: {cam_on_steps}/{total_steps} steps ({cam_on_steps/total_steps*100:.1f}%)")

    # Track camera switching
    cam_switches = 0
    for i in range(1, len(steps)):
        if steps[i]['current_camera'] != steps[i-1]['current_camera']:
            cam_switches += 1
    print(f"  Camera switches: {cam_switches}")

    # Camera distribution
    cam_views = {}
    for s in steps:
        if s['cameras_on']:
            cam = s['current_camera']
            cam_views[cam] = cam_views.get(cam, 0) + 1

    if cam_views:
        print(f"  Camera view distribution:")
        for cam, count in sorted(cam_views.items(), key=lambda x: -x[1]):
            print(f"    {cam}: {count} steps")

    # Animatronic position tracking
    print(f"\nAnimatronic Positions at Key Moments:")
    key_seconds = [0, 89, 179, 268, 357, 446, 535]
    for sec in key_seconds:
        matching = [s for s in steps if abs(s['second'] - sec) < 1]
        if matching:
            s = matching[0]
            print(f"  {sec:>3d}s: F={s['freddy_pos']:<6s} B={s['bonnie_pos']:<6s} "
                  f"C={s['chica_pos']:<6s} Fx={s['foxy_pos']:<4s}({s['foxy_sub']})")

    # Door usage efficiency
    door_closed_steps = sum(1 for s in steps if s['left_door'] or s['right_door'])
    print(f"\nDoor Usage:")
    print(f"  Steps with any door closed: {door_closed_steps}/{total_steps} "
          f"({door_closed_steps/total_steps*100:.1f}%)")

    left_closed = sum(1 for s in steps if s['left_door'])
    right_closed = sum(1 for s in steps if s['right_door'])
    print(f"  Left door closed: {left_closed} steps")
    print(f"  Right door closed: {right_closed} steps")

    # Reward breakdown
    print(f"\nReward Curve:")
    quarter = max(total_steps // 4, 1)
    for i in range(4):
        start = i * quarter
        end = min((i + 1) * quarter, total_steps)
        segment_rewards = [steps[j]['reward'] for j in range(start, end)]
        avg_reward = sum(segment_rewards) / len(segment_rewards) if segment_rewards else 0
        print(f"  Q{i+1} (steps {start}-{end}): avg reward {avg_reward:>+.4f}")


def main():
    parser = argparse.ArgumentParser(description='Analyze FNAF replay')
    parser.add_argument('replay_file', type=str)
    args = parser.parse_args()
    analyze(args.replay_file)


if __name__ == '__main__':
    main()
