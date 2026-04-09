"""
Demo script to watch the FNAF environment in action.
Runs a simple heuristic agent that demonstrates the environment.
"""

import gymnasium as gym
import fnaf_gymnasium
from fnaf_gymnasium.envs.fnaf_env import (
    ACTION_NOOP, ACTION_TOGGLE_CAMERAS, ACTION_TOGGLE_LEFT_DOOR,
    ACTION_TOGGLE_RIGHT_DOOR, ACTION_TOGGLE_LEFT_LIGHT, ACTION_TOGGLE_RIGHT_LIGHT,
)


def heuristic_agent(obs, info):
    """
    Simple rule-based agent for FNAF.

    Strategy:
    - Check cameras periodically to keep Foxy at bay
    - Close doors when animatronics are in doorways
    - Conserve power when safe
    """
    left_doorway = obs[73] > 0.5
    right_doorway = obs[74] > 0.5
    cameras_on = obs[3] > 0.5
    left_door_closed = obs[15] > 0.5
    right_door_closed = obs[16] > 0.5
    foxy_attacking = obs[75] > 0.5
    power = obs[1]

    # Emergency: close doors if threats at doorway
    if left_doorway and not left_door_closed:
        return ACTION_TOGGLE_LEFT_DOOR
    if right_doorway and not right_door_closed:
        return ACTION_TOGGLE_RIGHT_DOOR

    # Open doors if no threat (save power)
    if not left_doorway and left_door_closed:
        return ACTION_TOGGLE_LEFT_DOOR
    if not right_doorway and right_door_closed:
        return ACTION_TOGGLE_RIGHT_DOOR

    # Toggle cameras periodically to keep Foxy in check
    time_norm = obs[0]
    # Use cameras every ~10 seconds (check time modulo)
    should_check_cameras = int(time_norm * 535) % 10 < 3

    if should_check_cameras and not cameras_on:
        return ACTION_TOGGLE_CAMERAS
    elif not should_check_cameras and cameras_on:
        return ACTION_TOGGLE_CAMERAS

    return ACTION_NOOP


def main():
    env = gym.make('FnafNight-v0', render_mode='human', night=1)
    obs, info = env.reset(seed=42)

    total_reward = 0
    step = 0

    print("Starting FNAF Night 1 demo with heuristic agent...\n")

    while True:
        action = heuristic_agent(obs, info)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        step += 1

        # Render every 10 steps
        if step % 30 == 0:
            env.render()
            print(f"  Step {step} | Reward: {total_reward:.2f}\n")

        if terminated or truncated:
            env.render()
            print(f"\n{'='*50}")
            print(f"Game Over after {step} steps")
            print(f"Result: {info['game_over_reason']}")
            print(f"Total Reward: {total_reward:.2f}")
            print(f"Final Power: {info['power']:.1f}%")
            print(f"{'='*50}")
            break

    env.close()


if __name__ == '__main__':
    main()
