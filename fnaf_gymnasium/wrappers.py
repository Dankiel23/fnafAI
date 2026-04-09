"""
Custom Gymnasium wrappers for the FNAF environment.

These wrappers modify observation/reward/action spaces for
different training configurations.
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class FnafRewardShaping(gym.Wrapper):
    """
    Configurable reward shaping wrapper.

    Allows tuning reward components independently:
    - survival_bonus: reward per step for staying alive
    - death_penalty: penalty for dying
    - win_bonus: reward for surviving the night
    - power_penalty_scale: scale for power usage penalty
    - defensive_bonus: reward for closing doors against threats
    - waste_penalty: penalty for closing doors with no threat
    """

    def __init__(self, env, survival_bonus=0.01, death_penalty=-100.0,
                 win_bonus=100.0, power_penalty_scale=0.005,
                 defensive_bonus=0.02, waste_penalty=0.01):
        super().__init__(env)
        self.survival_bonus = survival_bonus
        self.death_penalty = death_penalty
        self.win_bonus = win_bonus
        self.power_penalty_scale = power_penalty_scale
        self.defensive_bonus = defensive_bonus
        self.waste_penalty = waste_penalty

    def step(self, action):
        obs, _, terminated, truncated, info = self.env.step(action)

        reward = 0.0
        game = self.env.unwrapped.game

        if terminated:
            if info.get('game_over_reason') == 'SURVIVED':
                reward = self.win_bonus
            else:
                reward = self.death_penalty
        else:
            reward += self.survival_bonus
            reward -= self.power_penalty_scale * info.get('power_usage', 1)

            left_threat = obs[73] > 0.5
            right_threat = obs[74] > 0.5
            left_door = obs[15] > 0.5
            right_door = obs[16] > 0.5

            if left_threat and left_door:
                reward += self.defensive_bonus
            if right_threat and right_door:
                reward += self.defensive_bonus
            if not left_threat and left_door:
                reward -= self.waste_penalty
            if not right_threat and right_door:
                reward -= self.waste_penalty

        return obs, reward, terminated, truncated, info


class FnafSparseReward(gym.Wrapper):
    """
    Sparse reward wrapper - only rewards at episode end.
    +1 for survival, -1 for death.
    """

    def step(self, action):
        obs, _, terminated, truncated, info = self.env.step(action)
        reward = 0.0
        if terminated:
            if info.get('game_over_reason') == 'SURVIVED':
                reward = 1.0
            else:
                reward = -1.0
        return obs, reward, terminated, truncated, info


class FnafTimePenalty(gym.Wrapper):
    """
    Adds a small time-based penalty that increases as time goes on,
    encouraging the agent to develop efficient strategies.
    """

    def __init__(self, env, scale=0.001):
        super().__init__(env)
        self.scale = scale

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        time_progress = obs[0]  # normalized time
        reward -= self.scale * time_progress
        return obs, reward, terminated, truncated, info


class FnafActionMask(gym.Wrapper):
    """
    Wrapper that provides action masking info.
    Useful for algorithms that support invalid action masking.

    Masks out redundant actions (e.g., switching cameras when cameras off).
    """

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        info['action_mask'] = self._get_action_mask(obs)
        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        info['action_mask'] = self._get_action_mask(obs)
        return obs, info

    def _get_action_mask(self, obs):
        """Return boolean mask of valid actions."""
        mask = np.ones(self.env.action_space.n, dtype=bool)
        cameras_on = obs[3] > 0.5
        power_out = obs[19] > 0.5

        if power_out:
            # Only NOOP is valid during power outage
            mask[:] = False
            mask[0] = True
            return mask

        # Camera switching only valid when cameras are on
        if not cameras_on:
            mask[6:17] = False

        return mask
