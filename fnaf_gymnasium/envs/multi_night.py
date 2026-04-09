"""
Multi-night environment that chains multiple nights together.

The agent must survive consecutive nights in a single episode,
similar to the actual FNAF game progression.
"""

import gymnasium as gym
import numpy as np
from typing import Optional

from .fnaf_env import FnafEnv
from .game_logic import FNAFGame


class FnafMultiNightEnv(FnafEnv):
    """
    Environment where the agent plays through multiple consecutive nights.

    The observation is augmented with the current night number.
    Power resets between nights but AI levels increase.

    Observation space: 78 dimensions (77 base + 1 for current night)
    """

    def __init__(self, render_mode=None, start_night=1, end_night=5,
                 step_duration=1.0, full_observability=True):
        self.start_night = start_night
        self.end_night = end_night
        self.current_night_num = start_night

        super().__init__(
            render_mode=render_mode,
            night=start_night,
            step_duration=step_duration,
            full_observability=full_observability,
        )

        # Extend observation space by 1 for night number
        self.obs_size = 78
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(self.obs_size,), dtype=np.float32
        )

        self.total_reward_all_nights = 0.0
        self.nights_survived = 0

    def _get_obs(self) -> np.ndarray:
        base_obs = super()._get_obs()
        # Append normalized night number
        night_obs = np.array([self.current_night_num / 7.0], dtype=np.float32)
        return np.concatenate([base_obs, night_obs])

    def _get_info(self) -> dict:
        info = super()._get_info()
        info['current_night_in_sequence'] = self.current_night_num
        info['nights_survived'] = self.nights_survived
        info['total_nights'] = self.end_night - self.start_night + 1
        return info

    def reset(self, seed=None, options=None):
        self.current_night_num = self.start_night
        self.night = self.start_night
        self.total_reward_all_nights = 0.0
        self.nights_survived = 0
        return super().reset(seed=seed, options=options)

    def step(self, action: int):
        obs, reward, terminated, truncated, info = super().step(action)

        if terminated:
            if self.game.game_over_reason.name == 'SURVIVED':
                self.nights_survived += 1
                reward += 50.0  # Bonus for completing a night

                if self.current_night_num < self.end_night:
                    # Advance to next night
                    self.current_night_num += 1
                    self.night = self.current_night_num
                    self.game = FNAFGame(night=self.current_night_num)
                    obs = self._get_obs()
                    info = self._get_info()
                    terminated = False  # Continue to next night
                else:
                    # Beat all nights!
                    reward += 200.0
                    info['all_nights_complete'] = True
            # If died, terminated stays True

        self.total_reward_all_nights += reward
        return obs, reward, terminated, truncated, info
