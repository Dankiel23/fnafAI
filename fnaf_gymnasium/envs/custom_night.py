"""
Custom night environment variant.

Allows setting arbitrary AI levels for each animatronic,
enabling fine-grained difficulty control for training.
"""

import gymnasium as gym
from typing import Optional

from .fnaf_env import FnafEnv
from .game_logic import FNAFGame


class FnafCustomNightEnv(FnafEnv):
    """
    FNAF environment with custom AI level configuration.

    Instead of selecting a night (1-7), directly set starting AI
    levels for each animatronic.

    Usage:
        env = FnafCustomNightEnv(
            freddy_ai=5,
            bonnie_ai=10,
            chica_ai=8,
            foxy_ai=12,
        )
    """

    def __init__(self, render_mode=None, freddy_ai=0, bonnie_ai=0,
                 chica_ai=0, foxy_ai=0, step_duration=1.0,
                 full_observability=True):
        super().__init__(
            render_mode=render_mode,
            night=7,  # Custom night uses slot 7
            step_duration=step_duration,
            full_observability=full_observability,
        )
        self.custom_freddy_ai = freddy_ai
        self.custom_bonnie_ai = bonnie_ai
        self.custom_chica_ai = chica_ai
        self.custom_foxy_ai = foxy_ai

    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)

        # Override AI levels with custom values
        self.game.freddy.current_ai_level = self.custom_freddy_ai
        self.game.bonnie.current_ai_level = self.custom_bonnie_ai
        self.game.chica.current_ai_level = self.custom_chica_ai
        self.game.foxy.current_ai_level = self.custom_foxy_ai

        # Re-generate observation with correct AI levels
        obs = self._get_obs()
        info = self._get_info()
        return obs, info


class FnafRandomDifficultyEnv(FnafEnv):
    """
    FNAF environment that randomizes difficulty each episode.

    Useful for training agents that generalize across all difficulty levels.
    Each reset picks random AI levels within specified bounds.
    """

    def __init__(self, render_mode=None, min_ai=0, max_ai=20,
                 step_duration=1.0, full_observability=True):
        super().__init__(
            render_mode=render_mode,
            night=7,
            step_duration=step_duration,
            full_observability=full_observability,
        )
        self.min_ai = min_ai
        self.max_ai = max_ai

    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)

        import random as rng
        if seed is not None:
            rng.seed(seed)

        self.game.freddy.current_ai_level = rng.randint(self.min_ai, self.max_ai)
        self.game.bonnie.current_ai_level = rng.randint(self.min_ai, self.max_ai)
        self.game.chica.current_ai_level = rng.randint(self.min_ai, self.max_ai)
        self.game.foxy.current_ai_level = rng.randint(self.min_ai, self.max_ai)

        obs = self._get_obs()
        info = self._get_info()
        return obs, info
