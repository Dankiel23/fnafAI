"""FNAF 1 Gymnasium Environment for Reinforcement Learning."""

from gymnasium.envs.registration import register
from .wrappers import (
    FnafRewardShaping,
    FnafSparseReward,
    FnafTimePenalty,
    FnafActionMask,
)

register(
    id='FnafNight-v0',
    entry_point='fnaf_gymnasium.envs:FnafEnv',
    max_episode_steps=535,  # One step per second, 535 seconds per night
)

register(
    id='FnafCustomNight-v0',
    entry_point='fnaf_gymnasium.envs.custom_night:FnafCustomNightEnv',
    max_episode_steps=535,
)

register(
    id='FnafRandomDifficulty-v0',
    entry_point='fnaf_gymnasium.envs.custom_night:FnafRandomDifficultyEnv',
    max_episode_steps=535,
)

register(
    id='FnafMultiNight-v0',
    entry_point='fnaf_gymnasium.envs.multi_night:FnafMultiNightEnv',
    max_episode_steps=2675,  # 535 seconds x 5 nights max
)

register(
    id='FnafCVReady-v0',
    entry_point='fnaf_gymnasium.envs.cv_env:FnafCVReadyEnv',
    max_episode_steps=535,
)

__all__ = [
    'FnafRewardShaping',
    'FnafSparseReward',
    'FnafTimePenalty',
    'FnafActionMask',
]
