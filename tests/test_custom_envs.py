"""Tests for custom environment variants."""

import pytest
import numpy as np
import gymnasium as gym

import fnaf_gymnasium
from fnaf_gymnasium.envs.custom_night import FnafCustomNightEnv, FnafRandomDifficultyEnv
from fnaf_gymnasium.envs.multi_night import FnafMultiNightEnv


class TestCustomNightEnv:

    def test_custom_ai_levels(self):
        env = FnafCustomNightEnv(freddy_ai=10, bonnie_ai=15, chica_ai=5, foxy_ai=20)
        obs, info = env.reset()
        assert env.game.freddy.current_ai_level == 10
        assert env.game.bonnie.current_ai_level == 15
        assert env.game.chica.current_ai_level == 5
        assert env.game.foxy.current_ai_level == 20
        env.close()

    def test_custom_night_registration(self):
        env = gym.make('FnafCustomNight-v0', freddy_ai=5, bonnie_ai=5)
        obs, info = env.reset()
        assert obs.shape == (77,)
        env.close()

    def test_custom_night_step(self):
        env = FnafCustomNightEnv(freddy_ai=0, bonnie_ai=0, chica_ai=0, foxy_ai=0)
        obs, _ = env.reset()
        obs2, reward, term, trunc, info = env.step(0)
        assert isinstance(reward, float)
        env.close()


class TestRandomDifficultyEnv:

    def test_random_ai_in_range(self):
        env = FnafRandomDifficultyEnv(min_ai=5, max_ai=10)
        for _ in range(20):
            obs, _ = env.reset()
            for anim in [env.game.freddy, env.game.bonnie, env.game.chica, env.game.foxy]:
                assert 5 <= anim.current_ai_level <= 10
        env.close()

    def test_random_difficulty_registration(self):
        env = gym.make('FnafRandomDifficulty-v0', min_ai=0, max_ai=5)
        obs, _ = env.reset()
        assert obs.shape == (77,)
        env.close()


class TestMultiNightEnv:

    def test_observation_shape(self):
        env = FnafMultiNightEnv(start_night=1, end_night=3)
        obs, info = env.reset()
        assert obs.shape == (78,)
        env.close()

    def test_starts_at_night_1(self):
        env = FnafMultiNightEnv(start_night=1, end_night=3)
        obs, info = env.reset()
        assert info['current_night_in_sequence'] == 1
        assert info['nights_survived'] == 0
        env.close()

    def test_night_encoded_in_obs(self):
        env = FnafMultiNightEnv(start_night=1, end_night=5)
        obs, _ = env.reset()
        # Last element should be night/7 = 1/7
        assert abs(obs[-1] - 1.0/7.0) < 0.01
        env.close()

    def test_step_works(self):
        env = FnafMultiNightEnv(start_night=1, end_night=2)
        obs, _ = env.reset()
        obs2, reward, term, trunc, info = env.step(0)
        assert obs2.shape == (78,)
        env.close()
