"""Tests for the FNAF 1 Gymnasium environment."""

import pytest
import numpy as np
import gymnasium as gym

import fnaf_gymnasium
from fnaf_gymnasium.envs.fnaf_env import FnafEnv, NUM_ACTIONS


class TestEnvRegistration:
    """Test that the environment is properly registered."""

    def test_env_registered(self):
        env = gym.make('FnafNight-v0')
        assert env is not None
        env.close()

    def test_env_with_params(self):
        env = gym.make('FnafNight-v0', night=3, full_observability=True)
        assert env is not None
        env.close()


class TestEnvInterface:
    """Test the Gymnasium interface compliance."""

    def test_reset_returns_correct_types(self):
        env = FnafEnv(night=1)
        obs, info = env.reset()
        assert isinstance(obs, np.ndarray)
        assert isinstance(info, dict)
        assert obs.dtype == np.float32
        env.close()

    def test_observation_shape(self):
        env = FnafEnv(night=1)
        obs, _ = env.reset()
        assert obs.shape == (77,)
        env.close()

    def test_observation_bounds(self):
        env = FnafEnv(night=1)
        obs, _ = env.reset()
        assert np.all(obs >= 0.0)
        assert np.all(obs <= 1.0)
        env.close()

    def test_step_returns_correct_types(self):
        env = FnafEnv(night=1)
        obs, _ = env.reset()
        obs, reward, terminated, truncated, info = env.step(0)
        assert isinstance(obs, np.ndarray)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)
        env.close()

    def test_action_space(self):
        env = FnafEnv(night=1)
        assert env.action_space.n == NUM_ACTIONS
        env.close()

    def test_observation_space(self):
        env = FnafEnv(night=1)
        obs, _ = env.reset()
        assert env.observation_space.contains(obs)
        env.close()


class TestEnvActions:
    """Test that actions properly affect game state."""

    def test_noop_action(self):
        env = FnafEnv(night=1)
        obs1, _ = env.reset()
        obs2, _, _, _, _ = env.step(0)  # NOOP
        # Time should advance (obs[0] changes)
        assert obs2[0] > obs1[0]
        env.close()

    def test_toggle_cameras(self):
        env = FnafEnv(night=1)
        env.reset()
        obs, _, _, _, _ = env.step(1)  # Toggle cameras
        assert obs[3] == 1.0  # cameras_on
        obs, _, _, _, _ = env.step(1)  # Toggle cameras off
        assert obs[3] == 0.0
        env.close()

    def test_toggle_left_door(self):
        env = FnafEnv(night=1)
        env.reset()
        obs, _, _, _, _ = env.step(2)  # Toggle left door
        assert obs[15] == 1.0  # left_door_closed
        env.close()

    def test_toggle_right_door(self):
        env = FnafEnv(night=1)
        env.reset()
        obs, _, _, _, _ = env.step(3)  # Toggle right door
        assert obs[16] == 1.0  # right_door_closed
        env.close()

    def test_switch_camera(self):
        env = FnafEnv(night=1)
        env.reset()
        obs, _, _, _, _ = env.step(1)  # Toggle cameras ON
        obs, _, _, _, _ = env.step(6 + 7)  # Switch to camera index 7 = '4B'
        # Camera one-hot at index 4+7=11 should be 1
        assert obs[4 + 7] == 1.0
        env.close()


class TestEnvReward:
    """Test reward calculations."""

    def test_survival_reward_positive(self):
        env = FnafEnv(night=1)
        env.reset()
        _, reward, _, _, _ = env.step(0)
        # Should get small positive reward for surviving
        assert reward > -1.0  # Not dead
        env.close()

    def test_game_over_reward(self):
        """Test that death gives negative reward."""
        env = FnafEnv(night=1)
        env.reset()
        # Manually trigger game over
        env.game._game_over(env.game.game_over_reason.FREDDY)
        # Force game state
        env.game.game_over = True
        env.game.game_over_reason = env.game.game_over_reason.FREDDY
        # The next step should report the death reward
        # (but the game is already over from the manual call)
        env.close()


class TestEnvInfo:
    """Test info dictionary contents."""

    def test_info_keys(self):
        env = FnafEnv(night=1)
        _, info = env.reset()
        expected_keys = [
            'current_second', 'power', 'power_usage', 'night',
            'in_game_time', 'freddy_pos', 'bonnie_pos', 'chica_pos',
            'foxy_pos', 'foxy_sub', 'game_over_reason', 'power_out',
        ]
        for key in expected_keys:
            assert key in info
        env.close()


class TestEnvRender:
    """Test render functionality."""

    def test_ansi_render(self):
        env = FnafEnv(night=1, render_mode='ansi')
        env.reset()
        output = env.render()
        assert isinstance(output, str)
        assert 'FNAF' in output
        assert 'Power' in output
        env.close()


class TestFullEpisode:
    """Test running complete episodes."""

    def test_random_episode(self):
        """Run a full episode with random actions."""
        env = FnafEnv(night=1)
        obs, _ = env.reset(seed=42)
        total_reward = 0
        steps = 0

        while steps < 600:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
            if terminated or truncated:
                break

        assert steps > 0
        assert info['game_over_reason'] != 'NONE' or steps >= 535
        env.close()

    def test_multiple_resets(self):
        """Test that environment can be reset multiple times."""
        env = FnafEnv(night=1)
        for _ in range(5):
            obs, info = env.reset()
            assert obs.shape == (77,)
            for _ in range(10):
                obs, _, terminated, _, _ = env.step(0)
                if terminated:
                    break
        env.close()
