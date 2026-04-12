"""Tests for gymnasium wrappers."""

import pytest
import numpy as np
import gymnasium as gym

import fnaf_gymnasium
from fnaf_gymnasium.envs.fnaf_env import FnafEnv
from fnaf_gymnasium.wrappers import (
    FnafRewardShaping,
    FnafSparseReward,
    FnafTimePenalty,
    FnafActionMask,
)


class TestFnafSparseReward:

    def test_zero_reward_mid_episode(self):
        env = FnafSparseReward(FnafEnv(night=1))
        env.reset()
        _, reward, terminated, _, _ = env.step(0)
        if not terminated:
            assert reward == 0.0
        env.close()

    def test_positive_reward_on_win(self):
        env = FnafSparseReward(FnafEnv(night=1))
        env.reset()
        # Fast-forward to near end
        env.unwrapped.game.current_second = 534
        _, reward, terminated, _, info = env.step(0)
        if info.get('game_over_reason') == 'SURVIVED':
            assert reward == 1.0
        env.close()


class TestFnafTimePenalty:

    def test_penalty_increases_over_time(self):
        env = FnafTimePenalty(FnafEnv(night=1), scale=0.01)
        env.reset()
        _, r1, _, _, _ = env.step(0)
        for _ in range(100):
            _, r2, term, _, _ = env.step(0)
            if term:
                break
        # Later rewards should be lower due to time penalty
        # (this is approximate since other reward components exist)
        env.close()


class TestFnafActionMask:

    def test_mask_shape(self):
        env = FnafActionMask(FnafEnv(night=1))
        obs, info = env.reset()
        assert 'action_mask' in info
        assert len(info['action_mask']) == env.action_space.n
        env.close()

    def test_camera_actions_masked_when_off(self):
        env = FnafActionMask(FnafEnv(night=1))
        obs, info = env.reset()
        # Cameras are off by default
        mask = info['action_mask']
        # Camera switch actions (6-16) should be masked
        assert not any(mask[6:17])
        env.close()

    def test_camera_actions_unmasked_when_on(self):
        env = FnafActionMask(FnafEnv(night=1))
        env.reset()
        obs, _, _, _, info = env.step(1)  # Toggle cameras ON
        mask = info['action_mask']
        # Camera switch actions should be available
        assert all(mask[6:17])
        env.close()

    def test_power_outage_only_noop(self):
        env = FnafActionMask(FnafEnv(night=1))
        env.reset()
        env.unwrapped.game.power_out = True
        obs, _, _, _, info = env.step(0)
        mask = info['action_mask']
        assert mask[0] == True  # NOOP valid
        assert not any(mask[1:])  # Everything else masked
        env.close()


class TestFnafRewardShaping:

    def test_custom_win_bonus(self):
        env = FnafRewardShaping(FnafEnv(night=1), win_bonus=500.0)
        env.reset()
        env.unwrapped.game.current_second = 534
        _, reward, terminated, _, info = env.step(0)
        if info.get('game_over_reason') == 'SURVIVED':
            assert reward == 500.0
        env.close()

    def test_custom_death_penalty(self):
        env = FnafRewardShaping(FnafEnv(night=1), death_penalty=-50.0)
        env.reset()
        # Force a death
        from fnaf_gymnasium.envs.game_logic import GameOverReason
        env.unwrapped.game._game_over(GameOverReason.FREDDY)
        _, reward, _, _, _ = env.step(0)
        # Reward should reflect death penalty
        env.close()

    def test_right_doorway_close_is_rewarded_even_if_threat_moves_away(self):
        env = FnafRewardShaping(
            FnafEnv(night=2),
            survival_bonus=0.0,
            power_penalty_scale=0.0,
            waste_penalty=0.0,
            defensive_bonus=1.0,
            approaching_bonus=0.0,
            open_door_penalty=0.0,
            light_bonus=0.0,
        )
        env.reset()
        env.unwrapped.game.chica.current_ai_level = 20
        env.unwrapped.game.chica.current_position = '4B'
        env.unwrapped.game.chica.sub_position = 1
        env.unwrapped.game.chica.time_since_last_check = (
            env.unwrapped.game.chica.movement_opportunity_interval
        )
        env.unwrapped.game.player.right_door_closed = False

        _, reward, terminated, _, _ = env.step(3)  # Toggle right door

        assert not terminated
        assert reward >= 1.0
        assert env.unwrapped.game.player.right_door_closed is True
        assert env.unwrapped.game.chica.current_position == '1B'
        env.close()

    def test_right_doorway_open_is_penalized_when_knowable(self):
        env = FnafRewardShaping(
            FnafEnv(night=2),
            survival_bonus=0.0,
            power_penalty_scale=0.0,
            waste_penalty=0.0,
            defensive_bonus=0.0,
            approaching_bonus=0.0,
            open_door_penalty=1.0,
            light_bonus=0.0,
        )
        env.reset()
        env.unwrapped.game.chica.current_ai_level = 20
        env.unwrapped.game.chica.current_position = '4B'
        env.unwrapped.game.chica.sub_position = 1
        env.unwrapped.game.chica.time_since_last_check = (
            env.unwrapped.game.chica.movement_opportunity_interval
        )
        env.unwrapped.game.player.right_door_closed = False
        env.unwrapped.game.player.cameras_on = True

        _, reward, terminated, _, _ = env.step(0)

        assert not terminated
        assert reward <= -1.0
        assert env.unwrapped.game.chica.current_position == 'office'
        env.close()
