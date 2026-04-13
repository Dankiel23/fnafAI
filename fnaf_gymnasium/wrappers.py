"""
Custom Gymnasium wrappers for the FNAF environment.

These wrappers modify observation/reward/action spaces for
different training configurations.
"""

import gymnasium as gym
import numpy as np

from .envs.game_logic import CAMERAS


class FnafRewardShaping(gym.Wrapper):
    """
    Configurable reward shaping wrapper.

    Allows tuning reward components independently:
    - survival_bonus: reward per step for staying alive
    - death_penalty: penalty for dying
    - win_bonus: reward for surviving the night
    - power_penalty_scale: scale for power usage penalty
    - defensive_bonus: reward for closing doors against threats in the doorway
    - approaching_bonus: reward for closing doors when a threat is in the hallway corner
    - waste_penalty: penalty for closing doors with no nearby threat at all
    """

    def __init__(self, env, survival_bonus=0.01, death_penalty=-100.0,
                 win_bonus=100.0, power_penalty_scale=0.005,
                 defensive_bonus=0.05, approaching_bonus=0.03,
                 waste_penalty=0.01, open_door_penalty=0.05,
                 light_bonus=0.03):
        super().__init__(env)
        self.survival_bonus = survival_bonus
        self.death_penalty = death_penalty
        self.win_bonus = win_bonus
        self.power_penalty_scale = power_penalty_scale
        self.defensive_bonus = defensive_bonus
        self.approaching_bonus = approaching_bonus
        self.waste_penalty = waste_penalty
        self.open_door_penalty = open_door_penalty
        self.light_bonus = light_bonus

    def _get_threat_level(self, game):
        """
        Returns (left_in_doorway, left_approaching, right_in_doorway, right_approaching).

        in_doorway: animatronic is at the door right now.
        approaching: animatronic is in the hallway corner and one move from the doorway.
        """
        left_doorway = game.is_animatronic_in_doorway('left')
        right_doorway = game.is_animatronic_in_doorway('right')

        left_approaching = (
            game.bonnie.current_position == '2B' and game.bonnie.sub_position == -1
        )
        right_approaching = (
            game.chica.current_position == '4B' and game.chica.sub_position == -1
        )

        return left_doorway, left_approaching, right_doorway, right_approaching

    def _door_and_light_state_from_obs(self, obs):
        """Extract door/light booleans from the flat observation."""
        return {
            'left_door': obs[15] > 0.5,
            'right_door': obs[16] > 0.5,
            'left_light': obs[17] > 0.5,
            'right_light': obs[18] > 0.5,
        }

    def step(self, action):
        game = self.env.unwrapped.game
        pre_left_doorway, pre_left_approaching, pre_right_doorway, pre_right_approaching = (
            self._get_threat_level(game)
        )

        obs, _, terminated, truncated, info = self.env.step(action)
        game = self.env.unwrapped.game
        reward = 0.0

        if terminated:
            if info.get('game_over_reason') == 'SURVIVED':
                reward = self.win_bonus
            else:
                reward = self.death_penalty
            return obs, reward, terminated, truncated, info

        reward += self.survival_bonus
        reward -= self.power_penalty_scale * info.get('power_usage', 1)

        post_left_doorway, post_left_approaching, post_right_doorway, post_right_approaching = (
            self._get_threat_level(game)
        )
        state = self._door_and_light_state_from_obs(obs)
        left_door = state['left_door']
        right_door = state['right_door']
        left_light = state['left_light']
        right_light = state['right_light']

        # Include pre-step threat state so last-moment correct reactions still
        # receive credit even when the animatronic moves away during the step.
        left_doorway = pre_left_doorway or post_left_doorway
        right_doorway = pre_right_doorway or post_right_doorway
        left_approaching = pre_left_approaching or post_left_approaching
        right_approaching = pre_right_approaching or post_right_approaching

        # Reward light usage more strongly for an actual doorway threat than for
        # a hallway-corner approach.
        if left_light and left_doorway:
            reward += self.light_bonus
        elif left_light and left_approaching:
            reward += self.light_bonus * 0.5

        if right_light and right_doorway:
            reward += self.light_bonus
        elif right_light and right_approaching:
            reward += self.light_bonus * 0.5

        if left_doorway and left_door:
            reward += self.defensive_bonus
        if right_doorway and right_door:
            reward += self.defensive_bonus

        if left_approaching and left_door:
            reward += self.approaching_bonus
        if right_approaching and right_door:
            reward += self.approaching_bonus

        # Penalize failing to close only when the threat was already present
        # before the action, so we do not punish the agent for threats that
        # appeared at the end of the step with no chance to react.
        if pre_left_doorway and not left_door:
            reward -= self.open_door_penalty
        if pre_right_doorway and not right_door:
            reward -= self.open_door_penalty
        if pre_left_approaching and not left_door:
            reward -= self.open_door_penalty * 0.5
        if pre_right_approaching and not right_door:
            reward -= self.open_door_penalty * 0.5

        if not left_doorway and not left_approaching and left_door:
            reward -= self.waste_penalty
        if not right_doorway and not right_approaching and right_door:
            reward -= self.waste_penalty

        return obs, reward, terminated, truncated, info


class FnafCVRewardShaping(FnafRewardShaping):
    """
    Reward shaping tailored for the CV-ready environment.

    This keeps the robust pre/post threat credit assignment from
    ``FnafRewardShaping``, adds a small bonus for actually refreshing memory
    through observation, and discourages camera-selection spam while the
    monitor is down.
    """

    def __init__(self, env, info_gain_bonus=0.005,
                 info_refresh_threshold=5.0,
                 stale_light_bonus=0.01,
                 stale_light_threshold=8.0,
                 light_scout_cooldown=5.0,
                 invalid_camera_penalty=0.05,
                 redundant_camera_penalty=0.005,
                 **kwargs):
        super().__init__(env, **kwargs)
        self.info_gain_bonus = info_gain_bonus
        self.info_refresh_threshold = info_refresh_threshold
        self.stale_light_bonus = stale_light_bonus
        self.stale_light_threshold = stale_light_threshold
        self.light_scout_cooldown = light_scout_cooldown
        self.invalid_camera_penalty = invalid_camera_penalty
        self.redundant_camera_penalty = redundant_camera_penalty
        self._last_left_scout_second = float('-inf')
        self._last_right_scout_second = float('-inf')

    def reset(self, **kwargs):
        self._last_left_scout_second = float('-inf')
        self._last_right_scout_second = float('-inf')
        return self.env.reset(**kwargs)

    def step(self, action):
        cv_env = self.env.unwrapped
        game = cv_env.game
        pre_left_doorway, pre_left_approaching, pre_right_doorway, pre_right_approaching = (
            self._get_threat_level(game)
        )
        pre_last_seen = dict(getattr(cv_env, '_last_seen_time', {}))
        pre_positions = dict(getattr(cv_env, '_last_known_positions', {}))
        pre_cameras_on = game.player.cameras_on
        pre_current_camera = game.player.current_camera
        pre_left_light = game.player.left_light_on
        pre_right_light = game.player.right_light_on

        obs, _, terminated, truncated, info = self.env.step(action)
        cv_env = self.env.unwrapped
        game = cv_env.game
        reward = 0.0

        if terminated:
            if info.get('game_over_reason') == 'SURVIVED':
                reward = self.win_bonus
            else:
                reward = self.death_penalty
            return obs, reward, terminated, truncated, info

        reward += self.survival_bonus
        reward -= self.power_penalty_scale * info.get('power_usage', 1)

        post_left_doorway, post_left_approaching, post_right_doorway, post_right_approaching = (
            self._get_threat_level(game)
        )
        state = self._door_and_light_state_from_obs(obs)
        left_door = state['left_door']
        right_door = state['right_door']
        left_light = state['left_light']
        right_light = state['right_light']

        left_doorway = pre_left_doorway or post_left_doorway
        right_doorway = pre_right_doorway or post_right_doorway
        left_approaching = pre_left_approaching or post_left_approaching
        right_approaching = pre_right_approaching or post_right_approaching

        if left_light and left_doorway:
            reward += self.light_bonus
        elif left_light and left_approaching:
            reward += self.light_bonus * 0.5

        if right_light and right_doorway:
            reward += self.light_bonus
        elif right_light and right_approaching:
            reward += self.light_bonus * 0.5

        if left_doorway and left_door:
            reward += self.defensive_bonus
        if right_doorway and right_door:
            reward += self.defensive_bonus

        if left_approaching and left_door:
            reward += self.approaching_bonus
        if right_approaching and right_door:
            reward += self.approaching_bonus

        if pre_left_doorway and not left_door:
            reward -= self.open_door_penalty
        if pre_right_doorway and not right_door:
            reward -= self.open_door_penalty
        if pre_left_approaching and not left_door:
            reward -= self.open_door_penalty * 0.5
        if pre_right_approaching and not right_door:
            reward -= self.open_door_penalty * 0.5

        if not left_doorway and not left_approaching and left_door:
            reward -= self.waste_penalty
        if not right_doorway and not right_approaching and right_door:
            reward -= self.waste_penalty

        post_last_seen = getattr(cv_env, '_last_seen_time', {})
        post_positions = getattr(cv_env, '_last_known_positions', {})
        info_gain = 0
        for name, post_seen in post_last_seen.items():
            if post_seen > pre_last_seen.get(name, float('-inf')):
                pre_seen = pre_last_seen.get(name, float('-inf'))
                pre_pos = pre_positions.get(name)
                post_pos = post_positions.get(name)
                refreshed_stale_memory = (
                    pre_seen == float('-inf')
                    or (game.current_second - pre_seen) >= self.info_refresh_threshold
                )
                changed_position = post_pos != pre_pos
                if refreshed_stale_memory or changed_position:
                    info_gain += 1
        reward += self.info_gain_bonus * info_gain

        bonnie_age = game.current_second - pre_last_seen.get('Bonnie', float('-inf'))
        chica_age = game.current_second - pre_last_seen.get('Chica', float('-inf'))
        if (not pre_cameras_on and not pre_left_light and left_light
                and bonnie_age >= self.stale_light_threshold
                and (game.current_second - self._last_left_scout_second) >= self.light_scout_cooldown):
            reward += self.stale_light_bonus
            self._last_left_scout_second = game.current_second

        if (not pre_cameras_on and not pre_right_light and right_light
                and chica_age >= self.stale_light_threshold
                and (game.current_second - self._last_right_scout_second) >= self.light_scout_cooldown):
            reward += self.stale_light_bonus
            self._last_right_scout_second = game.current_second

        if 6 <= action < 6 + len(CAMERAS):
            if not pre_cameras_on:
                reward -= self.invalid_camera_penalty
            elif pre_current_camera in CAMERAS and CAMERAS.index(pre_current_camera) == action - 6:
                reward -= self.redundant_camera_penalty

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
        time_progress = obs[0]
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
            mask[:] = False
            mask[0] = True
            return mask

        if not cameras_on:
            mask[6:17] = False

        return mask
