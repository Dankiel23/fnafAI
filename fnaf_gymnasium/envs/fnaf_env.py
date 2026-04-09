"""
FNAF 1 Gymnasium Environment.

Wraps the FNAF game logic into a standard Gymnasium environment
suitable for reinforcement learning.
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from typing import Optional

from .game_logic import FNAFGame, CAMERAS, GameOverReason, NIGHT_DURATION


# Action definitions
ACTION_NOOP = 0
ACTION_TOGGLE_CAMERAS = 1
ACTION_TOGGLE_LEFT_DOOR = 2
ACTION_TOGGLE_RIGHT_DOOR = 3
ACTION_TOGGLE_LEFT_LIGHT = 4
ACTION_TOGGLE_RIGHT_LIGHT = 5
# Actions 6-16: Switch to camera 0-10 (mapped to CAMERAS list)
ACTION_CAM_START = 6
ACTION_CAM_END = ACTION_CAM_START + len(CAMERAS) - 1

NUM_ACTIONS = ACTION_CAM_END + 1  # 17 actions total

# Position encoding for animatronics
POSITION_TO_IDX = {pos: i for i, pos in enumerate(CAMERAS + ['office'])}
NUM_POSITIONS = len(POSITION_TO_IDX)  # 12


class FnafEnv(gym.Env):
    """
    FNAF 1 Gymnasium Environment.

    Observation space: flat vector of normalized game state
    Action space: Discrete(17)
      0: No-op
      1: Toggle cameras
      2: Toggle left door
      3: Toggle right door
      4: Toggle left light
      5: Toggle right light
      6-16: Switch to camera [1A, 1B, 1C, 2A, 2B, 3, 4A, 4B, 5, 6, 7]

    Reward:
      - Small positive reward each step for surviving (+0.01)
      - Large positive reward for reaching 6AM (+100)
      - Large negative reward for death (-100)
      - Small negative reward proportional to power usage (-0.005 * multiplier)
      - Bonus for correct defensive actions
    """

    metadata = {'render_modes': ['human', 'ansi'], 'render_fps': 1}

    def __init__(self, render_mode: Optional[str] = None, night: int = 1,
                 step_duration: float = 1.0, full_observability: bool = True):
        """
        Args:
            render_mode: 'human' for text output, 'ansi' for string return
            night: Which night to simulate (1-7)
            step_duration: Seconds per step (default 1.0)
            full_observability: If True, agent sees all animatronic positions.
                              If False, only sees positions visible on current camera.
        """
        super().__init__()

        self.render_mode = render_mode
        self.night = night
        self.step_duration = step_duration
        self.full_observability = full_observability

        # Action space: 17 discrete actions
        self.action_space = spaces.Discrete(NUM_ACTIONS)

        # Observation space: flat normalized vector
        # Layout:
        #   [0]     : normalized time (0-1)
        #   [1]     : normalized power (0-1)
        #   [2]     : power usage multiplier / 4
        #   [3]     : cameras on (0/1)
        #   [4]     : current camera (one-hot, 11 values)
        #   [15]    : left door closed (0/1)
        #   [16]    : right door closed (0/1)
        #   [17]    : left light on (0/1)
        #   [18]    : right light on (0/1)
        #   [19]    : power outage active (0/1)
        #   [20-31] : freddy position (one-hot, 12 values)
        #   [32-43] : bonnie position (one-hot, 12 values)
        #   [44-55] : chica position (one-hot, 12 values)
        #   [56-67] : foxy position (one-hot, 12 values)
        #   [68]    : foxy sub-position / 3 (0, 0.33, 0.67, 1.0)
        #   [69]    : freddy AI level / 20
        #   [70]    : bonnie AI level / 20
        #   [71]    : chica AI level / 20
        #   [72]    : foxy AI level / 20
        #   [73]    : left doorway occupied (0/1)
        #   [74]    : right doorway occupied (0/1)
        #   [75]    : foxy attacking (0/1)
        #   [76]    : freddy waiting to move (0/1)
        # Total: 77

        self.obs_size = 77
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(self.obs_size,), dtype=np.float32
        )

        self.game: Optional[FNAFGame] = None

    def _get_obs(self) -> np.ndarray:
        """Build observation vector from game state."""
        obs = np.zeros(self.obs_size, dtype=np.float32)
        g = self.game

        # Time
        obs[0] = g.current_second / NIGHT_DURATION

        # Power
        obs[1] = max(0, g.player.power) / 100.0

        # Power usage multiplier
        obs[2] = g.get_power_usage_multiplier() / 4.0

        # Cameras on
        obs[3] = float(g.player.cameras_on)

        # Current camera (one-hot)
        cam_idx = CAMERAS.index(g.player.current_camera) if g.player.current_camera in CAMERAS else 0
        obs[4 + cam_idx] = 1.0

        # Door/light states
        obs[15] = float(g.player.left_door_closed)
        obs[16] = float(g.player.right_door_closed)
        obs[17] = float(g.player.left_light_on)
        obs[18] = float(g.player.right_light_on)

        # Power outage
        obs[19] = float(g.power_out)

        # Animatronic positions (one-hot encoded)
        for i, anim in enumerate([g.freddy, g.bonnie, g.chica, g.foxy]):
            base = 20 + i * NUM_POSITIONS
            if self.full_observability or self._can_see_animatronic(anim):
                pos_idx = POSITION_TO_IDX.get(anim.current_position, 0)
                obs[base + pos_idx] = 1.0
            # If not observable, all zeros (unknown)

        # Foxy sub-position
        obs[68] = max(0, g.foxy.sub_position) / 3.0

        # AI levels
        obs[69] = g.freddy.current_ai_level / 20.0
        obs[70] = g.bonnie.current_ai_level / 20.0
        obs[71] = g.chica.current_ai_level / 20.0
        obs[72] = g.foxy.current_ai_level / 20.0

        # Doorway occupation
        obs[73] = float(g.is_animatronic_in_doorway('left'))
        obs[74] = float(g.is_animatronic_in_doorway('right'))

        # Foxy attacking
        obs[75] = float(g.foxy_attacking or g.foxy_running)

        # Freddy waiting
        obs[76] = float(g.freddy_waiting)

        return obs

    def _can_see_animatronic(self, anim) -> bool:
        """Check if the player can see this animatronic (partial observability)."""
        if not self.game.player.cameras_on:
            # Can only see animatronics in doorways via light
            if anim.current_position == '2B' and anim.sub_position != -1 and self.game.player.left_light_on:
                return True
            if anim.current_position == '4B' and anim.sub_position != -1 and self.game.player.right_light_on:
                return True
            if anim.current_position == 'office':
                return True
            return False
        else:
            return anim.current_position == self.game.player.current_camera

    def _get_info(self) -> dict:
        """Return auxiliary info dict."""
        g = self.game
        return {
            'current_second': g.current_second,
            'power': g.player.power,
            'power_usage': g.get_power_usage_multiplier(),
            'night': self.night,
            'in_game_time': g.get_in_game_time(),
            'freddy_pos': g.freddy.current_position,
            'bonnie_pos': g.bonnie.current_position,
            'chica_pos': g.chica.current_position,
            'foxy_pos': g.foxy.current_position,
            'foxy_sub': g.foxy.sub_position,
            'game_over_reason': g.game_over_reason.name if g.game_over else 'NONE',
            'power_out': g.power_out,
        }

    def reset(self, seed=None, options=None):
        """Reset the environment to start a new night."""
        super().reset(seed=seed)

        night = self.night
        if options and 'night' in options:
            night = options['night']

        self.game = FNAFGame(night=night)
        obs = self._get_obs()
        info = self._get_info()

        return obs, info

    def step(self, action: int):
        """
        Execute one action and advance the game by step_duration seconds.

        Returns:
            observation, reward, terminated, truncated, info
        """
        assert self.game is not None, "Call reset() before step()"

        prev_power = self.game.player.power
        prev_second = self.game.current_second

        # Apply action
        self._apply_action(action)

        # Advance game
        self.game.step(self.step_duration)

        # Calculate reward
        reward = self._calculate_reward(prev_power, prev_second)

        # Check termination
        terminated = self.game.game_over
        truncated = False

        obs = self._get_obs()
        info = self._get_info()

        return obs, reward, terminated, truncated, info

    def _apply_action(self, action: int):
        """Apply the agent's action to the game."""
        if action == ACTION_NOOP:
            pass
        elif action == ACTION_TOGGLE_CAMERAS:
            self.game.toggle_cameras()
        elif action == ACTION_TOGGLE_LEFT_DOOR:
            self.game.toggle_left_door()
        elif action == ACTION_TOGGLE_RIGHT_DOOR:
            self.game.toggle_right_door()
        elif action == ACTION_TOGGLE_LEFT_LIGHT:
            self.game.toggle_left_light()
        elif action == ACTION_TOGGLE_RIGHT_LIGHT:
            self.game.toggle_right_light()
        elif ACTION_CAM_START <= action <= ACTION_CAM_END:
            cam_idx = action - ACTION_CAM_START
            if cam_idx < len(CAMERAS):
                self.game.switch_camera(CAMERAS[cam_idx])

    def _calculate_reward(self, prev_power: float, prev_second: float) -> float:
        """Calculate reward for the current step."""
        reward = 0.0
        g = self.game

        if g.game_over:
            if g.game_over_reason == GameOverReason.SURVIVED:
                reward += 100.0  # Victory!
            else:
                reward -= 100.0  # Death
        else:
            # Survival bonus per step
            reward += 0.01

            # Power conservation incentive
            power_usage = g.get_power_usage_multiplier()
            reward -= 0.005 * power_usage

            # Correct defensive action bonuses
            if g.is_animatronic_in_doorway('left') and g.player.left_door_closed:
                reward += 0.02  # Good - blocking a threat
            if g.is_animatronic_in_doorway('right') and g.player.right_door_closed:
                reward += 0.02  # Good - blocking a threat

            # Penalty for closing doors when no threat
            if g.player.left_door_closed and not g.is_animatronic_in_doorway('left'):
                reward -= 0.01  # Wasting power
            if g.player.right_door_closed and not g.is_animatronic_in_doorway('right'):
                reward -= 0.01  # Wasting power

            # Camera usage keeps Foxy at bay (small reward for periodic camera use)
            if g.player.cameras_on and g.foxy.current_position == '1C':
                reward += 0.005  # Keeping Foxy in check

        return reward

    def render(self):
        """Render the current game state."""
        if self.game is None:
            return

        g = self.game
        time_info = g.get_in_game_time()
        time_str = f"{time_info['hour']}:{time_info['minute']:02d} AM"

        lines = [
            f"=== FNAF Night {self.night} === {time_str} ===",
            f"Power: {g.player.power:.1f}% (Usage: {g.get_power_usage_multiplier()}x)",
            f"Cameras: {'ON' if g.player.cameras_on else 'OFF'} (Viewing: {g.player.current_camera})",
            f"Left Door: {'CLOSED' if g.player.left_door_closed else 'OPEN'}  |  Right Door: {'CLOSED' if g.player.right_door_closed else 'OPEN'}",
            f"Left Light: {'ON' if g.player.left_light_on else 'OFF'}  |  Right Light: {'ON' if g.player.right_light_on else 'OFF'}",
            f"---",
            f"Freddy: {g.freddy.current_position} (AI: {g.freddy.current_ai_level})",
            f"Bonnie: {g.bonnie.current_position} (AI: {g.bonnie.current_ai_level})",
            f"Chica:  {g.chica.current_position} (AI: {g.chica.current_ai_level})",
            f"Foxy:   {g.foxy.current_position} sub:{g.foxy.sub_position} (AI: {g.foxy.current_ai_level})",
        ]

        if g.power_out:
            lines.append(f"!!! POWER OUT !!! Phase: {g.power_outage_phase}")
        if g.game_over:
            lines.append(f"GAME OVER: {g.game_over_reason.name}")

        output = "\n".join(lines)

        if self.render_mode == 'human':
            print(output)
        elif self.render_mode == 'ansi':
            return output

    def close(self):
        """Clean up."""
        pass
