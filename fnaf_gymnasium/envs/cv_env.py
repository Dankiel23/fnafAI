"""
CV-Ready FNAF Environment.

Designed to match what a computer vision pipeline would actually
provide when playing the real FNAF game. Key differences from
the standard FnafEnv:

1. PARTIAL OBSERVABILITY: Only sees animatronics on the current camera
   or in doorways when lights are on - exactly what a player sees.

2. MEMORY: Tracks last-known position + time since last seen for each
   animatronic. A CV system would retain this between camera switches.

3. NO LEAKED INTERNALS: AI levels, foxy_attacking flag, freddy_waiting
   are hidden. The agent must infer danger from what it can observe.

4. OBSERVATION NOISE: Optional Gaussian noise on continuous values and
   dropout on binary values to simulate imperfect CV detection.

5. ANIMATRONIC VISIBLE FLAG: A dedicated signal for "something is on
   this camera right now" - the clearest thing CV would detect.

Training with this env produces a model whose inputs at inference time
can be directly filled from CV outputs with no mismatch.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Optional

from .game_logic import FNAFGame, CAMERAS, NIGHT_DURATION, GameOverReason

# Positions the agent can track
ALL_POSITIONS = CAMERAS + ['office']
POSITION_TO_IDX = {pos: i for i, pos in enumerate(ALL_POSITIONS)}
NUM_POSITIONS = len(ALL_POSITIONS)  # 12

# How quickly "time since last seen" saturates (seconds)
MEMORY_DECAY_MAX = 60.0

# Actions (same as FnafEnv)
ACTION_NOOP = 0
ACTION_TOGGLE_CAMERAS = 1
ACTION_TOGGLE_LEFT_DOOR = 2
ACTION_TOGGLE_RIGHT_DOOR = 3
ACTION_TOGGLE_LEFT_LIGHT = 4
ACTION_TOGGLE_RIGHT_LIGHT = 5
ACTION_CAM_START = 6
NUM_ACTIONS = ACTION_CAM_START + len(CAMERAS)  # 17

"""
Observation layout (total: 87 dimensions):

Player-controlled state (always accurate - CV reads these trivially):
  [0]     : normalized time (0-1)
  [1]     : power % (0-1)
  [2]     : power usage multiplier / 4
  [3]     : cameras on (0/1)
  [4-14]  : current camera one-hot (11 values)
  [15]    : left door closed (0/1)
  [16]    : right door closed (0/1)
  [17]    : left light on (0/1)
  [18]    : right light on (0/1)
  [19]    : power outage (0/1)

Current camera view (what CV sees RIGHT NOW):
  [20]    : something visible on current camera (0/1)
  [21]    : Freddy visible on current camera (0/1)
  [22]    : Bonnie visible on current camera (0/1)
  [23]    : Chica visible on current camera (0/1)
  [24]    : Foxy visible on current camera (0/1)
  [25]    : Foxy sub-position visible (0/1) - only when on cam 1C

Doorway checks via lights (what CV sees via lights):
  [26]    : left doorway occupied (0/1) - only valid when left light on
  [27]    : right doorway occupied (0/1) - only valid when right light on

Memory - last known position per animatronic (one-hot, 12 values each):
  [28-39] : Freddy last known position (all zeros = never seen)
  [40-51] : Bonnie last known position
  [52-63] : Chica last known position
  [64-75] : Foxy last known position

Memory - staleness per animatronic (0 = just seen, 1 = stale/never seen):
  [76]    : Freddy staleness
  [77]    : Bonnie staleness
  [78]    : Chica staleness
  [79]    : Foxy staleness

Memory - Foxy sub-position last seen:
  [80]    : Foxy last known sub-position / 3

Derived danger signals (agent must infer these, not given directly):
  [81]    : left door threat in memory (Bonnie/Foxy near left side)
  [82]    : right door threat in memory (Chica/Freddy near right side)
  [83]    : Foxy was at sub_position 2 last seen (near-ready to attack)
  [84]    : how long cameras have been off this stretch (0-1, 30s max)
  [85]    : how long left door has been open (0-1, 30s max)
  [86]    : how long right door has been open (0-1, 30s max)
"""

# Camera-to-animatronic visibility mapping (which animatronics can appear on which camera)
CAMERA_VISIBILITY = {
    '1A': ['Freddy', 'Bonnie', 'Chica'],
    '1B': ['Freddy', 'Bonnie', 'Chica'],
    '1C': ['Foxy'],
    '2A': ['Bonnie', 'Foxy'],
    '2B': ['Bonnie'],
    '3':  ['Bonnie'],
    '4A': ['Freddy', 'Chica'],
    '4B': ['Freddy', 'Chica'],
    '5':  ['Bonnie'],
    '6':  ['Chica'],
    '7':  ['Freddy', 'Chica'],
}

# Left-side cameras (Bonnie/Foxy territory)
LEFT_SIDE_CAMERAS = {'2A', '2B', '3', '1C'}
# Right-side cameras (Chica/Freddy territory)
RIGHT_SIDE_CAMERAS = {'4A', '4B', '6', '7'}


class FnafCVReadyEnv(gym.Env):
    """
    FNAF environment that trains agents on inputs matching
    what a computer vision pipeline would provide.

    Use this for training if you intend to deploy the model
    against the real game via screen capture.
    """

    metadata = {'render_modes': ['human', 'ansi'], 'render_fps': 1}

    def __init__(
        self,
        render_mode: Optional[str] = None,
        night: int = 1,
        step_duration: float = 1.0,
        obs_noise_std: float = 0.0,
        cv_detection_rate: float = 1.0,
    ):
        """
        Args:
            night: Which night to simulate (1-7)
            step_duration: Seconds per gym step
            obs_noise_std: Gaussian noise std on continuous observations (0 = no noise).
                          Set to ~0.02 to simulate imperfect CV readings.
            cv_detection_rate: Probability CV correctly detects an animatronic (0-1).
                              1.0 = perfect detection, 0.85 = realistic CV.
        """
        super().__init__()

        self.render_mode = render_mode
        self.night = night
        self.step_duration = step_duration
        self.obs_noise_std = obs_noise_std
        self.cv_detection_rate = cv_detection_rate

        self.action_space = spaces.Discrete(NUM_ACTIONS)
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(87,), dtype=np.float32
        )

        self.game: Optional[FNAFGame] = None

        # Memory state
        self._last_known_positions = {}   # name -> position str or None
        self._last_seen_time = {}         # name -> game second last seen
        self._last_known_foxy_sub = 0

        # Derived tracking
        self._cameras_off_duration = 0.0
        self._left_door_open_duration = 0.0
        self._right_door_open_duration = 0.0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        night = self.night
        if options and 'night' in options:
            night = options['night']

        self.game = FNAFGame(night=night)
        g = self.game

        # Bootstrap memory with known starting positions.
        # FNAF always starts animatronics at fixed locations (Freddy/Bonnie/Chica at 1A,
        # Foxy at 1C), so the agent "knows" these the same way a player glancing at
        # the cameras on night-start would. Without this, memory is blank, every door
        # close triggers a waste_penalty, and the agent learns to never close doors
        # before it can even attempt to learn camera management.
        for name, anim in [
            ('Freddy', g.freddy), ('Bonnie', g.bonnie),
            ('Chica', g.chica), ('Foxy', g.foxy),
        ]:
            self._last_known_positions[name] = anim.current_position
            self._last_seen_time[name] = 0.0  # fresh at game start

        self._last_known_foxy_sub = max(0, g.foxy.sub_position)
        self._cameras_off_duration = 0.0
        self._left_door_open_duration = 0.0
        self._right_door_open_duration = 0.0

        obs = self._get_obs()
        info = self._get_info()
        return obs, info

    def step(self, action: int):
        assert self.game is not None, "Call reset() before step()"

        self._apply_action(action)
        self.game.step(self.step_duration)

        # Update memory from current camera view
        self._update_memory()

        # Update derived timing trackers
        g = self.game
        if not g.player.cameras_on:
            self._cameras_off_duration += self.step_duration
        else:
            self._cameras_off_duration = 0.0

        if not g.player.left_door_closed:
            self._left_door_open_duration += self.step_duration
        else:
            self._left_door_open_duration = 0.0

        if not g.player.right_door_closed:
            self._right_door_open_duration += self.step_duration
        else:
            self._right_door_open_duration = 0.0

        reward = self._calculate_reward()
        terminated = self.game.game_over
        truncated = False

        obs = self._get_obs()
        info = self._get_info()
        return obs, reward, terminated, truncated, info

    def _apply_action(self, action: int):
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
        elif ACTION_CAM_START <= action < NUM_ACTIONS:
            cam_idx = action - ACTION_CAM_START
            self.game.switch_camera(CAMERAS[cam_idx])

    def _update_memory(self):
        """
        Update last-known positions based on what the agent can currently see.
        This mirrors what a CV pipeline would do: update state when you see
        something, retain last known state when you don't.
        """
        g = self.game
        animatronics = {
            'Freddy': g.freddy,
            'Bonnie': g.bonnie,
            'Chica': g.chica,
            'Foxy': g.foxy,
        }

        for name, anim in animatronics.items():
            if self._cv_can_see(name, anim):
                # Simulate imperfect CV detection rate
                if self.cv_detection_rate >= 1.0 or self.np_random.random() < self.cv_detection_rate:
                    self._last_known_positions[name] = anim.current_position
                    self._last_seen_time[name] = g.current_second
                    if name == 'Foxy':
                        self._last_known_foxy_sub = max(0, anim.sub_position)

    def _cv_can_see(self, name: str, anim) -> bool:
        """
        Determine if CV would currently detect this animatronic.
        Matches what a player can actually see on screen.
        """
        g = self.game

        # Always see animatronics in the office (they're right there)
        if anim.current_position == 'office':
            return True

        # See via lights: animatronic in doorway with corresponding light on
        if (anim.current_position == '2B' and anim.sub_position != -1
                and g.player.left_light_on):
            return True
        if (anim.current_position == '4B' and anim.sub_position != -1
                and g.player.right_light_on):
            return True

        # See via cameras: currently viewing the camera they're on
        if g.player.cameras_on:
            cam = g.player.current_camera
            # Check this animatronic can appear on this camera
            if (cam in CAMERA_VISIBILITY
                    and name in CAMERA_VISIBILITY[cam]
                    and anim.current_position == cam):
                return True
            # Foxy at 2A while running triggers on camera 2A
            if name == 'Foxy' and cam == '2A' and anim.current_position == '2A':
                return True

        return False

    def _get_obs(self) -> np.ndarray:
        obs = np.zeros(87, dtype=np.float32)
        g = self.game

        # --- Player-controlled state ---
        obs[0] = g.current_second / NIGHT_DURATION
        obs[1] = max(0.0, g.player.power) / 100.0
        obs[2] = g.get_power_usage_multiplier() / 4.0
        obs[3] = float(g.player.cameras_on)

        cam_idx = CAMERAS.index(g.player.current_camera) if g.player.current_camera in CAMERAS else 0
        obs[4 + cam_idx] = 1.0

        obs[15] = float(g.player.left_door_closed)
        obs[16] = float(g.player.right_door_closed)
        obs[17] = float(g.player.left_light_on)
        obs[18] = float(g.player.right_light_on)
        obs[19] = float(g.power_out)

        # --- Current camera view (what CV sees right now) ---
        if g.player.cameras_on:
            cam = g.player.current_camera
            animatronics = {
                'Freddy': (g.freddy, 21),
                'Bonnie': (g.bonnie, 22),
                'Chica':  (g.chica,  23),
                'Foxy':   (g.foxy,   24),
            }
            for name, (anim, idx) in animatronics.items():
                if self._cv_can_see(name, anim):
                    obs[20] = 1.0  # something is visible
                    obs[idx] = 1.0
            # Foxy sub-position visible on cam 1C
            if cam == '1C' and g.foxy.current_position == '1C':
                obs[25] = max(0, g.foxy.sub_position) / 3.0

        # --- Doorway checks (lights) ---
        if g.player.left_light_on:
            obs[26] = float(g.is_animatronic_in_doorway('left'))
        if g.player.right_light_on:
            obs[27] = float(g.is_animatronic_in_doorway('right'))

        # --- Memory: last known positions ---
        anim_names = ['Freddy', 'Bonnie', 'Chica', 'Foxy']
        for i, name in enumerate(anim_names):
            base = 28 + i * NUM_POSITIONS
            last_pos = self._last_known_positions.get(name)
            if last_pos is not None:
                pos_idx = POSITION_TO_IDX.get(last_pos, 0)
                obs[base + pos_idx] = 1.0

        # --- Memory: staleness ---
        for i, name in enumerate(anim_names):
            last_seen = self._last_seen_time.get(name, -MEMORY_DECAY_MAX)
            age = g.current_second - last_seen
            obs[76 + i] = min(1.0, age / MEMORY_DECAY_MAX)

        # --- Foxy last known sub-position ---
        obs[80] = self._last_known_foxy_sub / 3.0

        # --- Derived danger signals ---
        # Left threat: Bonnie or Foxy last known on left-side cameras
        left_pos = self._last_known_positions.get('Bonnie')
        foxy_pos = self._last_known_positions.get('Foxy')
        obs[81] = float(
            (left_pos in LEFT_SIDE_CAMERAS or left_pos == '2B') or
            (foxy_pos in LEFT_SIDE_CAMERAS and foxy_pos is not None)
        )

        # Right threat: Chica or Freddy last known on right-side cameras
        chica_pos = self._last_known_positions.get('Chica')
        freddy_pos = self._last_known_positions.get('Freddy')
        obs[82] = float(
            (chica_pos in RIGHT_SIDE_CAMERAS or chica_pos == '4B') or
            (freddy_pos in RIGHT_SIDE_CAMERAS and freddy_pos is not None)
        )

        # Foxy was near-ready to attack last time seen
        obs[83] = float(self._last_known_foxy_sub >= 2)

        # Timing signals
        obs[84] = min(1.0, self._cameras_off_duration / 30.0)
        obs[85] = min(1.0, self._left_door_open_duration / 30.0)
        obs[86] = min(1.0, self._right_door_open_duration / 30.0)

        # --- Optional noise (simulate imperfect CV readings) ---
        if self.obs_noise_std > 0:
            obs = self._add_noise(obs)

        return obs

    def _add_noise(self, obs: np.ndarray) -> np.ndarray:
        """Add noise to simulate imperfect CV readings."""
        obs = obs.copy()

        # Gaussian noise on continuous values (time, power, staleness, timers)
        continuous_indices = [0, 1, 2, 76, 77, 78, 79, 80, 84, 85, 86]
        for idx in continuous_indices:
            obs[idx] = float(np.clip(
                obs[idx] + self.np_random.normal(0, self.obs_noise_std),
                0.0, 1.0
            ))

        return obs

    def _calculate_reward(self) -> float:
        g = self.game
        reward = 0.0

        if g.game_over:
            if g.game_over_reason == GameOverReason.SURVIVED:
                reward += 100.0
            else:
                reward -= 100.0
            return reward

        # Survival bonus
        reward += 0.01

        # Power conservation
        reward -= 0.005 * g.get_power_usage_multiplier()

        # Reward for actively checking cameras (information gathering)
        if g.player.cameras_on:
            reward += 0.003

        # Reward for checking doorways with lights when memory is stale
        left_staleness = min(1.0, (g.current_second - self._last_seen_time.get('Bonnie', -MEMORY_DECAY_MAX)) / MEMORY_DECAY_MAX)
        right_staleness = min(1.0, (g.current_second - self._last_seen_time.get('Chica', -MEMORY_DECAY_MAX)) / MEMORY_DECAY_MAX)

        if g.player.left_light_on and left_staleness > 0.3:
            reward += 0.005  # Good - checking on a stale threat
        if g.player.right_light_on and right_staleness > 0.3:
            reward += 0.005

        # Reward correct door closures based on memory
        left_pos = self._last_known_positions.get('Bonnie')
        foxy_pos = self._last_known_positions.get('Foxy')
        left_threat = (left_pos in LEFT_SIDE_CAMERAS or left_pos == '2B' or
                       foxy_pos in LEFT_SIDE_CAMERAS)
        right_threat = (self._last_known_positions.get('Chica') in RIGHT_SIDE_CAMERAS or
                        self._last_known_positions.get('Freddy') in RIGHT_SIDE_CAMERAS)

        if left_threat and g.player.left_door_closed:
            reward += 0.02
        if right_threat and g.player.right_door_closed:
            reward += 0.02

        # Penalize closed doors when memory says no threat (wasting power)
        if not left_threat and g.player.left_door_closed:
            reward -= 0.01
        if not right_threat and g.player.right_door_closed:
            reward -= 0.01

        return reward

    def _get_info(self) -> dict:
        g = self.game
        return {
            'current_second': g.current_second,
            'power': g.player.power,
            'power_usage': g.get_power_usage_multiplier(),
            'night': self.night,
            'in_game_time': g.get_in_game_time(),
            'game_over_reason': g.game_over_reason.name if g.game_over else 'NONE',
            'power_out': g.power_out,
            'last_known_positions': dict(self._last_known_positions),
            'memory_staleness': {
                name: min(1.0, (g.current_second - self._last_seen_time.get(name, -MEMORY_DECAY_MAX)) / MEMORY_DECAY_MAX)
                for name in ['Freddy', 'Bonnie', 'Chica', 'Foxy']
            },
        }

    def render(self):
        if self.game is None:
            return
        g = self.game
        t = g.get_in_game_time()

        lines = [
            f"=== CV-Ready FNAF Night {self.night} === {t['hour']}:{t['minute']:02d}AM ===",
            f"Power: {g.player.power:.1f}% (x{g.get_power_usage_multiplier()})",
            f"Cameras: {'ON' if g.player.cameras_on else 'OFF'} ({g.player.current_camera})",
            f"Doors: L={'CLOSED' if g.player.left_door_closed else 'open'} R={'CLOSED' if g.player.right_door_closed else 'open'}",
            "--- Memory (last known) ---",
        ]
        for name in ['Freddy', 'Bonnie', 'Chica', 'Foxy']:
            pos = self._last_known_positions.get(name, 'unknown')
            age = g.current_second - self._last_seen_time.get(name, -MEMORY_DECAY_MAX)
            staleness = f"{min(age, MEMORY_DECAY_MAX):.0f}s ago" if pos else "never seen"
            lines.append(f"  {name:<8}: {str(pos):<8} ({staleness})")

        if g.game_over:
            lines.append(f"GAME OVER: {g.game_over_reason.name}")

        output = "\n".join(lines)
        if self.render_mode == 'human':
            print(output)
        elif self.render_mode == 'ansi':
            return output

    def close(self):
        pass
