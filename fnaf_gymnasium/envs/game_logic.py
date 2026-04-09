"""
Pure Python port of the FNAF 1 game logic from the TypeScript simulator.
No UI/DOM dependencies - just the core simulation engine.
"""

import random
import math
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Optional

from .constants import (
    CAMERAS, POSITIONS, CAMERA_NAMES,
    NIGHT_DURATION_SECONDS as NIGHT_DURATION,
    AI_INCREASE_2AM, AI_INCREASE_3AM, AI_INCREASE_4AM,
    POWER_DRAIN_SPACING, STARTING_POWER, MAX_POWER_MULTIPLIER,
    FREDDY_PATH, FREDDY_WAIT_BASE_FRAMES, FREDDY_WAIT_AI_MULTIPLIER, GAME_FPS,
    FOXY_COOLDOWN_MIN, FOXY_COOLDOWN_MAX, FOXY_ATTACK_COUNTDOWN, FOXY_RUN_DELAY,
    BONNIE_POSITIONS, CHICA_ADJACENCY,
    OFFICE_JUMPSCARE_TIMEOUT, FREDDY_OFFICE_JUMPSCARE_CHANCE,
    POWER_OUTAGE_CHANCE, POWER_OUTAGE_MAX_ATTEMPTS,
    FREDDY_INTERVAL, BONNIE_INTERVAL, CHICA_INTERVAL, FOXY_INTERVAL,
    TIME_CONVERSION_FACTOR,
)


class GameOverReason(IntEnum):
    NONE = 0
    SURVIVED = 1  # Reached 6AM
    FREDDY = 2
    BONNIE = 3
    CHICA = 4
    FOXY = 5
    POWER_OUTAGE_FREDDY = 6


@dataclass
class AnimatronicState:
    name: str
    current_position: str
    sub_position: int  # -1 = no sub-position, 0-2 for Foxy stages
    movement_opportunity_interval: float  # seconds between movement checks
    ai_levels: list  # Starting AI levels per night (index 0 unused, 1-7)
    current_ai_level: int = 0
    current_countdown: float = 0
    successful_movement_checks: int = 0
    failed_movement_checks: int = 0
    office_attempts: int = 0
    # Internal timers (in seconds)
    time_since_last_check: float = 0.0
    freddy_waiting_frames: float = 0  # Freddy's post-check waiting time
    freddy_waiting_destination: str = ''
    in_office: bool = False
    in_doorway: bool = False


@dataclass
class PlayerState:
    cameras_on: bool = False
    current_camera: str = '1A'
    left_door_closed: bool = False
    right_door_closed: bool = False
    left_light_on: bool = False
    right_light_on: bool = False
    power: float = STARTING_POWER
    cameras_toggled: int = 0
    cameras_looked_at: int = 0
    left_door_toggled: int = 0
    right_door_toggled: int = 0


class FNAFGame:
    """Core FNAF 1 game simulation engine."""

    def __init__(self, night: int = 1):
        self.night = max(1, min(night, 7))
        self.current_second: float = 0.0
        self.game_over_reason: GameOverReason = GameOverReason.NONE
        self.game_over: bool = False

        # Power outage state
        self.power_out: bool = False
        self.power_outage_phase: int = 0  # 0=waiting for freddy, 1=song, 2=darkness
        self.power_outage_timer: float = 0.0
        self.power_outage_attempts: int = 0

        # Foxy-specific timers
        self.foxy_cooldown_active: bool = False
        self.foxy_cooldown_remaining: float = 0.0
        self.foxy_attack_countdown: float = 0.0
        self.foxy_attacking: bool = False
        self.foxy_running_timer: float = 0.0
        self.foxy_running: bool = False

        # Freddy waiting to move (has passed check, waiting frames)
        self.freddy_waiting: bool = False
        self.freddy_wait_remaining: float = 0.0
        self.freddy_wait_destination: str = ''

        # Bonnie/Chica in-office jumpscare timers
        self.bonnie_jumpscare_timer: float = 0.0
        self.chica_jumpscare_timer: float = 0.0

        # Freddy in-office jumpscare timer
        self.freddy_office_timer: float = 0.0

        # Initialize player state
        self.player = PlayerState(power=STARTING_POWER)

        # Initialize animatronics
        self._init_animatronics()

    def _init_animatronics(self):
        """Create animatronic states with proper AI levels for the current night."""
        freddy_night4_ai = random.choice([1, 2])

        self.freddy = AnimatronicState(
            name='Freddy',
            current_position='1A',
            sub_position=-1,
            movement_opportunity_interval=FREDDY_INTERVAL,
            ai_levels=[None, 0, 0, 1, freddy_night4_ai, 3, 4, 20],
        )

        self.bonnie = AnimatronicState(
            name='Bonnie',
            current_position='1A',
            sub_position=-1,
            movement_opportunity_interval=BONNIE_INTERVAL,
            ai_levels=[None, 0, 3, 0, 2, 5, 10, 20],
        )

        self.chica = AnimatronicState(
            name='Chica',
            current_position='1A',
            sub_position=-1,
            movement_opportunity_interval=CHICA_INTERVAL,
            ai_levels=[None, 0, 1, 5, 4, 7, 12, 20],
        )

        self.foxy = AnimatronicState(
            name='Foxy',
            current_position='1C',
            sub_position=0,
            movement_opportunity_interval=FOXY_INTERVAL,
            ai_levels=[None, 0, 1, 2, 6, 5, 16, 20],
        )

        # Set starting AI levels
        for anim in [self.freddy, self.bonnie, self.chica, self.foxy]:
            anim.current_ai_level = anim.ai_levels[self.night] if self.night <= 7 else 1

        self.animatronics = [self.freddy, self.bonnie, self.chica, self.foxy]

    def _make_movement_check(self, animatronic: AnimatronicState) -> bool:
        """Roll a movement check. Returns True if the animatronic can move."""
        comparison = random.randint(1, 20)
        can_move = animatronic.current_ai_level >= comparison
        if can_move:
            animatronic.successful_movement_checks += 1
        else:
            animatronic.failed_movement_checks += 1
        return can_move

    def _increase_ai_level(self, animatronic: AnimatronicState):
        """Increase an animatronic's AI level by 1, max 20."""
        if animatronic.current_ai_level < 20:
            animatronic.current_ai_level += 1

    def _calculate_power_drain_multiplier(self) -> int:
        """Calculate power usage multiplier (1-4)."""
        usage = 1  # base
        if self.player.left_door_closed:
            usage += 1
        if self.player.right_door_closed:
            usage += 1
        if self.player.cameras_on:
            usage += 1
        if self.player.left_light_on:
            usage += 1
        if self.player.right_light_on:
            usage += 1
        return min(usage, MAX_POWER_MULTIPLIER)

    def _calculate_power_drain_per_second(self) -> float:
        """Calculate power drain per second."""
        default_drain = 0.1 * self._calculate_power_drain_multiplier()
        nightly_drain = 0.1 / POWER_DRAIN_SPACING[self.night] if self.night >= 1 else 0
        # The original drains this amount every 0.1 seconds, divided by 10
        # So per 0.1s: (default_drain + nightly_drain) / 10
        # Per second: (default_drain + nightly_drain)
        return default_drain + nightly_drain

    def _drain_power(self, dt: float):
        """Drain power over dt seconds."""
        if self.power_out:
            return
        self.player.power -= self._calculate_power_drain_per_second() * dt
        if self.player.power <= 0:
            self.player.power = 0
            self._trigger_power_outage()

    def _trigger_power_outage(self):
        """Start the power outage sequence."""
        self.power_out = True
        self.power_outage_phase = 0  # Waiting for Freddy to arrive
        self.power_outage_timer = 0.0
        self.power_outage_attempts = 0
        # Open all doors, disable everything
        self.player.left_door_closed = False
        self.player.right_door_closed = False
        self.player.cameras_on = False
        self.player.left_light_on = False
        self.player.right_light_on = False

    def _update_power_outage(self, dt: float):
        """Update the power outage Freddy sequence."""
        if not self.power_out or self.game_over:
            return

        self.power_outage_timer += dt

        if self.power_outage_phase == 0:
            # Freddy arrives: 20% chance every 5 seconds, max 20 seconds
            if self.power_outage_timer >= 5.0:
                self.power_outage_timer -= 5.0
                self.power_outage_attempts += 1
                if random.random() < POWER_OUTAGE_CHANCE or self.power_outage_attempts >= POWER_OUTAGE_MAX_ATTEMPTS:
                    self.power_outage_phase = 1
                    self.power_outage_timer = 0.0
                    self.power_outage_attempts = 0

        elif self.power_outage_phase == 1:
            # Song playing: 20% chance to end every 5 seconds, max 20 seconds
            if self.power_outage_timer >= 5.0:
                self.power_outage_timer -= 5.0
                self.power_outage_attempts += 1
                if random.random() < POWER_OUTAGE_CHANCE or self.power_outage_attempts >= POWER_OUTAGE_MAX_ATTEMPTS:
                    self.power_outage_phase = 2
                    self.power_outage_timer = 0.0
                    self.power_outage_attempts = 0

        elif self.power_outage_phase == 2:
            # Darkness: 20% chance every 2 seconds
            if self.power_outage_timer >= 2.0:
                self.power_outage_timer -= 2.0
                if random.random() < POWER_OUTAGE_CHANCE:
                    self._game_over(GameOverReason.POWER_OUTAGE_FREDDY)

    def _game_over(self, reason: GameOverReason):
        """End the game."""
        self.game_over = True
        self.game_over_reason = reason

    # ============================================================
    # FOXY LOGIC
    # ============================================================

    def _update_foxy(self, dt: float):
        """Process Foxy's movement logic."""
        if self.power_out or self.game_over:
            return
        if self.foxy.current_ai_level == 0:
            return

        # Handle Foxy running down the hall
        if self.foxy_running:
            self.foxy_running_timer -= dt
            if self.foxy_running_timer <= 0:
                self._foxy_reach_office()
            return

        # Handle Foxy attacking (25s countdown or camera 2A trigger)
        if self.foxy_attacking:
            self.foxy_attack_countdown -= dt
            if self.foxy_attack_countdown <= 0:
                self._foxy_attempt_jumpscare(from_camera=False)
            return

        # Handle cooldown after cameras go off
        if self.foxy_cooldown_active:
            self.foxy_cooldown_remaining -= dt
            if self.foxy_cooldown_remaining <= 0:
                self.foxy_cooldown_active = False
            else:
                return

        # Regular movement checks
        self.foxy.time_since_last_check += dt
        if self.foxy.time_since_last_check >= self.foxy.movement_opportunity_interval:
            self.foxy.time_since_last_check -= self.foxy.movement_opportunity_interval
            self._foxy_movement_check()

    def _foxy_movement_check(self):
        """Perform Foxy's movement check."""
        if self.foxy.current_position != '1C':
            return

        # Auto-fail while cameras are on
        if self.player.cameras_on:
            return

        can_move = self._make_movement_check(self.foxy)

        if can_move and self.foxy.sub_position < 2:
            # Advance one step in Pirate Cove
            self.foxy.sub_position += 1
        elif can_move and self.foxy.sub_position == 2:
            # Leave Pirate Cove!
            self.foxy.current_position = '2A'
            self.foxy.sub_position = -1
            self.foxy_attacking = True
            self.foxy_attack_countdown = FOXY_ATTACK_COUNTDOWN

    def _foxy_attempt_jumpscare(self, from_camera: bool = False):
        """Foxy attempts to reach the office."""
        self.foxy_attacking = False
        self.foxy.office_attempts += 1

        if from_camera:
            # 1.87 second delay when triggered by looking at cam 2A
            self.foxy_running = True
            self.foxy_running_timer = FOXY_RUN_DELAY
        else:
            self._foxy_reach_office()

    def _foxy_reach_office(self):
        """Foxy reaches the office."""
        self.foxy_running = False

        if self.player.left_door_closed:
            # Bash on door, drain power
            power_drain = 1 + (self.foxy.office_attempts - 1) * 5
            self.player.power -= power_drain
            if self.player.power <= 0:
                self.player.power = 0
                self._trigger_power_outage()
            # Return to Pirate Cove at step 0 or 1
            self.foxy.current_position = '1C'
            self.foxy.sub_position = random.randint(0, 1)
            self.foxy.time_since_last_check = 0.0
        else:
            self._game_over(GameOverReason.FOXY)

    def _on_cameras_off_foxy(self):
        """When cameras go off, Foxy gets a cooldown."""
        if self.foxy.current_position == '1C' and not self.foxy_attacking:
            self.foxy_cooldown_active = True
            self.foxy_cooldown_remaining = random.uniform(FOXY_COOLDOWN_MIN, FOXY_COOLDOWN_MAX)

    def _on_camera_view_2a(self):
        """When player looks at camera 2A and Foxy is attacking."""
        if self.foxy_attacking and self.foxy.current_position == '2A':
            self._foxy_attempt_jumpscare(from_camera=True)

    # ============================================================
    # FREDDY LOGIC
    # ============================================================

    def _update_freddy(self, dt: float):
        """Process Freddy's movement logic."""
        if self.power_out or self.game_over:
            return
        if self.freddy.current_ai_level == 0:
            return

        # Handle Freddy in office - 25% chance per second while cameras down
        if self.freddy.current_position == 'office':
            self.freddy_office_timer += dt
            if self.freddy_office_timer >= 1.0:
                self.freddy_office_timer -= 1.0
                if not self.player.cameras_on and random.random() < FREDDY_OFFICE_JUMPSCARE_CHANCE:
                    self._game_over(GameOverReason.FREDDY)
            return

        # Handle Freddy waiting to move after passing a check
        if self.freddy_waiting:
            self.freddy_wait_remaining -= dt / 60.0  # Convert to frames at 60fps
            # Actually, the waiting is in frames: 1000 - AI*100 frames at 60fps
            # So we need to count in seconds: (1000 - AI*100) / 60
            if self.freddy_wait_remaining <= 0 and not self.player.cameras_on:
                self.freddy.current_position = self.freddy_wait_destination
                self.freddy_waiting = False
                self.freddy.time_since_last_check = 0.0
            return

        # Regular movement checks
        self.freddy.time_since_last_check += dt
        if self.freddy.time_since_last_check >= self.freddy.movement_opportunity_interval:
            self.freddy.time_since_last_check -= self.freddy.movement_opportunity_interval
            self._freddy_movement_check()

    def _freddy_movement_check(self):
        """Perform Freddy's movement check."""
        # Can't leave stage if Bonnie or Chica still there
        if self.freddy.current_position == '1A':
            if self.bonnie.current_position == '1A' or self.chica.current_position == '1A':
                return

        # Auto-fail while cameras on (unless at 4B)
        if self.player.cameras_on and self.freddy.current_position != '4B':
            return

        # At 4B: auto-fail if camera is looking at 4B
        if self.freddy.current_position == '4B' and self.player.cameras_on and self.player.current_camera == '4B':
            return

        can_move = self._make_movement_check(self.freddy)
        if not can_move:
            return

        # Freddy at 4B - attempt office entry
        if self.freddy.current_position == '4B':
            if self.player.cameras_on and self.player.current_camera != '4B':
                # Cameras on, not looking at 4B
                if self.player.right_door_closed:
                    # Blocked! Go back to 4A
                    self.freddy.current_position = '4A'
                    self.freddy.office_attempts += 1
                else:
                    # Enter office!
                    self.freddy.current_position = 'office'
                    self.freddy.office_attempts += 1
                    self.freddy_office_timer = 0.0
            elif not self.player.cameras_on:
                # Cameras off at 4B - can't enter (Freddy needs cameras on to enter)
                self.freddy.office_attempts += 1
            return

        # Normal movement along Freddy's path
        if self.freddy.current_position in FREDDY_PATH:
            destination = FREDDY_PATH[self.freddy.current_position]
            wait_frames = max(0, FREDDY_WAIT_BASE_FRAMES - self.freddy.current_ai_level * FREDDY_WAIT_AI_MULTIPLIER)
            wait_seconds = wait_frames / GAME_FPS

            if wait_seconds <= 0:
                # Move immediately
                self.freddy.current_position = destination
            else:
                # Set up waiting
                self.freddy_waiting = True
                self.freddy_wait_remaining = wait_seconds
                self.freddy_wait_destination = destination

    # ============================================================
    # BONNIE LOGIC
    # ============================================================

    def _update_bonnie(self, dt: float):
        """Process Bonnie's movement logic."""
        if self.power_out or self.game_over:
            return
        if self.bonnie.current_ai_level == 0:
            return

        # Handle Bonnie in office (30s timer)
        if self.bonnie.current_position == 'office':
            self.bonnie_jumpscare_timer += dt
            if self.bonnie_jumpscare_timer >= OFFICE_JUMPSCARE_TIMEOUT:
                self._game_over(GameOverReason.BONNIE)
            elif not self.player.cameras_on:
                # Cameras down while Bonnie in office = instant death
                # (Actually triggered on cameras-off event, handled in step)
                pass
            return

        # Regular movement checks
        self.bonnie.time_since_last_check += dt
        if self.bonnie.time_since_last_check >= self.bonnie.movement_opportunity_interval:
            self.bonnie.time_since_last_check -= self.bonnie.movement_opportunity_interval
            self._bonnie_movement_check()

    def _bonnie_movement_check(self):
        """Perform Bonnie's movement check."""
        can_move = self._make_movement_check(self.bonnie)
        if not can_move:
            return

        # At hall corner (2B) but not in doorway yet
        if self.bonnie.current_position == '2B' and self.bonnie.sub_position == -1:
            self.bonnie.sub_position = 1  # Move to doorway
            return

        # In doorway at 2B
        if self.bonnie.current_position == '2B' and self.bonnie.sub_position != -1:
            if self.player.left_door_closed:
                # Blocked - return to dining room
                self.bonnie.current_position = '1B'
                self.bonnie.sub_position = -1
            else:
                # Enter office!
                self.bonnie.current_position = 'office'
                self.bonnie.sub_position = -1
                self.bonnie.office_attempts += 1
                self.bonnie_jumpscare_timer = 0.0
                # Immediate jumpscare if cameras are already down
                if not self.player.cameras_on:
                    self._game_over(GameOverReason.BONNIE)
            return

        # Normal movement - random position on left side
        possible = BONNIE_POSITIONS
        self.bonnie.current_position = random.choice(possible)
        self.bonnie.sub_position = -1

    # ============================================================
    # CHICA LOGIC
    # ============================================================

    def _update_chica(self, dt: float):
        """Process Chica's movement logic."""
        if self.power_out or self.game_over:
            return
        if self.chica.current_ai_level == 0:
            return

        # Handle Chica in office (30s timer)
        if self.chica.current_position == 'office':
            self.chica_jumpscare_timer += dt
            if self.chica_jumpscare_timer >= OFFICE_JUMPSCARE_TIMEOUT:
                self._game_over(GameOverReason.CHICA)
            return

        # Regular movement checks
        self.chica.time_since_last_check += dt
        if self.chica.time_since_last_check >= self.chica.movement_opportunity_interval:
            self.chica.time_since_last_check -= self.chica.movement_opportunity_interval
            self._chica_movement_check()

    def _chica_movement_check(self):
        """Perform Chica's movement check."""
        can_move = self._make_movement_check(self.chica)
        if not can_move:
            return

        # At hall corner (4B) but not in doorway yet
        if self.chica.current_position == '4B' and self.chica.sub_position == -1:
            self.chica.sub_position = 1  # Move to doorway
            return

        # In doorway at 4B
        if self.chica.current_position == '4B' and self.chica.sub_position != -1:
            if self.player.right_door_closed:
                # Blocked - return to dining room
                self.chica.current_position = '1B'
                self.chica.sub_position = -1
            else:
                # Enter office!
                self.chica.current_position = 'office'
                self.chica.sub_position = -1
                self.chica.office_attempts += 1
                self.chica_jumpscare_timer = 0.0
                # Immediate jumpscare if cameras are already down
                if not self.player.cameras_on:
                    self._game_over(GameOverReason.CHICA)
            return

        # Adjacent movement for Chica
        new_pos = self._calculate_chica_position()
        if new_pos:
            self.chica.current_position = new_pos
            self.chica.sub_position = -1

    def _calculate_chica_position(self) -> Optional[str]:
        """Calculate Chica's next position (adjacent rooms only)."""
        pos = self.chica.current_position
        if pos in CHICA_ADJACENCY:
            return random.choice(CHICA_ADJACENCY[pos])
        return None

    # ============================================================
    # MAIN GAME STEP
    # ============================================================

    def step(self, dt: float = 1.0):
        """
        Advance the game by dt seconds.
        Call this after applying player actions.
        """
        if self.game_over:
            return

        # Advance time
        old_second = int(self.current_second)
        self.current_second += dt
        new_second = int(self.current_second)

        # Check AI level increases
        for sec in range(old_second + 1, new_second + 1):
            if sec == AI_INCREASE_2AM:
                self._increase_ai_level(self.bonnie)
            if sec == AI_INCREASE_3AM:
                self._increase_ai_level(self.bonnie)
                self._increase_ai_level(self.chica)
                self._increase_ai_level(self.foxy)
            if sec == AI_INCREASE_4AM:
                self._increase_ai_level(self.bonnie)

        # Check for 6AM
        if self.current_second >= NIGHT_DURATION:
            self._game_over(GameOverReason.SURVIVED)
            return

        # Drain power
        if not self.power_out:
            self._drain_power(dt)

        # Update power outage sequence
        if self.power_out:
            self._update_power_outage(dt)
            return

        # Update animatronics
        self._update_freddy(dt)
        self._update_bonnie(dt)
        self._update_chica(dt)
        self._update_foxy(dt)

        # Bonnie/Chica office jumpscares are handled in toggle_cameras
        # (cameras-off event) and via their 30s timers in _update_bonnie/_update_chica

    # ============================================================
    # PLAYER ACTIONS
    # ============================================================

    def toggle_cameras(self):
        """Toggle cameras on/off."""
        if self.power_out or self.game_over:
            return
        was_on = self.player.cameras_on
        self.player.cameras_on = not self.player.cameras_on
        self.player.cameras_toggled += 1

        if was_on and not self.player.cameras_on:
            # Cameras just went off - trigger Foxy cooldown
            self._on_cameras_off_foxy()
            # Bonnie/Chica in office jumpscare on cameras-off
            if self.bonnie.current_position == 'office':
                self._game_over(GameOverReason.BONNIE)
            if not self.game_over and self.chica.current_position == 'office':
                self._game_over(GameOverReason.CHICA)

    def switch_camera(self, camera: str):
        """Switch to a specific camera."""
        if camera not in CAMERAS or self.power_out or self.game_over:
            return
        self.player.current_camera = camera
        self.player.cameras_looked_at += 1

        # Check if looking at 2A triggers Foxy
        if camera == '2A':
            self._on_camera_view_2a()

    def toggle_left_door(self):
        """Toggle left door open/closed."""
        if self.power_out or self.game_over:
            return
        # Can't toggle if an animatronic is in the office
        if self.bonnie.current_position == 'office' or self.chica.current_position == 'office' or self.freddy.current_position == 'office':
            return
        self.player.left_door_closed = not self.player.left_door_closed
        self.player.left_door_toggled += 1

    def toggle_right_door(self):
        """Toggle right door open/closed."""
        if self.power_out or self.game_over:
            return
        if self.bonnie.current_position == 'office' or self.chica.current_position == 'office' or self.freddy.current_position == 'office':
            return
        self.player.right_door_closed = not self.player.right_door_closed
        self.player.right_door_toggled += 1

    def toggle_left_light(self):
        """Toggle left light on/off."""
        if self.power_out or self.game_over:
            return
        self.player.left_light_on = not self.player.left_light_on

    def toggle_right_light(self):
        """Toggle right light on/off."""
        if self.power_out or self.game_over:
            return
        self.player.right_light_on = not self.player.right_light_on

    # ============================================================
    # STATE QUERY
    # ============================================================

    def get_in_game_time(self) -> dict:
        """Get the current in-game time."""
        in_game_minutes = max(0, math.floor(self.current_second * TIME_CONVERSION_FACTOR))
        hour = max(0, math.floor(in_game_minutes / 60))
        if hour == 0:
            hour = 12
        minute = in_game_minutes % 60
        return {'hour': hour, 'minute': minute}

    def is_animatronic_in_doorway(self, side: str) -> bool:
        """Check if an animatronic is in the specified doorway."""
        if side == 'left':
            # Bonnie at 2B with sub_position != -1, or Foxy running
            return (self.bonnie.current_position == '2B' and self.bonnie.sub_position != -1) or \
                   (self.foxy.current_position == '2A')
        elif side == 'right':
            return (self.chica.current_position == '4B' and self.chica.sub_position != -1) or \
                   (self.freddy.current_position == '4B')
        return False

    def get_power_usage_multiplier(self) -> int:
        """Get current power usage multiplier."""
        return self._calculate_power_drain_multiplier()
