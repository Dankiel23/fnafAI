"""Tests for the FNAF 1 game logic engine."""

import pytest
import random
from fnaf_gymnasium.envs.game_logic import (
    FNAFGame, GameOverReason, NIGHT_DURATION,
    AI_INCREASE_2AM, AI_INCREASE_3AM, AI_INCREASE_4AM,
)


class TestGameInitialization:
    """Test game initialization and starting conditions."""

    def test_default_night_1(self):
        game = FNAFGame(night=1)
        assert game.night == 1
        assert game.current_second == 0.0
        assert game.game_over is False
        assert game.power_out is False

    def test_starting_positions(self):
        game = FNAFGame(night=1)
        assert game.freddy.current_position == '1A'
        assert game.bonnie.current_position == '1A'
        assert game.chica.current_position == '1A'
        assert game.foxy.current_position == '1C'
        assert game.foxy.sub_position == 0

    def test_night_1_ai_levels(self):
        game = FNAFGame(night=1)
        assert game.freddy.current_ai_level == 0
        assert game.bonnie.current_ai_level == 0
        assert game.chica.current_ai_level == 0
        assert game.foxy.current_ai_level == 0

    def test_night_2_ai_levels(self):
        game = FNAFGame(night=2)
        assert game.freddy.current_ai_level == 0
        assert game.bonnie.current_ai_level == 3
        assert game.chica.current_ai_level == 1
        assert game.foxy.current_ai_level == 1

    def test_starting_power(self):
        game = FNAFGame(night=1)
        assert game.player.power == 99.0

    def test_player_defaults(self):
        game = FNAFGame(night=1)
        assert game.player.cameras_on is False
        assert game.player.left_door_closed is False
        assert game.player.right_door_closed is False
        assert game.player.current_camera == '1A'


class TestPowerMechanics:
    """Test power drain calculations."""

    def test_base_power_drain_multiplier(self):
        game = FNAFGame(night=1)
        # Nothing active - base multiplier is 1
        assert game._calculate_power_drain_multiplier() == 1

    def test_power_drain_with_doors(self):
        game = FNAFGame(night=1)
        game.player.left_door_closed = True
        assert game._calculate_power_drain_multiplier() == 2
        game.player.right_door_closed = True
        assert game._calculate_power_drain_multiplier() == 3

    def test_power_drain_max_4(self):
        game = FNAFGame(night=1)
        game.player.left_door_closed = True
        game.player.right_door_closed = True
        game.player.cameras_on = True
        game.player.left_light_on = True
        game.player.right_light_on = True
        assert game._calculate_power_drain_multiplier() == 4

    def test_power_drains_over_time(self):
        game = FNAFGame(night=1)
        initial_power = game.player.power
        game.step(1.0)
        assert game.player.power < initial_power

    def test_power_outage_triggers(self):
        game = FNAFGame(night=1)
        game.player.power = 0.01
        game.step(1.0)
        assert game.power_out is True


class TestAILevelIncreases:
    """Test AI level increases at 2AM, 3AM, 4AM."""

    def test_bonnie_increases_at_2am(self):
        game = FNAFGame(night=1)
        initial = game.bonnie.current_ai_level
        game.current_second = AI_INCREASE_2AM - 1
        game.step(1.0)
        assert game.bonnie.current_ai_level == initial + 1

    def test_ai_increases_at_3am(self):
        game = FNAFGame(night=1)
        bonnie_init = game.bonnie.current_ai_level
        chica_init = game.chica.current_ai_level
        foxy_init = game.foxy.current_ai_level
        game.current_second = AI_INCREASE_3AM - 1
        game.step(1.0)
        assert game.bonnie.current_ai_level == bonnie_init + 1
        assert game.chica.current_ai_level == chica_init + 1
        assert game.foxy.current_ai_level == foxy_init + 1

    def test_freddy_no_increase(self):
        game = FNAFGame(night=1)
        initial = game.freddy.current_ai_level
        # Run through all AI increase times
        game.current_second = AI_INCREASE_2AM - 1
        game.step(1.0)
        game.step(1.0)
        assert game.freddy.current_ai_level == initial


class TestPlayerActions:
    """Test player action mechanics."""

    def test_toggle_cameras(self):
        game = FNAFGame(night=1)
        assert game.player.cameras_on is False
        game.toggle_cameras()
        assert game.player.cameras_on is True
        game.toggle_cameras()
        assert game.player.cameras_on is False

    def test_switch_camera(self):
        game = FNAFGame(night=1)
        game.toggle_cameras()
        game.switch_camera('4B')
        assert game.player.current_camera == '4B'

    def test_invalid_camera_ignored(self):
        game = FNAFGame(night=1)
        game.switch_camera('invalid')
        assert game.player.current_camera == '1A'

    def test_switch_camera_ignored_when_cameras_off(self):
        game = FNAFGame(night=1)
        game.switch_camera('4B')
        assert game.player.current_camera == '1A'

    def test_toggle_doors(self):
        game = FNAFGame(night=1)
        game.toggle_left_door()
        assert game.player.left_door_closed is True
        game.toggle_right_door()
        assert game.player.right_door_closed is True

    def test_toggle_lights(self):
        game = FNAFGame(night=1)
        game.toggle_left_light()
        assert game.player.left_light_on is True
        game.toggle_right_light()
        assert game.player.right_light_on is True

    def test_actions_disabled_during_power_outage(self):
        game = FNAFGame(night=1)
        game.power_out = True
        game.toggle_cameras()
        assert game.player.cameras_on is False
        game.toggle_left_door()
        assert game.player.left_door_closed is False


class TestGameCompletion:
    """Test game over conditions."""

    def test_survive_night(self):
        """Test that reaching 535 seconds wins the game."""
        game = FNAFGame(night=1)
        game.current_second = NIGHT_DURATION - 1
        game.step(1.0)
        assert game.game_over is True
        assert game.game_over_reason == GameOverReason.SURVIVED

    def test_full_night_simulation(self):
        """Simulate a full night with random actions."""
        random.seed(42)
        game = FNAFGame(night=1)
        steps = 0
        while not game.game_over and steps < 600:
            # Random actions
            action = random.randint(0, 5)
            if action == 1:
                game.toggle_cameras()
            elif action == 2:
                game.toggle_left_door()
            elif action == 3:
                game.toggle_right_door()
            game.step(1.0)
            steps += 1
        assert game.game_over is True


class TestMovementChecks:
    """Test animatronic movement check mechanics."""

    def test_ai_level_0_never_moves(self):
        """At AI level 0, movement should never succeed."""
        game = FNAFGame(night=1)
        game.freddy.current_ai_level = 0
        for _ in range(100):
            assert game._make_movement_check(game.freddy) is False

    def test_ai_level_20_always_moves(self):
        """At AI level 20, movement should always succeed."""
        game = FNAFGame(night=1)
        game.freddy.current_ai_level = 20
        for _ in range(100):
            assert game._make_movement_check(game.freddy) is True


class TestFreddyLogic:
    """Test Freddy-specific mechanics."""

    def test_freddy_cant_leave_with_others_on_stage(self):
        game = FNAFGame(night=1)
        game.freddy.current_ai_level = 20
        # Both Bonnie and Chica on stage - Freddy can't leave
        assert game.bonnie.current_position == '1A'
        assert game.chica.current_position == '1A'
        game._freddy_movement_check()
        assert game.freddy.current_position == '1A'

    def test_freddy_path(self):
        """Test Freddy follows his set path."""
        game = FNAFGame(night=1)
        game.freddy.current_ai_level = 20
        # Move Bonnie and Chica off stage
        game.bonnie.current_position = '1B'
        game.chica.current_position = '1B'

        # Freddy's path: 1A -> 1B -> 7 -> 6 -> 4A -> 4B
        expected_path = ['1B', '7', '6', '4A', '4B']
        for expected in expected_path:
            game._freddy_movement_check()
            # With AI 20, wait time is 0, so should move immediately
            if game.freddy_waiting:
                game.freddy.current_position = game.freddy_wait_destination
                game.freddy_waiting = False
            assert game.freddy.current_position == expected


class TestFoxyLogic:
    """Test Foxy-specific mechanics."""

    def test_foxy_starts_at_pirate_cove(self):
        game = FNAFGame(night=1)
        assert game.foxy.current_position == '1C'
        assert game.foxy.sub_position == 0

    def test_foxy_auto_fails_with_cameras(self):
        """Foxy should auto-fail while cameras are on."""
        game = FNAFGame(night=2)
        game.player.cameras_on = True
        game.foxy.current_ai_level = 20
        initial_sub = game.foxy.sub_position
        game._foxy_movement_check()
        assert game.foxy.sub_position == initial_sub  # Didn't advance

    def test_foxy_door_bash_power_drain(self):
        """Test Foxy drains power on door bash."""
        game = FNAFGame(night=1)
        game.player.left_door_closed = True
        game.foxy.current_position = '2A'
        game.foxy.office_attempts = 1
        initial_power = game.player.power
        game._foxy_reach_office()
        # First attempt: 1 + (1-1)*5 = 1% drain
        assert game.player.power == initial_power - 1
        assert game.foxy.current_position == '1C'
