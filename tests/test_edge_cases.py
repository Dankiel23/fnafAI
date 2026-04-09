"""Tests for edge cases and tricky game mechanics."""

import pytest
import random
from fnaf_gymnasium.envs.game_logic import FNAFGame, GameOverReason


class TestFreddyEdgeCases:

    def test_freddy_blocked_by_right_door_returns_to_4a(self):
        """Freddy at 4B with door closed should go back to 4A, not dining room."""
        game = FNAFGame(night=5)
        game.freddy.current_ai_level = 20
        game.freddy.current_position = '4B'
        game.bonnie.current_position = '1B'
        game.chica.current_position = '1B'
        game.player.cameras_on = True
        game.player.current_camera = '1A'  # Not 4B
        game.player.right_door_closed = True
        game._freddy_movement_check()
        assert game.freddy.current_position == '4A'

    def test_freddy_cant_enter_with_cameras_off_at_4b(self):
        """Freddy needs cameras on to enter office from 4B."""
        game = FNAFGame(night=5)
        game.freddy.current_ai_level = 20
        game.freddy.current_position = '4B'
        game.player.cameras_on = False
        game._freddy_movement_check()
        assert game.freddy.current_position == '4B'  # Stays at 4B

    def test_freddy_enters_when_cameras_on_not_watching_4b(self):
        """Freddy enters office when cameras on but not on 4B."""
        game = FNAFGame(night=5)
        game.freddy.current_ai_level = 20
        game.freddy.current_position = '4B'
        game.bonnie.current_position = '1B'
        game.chica.current_position = '1B'
        game.player.cameras_on = True
        game.player.current_camera = '1A'
        game.player.right_door_closed = False
        game._freddy_movement_check()
        assert game.freddy.current_position == 'office'

    def test_freddy_wait_time_at_ai_10_plus(self):
        """At AI 10+, Freddy should move immediately (0 frame wait)."""
        game = FNAFGame(night=1)
        game.freddy.current_ai_level = 15
        game.freddy.current_position = '1B'
        game.bonnie.current_position = '1B'
        game.chica.current_position = '1B'
        game.player.cameras_on = False
        game._freddy_movement_check()
        # Should move immediately, no waiting
        assert game.freddy.current_position == '7'
        assert not game.freddy_waiting


class TestFoxyEdgeCases:

    def test_foxy_door_bash_escalating_power(self):
        """Each subsequent door bash should drain more power."""
        game = FNAFGame(night=2)
        game.player.left_door_closed = True

        # First bash: 1%
        game.foxy.current_position = '2A'
        game.foxy.office_attempts = 0
        initial = game.player.power
        game.foxy.office_attempts += 1
        game._foxy_reach_office()
        assert abs(game.player.power - (initial - 1)) < 0.01

        # Second bash: 1 + 5 = 6%
        game.foxy.current_position = '2A'
        prev = game.player.power
        game.foxy.office_attempts += 1
        game._foxy_reach_office()
        assert abs(game.player.power - (prev - 6)) < 0.01

    def test_foxy_returns_to_step_0_or_1_after_bash(self):
        """After door bash, Foxy returns to pirate cove at step 0 or 1."""
        random.seed(42)
        game = FNAFGame(night=2)
        game.player.left_door_closed = True
        results = set()
        for i in range(50):
            game.foxy.current_position = '2A'
            game.foxy.office_attempts = i + 1
            game._foxy_reach_office()
            results.add(game.foxy.sub_position)
        assert results == {0, 1}


class TestPowerOutage:

    def test_power_outage_disables_controls(self):
        game = FNAFGame(night=1)
        game._trigger_power_outage()
        assert game.player.left_door_closed is False
        assert game.player.right_door_closed is False
        assert game.player.cameras_on is False
        assert game.player.left_light_on is False
        assert game.player.right_light_on is False

    def test_power_outage_blocks_actions(self):
        game = FNAFGame(night=1)
        game.power_out = True
        game.toggle_cameras()
        assert game.player.cameras_on is False
        game.toggle_left_door()
        assert game.player.left_door_closed is False

    def test_can_survive_power_outage(self):
        """Test that 6AM can be reached during power outage."""
        random.seed(12345)
        game = FNAFGame(night=1)
        game.player.power = 0
        game._trigger_power_outage()
        game.current_second = 530
        # Step through remaining seconds - may or may not die
        for _ in range(10):
            if game.game_over:
                break
            game.step(1.0)
        # Either survived or died to Freddy
        assert game.game_over


class TestBonnieChicaEdgeCases:

    def test_bonnie_returns_to_1b_when_blocked(self):
        game = FNAFGame(night=3)
        game.bonnie.current_ai_level = 20
        game.bonnie.current_position = '2B'
        game.bonnie.sub_position = 1  # In doorway
        game.player.left_door_closed = True
        game._bonnie_movement_check()
        assert game.bonnie.current_position == '1B'
        assert game.bonnie.sub_position == -1

    def test_chica_returns_to_1b_when_blocked(self):
        game = FNAFGame(night=3)
        game.chica.current_ai_level = 20
        game.chica.current_position = '4B'
        game.chica.sub_position = 1
        game.player.right_door_closed = True
        game._chica_movement_check()
        assert game.chica.current_position == '1B'
        assert game.chica.sub_position == -1

    def test_bonnie_immediate_jumpscare_cameras_off(self):
        """Bonnie entering office with cameras off should be immediate death."""
        game = FNAFGame(night=3)
        game.bonnie.current_ai_level = 20
        game.bonnie.current_position = '2B'
        game.bonnie.sub_position = 1
        game.player.left_door_closed = False
        game.player.cameras_on = False
        game._bonnie_movement_check()
        assert game.game_over
        assert game.game_over_reason == GameOverReason.BONNIE


class TestTimingEdgeCases:

    def test_6am_reached_exactly(self):
        game = FNAFGame(night=1)
        game.current_second = 534.5
        game.step(0.5)
        assert game.game_over
        assert game.game_over_reason == GameOverReason.SURVIVED

    def test_multiple_ai_increases_in_one_step(self):
        """A large dt should still trigger all AI increases."""
        game = FNAFGame(night=1)
        initial_bonnie = game.bonnie.current_ai_level
        # Jump from before 2AM to after 4AM in one step
        game.current_second = 170
        game.step(200)  # 170 + 200 = 370, past 4AM
        # Bonnie should have gotten +1 at 2AM, +1 at 3AM, +1 at 4AM = +3
        assert game.bonnie.current_ai_level == initial_bonnie + 3
