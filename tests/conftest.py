"""Pytest configuration and shared fixtures for FNAF tests."""

import pytest
import gymnasium as gym
import fnaf_gymnasium
from fnaf_gymnasium.envs.game_logic import FNAFGame
from fnaf_gymnasium.envs.fnaf_env import FnafEnv


@pytest.fixture
def game_night1():
    """Fresh FNAF game on Night 1."""
    return FNAFGame(night=1)


@pytest.fixture
def game_night5():
    """Fresh FNAF game on Night 5 (hard difficulty)."""
    return FNAFGame(night=5)


@pytest.fixture
def env_night1():
    """Fresh gymnasium environment on Night 1."""
    env = FnafEnv(night=1)
    yield env
    env.close()


@pytest.fixture
def env_night1_partial():
    """Night 1 environment with partial observability."""
    env = FnafEnv(night=1, full_observability=False)
    yield env
    env.close()


@pytest.fixture
def game_all_active():
    """Game with all animatronics at AI level 20."""
    game = FNAFGame(night=7)
    for anim in game.animatronics:
        anim.current_ai_level = 20
    return game


@pytest.fixture
def game_foxy_only():
    """Game with only Foxy active."""
    game = FNAFGame(night=1)
    game.freddy.current_ai_level = 0
    game.bonnie.current_ai_level = 0
    game.chica.current_ai_level = 0
    game.foxy.current_ai_level = 10
    return game
