"""
Alternative observation space builders for different RL approaches.

Provides different ways to represent the FNAF game state as
observations for the agent.
"""

import numpy as np
from .game_logic import FNAFGame, CAMERAS

POSITION_TO_IDX = {pos: i for i, pos in enumerate(CAMERAS + ['office'])}
NUM_POSITIONS = len(POSITION_TO_IDX)


def build_flat_observation(game: FNAFGame, full_obs: bool = True) -> np.ndarray:
    """
    Build a 77-dimensional flat observation vector.
    This is the default observation format.
    """
    from .fnaf_env import FnafEnv
    # Delegate to the env's method via a temporary instance approach
    # This is the canonical implementation
    obs = np.zeros(77, dtype=np.float32)
    g = game

    obs[0] = g.current_second / 535.0
    obs[1] = max(0, g.player.power) / 100.0
    obs[2] = g.get_power_usage_multiplier() / 4.0
    obs[3] = float(g.player.cameras_on)

    cam_idx = CAMERAS.index(g.player.current_camera) if g.player.current_camera in CAMERAS else 0
    obs[4 + cam_idx] = 1.0

    obs[15] = float(g.player.left_door_closed)
    obs[16] = float(g.player.right_door_closed)
    obs[17] = float(g.player.left_light_on)
    obs[18] = float(g.player.right_light_on)
    obs[19] = float(g.power_out)

    for i, anim in enumerate([g.freddy, g.bonnie, g.chica, g.foxy]):
        base = 20 + i * NUM_POSITIONS
        if full_obs:
            pos_idx = POSITION_TO_IDX.get(anim.current_position, 0)
            obs[base + pos_idx] = 1.0

    obs[68] = max(0, g.foxy.sub_position) / 3.0
    obs[69] = g.freddy.current_ai_level / 20.0
    obs[70] = g.bonnie.current_ai_level / 20.0
    obs[71] = g.chica.current_ai_level / 20.0
    obs[72] = g.foxy.current_ai_level / 20.0
    obs[73] = float(g.is_animatronic_in_doorway('left'))
    obs[74] = float(g.is_animatronic_in_doorway('right'))
    obs[75] = float(g.foxy_attacking or g.foxy_running)
    obs[76] = float(g.freddy_waiting)

    return obs


def build_compact_observation(game: FNAFGame) -> np.ndarray:
    """
    Build a compact 25-dimensional observation.
    Uses integer position indices instead of one-hot encoding.
    Better for smaller networks or simpler algorithms.
    """
    obs = np.zeros(25, dtype=np.float32)
    g = game

    obs[0] = g.current_second / 535.0
    obs[1] = max(0, g.player.power) / 100.0
    obs[2] = g.get_power_usage_multiplier() / 4.0
    obs[3] = float(g.player.cameras_on)
    obs[4] = CAMERAS.index(g.player.current_camera) / len(CAMERAS) if g.player.current_camera in CAMERAS else 0
    obs[5] = float(g.player.left_door_closed)
    obs[6] = float(g.player.right_door_closed)
    obs[7] = float(g.player.left_light_on)
    obs[8] = float(g.player.right_light_on)
    obs[9] = float(g.power_out)

    # Position as normalized index
    obs[10] = POSITION_TO_IDX.get(g.freddy.current_position, 0) / NUM_POSITIONS
    obs[11] = POSITION_TO_IDX.get(g.bonnie.current_position, 0) / NUM_POSITIONS
    obs[12] = POSITION_TO_IDX.get(g.chica.current_position, 0) / NUM_POSITIONS
    obs[13] = POSITION_TO_IDX.get(g.foxy.current_position, 0) / NUM_POSITIONS

    obs[14] = max(0, g.foxy.sub_position) / 3.0
    obs[15] = g.freddy.current_ai_level / 20.0
    obs[16] = g.bonnie.current_ai_level / 20.0
    obs[17] = g.chica.current_ai_level / 20.0
    obs[18] = g.foxy.current_ai_level / 20.0
    obs[19] = float(g.is_animatronic_in_doorway('left'))
    obs[20] = float(g.is_animatronic_in_doorway('right'))
    obs[21] = float(g.foxy_attacking or g.foxy_running)
    obs[22] = float(g.freddy_waiting)
    obs[23] = float(g.bonnie.current_position == 'office')
    obs[24] = float(g.chica.current_position == 'office')

    return obs


def build_image_observation(game: FNAFGame, width: int = 11, height: int = 7) -> np.ndarray:
    """
    Build a 2D grid representation of the game state.
    Useful for CNN-based policies.

    Returns shape (channels, height, width) normalized to [0, 1].
    Channels:
      0: Camera map (which camera is being viewed)
      1: Freddy position
      2: Bonnie position
      3: Chica position
      4: Foxy position
      5: Door/light/power state (encoded as scalar features)
    """
    channels = 6
    obs = np.zeros((channels, height, width), dtype=np.float32)

    # Map cameras to grid positions (rough layout)
    cam_grid = {
        '1A': (3, 5), '1B': (3, 4), '1C': (1, 2),
        '2A': (2, 3), '2B': (1, 3), '3': (2, 2),
        '4A': (4, 3), '4B': (5, 3), '5': (4, 2),
        '6': (5, 4), '7': (4, 5),
        'office': (3, 1),
    }

    # Channel 0: Current camera view
    if game.player.cameras_on and game.player.current_camera in cam_grid:
        y, x = cam_grid[game.player.current_camera]
        if 0 <= y < height and 0 <= x < width:
            obs[0, y, x] = 1.0

    # Channels 1-4: Animatronic positions
    for ch, anim in enumerate([game.freddy, game.bonnie, game.chica, game.foxy], start=1):
        if anim.current_position in cam_grid:
            y, x = cam_grid[anim.current_position]
            if 0 <= y < height and 0 <= x < width:
                obs[ch, y, x] = anim.current_ai_level / 20.0

    # Channel 5: Game state features encoded spatially
    obs[5, 0, 0] = game.player.power / 100.0
    obs[5, 0, 1] = game.current_second / 535.0
    obs[5, 0, 2] = float(game.player.left_door_closed)
    obs[5, 0, 3] = float(game.player.right_door_closed)
    obs[5, 0, 4] = float(game.player.cameras_on)
    obs[5, 0, 5] = game.get_power_usage_multiplier() / 4.0

    return obs
