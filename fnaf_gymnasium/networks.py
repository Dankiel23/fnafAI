"""
Custom neural network architectures for FNAF RL agents.

Provides policy network configurations optimized for the
FNAF observation and action spaces.
"""

from typing import Dict, List, Type

import torch as th
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from gymnasium import spaces


class FnafFeatureExtractor(BaseFeaturesExtractor):
    """
    Custom feature extractor that processes different parts
    of the FNAF observation differently.

    Splits the observation into:
    - Continuous features (time, power, usage)
    - Binary features (doors, lights, cameras)
    - Position one-hot encodings (animatronic positions)
    - AI level features

    Each group gets its own processing, then they're concatenated.
    """

    def __init__(self, observation_space: spaces.Box, features_dim: int = 128):
        super().__init__(observation_space, features_dim)

        obs_size = observation_space.shape[0]

        # Continuous features: time, power, usage (3 values)
        self.continuous_net = nn.Sequential(
            nn.Linear(3, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
        )

        # Binary state features: cameras_on, doors, lights, power_out (6 values)
        self.binary_net = nn.Sequential(
            nn.Linear(6, 16),
            nn.ReLU(),
        )

        # Camera one-hot (11 values)
        self.camera_net = nn.Sequential(
            nn.Linear(11, 16),
            nn.ReLU(),
        )

        # Position encodings: 4 animatronics x 12 positions = 48 values
        self.position_net = nn.Sequential(
            nn.Linear(48, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
        )

        # AI levels + foxy sub + threats + attack states (9 values)
        self.ai_net = nn.Sequential(
            nn.Linear(9, 24),
            nn.ReLU(),
        )

        # Combine all (32 + 16 + 16 + 32 + 24 = 120)
        combined_size = 32 + 16 + 16 + 32 + 24
        self.combiner = nn.Sequential(
            nn.Linear(combined_size, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        # Split observation into groups
        continuous = observations[:, :3]          # time, power, usage
        cameras_on = observations[:, 3:4]         # cameras_on
        camera_hot = observations[:, 4:15]        # camera one-hot
        binary = observations[:, 15:20]           # doors, lights, power_out
        binary_all = th.cat([cameras_on, binary], dim=1)
        positions = observations[:, 20:68]        # 4 x 12 position one-hots
        ai_state = observations[:, 68:77]         # AI levels, foxy sub, threats, etc.

        cont_out = self.continuous_net(continuous)
        bin_out = self.binary_net(binary_all)
        cam_out = self.camera_net(camera_hot)
        pos_out = self.position_net(positions)
        ai_out = self.ai_net(ai_state)

        combined = th.cat([cont_out, bin_out, cam_out, pos_out, ai_out], dim=1)
        return self.combiner(combined)


class FnafCVFeatureExtractor(BaseFeaturesExtractor):
    """
    Feature extractor for the 87-dim CV-ready observation space.

    Processes each semantic group separately before combining:
      - continuous state  : time, power, usage                [0-2]
      - binary state      : cameras_on, doors, lights, power  [3,15-19]
      - camera nav        : which camera (one-hot)            [4-14]
      - camera view       : what CV sees right now            [20-27]
      - memory positions  : 4 x 12 last-known one-hots        [28-75]
      - memory meta       : staleness + foxy sub-pos          [76-80]
      - danger/timing     : derived signals                   [81-86]
    """

    def __init__(self, observation_space: spaces.Box, features_dim: int = 192):
        super().__init__(observation_space, features_dim)

        self.continuous_net = nn.Sequential(
            nn.Linear(3, 16), nn.ReLU(),
        )
        self.binary_net = nn.Sequential(
            nn.Linear(6, 16), nn.ReLU(),
        )
        self.camera_nav_net = nn.Sequential(
            nn.Linear(11, 16), nn.ReLU(),
        )
        self.camera_view_net = nn.Sequential(
            nn.Linear(8, 24), nn.ReLU(),
        )
        # 4 animatronics × 12 positions = 48 values
        self.memory_pos_net = nn.Sequential(
            nn.Linear(48, 64), nn.ReLU(),
            nn.Linear(64, 48), nn.ReLU(),
        )
        # staleness (4) + foxy sub-pos (1) = 5
        self.memory_meta_net = nn.Sequential(
            nn.Linear(5, 16), nn.ReLU(),
        )
        # danger signals (3) + timing signals (3) = 6
        self.danger_net = nn.Sequential(
            nn.Linear(6, 16), nn.ReLU(),
        )

        # 16+16+16+24+48+16+16 = 152
        combined_size = 16 + 16 + 16 + 24 + 48 + 16 + 16
        self.combiner = nn.Sequential(
            nn.Linear(combined_size, features_dim), nn.ReLU(),
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        continuous   = observations[:, 0:3]
        cameras_on   = observations[:, 3:4]
        camera_nav   = observations[:, 4:15]
        doors_lights = observations[:, 15:20]       # doors, lights, power_out
        binary       = th.cat([cameras_on, doors_lights], dim=1)  # 6 values
        camera_view  = observations[:, 20:28]
        memory_pos   = observations[:, 28:76]
        memory_meta  = observations[:, 76:81]
        danger       = observations[:, 81:87]

        x = th.cat([
            self.continuous_net(continuous),
            self.binary_net(binary),
            self.camera_nav_net(camera_nav),
            self.camera_view_net(camera_view),
            self.memory_pos_net(memory_pos),
            self.memory_meta_net(memory_meta),
            self.danger_net(danger),
        ], dim=1)
        return self.combiner(x)


def get_policy_kwargs(arch: str = 'default') -> dict:
    """
    Get policy keyword arguments for different architectures.

    Args:
        arch: Architecture name:
            - 'default': Standard MLP (good baseline)
            - 'small': Smaller network (faster, less capacity)
            - 'large': Larger network (more capacity)
            - 'custom': Uses FnafFeatureExtractor (77-dim full-obs env)
            - 'cv': Uses FnafCVFeatureExtractor (87-dim CV env)

    Returns:
        Dict of policy kwargs to pass to SB3 algorithm
    """
    if arch == 'default':
        return {
            'net_arch': dict(pi=[128, 128], vf=[128, 128]),
        }
    elif arch == 'small':
        return {
            'net_arch': dict(pi=[64, 64], vf=[64, 64]),
        }
    elif arch == 'large':
        return {
            'net_arch': dict(pi=[256, 256, 128], vf=[256, 256, 128]),
        }
    elif arch == 'custom':
        return {
            'features_extractor_class': FnafFeatureExtractor,
            'features_extractor_kwargs': {'features_dim': 128},
            'net_arch': dict(pi=[64, 64], vf=[64, 64]),
        }
    elif arch == 'cv':
        return {
            'features_extractor_class': FnafCVFeatureExtractor,
            'features_extractor_kwargs': {'features_dim': 192},
            'net_arch': dict(pi=[128, 128], vf=[128, 128]),
        }
    else:
        raise ValueError(f"Unknown architecture: {arch}")
