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


def get_policy_kwargs(arch: str = 'default') -> dict:
    """
    Get policy keyword arguments for different architectures.

    Args:
        arch: Architecture name:
            - 'default': Standard MLP (good baseline)
            - 'small': Smaller network (faster, less capacity)
            - 'large': Larger network (more capacity)
            - 'custom': Uses FnafFeatureExtractor

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
    else:
        raise ValueError(f"Unknown architecture: {arch}")
