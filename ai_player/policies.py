"""Multi-modal feature extractor for Stable-Baselines3.

SB3's built-in ``CombinedExtractor`` flattens every non-image key, which would
hand a 32x30 spectrogram to the policy as 960 unstructured numbers. The
extractor below keeps the two modalities separate for as long as it is useful:

    frames  (C, 84, 84) --> Nature-DQN conv stack --> 512
    audio   (n_mels, T)  --> small MLP            --> 128
                                   concat --> Linear --> features_dim

Both branches then feed one fused trunk, which the PPO/DQN heads sit on top of.
Late fusion like this is the standard recipe for image+audio control: the
convolutions stay free to learn spatial filters without competing with the
audio weights.
"""

from __future__ import annotations

import gymnasium as gym
import torch
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn


class MultiModalExtractor(BaseFeaturesExtractor):
    """Fuse stacked grayscale frames with a log-mel spectrogram."""

    def __init__(
        self,
        observation_space: gym.spaces.Dict,
        features_dim: int = 512,
        cnn_output_dim: int = 512,
        audio_output_dim: int = 128,
    ) -> None:
        super().__init__(observation_space, features_dim)
        if "frames" not in observation_space.spaces:
            raise ValueError("MultiModalExtractor requires a 'frames' observation key")

        frame_space = observation_space.spaces["frames"]
        n_input_channels = frame_space.shape[0]
        # Mnih et al. (2015) conv stack: still the best speed/quality trade-off
        # for 84x84 control inputs.
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            sample = torch.zeros(1, *frame_space.shape, dtype=torch.float32)
            n_flatten = self.cnn(sample).shape[1]
        self.cnn_head = nn.Sequential(nn.Linear(n_flatten, cnn_output_dim), nn.ReLU())

        self.audio_space = observation_space.spaces.get("audio")
        fused_dim = cnn_output_dim
        if self.audio_space is not None:
            audio_dim = int(torch.tensor(self.audio_space.shape).prod().item())
            self.audio_net = nn.Sequential(
                nn.Flatten(),
                nn.Linear(audio_dim, 256),
                nn.ReLU(),
                nn.Linear(256, audio_output_dim),
                nn.ReLU(),
            )
            fused_dim += audio_output_dim
        else:
            self.audio_net = None

        self.fusion = nn.Sequential(nn.Linear(fused_dim, features_dim), nn.ReLU())

    def forward(self, observations: dict[str, torch.Tensor]) -> torch.Tensor:
        # Frames arrive as uint8 0..255; scaling here (instead of in the env)
        # keeps the replay buffer and rollout storage 4x smaller.
        frames = observations["frames"].float() / 255.0
        features = self.cnn_head(self.cnn(frames))
        if self.audio_net is not None:
            audio = observations["audio"].float()
            features = torch.cat([features, self.audio_net(audio)], dim=1)
        return self.fusion(features)
