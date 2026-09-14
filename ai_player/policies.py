"""Multi-modal feature extractor for Stable-Baselines3.

SB3's built-in ``CombinedExtractor`` flattens every non-image key, which would
hand a 32x30 spectrogram to the policy as 960 unstructured numbers. The
extractor below keeps the two modalities separate for as long as it is useful:

    frames  (C, H, W)   --> Vision Backbone (Nature-DQN / ConvNeXt / ResNet) --> cnn_output_dim
    audio   (n_mels, T)  --> small MLP                                      --> 128
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


class ConvNeXtBlock(nn.Module):
    """A lightweight 2D ConvNeXt block: 7x7 depthwise conv, LayerNorm, 1x1 convs, GELU, residual."""

    def __init__(self, dim: int, drop_path: float = 0.0) -> None:
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = nn.GroupNorm(1, dim)  # Channels-first LayerNorm equivalent
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_tensor = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)
        return input_tensor + x


class ConvNeXtBackbone(nn.Module):
    """Lightweight ConvNeXt feature extractor for arbitrary input sizes (84x84, 128x128, 256x256)."""

    def __init__(self, in_channels: int, out_dim: int = 512) -> None:
        super().__init__()
        # Stage 1: Stem downsampling (4x stride)
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=4, stride=4),
            nn.GroupNorm(1, 64),
        )
        self.stage1 = nn.Sequential(
            ConvNeXtBlock(64),
            ConvNeXtBlock(64),
        )
        # Stage 2: Downsample 2x
        self.downsample2 = nn.Sequential(
            nn.GroupNorm(1, 64),
            nn.Conv2d(64, 128, kernel_size=2, stride=2),
        )
        self.stage2 = nn.Sequential(
            ConvNeXtBlock(128),
            ConvNeXtBlock(128),
        )
        # Stage 3: Downsample 2x
        self.downsample3 = nn.Sequential(
            nn.GroupNorm(1, 128),
            nn.Conv2d(128, 256, kernel_size=2, stride=2),
        )
        self.stage3 = nn.Sequential(
            ConvNeXtBlock(256),
            ConvNeXtBlock(256),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stage1(x)
        x = self.downsample2(x)
        x = self.stage2(x)
        x = self.downsample3(x)
        x = self.stage3(x)
        x = self.pool(x)
        return self.head(x)


class ResidualBlock(nn.Module):
    """Standard ResNet basic residual block with skip connection."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        return self.relu(out)


class ResNetBackbone(nn.Module):
    """Lightweight ResNet-style feature extractor for 84x84, 128x128, 256x256 and arbitrary sizes."""

    def __init__(self, in_channels: int, out_dim: int = 512) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.layer1 = nn.Sequential(
            ResidualBlock(64, 64, stride=1),
            ResidualBlock(64, 64, stride=1),
        )
        self.layer2 = nn.Sequential(
            ResidualBlock(64, 128, stride=2),
            ResidualBlock(128, 128, stride=1),
        )
        self.layer3 = nn.Sequential(
            ResidualBlock(128, 256, stride=2),
            ResidualBlock(256, 256, stride=1),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, out_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.pool(x)
        return self.head(x)


class NatureCNNBackbone(nn.Module):
    """Mnih et al. (2015) conv stack for classic fast 84x84 (or arbitrary size with adaptive pool)."""

    def __init__(self, in_channels: int, sample_shape: tuple[int, ...], out_dim: int = 512) -> None:
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            sample = torch.zeros(1, *sample_shape, dtype=torch.float32)
            n_flatten = self.cnn(sample).shape[1]
        self.head = nn.Sequential(nn.Linear(n_flatten, out_dim), nn.ReLU())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.cnn(x))


def build_vision_backbone(
    backbone_type: str,
    in_channels: int,
    frame_shape: tuple[int, ...],
    out_dim: int = 512,
) -> nn.Module:
    """Build the requested vision backbone (nature_cnn, convnext, resnet)."""
    arch = backbone_type.lower()
    if arch in {"nature_cnn", "nature", "dqn"}:
        return NatureCNNBackbone(in_channels, frame_shape, out_dim=out_dim)
    elif arch in {"convnext", "convnext_tiny"}:
        return ConvNeXtBackbone(in_channels, out_dim=out_dim)
    elif arch in {"resnet", "resnet18", "resnet_tiny"}:
        return ResNetBackbone(in_channels, out_dim=out_dim)
    raise ValueError(
        f"Unknown backbone architecture {backbone_type!r}. Supported: nature_cnn, convnext, resnet"
    )


class MultiModalExtractor(BaseFeaturesExtractor):
    """Fuse stacked grayscale or RGB frames with an optional log-mel spectrogram.

    Supports configurable vision backbones: 'nature_cnn', 'convnext', 'resnet'.
    """

    def __init__(
        self,
        observation_space: gym.spaces.Dict,
        features_dim: int = 512,
        cnn_output_dim: int = 512,
        audio_output_dim: int = 128,
        backbone: str = "nature_cnn",
    ) -> None:
        super().__init__(observation_space, features_dim)
        if "frames" not in observation_space.spaces:
            raise ValueError("MultiModalExtractor requires a 'frames' observation key")

        frame_space = observation_space.spaces["frames"]
        n_input_channels = frame_space.shape[0]

        self.vision_backbone = build_vision_backbone(
            backbone,
            n_input_channels,
            frame_space.shape,
            out_dim=cnn_output_dim,
        )

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

    @property
    def cnn(self) -> nn.Module:
        """Backward compatibility alias for code expecting .cnn attribute."""
        if hasattr(self.vision_backbone, "cnn"):
            return self.vision_backbone.cnn
        return self.vision_backbone

    @property
    def cnn_head(self) -> nn.Module:
        """Backward compatibility alias for code expecting .cnn_head attribute."""
        if hasattr(self.vision_backbone, "head"):
            return self.vision_backbone.head
        return nn.Identity()

    def forward(self, observations: dict[str, torch.Tensor]) -> torch.Tensor:
        # Frames arrive as uint8 0..255; scaling here (instead of in the env)
        # keeps the replay buffer and rollout storage 4x smaller.
        frames = observations["frames"].float() / 255.0
        features = self.vision_backbone(frames)
        if self.audio_net is not None:
            audio = observations["audio"].float()
            features = torch.cat([features, self.audio_net(audio)], dim=1)
        return self.fusion(features)
