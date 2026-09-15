from __future__ import annotations

import gymnasium as gym
import numpy as np
import pytest
import torch

from ai_player.policies import (
    ConvNeXtBlock,
    MultiModalExtractor,
    ResidualBlock,
    build_vision_backbone,
)


def test_convnext_block_residual_identity_shape():
    block = ConvNeXtBlock(dim=64)
    x = torch.randn(2, 64, 28, 28)
    out = block(x)
    assert out.shape == (2, 64, 28, 28)
    assert torch.isfinite(out).all()


def test_residual_block_shape_and_downsample():
    # Stride 1, same dim
    res1 = ResidualBlock(64, 64, stride=1)
    x1 = torch.randn(2, 64, 32, 32)
    assert res1(x1).shape == (2, 64, 32, 32)

    # Stride 2, increase dim
    res2 = ResidualBlock(64, 128, stride=2)
    x2 = torch.randn(2, 64, 32, 32)
    assert res2(x2).shape == (2, 128, 16, 16)


@pytest.mark.parametrize("backbone_type", ["nature_cnn", "convnext", "resnet"])
def test_vision_backbone_forward_arbitrary_sizes(backbone_type):
    in_channels = 4
    for h, w in [(84, 84), (128, 128), (256, 256)]:
        sample_shape = (in_channels, h, w)
        backbone = build_vision_backbone(backbone_type, in_channels, sample_shape, out_dim=256)
        x = torch.randn(2, in_channels, h, w)
        out = backbone(x)
        assert out.shape == (2, 256)
        assert torch.isfinite(out).all()


def test_build_vision_backbone_invalid():
    with pytest.raises(ValueError, match="Unknown backbone architecture"):
        build_vision_backbone("unknown_arch", 4, (4, 84, 84))


def test_multimodal_extractor_convnext_rgb_256():
    # 4 frames stacked RGB -> 12 channels
    obs_space = gym.spaces.Dict({
        "frames": gym.spaces.Box(0, 255, shape=(12, 256, 256), dtype=np.uint8),
        "audio": gym.spaces.Box(0.0, 1.0, shape=(32, 30), dtype=np.float32),
    })

    extractor = MultiModalExtractor(
        obs_space,
        features_dim=1024,
        cnn_output_dim=512,
        audio_output_dim=128,
        backbone="convnext",
    )

    batch = {
        "frames": torch.randint(0, 255, (2, 12, 256, 256), dtype=torch.uint8),
        "audio": torch.rand(2, 32, 30, dtype=torch.float32),
    }

    out = extractor(batch)
    assert out.shape == (2, 1024)
    assert torch.isfinite(out).all()
    # Check backward compatibility aliases
    assert extractor.cnn is not None
    assert extractor.cnn_head is not None


def test_multimodal_extractor_resnet_grayscale():
    obs_space = gym.spaces.Dict({
        "frames": gym.spaces.Box(0, 255, shape=(4, 128, 128), dtype=np.uint8),
    })

    extractor = MultiModalExtractor(
        obs_space,
        features_dim=512,
        backbone="resnet",
    )

    batch = {
        "frames": torch.randint(0, 255, (2, 4, 128, 128), dtype=torch.uint8),
    }

    out = extractor(batch)
    assert out.shape == (2, 512)
    assert torch.isfinite(out).all()
    assert extractor.audio_net is None


def test_multimodal_extractor_with_cognitive_observation():
    obs_space = gym.spaces.Dict({
        "frames": gym.spaces.Box(0, 255, shape=(4, 84, 84), dtype=np.uint8),
        "audio": gym.spaces.Box(0.0, 1.0, shape=(32, 30), dtype=np.float32),
        "cognitive": gym.spaces.Box(0.0, 1.0, shape=(8,), dtype=np.float32),
    })

    extractor = MultiModalExtractor(
        obs_space,
        features_dim=256,
        cnn_output_dim=128,
        audio_output_dim=64,
        backbone="nature_cnn",
    )

    batch = {
        "frames": torch.randint(0, 255, (2, 4, 84, 84), dtype=torch.uint8),
        "audio": torch.rand(2, 32, 30, dtype=torch.float32),
        "cognitive": torch.rand(2, 8, dtype=torch.float32),
    }

    out = extractor(batch)
    assert out.shape == (2, 256)
    assert torch.isfinite(out).all()
    assert extractor.cognitive_net is not None
