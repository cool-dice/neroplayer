from __future__ import annotations

import numpy as np
import pytest

from ai_player.config import AudioConfig, Region, VisionConfig
from ai_player.perception import (
    AudioFeatureExtractor,
    FrameProcessor,
    FrameStack,
    LoopbackAudioCapture,
    SilentAudioCapture,
    build_audio_source,
    crop_region,
)


def test_frame_processor_downscales_to_grayscale():
    vision = VisionConfig()
    frame = np.random.default_rng(0).integers(0, 255, (240, 320, 3), dtype=np.uint8)

    processed = FrameProcessor(vision).process(frame)

    assert processed.shape == (vision.height, vision.width)
    assert processed.dtype == np.uint8


def test_frame_processor_keeps_colour_when_requested():
    vision = VisionConfig(grayscale=False, frame_stack=2)
    frame = np.zeros((100, 100, 3), dtype=np.uint8)

    processed = FrameProcessor(vision).process(frame)

    assert processed.shape == (vision.height, vision.width, 3)
    assert vision.observation_shape == (6, vision.height, vision.width)


def test_frame_stack_rolls_oldest_frame_out():
    vision = VisionConfig(frame_stack=3)
    stack = FrameStack(vision)
    first = np.full((vision.height, vision.width), 10, dtype=np.uint8)

    obs = stack.reset(first)
    assert obs.shape == vision.observation_shape
    assert np.all(obs == 10)

    obs = stack.push(np.full((vision.height, vision.width), 20, dtype=np.uint8))
    assert np.all(obs[-1] == 20)
    assert np.all(obs[0] == 10)

    for value in (30, 40, 50):
        obs = stack.push(np.full((vision.height, vision.width), value, dtype=np.uint8))
    assert np.all(obs[0] == 30) and np.all(obs[-1] == 50)


def test_audio_features_have_fixed_shape_and_range():
    audio = AudioConfig()
    extractor = AudioFeatureExtractor(audio)
    t = np.arange(audio.window_samples, dtype=np.float32) / audio.sample_rate
    tone = 0.5 * np.sin(2 * np.pi * 440.0 * t)

    features = extractor.extract(tone)

    assert features.shape == audio.observation_shape == (32, 30)
    assert features.dtype == np.float32
    assert features.min() >= 0.0 and features.max() <= 1.0
    assert features.max() > 0.0


def test_audio_features_pad_short_and_silent_windows():
    audio = AudioConfig()
    extractor = AudioFeatureExtractor(audio)

    short = extractor.extract(np.ones(100, dtype=np.float32))
    silence = extractor.extract(np.zeros(audio.window_samples, dtype=np.float32))

    assert short.shape == audio.observation_shape
    assert silence.shape == audio.observation_shape
    assert not np.any(silence)


def test_disabled_audio_yields_a_silent_source():
    audio = AudioConfig(enabled=False)

    assert isinstance(build_audio_source(audio), SilentAudioCapture)


def test_unavailable_loopback_degrades_to_silence(monkeypatch):
    audio = AudioConfig()

    def explode(self):
        raise AssertionError  # soundcard raises bare assertions with no message

    monkeypatch.setattr(LoopbackAudioCapture, "_resolve_microphone", explode)

    with pytest.warns(RuntimeWarning, match="AssertionError"):
        source = build_audio_source(audio)

    # Training must survive a machine with no loopback device.
    assert isinstance(source, SilentAudioCapture)
    assert not np.any(source.read_window())


def test_recorder_failure_mid_run_degrades_to_silence():
    audio = AudioConfig()
    source = LoopbackAudioCapture(audio)
    source._buffer = np.ones(audio.window_samples, dtype=np.float32)
    source._error = OSError("device disconnected")

    with pytest.warns(RuntimeWarning, match="device disconnected"):
        first = source.read_window()
    second = source.read_window()

    # Losing the audio device must not abort a long training run.
    assert not np.any(first)
    assert not np.any(second)


def test_silent_capture_matches_window_length():
    audio = AudioConfig()
    source = SilentAudioCapture(audio)
    source.start()

    window = source.read_window()
    source.stop()

    assert window.shape == (audio.window_samples,)


def test_crop_region_clamps_to_frame():
    frame = np.zeros((50, 60, 3), dtype=np.uint8)

    patch = crop_region(frame, Region(left=40, top=40, width=100, height=100))

    assert patch.shape == (10, 20, 3)
