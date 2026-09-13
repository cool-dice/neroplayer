from dataclasses import replace

import numpy as np

from config import CONFIG
from perception import AudioCapture, ScreenCapture


class CountingSource:
    def __init__(self):
        self.n = 0

    def grab(self):
        self.n += 1
        return np.full((60, 80, 3), self.n * 10, dtype=np.uint8)

    def close(self):
        pass


def test_frame_stack_rolls_oldest_frame_out():
    cfg = replace(CONFIG.screen, frame_width=8, frame_height=8, frame_stack=3)
    screen = ScreenCapture(cfg, source=CountingSource())

    first = screen.reset()
    assert first.shape == (3, 8, 8) and first.dtype == np.uint8
    assert np.all(first == 10)  # reset() fills the stack with the first frame

    second = screen.observe()
    assert [int(f.mean()) for f in second] == [10, 10, 20]
    assert screen.last_raw_frame.shape == (60, 80, 3)


def test_spectrogram_shape_and_range():
    cfg = CONFIG.audio
    audio = AudioCapture(cfg, start=False)
    rng = np.random.default_rng(0)
    signal = rng.uniform(-0.5, 0.5, cfg.window_samples).astype(np.float32)

    spec = audio.spectrogram(signal)
    assert spec.shape == audio.spectrogram_shape
    assert spec.dtype == np.float32
    assert spec.min() >= 0.0 and spec.max() <= 1.0
    assert spec.max() > 0.0  # noise is not silence

    silence = audio.spectrogram(np.zeros(cfg.window_samples, dtype=np.float32))
    assert np.all(silence == 0.0)


def test_disabled_audio_returns_zero_vector():
    audio = AudioCapture(replace(CONFIG.audio, enabled=False))
    obs = audio.observe()
    assert obs.shape == audio.observation_shape
    assert obs.dtype == np.float32
    assert not obs.any()
    audio.close()
