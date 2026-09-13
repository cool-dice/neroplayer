"""Screen and loopback-audio perception for the agent."""

from __future__ import annotations

from collections import deque
from typing import Any

import cv2
import numpy as np

from config import GameConfig


class ScreenCapture:
    """Capture a game window region and build channel-first frame stacks."""

    def __init__(self, config: GameConfig) -> None:
        self.config = config
        self._frames: deque[np.ndarray] = deque(maxlen=config.frame_stack)
        self._mss: Any | None = None

    def __enter__(self) -> ScreenCapture:
        self.start()
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def start(self) -> None:
        if self._mss is not None:
            return
        try:
            import mss
        except ImportError as exc:
            raise RuntimeError("Install 'mss' to enable screen capture.") from exc
        self._mss = mss.mss()

    def capture_raw(self) -> np.ndarray:
        """Return one BGR desktop capture."""
        self.start()
        assert self._mss is not None
        bgra = np.asarray(
            self._mss.grab(self.config.capture_region.as_mss_dict()),
            dtype=np.uint8,
        )
        return cv2.cvtColor(bgra, cv2.COLOR_BGRA2BGR)

    def process(self, bgr_frame: np.ndarray) -> np.ndarray:
        """Convert BGR to an 84x84-style grayscale observation frame."""
        gray = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2GRAY)
        return cv2.resize(
            gray,
            (self.config.image_width, self.config.image_height),
            interpolation=cv2.INTER_AREA,
        )

    def update_stack(self, bgr_frame: np.ndarray) -> np.ndarray:
        processed = self.process(bgr_frame)
        if not self._frames:
            self._frames.extend(processed.copy() for _ in range(self.config.frame_stack))
        else:
            self._frames.append(processed)
        return np.stack(tuple(self._frames), axis=0).astype(np.uint8, copy=False)

    def reset_stack(self, bgr_frame: np.ndarray) -> np.ndarray:
        self._frames.clear()
        return self.update_stack(bgr_frame)

    def close(self) -> None:
        if self._mss is not None:
            self._mss.close()
            self._mss = None


class AudioCapture:
    """Capture Windows loopback audio and return fixed-size mel statistics."""

    def __init__(self, config: GameConfig) -> None:
        self.config = config
        self._microphone: Any | None = None

    def start(self) -> None:
        if self._microphone is not None:
            return
        try:
            import soundcard as sc
        except ImportError as exc:
            raise RuntimeError("Install 'soundcard' to enable audio capture.") from exc

        speaker = sc.default_speaker()
        if speaker is None:
            raise RuntimeError("No default output device was found.")
        self._microphone = sc.get_microphone(
            id=str(speaker.name),
            include_loopback=True,
        )

    def capture_features(self) -> np.ndarray:
        """Record a short segment and summarize its mel-spectrogram."""
        self.start()
        assert self._microphone is not None
        sample_count = max(
            1,
            round(self.config.audio_sample_rate * self.config.audio_duration_seconds),
        )
        with self._microphone.recorder(
            samplerate=self.config.audio_sample_rate,
            channels=self.config.audio_channels,
        ) as recorder:
            samples = recorder.record(numframes=sample_count)
        return self.extract_features(np.asarray(samples, dtype=np.float32))

    def extract_features(self, samples: np.ndarray) -> np.ndarray:
        """Convert mono/stereo samples into normalized mel-band statistics."""
        try:
            import librosa
        except ImportError as exc:
            raise RuntimeError("Install 'librosa' to extract audio features.") from exc

        if samples.ndim == 2:
            samples = samples.mean(axis=1)
        samples = np.nan_to_num(samples.reshape(-1), copy=False)
        if samples.size == 0 or np.max(np.abs(samples)) < 1e-8:
            return np.zeros(self.config.audio_feature_size, dtype=np.float32)

        n_fft = min(1024, max(64, 2 ** int(np.floor(np.log2(samples.size)))))
        hop_length = max(16, n_fft // 4)
        mel = librosa.feature.melspectrogram(
            y=samples,
            sr=self.config.audio_sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=self.config.audio_mel_bands,
            power=2.0,
        )
        db = librosa.power_to_db(mel, ref=np.max)
        features = np.concatenate((db.mean(axis=1), db.std(axis=1)))
        # Bound values for a stable Box observation space.
        return np.clip(features / 80.0, -1.0, 1.0).astype(np.float32)
