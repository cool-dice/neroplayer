"""Perception: turns raw screen pixels and system audio into RL observations.

Two independent sensors live here:

* :class:`ScreenCapture` grabs the game viewport with ``mss`` (fast, no
  window-handle juggling), converts it to grayscale, downsamples it to
  ``frame_height x frame_width`` and stacks the last ``frame_stack`` frames
  so a single observation encodes motion (velocity/direction of sprites).
* :class:`AudioCapture` records the *loopback* of the default speaker with
  ``soundcard`` on a background thread and converts the trailing window into
  a normalised log-mel spectrogram with ``librosa``.

Both expose ``reset()`` / ``observe()`` and never raise from ``observe()`` so
one flaky sensor cannot kill a training run; failures degrade to zeros and a
logged warning.
"""

from __future__ import annotations

import logging
import threading
from collections import deque
from typing import Protocol

import cv2
import numpy as np

from config import AudioConfig, ScreenConfig

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Screen
# --------------------------------------------------------------------------- #
class FrameSource(Protocol):
    """Anything that can return a BGR frame of the capture region."""

    def grab(self) -> np.ndarray: ...

    def close(self) -> None: ...


class MSSFrameSource:
    """Grabs the configured region with ``mss`` and returns a BGR uint8 array."""

    def __init__(self, cfg: ScreenConfig) -> None:
        import mss  # imported lazily: needs a display server

        self._sct = mss.mss()
        self._region = cfg.capture_region.as_mss()

    def grab(self) -> np.ndarray:
        shot = self._sct.grab(self._region)
        # mss returns BGRA; drop alpha so downstream code can assume BGR.
        return np.asarray(shot, dtype=np.uint8)[:, :, :3]

    def close(self) -> None:
        self._sct.close()


class ScreenCapture:
    """Screen sensor producing stacked grayscale frames.

    ``observe()`` returns an array of shape ``(frame_stack, H, W)`` with dtype
    ``uint8``, which Stable-Baselines3 recognises as a channel-first image and
    routes through its NatureCNN feature extractor.
    """

    def __init__(self, cfg: ScreenConfig, source: FrameSource | None = None) -> None:
        self.cfg = cfg
        self._source = source or MSSFrameSource(cfg)
        self._frames: deque[np.ndarray] = deque(maxlen=cfg.frame_stack)
        self.last_raw_frame: np.ndarray | None = None

    @property
    def observation_shape(self) -> tuple[int, int, int]:
        return (self.cfg.frame_stack, self.cfg.frame_height, self.cfg.frame_width)

    def grab_raw(self) -> np.ndarray:
        """Capture and cache the full-resolution BGR frame (used by the observer)."""
        self.last_raw_frame = self._source.grab()
        return self.last_raw_frame

    def preprocess(self, frame_bgr: np.ndarray) -> np.ndarray:
        """Grayscale + resize a BGR frame to the agent's input resolution."""
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        # INTER_AREA is the least aliasing-prone choice when shrinking.
        return cv2.resize(
            gray, (self.cfg.frame_width, self.cfg.frame_height), interpolation=cv2.INTER_AREA
        )

    def reset(self) -> np.ndarray:
        """Fill the frame stack with the current frame and return the observation."""
        frame = self.preprocess(self.grab_raw())
        self._frames.clear()
        for _ in range(self.cfg.frame_stack):
            self._frames.append(frame)
        return self._stacked()

    def observe(self) -> np.ndarray:
        """Grab one new frame, push it onto the stack and return the stack."""
        try:
            frame = self.preprocess(self.grab_raw())
        except Exception:  # pragma: no cover - hardware dependent
            logger.exception("Screen capture failed; repeating last frame")
            frame = self._frames[-1] if self._frames else np.zeros(
                (self.cfg.frame_height, self.cfg.frame_width), dtype=np.uint8
            )
        self._frames.append(frame)
        return self._stacked()

    def _stacked(self) -> np.ndarray:
        return np.stack(self._frames, axis=0).astype(np.uint8)

    def close(self) -> None:
        self._source.close()


# --------------------------------------------------------------------------- #
# Audio
# --------------------------------------------------------------------------- #
class AudioCapture:
    """Loopback audio sensor producing a normalised log-mel spectrogram.

    Recording runs on a daemon thread that continuously appends PCM blocks to
    a ring buffer, so ``observe()`` is non-blocking and always sees the most
    recent ``window_seconds`` of sound regardless of the RL step rate.
    """

    def __init__(self, cfg: AudioConfig, start: bool = True) -> None:
        self.cfg = cfg
        self._buffer = np.zeros(cfg.window_samples, dtype=np.float32)
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._mic = None
        self.available = False

        if cfg.enabled and start:
            self._start()

    @property
    def observation_shape(self) -> tuple[int, int]:
        return (self.cfg.n_mels, self.cfg.n_frames)

    # -- lifecycle -------------------------------------------------------- #
    def _start(self) -> None:
        try:
            import soundcard as sc

            speaker = sc.default_speaker()
            # include_loopback=True exposes the speaker as a virtual microphone
            # (WASAPI loopback on Windows), i.e. "what the game is playing".
            self._mic = sc.get_microphone(speaker.name, include_loopback=True)
        except Exception as exc:  # pragma: no cover - hardware dependent
            logger.warning("Loopback audio unavailable (%s); audio observation will be zeros", exc)
            return

        self._thread = threading.Thread(target=self._record_loop, name="audio-capture", daemon=True)
        self._thread.start()
        self.available = True

    def _record_loop(self) -> None:  # pragma: no cover - hardware dependent
        # A block of ~1/4 window keeps latency low without hammering the API.
        block = max(256, self.cfg.window_samples // 4)
        try:
            with self._mic.recorder(samplerate=self.cfg.sample_rate, channels=self.cfg.channels) as rec:
                while not self._stop.is_set():
                    data = rec.record(numframes=block)  # (block, channels) float32
                    mono = data.mean(axis=1).astype(np.float32)
                    with self._lock:
                        self._buffer = np.roll(self._buffer, -len(mono))
                        self._buffer[-len(mono):] = mono
        except Exception:
            logger.exception("Audio recorder thread died; audio observation frozen")
            self.available = False

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)

    # -- observations ----------------------------------------------------- #
    def reset(self) -> np.ndarray:
        """Clear stale audio from the previous episode and return an observation."""
        with self._lock:
            self._buffer[:] = 0.0
        return self.observe()

    def observe(self) -> np.ndarray:
        """Return the log-mel spectrogram of the trailing window, scaled to [0, 1]."""
        with self._lock:
            window = self._buffer.copy()
        if not self.available:
            return np.zeros(self.observation_shape, dtype=np.float32)
        try:
            return self.spectrogram(window)
        except Exception:  # pragma: no cover
            logger.exception("Spectrogram computation failed")
            return np.zeros(self.observation_shape, dtype=np.float32)

    def spectrogram(self, samples: np.ndarray) -> np.ndarray:
        """Compute the normalised log-mel spectrogram of a 1-D float32 signal."""
        import librosa  # heavy import; only needed when audio is live

        mel = librosa.feature.melspectrogram(
            y=samples,
            sr=self.cfg.sample_rate,
            n_fft=self.cfg.n_fft,
            hop_length=self.cfg.hop_length,
            n_mels=self.cfg.n_mels,
        )
        db = librosa.power_to_db(mel, ref=1.0, top_db=None)
        norm = (db - self.cfg.min_db) / (self.cfg.max_db - self.cfg.min_db)
        norm = np.clip(norm, 0.0, 1.0).astype(np.float32)
        # librosa's frame count can differ by one from the analytic value at
        # window edges; pin the shape so the observation space is static.
        return _fit_width(norm, self.cfg.n_frames)


def _fit_width(arr: np.ndarray, width: int) -> np.ndarray:
    """Crop or zero-pad the last axis of ``arr`` to exactly ``width`` columns."""
    if arr.shape[-1] == width:
        return arr
    if arr.shape[-1] > width:
        return arr[..., :width]
    pad = width - arr.shape[-1]
    return np.pad(arr, [(0, 0)] * (arr.ndim - 1) + [(0, pad)])
