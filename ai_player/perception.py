"""Perception: turn raw screen pixels and system audio into agent observations.

Two independent sensor pipelines live here:

* **Vision** -- ``ScreenCapture`` grabs a desktop region with ``mss`` (a few
  hundred microseconds per grab), ``FrameProcessor`` shrinks it to 84x84
  grayscale, and ``FrameStack`` keeps the last N frames so the policy can infer
  motion from a single observation.
* **Audio** -- ``LoopbackAudioCapture`` records whatever the speakers are
  playing on a background thread, and ``AudioFeatureExtractor`` converts the
  most recent window into a log-mel spectrogram. Games leak a lot of state
  through sound (jump, hit, coin, death jingle) that pixels alone miss.

Both pipelines expose a small protocol plus a headless stub, so the environment
can run on a machine with no display and no sound card (CI, unit tests, the
bundled mock game).
"""

from __future__ import annotations

import threading
import warnings
from typing import Protocol, runtime_checkable

import cv2
import numpy as np

from .config import AudioConfig, CaptureConfig, Region, VisionConfig

BGRFrame = np.ndarray  # (H, W, 3) uint8
StackedFrames = np.ndarray  # (C, H, W) uint8
AudioFeatures = np.ndarray  # (n_mels, n_frames) float32 in [0, 1]


# ---------------------------------------------------------------------------
# Vision
# ---------------------------------------------------------------------------
@runtime_checkable
class FrameSource(Protocol):
    """Anything that can hand back the current game frame as BGR pixels."""

    def grab(self) -> BGRFrame: ...

    def close(self) -> None: ...


class ScreenCapture:
    """Fast desktop region capture backed by ``mss``.

    ``mss`` handles are not thread-safe, so one instance belongs to one thread.
    The handle is created lazily on the first grab, which keeps the object
    importable and constructible on machines without a display.
    """

    def __init__(self, capture: CaptureConfig) -> None:
        self._config = capture
        self._monitor = capture.region.as_mss_monitor()
        self._sct = None

    def _ensure_handle(self) -> None:
        if self._sct is not None:
            return
        try:
            import mss
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise RuntimeError(
                "mss is required for screen capture. Install it with "
                "`pip install mss`, or run with --mock to use the bundled game."
            ) from exc
        # mss >= 10 renamed the entry point to MSS and deprecated the old name.
        self._sct = getattr(mss, "MSS", mss.mss)()

    def grab(self) -> BGRFrame:
        """Return the captured region as a contiguous BGR uint8 array."""
        self._ensure_handle()
        raw = self._sct.grab(self._monitor)  # BGRA
        frame = np.asarray(raw, dtype=np.uint8)
        return np.ascontiguousarray(frame[:, :, :3])

    def close(self) -> None:
        if self._sct is not None:
            self._sct.close()
            self._sct = None

    def __enter__(self) -> ScreenCapture:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


class FrameProcessor:
    """Downscale and (optionally) grayscale a frame for the CNN.

    84x84 grayscale is the DQN-era default: small enough that a conv stack is
    cheap at 10-60 Hz, large enough to keep sprites distinguishable.
    """

    def __init__(self, vision: VisionConfig) -> None:
        self._vision = vision

    def process(self, frame: BGRFrame) -> np.ndarray:
        """Return an (H, W) or (H, W, 3) uint8 array at the configured size."""
        if self._vision.grayscale:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # INTER_AREA is the right filter for shrinking: it averages instead of
        # point-sampling, so thin sprites survive an aggressive downscale.
        return cv2.resize(
            frame,
            (self._vision.width, self._vision.height),
            interpolation=cv2.INTER_AREA,
        )


class FrameStack:
    """Rolling buffer of the last N processed frames, channels-first.

    A single frame is not a Markov state: a ball mid-screen could be moving
    either way. Stacking recovers velocity without giving the policy memory.
    """

    def __init__(self, vision: VisionConfig) -> None:
        self._vision = vision
        self._buffer = np.zeros(vision.observation_shape, dtype=np.uint8)

    @property
    def shape(self) -> tuple[int, int, int]:
        return self._vision.observation_shape

    def reset(self, frame: np.ndarray) -> StackedFrames:
        """Fill every slot with ``frame`` so episodes start from a valid state."""
        planes = self._to_planes(frame)
        self._buffer = np.concatenate([planes] * self._vision.frame_stack, axis=0)
        return self.observation

    def push(self, frame: np.ndarray) -> StackedFrames:
        """Append a frame, dropping the oldest one."""
        planes = self._to_planes(frame)
        self._buffer = np.concatenate([self._buffer[planes.shape[0] :], planes], axis=0)
        return self.observation

    @property
    def observation(self) -> StackedFrames:
        return self._buffer.copy()

    @staticmethod
    def _to_planes(frame: np.ndarray) -> np.ndarray:
        """Normalise a processed frame to channels-first (C, H, W)."""
        if frame.ndim == 2:
            return frame[np.newaxis, ...]
        return np.transpose(frame, (2, 0, 1))


def crop_region(frame: BGRFrame, region: Region) -> BGRFrame:
    """Crop ``region`` from ``frame``, clamped to the frame bounds."""
    height, width = frame.shape[:2]
    top = max(0, min(region.top, height))
    left = max(0, min(region.left, width))
    bottom = max(top, min(region.top + region.height, height))
    right = max(left, min(region.left + region.width, width))
    return frame[top:bottom, left:right]


# ---------------------------------------------------------------------------
# Audio
# ---------------------------------------------------------------------------
@runtime_checkable
class AudioSource(Protocol):
    """Anything that can hand back the most recent mono audio window."""

    def start(self) -> None: ...

    def read_window(self) -> np.ndarray: ...

    def stop(self) -> None: ...


class AudioFeatureExtractor:
    """Log-mel spectrogram with a fixed output shape.

    The shape guarantee matters: ``observation_space`` is declared once at env
    construction, so a short or ragged buffer must never change the array shape.
    """

    def __init__(self, audio: AudioConfig) -> None:
        self._audio = audio
        self._shape = audio.observation_shape

    @property
    def shape(self) -> tuple[int, int]:
        return self._shape

    def extract(self, samples: np.ndarray) -> AudioFeatures:
        """Convert mono float samples into a normalised (n_mels, n_frames) map."""
        cfg = self._audio
        samples = np.asarray(samples, dtype=np.float32).reshape(-1)
        if samples.size < cfg.window_samples:
            samples = np.pad(samples, (cfg.window_samples - samples.size, 0))
        samples = samples[-cfg.window_samples :]

        if not np.any(samples):  # silence: skip the FFT entirely
            return np.zeros(self._shape, dtype=np.float32)

        import librosa

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mel = librosa.feature.melspectrogram(
                y=samples,
                sr=cfg.sample_rate,
                n_fft=cfg.n_fft,
                hop_length=cfg.hop_length,
                n_mels=cfg.n_mels,
                center=False,
            )
            mel_db = librosa.power_to_db(mel, ref=np.max, top_db=cfg.top_db)

        # Map [-top_db, 0] dB onto [0, 1]; networks prefer bounded inputs, and a
        # fixed reference keeps the scale stable across episodes.
        features = (mel_db + cfg.top_db) / cfg.top_db
        features = np.clip(features, 0.0, 1.0).astype(np.float32)
        return self._fit_shape(features)

    def _fit_shape(self, features: np.ndarray) -> AudioFeatures:
        n_mels, n_frames = self._shape
        out = np.zeros(self._shape, dtype=np.float32)
        rows = min(n_mels, features.shape[0])
        cols = min(n_frames, features.shape[1])
        out[:rows, -cols:] = features[:rows, -cols:]
        return out


class LoopbackAudioCapture:
    """Record system output ("what the speakers play") via ``soundcard``.

    Recording runs on a daemon thread into a ring buffer so that
    ``read_window`` never blocks the RL step loop -- a blocking read would stall
    the agent and desynchronise it from the game.
    """

    def __init__(self, audio: AudioConfig, device_name: str | None = None) -> None:
        self._audio = audio
        self._device_name = device_name
        self._buffer = np.zeros(audio.window_samples, dtype=np.float32)
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._error: BaseException | None = None
        self._degraded = False

    def start(self) -> None:
        if self._thread is not None:
            return
        self._resolve_microphone()  # fail fast, on the caller's thread
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._record_loop, name="loopback-audio", daemon=True)
        self._thread.start()

    def _resolve_microphone(self):
        try:
            import soundcard
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise RuntimeError(
                "soundcard is required for audio capture. Install it, or set "
                "audio.enabled = false in the config."
            ) from exc

        if self._device_name:
            return soundcard.get_microphone(self._device_name, include_loopback=True)
        # The loopback "microphone" that shares the default speaker's name is
        # the standard way to tap system output on Windows (WASAPI).
        speaker = soundcard.default_speaker()
        return soundcard.get_microphone(str(speaker.name), include_loopback=True)

    def _record_loop(self) -> None:
        cfg = self._audio
        try:
            mic = self._resolve_microphone()
            with mic.recorder(samplerate=cfg.sample_rate, channels=1, blocksize=cfg.block_size) as rec:
                while not self._stop_event.is_set():
                    chunk = rec.record(numframes=cfg.block_size)
                    mono = np.asarray(chunk, dtype=np.float32).reshape(-1)
                    with self._lock:
                        self._buffer = np.concatenate([self._buffer, mono])[-cfg.window_samples :]
        except BaseException as exc:
            self._error = exc

    def read_window(self) -> np.ndarray:
        """Return the most recent ``window_samples`` of mono audio.

        If the recording thread died (device unplugged, session stolen by an
        exclusive-mode app) this degrades to silence and warns once. Aborting a
        multi-hour training run over a lost audio device would be worse than
        finishing it with a dead input.
        """
        if self._error is not None:
            error, self._error = self._error, None
            self._degraded = True
            warnings.warn(
                f"Loopback audio capture stopped ({error}); returning silence.",
                RuntimeWarning,
                stacklevel=2,
            )
        if self._degraded:
            return np.zeros(self._audio.window_samples, dtype=np.float32)
        with self._lock:
            return self._buffer.copy()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None

    def __enter__(self) -> LoopbackAudioCapture:
        self.start()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.stop()


class SilentAudioCapture:
    """No-op audio source used when audio is disabled or unavailable."""

    def __init__(self, audio: AudioConfig) -> None:
        self._window = np.zeros(audio.window_samples, dtype=np.float32)

    def start(self) -> None:
        return None

    def read_window(self) -> np.ndarray:
        return self._window.copy()

    def stop(self) -> None:
        return None


def build_audio_source(audio: AudioConfig, device_name: str | None = None) -> AudioSource:
    """Return a loopback recorder, or a silent stub when audio is off.

    Falls back to silence (with a warning) instead of crashing when no loopback
    device can be opened -- training should survive a missing sound card.
    """
    if not audio.enabled:
        return SilentAudioCapture(audio)
    source = LoopbackAudioCapture(audio, device_name)
    try:
        source.start()
    except BaseException as exc:
        # Some backends raise bare AssertionErrors with no message (soundcard
        # does exactly that when PulseAudio is missing), so name the type too.
        reason = f"{type(exc).__name__}: {exc}" if str(exc) else type(exc).__name__
        warnings.warn(
            f"Falling back to silent audio ({reason}). Check that a loopback "
            "device exists, or set audio.enabled = false to stop trying.",
            RuntimeWarning,
            stacklevel=2,
        )
        return SilentAudioCapture(audio)
    return source
