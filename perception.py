"""Root alias exposing perception sensors from ai_player.perception."""

from ai_player.perception import (  # noqa: F401
    AudioCapture,
    FrameSource,
    MSSScreenCapture,
    ScreenCapture,
    SyntheticAudioCapture,
    SyntheticScreenCapture,
    build_audio_capture,
    build_screen_capture,
)
