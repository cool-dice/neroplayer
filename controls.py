"""Root alias exposing game input controls from ai_player.controls."""

from ai_player.controls import (  # noqa: F401
    DirectInputController,
    GameController,
    MockController,
    build_controller,
)
