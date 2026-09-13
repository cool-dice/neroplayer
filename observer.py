"""Root alias exposing game observer and critic rewards from ai_player.observer."""

from ai_player.observer import (  # noqa: F401
    GameObserver,
    ObservationFeedback,
    build_observer,
)
