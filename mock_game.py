"""Root alias exposing the mock arcade game from ai_player.mock_game."""

import sys

from ai_player.mock_game import (  # noqa: F401
    Hazard,
    MockArcadeGame,
    make_mock_game,
    mock_config,
)

if __name__ == "__main__":
    from ai_player.mock_game import main

    sys.exit(main())
