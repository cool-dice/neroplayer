"""AI player: a vision + audio driven RL agent for windowed games."""

from __future__ import annotations

from .config import AppConfig, load_config
from .environment import GameEnv, make_env
from .observer import GameObserver

__all__ = ["AppConfig", "GameEnv", "GameObserver", "load_config", "make_env"]
__version__ = "0.1.0"
