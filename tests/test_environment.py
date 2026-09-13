from dataclasses import replace

import numpy as np

from config import CONFIG, Region
from environment import WindowsGameEnv
from observer import GameStateObserver


class FakeScreen:
    def __init__(self, config):
        self.config = config
        self.frame = np.zeros((8, 8, 3), dtype=np.uint8)
        self.stack = np.zeros(
            (config.frame_stack, config.image_height, config.image_width),
            dtype=np.uint8,
        )
        self.closed = False

    def capture_raw(self):
        return self.frame.copy()

    def reset_stack(self, _frame):
        return self.stack.copy()

    def update_stack(self, _frame):
        return self.stack.copy()

    def close(self):
        self.closed = True


class FakeAudio:
    def __init__(self, config):
        self.features = np.zeros(config.audio_feature_size, dtype=np.float32)

    def capture_features(self):
        return self.features.copy()


class FakeController:
    action_count = 3

    def __init__(self):
        self.actions = []
        self.restarts = 0
        self.released = False

    def execute(self, action):
        self.actions.append(action)

    def restart(self):
        self.restarts += 1

    def release_all(self):
        self.released = True


def test_environment_reset_step_and_truncation(tmp_path):
    config = replace(
        CONFIG,
        image_width=8,
        image_height=8,
        frame_stack=2,
        target_fps=1_000_000,
        max_episode_steps=1,
        score_region=Region(0, 0, 2, 2),
        game_over_region=Region(2, 2, 2, 2),
        game_over_template=tmp_path / "missing.png",
    )
    screen = FakeScreen(config)
    controller = FakeController()
    env = WindowsGameEnv(
        config,
        screen=screen,
        audio=FakeAudio(config),
        controller=controller,
        observer=GameStateObserver(config),
    )

    observation, reset_info = env.reset(seed=7)
    next_observation, reward, terminated, truncated, info = env.step(2)

    assert env.observation_space.contains(observation)
    assert env.observation_space.contains(next_observation)
    assert reset_info == {"episode_step": 0}
    assert controller.restarts == 1
    assert controller.actions == [2]
    assert reward == config.living_reward
    assert terminated is False
    assert truncated is True
    assert info["episode_step"] == 1

    env.close()
    assert screen.closed is True
    assert controller.released is True
