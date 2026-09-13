from __future__ import annotations

import cv2
import numpy as np
import pytest

from ai_player.config import Region, RewardConfig
from ai_player.mock_game import MockArcadeGame, mock_config
from ai_player.observer import (
    ColorGameOverDetector,
    GameObserver,
    PixelChangeScoreSignal,
    TemplateGameOverDetector,
    build_game_over_detector,
    build_score_signal,
)


@pytest.fixture
def reward() -> RewardConfig:
    return mock_config().reward


def playing_frame(game: MockArcadeGame | None = None) -> np.ndarray:
    game = game or MockArcadeGame(seed=1)
    return game.render()


def game_over_frame() -> np.ndarray:
    game = MockArcadeGame(seed=1)
    game.game_over = True
    return game.render()


def test_colour_detector_separates_play_from_game_over(reward):
    detector = ColorGameOverDetector(reward)

    playing, play_conf = detector.detect(playing_frame())
    over, over_conf = detector.detect(game_over_frame())

    assert playing is False
    assert over is True
    assert over_conf > play_conf


def test_template_detector_matches_saved_banner(tmp_path, reward):
    banner = reward.game_over_region.crop(game_over_frame())
    template_path = tmp_path / "game_over.png"
    cv2.imwrite(str(template_path), cv2.cvtColor(banner, cv2.COLOR_BGR2GRAY))
    reward.game_over_template = str(template_path)

    detector = TemplateGameOverDetector(template_path, reward)

    assert detector.detect(game_over_frame())[0] is True
    assert detector.detect(playing_frame())[0] is False
    assert isinstance(build_game_over_detector(reward), TemplateGameOverDetector)


def test_missing_template_falls_back_to_colour(reward):
    reward.game_over_template = "assets/does-not-exist.png"

    with pytest.warns(RuntimeWarning):
        detector = build_game_over_detector(reward)

    assert isinstance(detector, ColorGameOverDetector)


def test_pixel_score_signal_fires_only_when_the_box_changes(reward):
    signal = PixelChangeScoreSignal(reward)
    game = MockArcadeGame(seed=2)

    assert signal.update(game.render()) == (0.0, None)  # first frame primes it
    assert signal.update(game.render())[0] == 0.0

    game.score = 7
    points, _ = signal.update(game.render())

    assert points == 1.0


def test_build_score_signal_rejects_unknown_mode(reward):
    reward.score_mode = "telepathy"

    with pytest.raises(ValueError, match="Unknown score_mode"):
        build_score_signal(reward)


def test_observer_rewards_survival_and_scoring(reward):
    observer = GameObserver(reward, score_signal=PixelChangeScoreSignal(reward))
    game = MockArcadeGame(seed=3)

    observer.reset()
    first = observer.evaluate(game.render())
    assert first.reward == pytest.approx(reward.step_reward)
    assert first.terminated is False

    game.score += 1
    scored = observer.evaluate(game.render())

    assert scored.reward == pytest.approx(reward.step_reward + reward.score_reward)
    assert observer.episode_points == 1.0


def test_observer_debounces_game_over_before_terminating(reward):
    reward.detection_patience = 2
    observer = GameObserver(reward, score_signal=PixelChangeScoreSignal(reward))
    observer.reset()
    frame = game_over_frame()

    first = observer.evaluate(frame)
    second = observer.evaluate(frame)

    assert first.terminated is False
    assert second.terminated is True
    assert second.reward == pytest.approx(reward.game_over_penalty)


def test_observer_streak_resets_on_a_playing_frame(reward):
    reward.detection_patience = 2
    observer = GameObserver(reward, score_signal=PixelChangeScoreSignal(reward))
    observer.reset()

    observer.evaluate(game_over_frame())
    observer.evaluate(playing_frame())
    verdict = observer.evaluate(game_over_frame())

    assert verdict.terminated is False


def test_observer_handles_an_empty_detection_region(reward):
    reward.game_over_region = Region(left=10_000, top=10_000, width=10, height=10)
    observer = GameObserver(reward, score_signal=PixelChangeScoreSignal(reward))
    observer.reset()

    verdict = observer.evaluate(playing_frame())

    assert verdict.terminated is False
