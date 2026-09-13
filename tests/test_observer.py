from __future__ import annotations

import cv2
import numpy as np
import pytest

from ai_player.config import Region, RewardConfig
from ai_player.mock_game import MockArcadeGame, build_mock_backends, mock_config
from ai_player.observer import (
    ColorGameOverDetector,
    GameObserver,
    PixelChangeScoreSignal,
    TemplateGameOverDetector,
    build_game_over_detector,
    build_score_signal,
)
from ai_player.perception import crop_region


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


def test_colour_detector_ignores_sprites_crossing_the_region(reward):
    """Regression: obstacles drifting through the ROI must not end the episode.

    A mean-colour test fired on 7% of live gameplay frames here, because two
    near-matching channels diluted the one that was far off.
    """
    detector = ColorGameOverDetector(reward)
    game, frames, _audio, controller = build_mock_backends(mock_config(), seed=5)

    alive_detections, banner_detections, banner_frames = 0, 0, 0
    for episode in range(6):
        game.seed(300 + episode)
        for _ in range(120):
            controller.act(2 if game.ticks % 3 else 0)
            fired, _confidence = detector.detect(frames.grab())
            if game.game_over:
                banner_frames += 1
                banner_detections += int(fired)
                controller.restart()
            else:
                alive_detections += int(fired)

    assert alive_detections == 0
    assert banner_frames > 0
    assert banner_detections == banner_frames


def test_colour_detector_confidence_is_banner_coverage(reward):
    detector = ColorGameOverDetector(reward)

    _fired, playing = detector.detect(playing_frame())
    _fired, banner = detector.detect(game_over_frame())

    assert playing < 0.1
    # The banner carries white "GAME OVER" text, so coverage is high, not 1.0.
    assert 0.8 < banner <= 1.0


def test_template_detector_matches_saved_banner(tmp_path, reward):
    banner = crop_region(game_over_frame(), reward.game_over_region)
    template_path = tmp_path / "game_over.png"
    cv2.imwrite(str(template_path), cv2.cvtColor(banner, cv2.COLOR_BGR2GRAY))
    reward.game_over_template = str(template_path)

    detector = TemplateGameOverDetector(template_path, reward)

    assert detector.detect(game_over_frame())[0] is True
    assert detector.detect(playing_frame())[0] is False
    assert isinstance(build_game_over_detector(reward), TemplateGameOverDetector)


def test_template_detector_handles_a_solid_colour_banner(tmp_path, reward):
    """Regression: a flat template has no variance, so correlation is useless.

    ``TM_CCOEFF_NORMED`` scores 0.0 for a perfect match on such a template,
    which would leave the episode unable to ever terminate.
    """
    region = reward.game_over_region
    template_path = tmp_path / "flat.png"
    cv2.imwrite(str(template_path), np.full((region.height, region.width), 90, np.uint8))
    detector = TemplateGameOverDetector(template_path, reward)

    banner = np.zeros_like(game_over_frame())
    banner[:] = 90
    dark = np.zeros_like(banner)

    assert detector.detect(banner) == (True, pytest.approx(1.0, abs=1e-3))
    assert detector.detect(dark)[0] is False


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


def test_pixel_score_signal_catches_single_digit_increments(reward):
    """A 5 -> 6 repaint moves few box pixels but much of the glyph ink."""
    signal = PixelChangeScoreSignal(reward)
    game = MockArcadeGame(seed=4)
    game.score = 5
    signal.update(game.render())

    game.score = 6
    points, _ = signal.update(game.render())

    assert points == 1.0


def test_pixel_score_signal_fires_once_per_scoring_frame(reward):
    """No missed frames and no phantom points against the game's own counter."""
    signal = PixelChangeScoreSignal(reward)
    game = MockArcadeGame(seed=5)
    # Ignore collisions: this measures the detector, not a dodging policy.
    game._check_collision = lambda: None
    signal.update(game.render())

    scoring_frames = detections = mismatches = 0
    previous = game.score
    for _ in range(150):
        game.tick()
        points, _ = signal.update(game.render())
        scored = game.score > previous
        previous = game.score
        scoring_frames += int(scored)
        detections += int(points)
        mismatches += int(bool(points) != scored)

    assert scoring_frames > 0
    assert detections == scoring_frames
    assert mismatches == 0


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
