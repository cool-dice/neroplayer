from __future__ import annotations

import json
import time

import cv2
import numpy as np
import pytest

from ai_player.cognitive import CognitiveState, CognitiveSupervisor
from ai_player.config import HUDConfig, Region, RewardConfig
from ai_player.mock_game import MockArcadeGame, build_mock_backends, mock_config
from ai_player.observer import (
    AutonomousGameOverDetector,
    ColorGameOverDetector,
    GameObserver,
    PixelChangeScoreSignal,
    TemplateGameOverDetector,
    VLMObserverInterface,
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
    reward.game_over_mode = "color"
    reward.game_over_template = "assets/does-not-exist.png"

    detector = build_game_over_detector(reward)

    assert isinstance(detector, ColorGameOverDetector)


def test_missing_template_in_auto_mode_falls_back_to_autonomous(reward):
    reward.game_over_mode = "auto"
    reward.game_over_template = "assets/does-not-exist.png"

    with pytest.warns(RuntimeWarning, match="falling back to autonomous game-over detector"):
        detector = build_game_over_detector(reward)

    assert isinstance(detector, AutonomousGameOverDetector)


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


def test_pixel_score_signal_handles_a_dark_on_light_hud(reward):
    """Inverting the HUD must not change what counts as glyph ink."""

    def inverted(game: MockArcadeGame) -> np.ndarray:
        frame = game.render()
        box = reward.score_region
        patch = frame[box.top : box.top + box.height, box.left : box.left + box.width]
        frame[box.top : box.top + box.height, box.left : box.left + box.width] = 255 - patch
        return frame

    signal = PixelChangeScoreSignal(reward)
    game = MockArcadeGame(seed=4)
    game.score = 5
    signal.update(inverted(game))

    game.score = 6
    points, _ = signal.update(inverted(game))

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


def test_observer_detects_idle_and_applies_penalty(reward):
    reward.idle_diff_threshold = 0.05
    reward.idle_penalty = -0.5
    reward.movement_reward = 0.2
    observer = GameObserver(reward, score_signal=PixelChangeScoreSignal(reward))
    observer.reset()

    # Frame 1 primes the baseline; diff is 1.0 (moving)
    f1 = np.full((100, 100, 3), 120, dtype=np.uint8)
    v1 = observer.evaluate(f1)
    assert v1.is_idle is False
    assert v1.reward == pytest.approx(reward.step_reward + reward.movement_reward)

    # Frame 2 is identical -> diff is 0.0 -> idle detected -> penalty applied
    f2 = f1.copy()
    v2 = observer.evaluate(f2)
    assert v2.is_idle is True
    assert v2.frame_diff < 0.05
    assert v2.reward == pytest.approx(reward.step_reward + reward.idle_penalty)

    # Frame 3 changes significantly -> moving again -> movement reward applied
    f3 = np.full((100, 100, 3), 200, dtype=np.uint8)
    v3 = observer.evaluate(f3)
    assert v3.is_idle is False
    assert v3.frame_diff > 0.05
    assert v3.reward == pytest.approx(reward.step_reward + reward.movement_reward)


def test_autonomous_game_over_detector_detects_game_over_texts(reward):
    detector = AutonomousGameOverDetector(reward)

    for color in [(255, 255, 255), (0, 0, 255), (0, 255, 255)]:
        frame = np.random.randint(15, 50, (240, 320, 3), dtype=np.uint8)
        cv2.putText(frame, "GAME OVER", (45, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        detected, conf = detector.detect(frame)
        assert detected is True, f"Failed for color {color}"
        assert conf >= reward.game_over_threshold


def test_autonomous_game_over_detector_detects_continue_screen(reward):
    detector = AutonomousGameOverDetector(reward)
    frame = np.random.randint(10, 40, (240, 320, 3), dtype=np.uint8)
    cv2.putText(frame, "CONTINUE", (55, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    detected, conf = detector.detect(frame)
    assert detected is True
    assert conf >= reward.game_over_threshold


def test_autonomous_game_over_detector_ignores_normal_play_and_noise(reward):
    # Set mock color away from random noise so color fallback doesn't trigger
    reward.game_over_color_bgr = (255, 0, 128)
    detector = AutonomousGameOverDetector(reward)

    # Normal gameplay frames
    for seed in range(5):
        game = MockArcadeGame(seed=seed)
        play_frame = game.render()
        detected, conf = detector.detect(play_frame)
        assert detected is False
        assert conf < reward.game_over_threshold

    # Pure random noise frames
    rng = np.random.RandomState(42)
    for _ in range(5):
        noise_frame = rng.randint(0, 256, (240, 320, 3), dtype=np.uint8)
        detected, conf = detector.detect(noise_frame)
        assert detected is False
        assert conf < reward.game_over_threshold


def test_autonomous_game_over_detector_detects_dimming(reward):
    # Set mock color away
    reward.game_over_color_bgr = (255, 0, 128)
    detector = AutonomousGameOverDetector(reward)

    black_frame = np.zeros((240, 320, 3), dtype=np.uint8)
    # First frame initializes diff
    detector.detect(black_frame)
    # Second stationary black frame triggers blackout dimming
    detected, conf = detector.detect(black_frame)
    assert detected is True
    assert conf >= reward.game_over_threshold


def test_build_game_over_detector_modes(reward):
    # Auto without template -> AutonomousGameOverDetector
    reward.game_over_mode = "auto"
    reward.game_over_template = None
    assert isinstance(build_game_over_detector(reward), AutonomousGameOverDetector)

    # Color mode -> ColorGameOverDetector
    reward.game_over_mode = "color"
    assert isinstance(build_game_over_detector(reward), ColorGameOverDetector)

    # Text mode -> AutonomousGameOverDetector
    reward.game_over_mode = "text"
    assert isinstance(build_game_over_detector(reward), AutonomousGameOverDetector)

    # Template mode without template raises ValueError
    reward.game_over_mode = "template"
    reward.game_over_template = None
    with pytest.raises(ValueError, match="game_over_template is not specified"):
        build_game_over_detector(reward)


def test_vlm_observer_supports_openai_chat_completions(monkeypatch):
    from ai_player.config import HUDConfig

    hud_cfg = HUDConfig(
        use_vlm=True,
        vlm_endpoint="http://localhost:1234/v1/chat/completions",
        vlm_model="qwen/qwen3-vl-8b",
    )
    vlm = VLMObserverInterface(hud_cfg)

    captured_payload = {}

    class MockResponse:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def read(self):
            content_str = (
                '```json\n{"score": [10, 10, 100, 30], "lives": [200, 10, 60, 30], '
                '"game_over": [50, 80, 200, 60]}\n```'
            )
            res = {"choices": [{"message": {"content": content_str}}]}
            return json.dumps(res).encode("utf-8")

    def mock_urlopen(req, timeout=None):
        nonlocal captured_payload
        captured_payload = json.loads(req.data.decode("utf-8"))
        return MockResponse()

    monkeypatch.setattr("urllib.request.urlopen", mock_urlopen)

    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    res = vlm.detect_hud(frame)

    assert "messages" in captured_payload
    assert captured_payload["messages"][0]["role"] == "user"
    assert res is not None
    assert res.score_region == Region(left=10, top=10, width=100, height=30)
    assert res.lives_region == Region(left=200, top=10, width=60, height=30)
    assert res.game_over_region == Region(left=50, top=80, width=200, height=60)


# ---------------------------------------------------------------------------
# Cognitive supervisor (VLM sentinel) fusion
# ---------------------------------------------------------------------------


class ScriptedSupervisor(CognitiveSupervisor):
    def __init__(self, cstate: CognitiveState | None = None, config: HUDConfig | None = None):
        super().__init__(config)
        self.state = cstate or CognitiveState()

    @property
    def current_state(self) -> CognitiveState:
        return self.state


def _fresh(**kwargs) -> CognitiveState:
    kwargs.setdefault("valid", True)
    kwargs.setdefault("timestamp", time.time())
    return CognitiveState(**kwargs)


def _blank_frame(value: int = 128) -> np.ndarray:
    return np.full((240, 320, 3), value, dtype=np.uint8)


def test_autonomous_game_over_detector_incorporates_supervisor(reward):
    sup = ScriptedSupervisor(_fresh(state="gameplay", confidence=0.0))
    detector = AutonomousGameOverDetector(reward, supervisor=sup)
    frame = _blank_frame()
    assert detector.detect(frame)[0] is False

    sup.state = _fresh(state="game_over", is_game_over=True, confidence=0.95, suggested_action="press_space")
    detected, conf = detector.detect(frame)
    assert detected is True
    assert conf >= 0.95


def test_autonomous_game_over_detector_vlm_instant_trigger_uses_defeat_label(reward):
    sup = ScriptedSupervisor(_fresh(state="defeat", confidence=0.88))
    detector = AutonomousGameOverDetector(reward, supervisor=sup)
    detected, conf = detector.detect(_blank_frame(100))
    assert detected is True
    assert conf >= 0.95


def test_autonomous_game_over_detector_ignores_stale_and_placeholder_verdicts(reward):
    frame = _blank_frame()
    stale = _fresh(state="game_over", is_game_over=True, confidence=0.99, timestamp=time.time() - 100.0)
    detector = AutonomousGameOverDetector(reward, supervisor=ScriptedSupervisor(stale))
    assert detector.detect(frame)[0] is False

    placeholder = CognitiveState(state="game_over", is_game_over=True, confidence=0.99)
    detector = AutonomousGameOverDetector(reward, supervisor=ScriptedSupervisor(placeholder))
    assert detector.detect(frame)[0] is False


def test_vlm_game_over_threshold_is_taken_from_config(reward):
    cfg = HUDConfig(vlm_game_over_confidence=0.9)
    sup = ScriptedSupervisor(_fresh(state="game_over", is_game_over=True, confidence=0.8), config=cfg)
    detector = AutonomousGameOverDetector(reward, supervisor=sup)
    # Below the instant threshold the verdict joins the debounced pool with its
    # own confidence (not an inflated one) and is judged by game_over_threshold.
    detected, conf = detector.detect(_blank_frame())
    assert conf == pytest.approx(0.8)
    assert detected is (reward.game_over_threshold <= 0.8)

    sup.state = _fresh(state="game_over", is_game_over=True, confidence=0.95)
    assert detector.detect(_blank_frame()) == (True, pytest.approx(0.95))

    # The observer only bypasses detection_patience above the configured threshold.
    reward.detection_patience = 2
    observer = GameObserver(reward, supervisor=sup)
    sup.state = _fresh(state="game_over", is_game_over=True, confidence=0.8)
    assert observer.evaluate(_blank_frame()).terminated is False
    observer.reset()
    sup.state = _fresh(state="game_over", is_game_over=True, confidence=0.95)
    assert observer.evaluate(_blank_frame()).terminated is True


def test_game_observer_passes_supervisor_telemetry(reward):
    sup = ScriptedSupervisor(
        _fresh(state="game_over", is_game_over=True, confidence=0.99, description="Player defeated")
    )
    observer = GameObserver(reward, supervisor=sup)
    verdict = observer.evaluate(_blank_frame())
    assert verdict.terminated is True
    assert verdict.game_over_confidence >= 0.95
    assert verdict.info["vlm_state"] == "game_over"
    assert verdict.info["vlm_is_game_over"] == 1.0
    assert verdict.info["vlm_fresh"] == 1.0

    sup.state = _fresh(state="gameplay", confidence=0.7, timestamp=time.time() - 100.0)
    observer.reset()
    verdict = observer.evaluate(_blank_frame())
    assert verdict.terminated is False
    assert verdict.info["vlm_fresh"] == 0.0


def test_game_observer_stale_verdict_does_not_bypass_patience(reward):
    reward.detection_patience = 3
    stale = _fresh(state="game_over", is_game_over=True, confidence=0.99, timestamp=time.time() - 100.0)
    observer = GameObserver(reward, supervisor=ScriptedSupervisor(stale))
    frame = game_over_frame()
    verdicts = [observer.evaluate(frame) for _ in range(3)]
    assert [v.terminated for v in verdicts] == [False, False, True]


def test_game_observer_dynamic_hud_update(reward):
    observer = GameObserver(reward)
    assert observer._reward.score_region.left == reward.score_region.left

    changed = observer.update_dynamic_hud({"score": [50, 40, 150, 35], "game_over": [100, 120, 300, 100]})
    assert changed is True
    assert observer._reward.score_region == Region(left=50, top=40, width=150, height=35)
    assert observer._reward.game_over_region.left == 100
    assert observer._reward.game_over_region.top == 120


def test_game_observer_dynamic_hud_update_is_idempotent(reward, monkeypatch):
    import ai_player.observer as observer_mod

    calls = {"score": 0, "game_over": 0}
    real_score, real_go = observer_mod.build_score_signal, observer_mod.build_game_over_detector

    def counting_score(*args, **kwargs):
        calls["score"] += 1
        return real_score(*args, **kwargs)

    def counting_go(*args, **kwargs):
        calls["game_over"] += 1
        return real_go(*args, **kwargs)

    monkeypatch.setattr(observer_mod, "build_score_signal", counting_score)
    monkeypatch.setattr(observer_mod, "build_game_over_detector", counting_go)

    sup = ScriptedSupervisor()
    observer = GameObserver(reward, supervisor=sup)
    calls["score"] = calls["game_over"] = 0

    layout = {"score": [50, 40, 150, 35], "game_over": [100, 120, 300, 100]}
    results = [observer.update_dynamic_hud(layout) for _ in range(5)]
    assert results == [True, False, False, False, False]
    assert calls == {"score": 1, "game_over": 1}
    assert sup.dynamic_hud_layout["score"] == [50, 40, 150, 35]

    assert observer.update_dynamic_hud({"score": [60, 40, 150, 35]}) is True
    assert calls == {"score": 2, "game_over": 1}
    assert observer.update_dynamic_hud({}) is False
    assert observer.update_dynamic_hud({"score": [0, 0, 0, 0]}) is False


def test_game_observer_forget_hud_clears_auto_detected_state(reward):
    from ai_player.config import HUDConfig

    observer = GameObserver(reward, hud_config=HUDConfig(auto_detect=True))
    observer._detected_hud = object()  # type: ignore[assignment]
    observer._applied_hud_boxes["score"] = [1, 2, 3, 4]
    observer._reward.game_over_template = "stale.png"
    observer.forget_hud()
    assert observer.detected_hud is None
    assert observer._applied_hud_boxes == {}
    assert observer._reward.game_over_template is None


def test_edge_keywords_follow_configured_game_over_keywords(reward):
    reward.game_over_keywords = ["YOU DIED"]
    detector = AutonomousGameOverDetector(reward)
    assert "CONTINUE" not in detector._edge_keywords
    assert not any("CONTINUE" in kw for kw in detector._edge_keywords)

    reward.game_over_keywords = ["RETRY", "GAME OVER"]
    detector = AutonomousGameOverDetector(reward)
    assert "RETRY 9" in detector._edge_keywords
    assert "RETRY?" in detector._edge_keywords
    assert "CONTINUE 9" not in detector._edge_keywords


def test_vlm_observer_rescales_boxes_from_downscaled_image(monkeypatch):
    hud_cfg = HUDConfig(
        use_vlm=True,
        vlm_endpoint="http://localhost:11434/api/generate",
        vlm_max_image_dim=640,
    )
    vlm = VLMObserverInterface(hud_cfg)
    sent = {}

    class MockResponse:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def read(self):
            body = {"score": [10, 10, 100, 30], "lives": [200, 10, 60, 30], "game_over": [50, 80, 200, 60]}
            return json.dumps({"response": json.dumps(body)}).encode("utf-8")

    def mock_urlopen(req, timeout=None):
        sent["payload"] = json.loads(req.data.decode("utf-8"))
        return MockResponse()

    monkeypatch.setattr("urllib.request.urlopen", mock_urlopen)

    res = vlm.detect_hud(np.zeros((720, 1280, 3), dtype=np.uint8))
    assert "640x360" in sent["payload"]["prompt"]
    assert res is not None
    assert res.score_region == Region(left=20, top=20, width=200, height=60)
    assert res.lives_region == Region(left=400, top=20, width=120, height=60)
    assert res.game_over_region == Region(left=100, top=160, width=400, height=120)
