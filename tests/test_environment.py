from __future__ import annotations

import json
import time

import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from ai_player.cognitive import CognitiveState, CognitiveSupervisor
from ai_player.config import AppConfig, HUDConfig
from ai_player.controls import NullController
from ai_player.environment import GameEnv, make_env
from ai_player.mock_game import build_mock_backends, mock_config


@pytest.fixture
def env() -> GameEnv:
    environment = make_env(mock=True, seed=7)
    yield environment
    environment.close()


def test_env_passes_the_gymnasium_api_checker(env):
    # Skips render-mode checks because the mock env has no display backend.
    check_env(env, skip_render_check=True)


def test_observation_matches_the_declared_space(env):
    obs, info = env.reset(seed=1)

    assert env.observation_space.contains(obs)
    assert obs["frames"].shape == env.config.vision.observation_shape
    assert obs["audio"].shape == env.config.audio.observation_shape
    assert info["episode_index"] == 1


def test_step_returns_the_five_tuple_and_info(env):
    env.reset(seed=1)

    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())

    assert env.observation_space.contains(obs)
    assert isinstance(reward, float)
    assert terminated is False and truncated is False
    assert {"score", "points", "episode_steps", "episode_reward"} <= set(info)


def test_audio_key_disappears_when_audio_is_disabled():
    config = AppConfig()
    config.audio.enabled = False
    env = make_env(config, mock=True, seed=0)
    try:
        obs, _ = env.reset()
        assert set(obs) == {"frames"}
        assert "audio" not in env.observation_space.spaces
    finally:
        env.close()


def test_crashing_terminates_with_the_penalty(env):
    env.reset(seed=2)
    reward = 0.0
    terminated = False
    # Action 2 is the no-op, so the agent never dodges and always crashes.
    for _ in range(env.config.env.max_episode_steps):
        _obs, reward, terminated, _truncated, _info = env.step(2)
        if terminated:
            break

    assert terminated is True
    assert reward == pytest.approx(env.config.reward.game_over_penalty)


def test_reset_clears_the_game_over_screen(env):
    env.reset(seed=3)
    for _ in range(200):
        if env.step(2)[2]:
            break

    obs, info = env.reset()

    assert env.observation_space.contains(obs)
    assert info["episode_index"] == 2
    # A fresh episode must not immediately report termination.
    assert env.step(0)[2] is False


def test_episode_truncates_at_the_step_limit():
    config = mock_config()
    config.env.max_episode_steps = 5
    game, frames, audio, controller = build_mock_backends(config, seed=4)
    # A game that never ends isolates truncation from termination.
    game._check_collision = lambda: None
    env = GameEnv(config, frame_source=frames, audio_source=audio, controller=controller)
    try:
        env.reset()
        truncated = False
        for _ in range(5):
            _obs, _reward, terminated, truncated, _info = env.step(2)
        assert terminated is False
        assert truncated is True
    finally:
        env.close()


def test_actions_reach_the_controller():
    config = mock_config()
    controller = NullController(config.control)
    _game, frames, audio, _mock_controller = build_mock_backends(config, seed=5)
    env = GameEnv(config, frame_source=frames, audio_source=audio, controller=controller)
    try:
        env.reset()
        env.step(0)
        env.step(1)
    finally:
        env.close()

    assert controller.history == [0, 1]
    assert controller.restarts == 1


def test_frame_stack_encodes_motion(env):
    obs, _ = env.reset(seed=6)
    assert np.all(obs["frames"][0] == obs["frames"][-1])

    for _ in range(3):
        obs, *_ = env.step(2)

    assert not np.all(obs["frames"][0] == obs["frames"][-1])


def test_render_returns_rgb_frames():
    env = make_env(mock=True, render_mode="rgb_array", seed=8)
    try:
        env.reset()
        env.step(0)
        frame = env.render()
        hud = env.render_hud()
    finally:
        env.close()

    region = env.config.capture.region
    assert frame.shape == (region.height, region.width, 3)
    assert hud.shape == (region.height, region.width, 3)


def test_env_tracks_real_fps_and_renders_hud_details():
    config = mock_config()
    config.control.action_keys = ["up", "down", "up+down", None]
    env = make_env(config, mock=True, render_mode="rgb_array", seed=10)
    try:
        env.reset()
        for action in range(4):
            _obs, _rew, _term, _trunc, info = env.step(action)
            assert "action" in info
            assert info["action"] == action
            assert env.last_info["real_fps"] >= 0.0

        hud = env.render_hud()
        assert hud is not None
        assert hud.shape == (config.capture.region.height, config.capture.region.width, 3)
    finally:
        env.close()


def test_env_tracks_latency_profiler():
    config = mock_config()
    env = make_env(config, mock=True, render_mode="rgb_array", seed=11)
    try:
        env.reset()
        _obs, _rew, _term, _trunc, _info = env.step(0)
        assert "latency_ms" in env.last_info
        lat = env.last_info["latency_ms"]
        assert "grab" in lat
        assert "act" in lat
        assert "proc" in lat
        assert "obs" in lat
        assert "render" in lat
        assert isinstance(lat["grab"], float)
        assert isinstance(lat["act"], float)
    finally:
        env.close()


def test_env_exposes_player_bbox_and_entities():
    config = mock_config()
    config.tracker.enabled = True
    config.tracker.auto_probe = True
    env = make_env(config, mock=True, render_mode="rgb_array", seed=12)
    try:
        _obs, _info = env.reset()
        _obs, _rew, _term, _trunc, step_info = env.step(0)
        assert "entities" in step_info
        assert "collisions" in step_info
        assert isinstance(step_info["entities"], list)
        assert isinstance(step_info["collisions"], list)
        hud = env.render_hud()
        assert hud is not None
    finally:
        env.close()


def test_env_renders_rgb_hud_inset():
    config = mock_config()
    config.vision.grayscale = False
    config.vision.width = 128
    config.vision.height = 128
    env = make_env(config, mock=True, render_mode="rgb_array", seed=13)
    try:
        env.reset()
        env.step(0)
        hud = env.render_hud()
        assert hud is not None
        assert hud.shape == (config.capture.region.height, config.capture.region.width, 3)
    finally:
        env.close()


def test_env_with_mouse_actions_in_telemetry_and_step():
    config = mock_config()
    config.control.mouse_enabled = True
    config.control.action_keys = [
        None,
        "up",
        "mouse_left",
        "aim_right",
        "up+mouse_left",
    ]
    env = make_env(config, mock=True, render_mode="rgb_array", seed=14)
    try:
        env.reset()
        # Step with mouse click action
        _obs, _rew, _term, _trunc, info_click = env.step(2)
        assert "mouse" in info_click
        assert info_click["mouse"]["clicks"] == ["left"]

        # Step with mouse aim action
        _obs, _rew, _term, _trunc, info_aim = env.step(3)
        assert info_aim["mouse"]["aim"] == (config.control.aim_step, 0)

        # Step with hybrid key + mouse
        _obs, _rew, _term, _trunc, info_hybrid = env.step(4)
        assert info_hybrid["mouse"]["clicks"] == ["left"]

        # Render HUD should succeed without error and include telemetry
        hud = env.render_hud()
        assert hud is not None
        assert hud.shape == (config.capture.region.height, config.capture.region.width, 3)
    finally:
        env.close()


# ---------------------------------------------------------------------------
# Cognitive supervisor / VLM sentinel integration
# ---------------------------------------------------------------------------


class ScriptedSupervisor(CognitiveSupervisor):
    """Supervisor whose verdict is set by the test instead of a VLM."""

    def __init__(self, cstate: CognitiveState | None = None):
        super().__init__()
        self.state = cstate or CognitiveState()

    @property
    def current_state(self) -> CognitiveState:
        return self.state


def _fresh(**kwargs) -> CognitiveState:
    kwargs.setdefault("valid", True)
    kwargs.setdefault("timestamp", time.time())
    return CognitiveState(**kwargs)


def _sentinel_env(config, supervisor, *, seed: int, render_mode=None):
    controller = NullController(config.control)
    _game, frames, audio, _mock_controller = build_mock_backends(config, seed=seed)
    env = GameEnv(
        config,
        frame_source=frames,
        audio_source=audio,
        controller=controller,
        supervisor=supervisor,
        render_mode=render_mode,
    )
    return env, controller


def test_env_auto_menu_navigation_and_vlm_telemetry():
    config = mock_config()
    config.hud.auto_menu_nav = True
    sup = ScriptedSupervisor(
        _fresh(
            state="menu",
            confidence=0.95,
            suggested_action="press_start",
            description="Title menu screen",
        )
    )
    env, controller = _sentinel_env(config, sup, seed=15, render_mode="rgb_array")
    try:
        env.reset()
        restarts_before = controller.restarts
        _obs, _rew, _term, _trunc, step_info = env.step(0)

        assert controller.restarts == restarts_before + 1
        assert step_info["vlm_state"] == "menu"
        assert step_info["vlm_goal"] == "press_start"
        assert step_info["vlm_desc"] == "Title menu screen"
        assert step_info["vlm_fresh"] is True
        assert step_info["menu_nav_taps"] == 1
        assert env.render_hud() is not None
    finally:
        env.close()


def test_env_menu_nav_taps_once_per_verdict_and_respects_cooldown():
    config = mock_config()
    config.hud.auto_menu_nav = True
    config.hud.menu_nav_cooldown = 0.0
    verdict_ts = time.time()
    sup = ScriptedSupervisor(_fresh(state="menu", confidence=0.9, timestamp=verdict_ts))
    env, controller = _sentinel_env(config, sup, seed=18)
    try:
        env.reset()
        base = controller.restarts
        for _ in range(10):
            env.step(0)
        # Same verdict -> a single tap even with no cooldown.
        assert controller.restarts == base + 1

        # A newer verdict with the cooldown still running -> no tap.
        config.hud.menu_nav_cooldown = 60.0
        sup.state = _fresh(state="menu", confidence=0.9, timestamp=verdict_ts + 0.5)
        env.step(0)
        assert controller.restarts == base + 1

        # Cooldown elapsed and verdict is newer -> second tap.
        config.hud.menu_nav_cooldown = 0.0
        env.step(0)
        assert controller.restarts == base + 2
        assert env.last_info["menu_nav_taps"] == 2
    finally:
        env.close()


def test_env_menu_nav_is_opt_in():
    config = mock_config()
    assert config.hud.auto_menu_nav is False
    sup = ScriptedSupervisor(_fresh(state="menu", confidence=0.99))
    env, controller = _sentinel_env(config, sup, seed=19)
    try:
        env.reset()
        base = controller.restarts
        env.step(0)
        assert controller.restarts == base
    finally:
        env.close()


def test_env_cognitive_obs_vector_and_dynamic_hud():
    config = mock_config()
    config.hud.cognitive_obs = True
    config.hud.cognitive_dim = 8
    sup = ScriptedSupervisor(
        _fresh(
            state="gameplay",
            confidence=0.92,
            suggested_action="up",
            description="Active play",
            hud_layout={
                "score": [25, 20, 110, 30],
                "hp": [25, 60, 90, 20],
                "ammo": [25, 85, 70, 15],
            },
            vital_stats={"hp_ratio": 0.75, "ammo_ratio": 0.40, "lives": 3.0, "danger_level": 0.15},
        )
    )
    env, _controller = _sentinel_env(config, sup, seed=16, render_mode="rgb_array")
    try:
        assert "cognitive" in env.observation_space.spaces
        obs, _info = env.reset()
        cog_vec = obs["cognitive"]
        assert cog_vec.shape == (8,)
        assert cog_vec[0] == 0.0
        assert cog_vec[1] == pytest.approx(0.75)
        assert cog_vec[2] == pytest.approx(0.40)
        assert cog_vec[3] == pytest.approx(3.0 / config.hud.cognitive_max_lives)
        assert cog_vec[4] == pytest.approx(0.15)
        assert cog_vec[5] > 0.0
        assert cog_vec[6] == pytest.approx(0.92)
        assert cog_vec[7] == 0.0

        _obs, _rew, _term, _trunc, _step_info = env.step(0)
        hud = env.render_hud()
        assert hud.shape == (config.capture.region.height, config.capture.region.width, 3)
        assert env._observer._reward.score_region.left == 25
        assert env._observer._reward.score_region.top == 20
    finally:
        env.close()


def test_env_cognitive_vector_lives_are_monotonic_and_actions_match_whole_tokens():
    config = mock_config()
    config.hud.cognitive_obs = True
    config.hud.cognitive_max_lives = 5.0
    config.control.action_keys = [None, "a", "up"]
    config.control.auto_combos = False
    sup = ScriptedSupervisor()
    env, _controller = _sentinel_env(config, sup, seed=20)
    try:
        env.reset()
        ratios = []
        for lives in (1.0, 2.0, 5.0):
            sup.state = _fresh(state="gameplay", vital_stats={"lives": lives})
            ratios.append(float(env._build_cognitive_vector()[3]))
        assert ratios == pytest.approx([0.2, 0.4, 1.0])

        # "press_start" contains the letter "a" but does not name the key "a".
        sup.state = _fresh(state="gameplay", suggested_action="press_start")
        assert env._build_cognitive_vector()[5] == pytest.approx(0.5)
        sup.state = _fresh(state="gameplay", suggested_action="move up")
        assert env._build_cognitive_vector()[5] == pytest.approx(3 / 3)
        sup.state = _fresh(state="gameplay", suggested_action=None)
        assert env._build_cognitive_vector()[5] == 0.0
    finally:
        env.close()


def test_env_ignores_stale_or_placeholder_verdicts():
    config = mock_config()
    config.hud.auto_menu_nav = True
    config.hud.cognitive_obs = True
    stale = _fresh(
        state="game_over",
        is_game_over=True,
        confidence=0.99,
        vital_stats={"hp_ratio": 0.1, "lives": 1.0},
        timestamp=time.time() - 100.0,
    )
    sup = ScriptedSupervisor(stale)
    env, controller = _sentinel_env(config, sup, seed=21, render_mode="rgb_array")
    try:
        obs, _info = env.reset()
        assert obs["cognitive"].tolist() == pytest.approx(list(GameEnv._COGNITIVE_DEFAULT))

        base = controller.restarts
        sup.state = _fresh(state="menu", confidence=0.99, timestamp=time.time() - 100.0)
        _obs, _rew, term, _trunc, info = env.step(0)
        assert term is False
        assert controller.restarts == base
        assert info["vlm_fresh"] is False

        sup.state = CognitiveState(state="game_over", is_game_over=True, confidence=0.99)  # valid=False
        _obs, _rew, term, _trunc, info = env.step(0)
        assert term is False
        assert info["vlm_fresh"] is False
        assert env.render_hud() is not None
    finally:
        env.close()


def test_env_instant_vlm_game_over_termination():
    config = mock_config()
    assert config.reward.detection_patience >= 2
    sup = ScriptedSupervisor()
    env, _controller = _sentinel_env(config, sup, seed=17)
    try:
        env.reset()
        sup.state = _fresh(state="game_over", is_game_over=True, confidence=0.95, description="dead")
        _obs, rew, term, _trunc, info = env.step(0)
        assert term is True
        assert rew == pytest.approx(config.reward.game_over_penalty)
        assert info["vlm_is_game_over"] is True

        # Below the configured confidence the heuristic detector's patience still applies.
        env.reset()
        sup.state = _fresh(state="game_over", is_game_over=True, confidence=0.5)
        _obs, _rew, term, _trunc, _info = env.step(0)
        assert term is False
    finally:
        env.close()


class _FrameAwareVLM:
    """Fake endpoint whose verdict depends on the frame the sentinel currently holds."""

    def __init__(self) -> None:
        self.supervisor: CognitiveSupervisor | None = None
        self.terminal_frame: np.ndarray | None = None
        self.calls = 0

    def __call__(self, req, timeout=None):
        self.calls += 1
        with self.supervisor._lock:
            held = self.supervisor._latest_frame
        stale = (
            self.terminal_frame is not None and held is not None and np.array_equal(held, self.terminal_frame)
        )
        body = {"state": "game_over" if stale else "gameplay", "game_over": stale, "confidence": 0.99}
        payload = json.dumps({"response": json.dumps(body)}).encode()

        class _Resp:
            def __enter__(self_inner):
                return self_inner

            def __exit__(self_inner, *exc):
                return False

            def read(self_inner):
                return payload

        return _Resp()


def test_env_reset_is_not_blocked_by_stale_sentinel_verdict(monkeypatch):
    fake = _FrameAwareVLM()
    monkeypatch.setattr("urllib.request.urlopen", fake)

    config = mock_config()
    config.hud.use_vlm = True
    config.hud.vlm_sentinel_interval = 0.05
    config.env.reset_timeout = 1.0
    controller = NullController(config.control)
    _game, frames, audio, _mock_controller = build_mock_backends(config, seed=22)
    env = GameEnv(config, frame_source=frames, audio_source=audio, controller=controller)
    fake.supervisor = env.supervisor
    try:
        env.reset()
        env.step(0)
        fake.terminal_frame = env.last_frame
        deadline = time.time() + 3.0
        while time.time() < deadline and not env.supervisor.is_game_over:
            time.sleep(0.01)
        assert env.supervisor.is_game_over is True

        restarts_before = controller.restarts
        t0 = time.perf_counter()
        env.reset()
        elapsed = time.perf_counter() - t0
        assert elapsed < 0.6, f"reset blocked on the stale verdict for {elapsed:.2f}s"
        assert controller.restarts == restarts_before + 1

        _obs, _rew, term, _trunc, _info = env.step(0)
        assert term is False
    finally:
        env.close()


def test_env_does_not_stop_an_injected_supervisor_on_close():
    config = mock_config()
    sup = CognitiveSupervisor(HUDConfig(use_vlm=False), interval=0.05)
    sup.start()
    try:
        env, _controller = _sentinel_env(config, sup, seed=23)
        env.reset()
        env.close()
        assert sup.is_running is True

        owned_env = GameEnv(
            config,
            frame_source=build_mock_backends(config, seed=24)[1],
            audio_source=build_mock_backends(config, seed=24)[2],
            controller=NullController(config.control),
        )
        owned = owned_env.supervisor
        owned.start()
        owned_env.close()
        assert owned.is_running is False
    finally:
        sup.stop()
