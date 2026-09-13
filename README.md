# Screen-Playing RL Agent

A minimal, modular pipeline for training a Deep Reinforcement Learning agent
that plays a **windowed game on Windows** the way a human does: it looks at the
screen, listens to the speakers, and presses keys. No game memory access, no
game-specific API.

```
 ┌──────────────┐   mss + cv2    ┌─────────────┐
 │  Game window │ ─────────────► │ perception  │──► image  (4 x 84 x 84 uint8)
 │              │  soundcard +   │             │──► audio  (32 x 11 log-mel)
 │              │  librosa       └─────────────┘  flattened  │
 │              │                ┌─────────────┐            ▼
 │              │ ◄───────────── │  controls   │ ◄─── PPO / DQN (Stable-Baselines3)
 │              │ pydirectinput  └─────────────┘            ▲
 │              │                ┌─────────────┐            │
 │              │ ─────────────► │  observer   │──► reward, terminated
 └──────────────┘  cv2 template  └─────────────┘
                   / OCR
```

## Project layout

| File | Role |
| --- | --- |
| `config.py` | Single source of truth: capture region, 84x84 frame size, frame stack, FPS, audio sample rate / mel settings, key mapping, reward shaping, PPO/DQN hyper-parameters. |
| `perception.py` | `ScreenCapture` (mss → grayscale → resize → frame stack) and `AudioCapture` (background loopback recorder → log-mel spectrogram). |
| `controls.py` | `GameController`: discrete action → timed `pydirectinput` key tap; restart sequence; dry-run mode on non-Windows hosts. |
| `observer.py` | `GameObserver`: the reward function. Game-over via `cv2.matchTemplate` (or mean-colour fallback) → `terminated=True`, `-100`. Score via Tesseract OCR (or pixel-change fallback) → `+10` per point, `+0.1` per surviving step. |
| `environment.py` | `GameEnv(gymnasium.Env)` with a `Dict(image, audio)` observation space and `Discrete` action space; wall-clock paced `step()`. |
| `train.py` | Builds the env, a `MultiInputPolicy` PPO/DQN model, periodic checkpoints, graceful save on Ctrl+C. |
| `play.py` | Runs a saved model for a few episodes. |
| `mock_game.py` | Built-in dodge arcade used by `--mock` so the PPO loop can run without a real window. |
| `tools/calibrate.py` | Screenshots the capture region with the score / game-over ROIs drawn on it; can crop the game-over template. |
| `assets/templates/` | Put `game_over.png` here (see the README inside). |

## Requirements

* Windows 10/11 (for `pydirectinput` and WASAPI loopback audio). The code
  runs on Linux/macOS in *dry-run* mode for development and unit testing.
* Python 3.10+
* Optional: [Tesseract OCR](https://github.com/UB-Mannheim/tesseract/wiki) on
  `PATH` for numeric score reading. Without it the observer rewards any change
  in the score region instead.

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

For GPU training replace the `torch` line with the CUDA wheel from
[pytorch.org](https://pytorch.org/get-started/locally/).

## Smoke-test the pipeline (any OS)

No game window and no Windows APIs — trains PPO for a few hundred steps on the
built-in dodge arcade. This is the fastest way to confirm Gymnasium, the
CNN+MLP `MultiInputPolicy`, and the OpenCV observer:

```bash
python -m pytest
python train.py --mock --smoke
python play.py --mock --episodes 1
```

Weights land in `models/game_agent.zip`.

## Quick start

1. **Open the game** in a window that does not move or resize (browser games:
   zoom to 100%, keep the tab in the foreground).
2. **Calibrate** the capture region and ROIs:

   ```powershell
   python tools/calibrate.py
   ```

   Open `assets/captures/calibration.png`, then edit `ScreenConfig.capture_region`,
   `RewardConfig.score_region` and `RewardConfig.game_over_region` in
   `config.py` until the green (score) and red (game over) boxes frame the right
   UI elements. Repeat until it looks right. `--live` shows the 84x84 view the
   agent actually receives.
3. **Capture the game-over template** while the game-over screen is showing:

   ```powershell
   python tools/calibrate.py --template
   ```

4. **Map the keys** in `ControlConfig.actions` / `restart_keys` (key names
   follow the pyautogui convention: `"up"`, `"space"`, `"z"`, ...).
5. **Validate the environment** against the Gymnasium API:

   ```powershell
   python train.py --check-env
   ```

6. **Train**:

   ```powershell
   python train.py --timesteps 200000
   tensorboard --logdir logs/tensorboard
   ```

   You have three seconds after launch to click into the game window. Ctrl+C
   saves `models/game_agent.zip` before exiting; checkpoints land in
   `models/checkpoints/` every 5 000 steps.
7. **Watch it play**:

   ```powershell
   python play.py --episodes 5
   ```

## How the RL loop works

* **Observation.** Four consecutive 84x84 grayscale frames are stacked so the
  policy can infer motion from a single observation (the classic Atari trick).
  A 0.25 s log-mel spectrogram (32 mel bands x 11 frames, flattened to a
  352-vector) of the system audio is attached as a second modality; SB3's
  `CombinedExtractor` sends the image through a NatureCNN and the spectrogram
  through an MLP branch, then concatenates both.
* **Action.** `Discrete(len(actions))`. Each action is a short key *tap*
  (`key_hold_seconds`), including an explicit no-op so the agent can learn to
  wait.
* **Real-time pacing.** `step()` sleeps until one frame period
  (`1 / target_fps`) has elapsed since the previous step so the game has time
  to react and the decision frequency stays stable when inference time jitters.
* **Reward.** `+0.1` per step alive, `+10` per point scored, `-100` and
  `terminated=True` when the game-over screen is confirmed for two consecutive
  frames. Episodes are also `truncated` after `max_episode_steps`.
* **Reset.** Sends `restart_keys`, waits `restart_delay_seconds`, refills the
  frame stack and snapshots the score region.

Because the environment is a live game, only one instance can exist per
machine and every episode is non-deterministic. Keep `n_steps` small, use
frame stacking, and expect wall-clock time per step to dominate training
throughput.

## Adapting to your game

| Want to... | Change |
| --- | --- |
| Use different keys / more actions | `ControlConfig.actions` and `action_names` |
| Add mouse actions | Extend `GameController.execute` with `pydirectinput.moveRel/click` |
| Detect game over on a colour flash instead of a banner | Delete the template and tune `game_over_region` / `game_over_color_bgr` |
| Reward something other than the score (e.g. a health bar) | Add a method to `GameObserver` and fold it into `evaluate()` |
| Drop the audio modality | `python train.py --no-audio` (the observation stays a Dict; the audio entry becomes zeros) |
| Use a different monitor / resolution | `ScreenConfig.capture_region` and `ControlConfig.focus_click` |

## Development without a game

`GameEnv` accepts injected `screen`, `audio`, `controller` and `observer`
instances, so the whole loop can be exercised with synthetic frames and a
dry-run controller on any OS. `python train.py --dry-run --no-audio --check-env`
runs the SB3 environment checker against whatever is on screen without sending
a single keystroke.

The test suite drives the environment with a scripted fake game (score flips,
game-over banner) and runs a few PPO updates through the `MultiInputPolicy`:

```bash
pip install -r requirements-dev.txt
pytest
ruff check . && ruff format --check .
```
