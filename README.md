# AI Player — vision + audio reinforcement learning for windowed games

An RL agent that plays a game it has no API access to. It sees the game by
capturing a region of the screen, hears it by tapping the system audio
loopback, presses keys through the Win32 `SendInput` API, and figures out its
own reward by reading the score box and the game-over screen with OpenCV.

Built on Gymnasium + Stable-Baselines3 (PPO or DQN) with a multi-modal policy:
a Nature-DQN convolutional stack over stacked 84×84 frames, fused with a small
MLP over a log-mel spectrogram of the last half second of audio.

The pipeline ships with a **bundled simulated game**, so you can run and test
everything end to end on any OS before pointing it at a real window:

```bash
pip install -r requirements.txt
python train.py --mock --timesteps 20000      # trains against the simulator
```

## How it fits together

```
  ┌───────────────┐    mss + cv2     ┌──────────────┐
  │               │ ───────────────► │  perception  │──► frames  (4 x 84 x 84 uint8)
  │  game window  │   soundcard +    │              │──► audio   (32 x 30 log-mel)
  │               │   librosa        └──────────────┘         │
  │               │                  ┌──────────────┐         ▼
  │               │ ◄─────────────── │   controls   │ ◄── PPO / DQN  (MultiModalExtractor:
  │               │  pydirectinput   └──────────────┘         ▲        CNN + audio MLP)
  │               │                  ┌──────────────┐         │
  │               │ ───────────────► │   observer   │──► reward, terminated
  └───────────────┘  matchTemplate   └──────────────┘
                     + OCR / pixels
```

## Why the pieces exist

| Module | Responsibility |
| --- | --- |
| `ai_player/config.py` | Every tunable: capture region, 84×84 frame size, target FPS, audio window, key map, reward weights. Serialises to JSON. |
| `ai_player/perception.py` | `mss` screen capture, OpenCV grayscale/downscale/frame-stacking, `soundcard` loopback recording, `librosa` mel-spectrogram features. |
| `ai_player/controls.py` | `pydirectinput` keyboard driver (tap or hold), restart sequences, guaranteed key release. |
| `ai_player/observer.py` | The critic. Detects game over (`cv2.matchTemplate` or a banner-colour coverage check) and score gains (Tesseract OCR or a pixel-signature fallback), then shapes the reward. |
| `ai_player/environment.py` | The `gymnasium.Env`: Dict observation space (image + audio), `Discrete` actions, fixed-rate stepping, episode resets. |
| `ai_player/policies.py` | `MultiModalExtractor`: Vision backbone (Nature-DQN / ConvNeXt / ResNet) + audio MLP branch → fused feature vector for SB3. |
| `ai_player/tracker.py` | Controllability probe (avatar auto-detection via differential optical flow) + foreground entity tracker (projectiles, threats, collectibles, causal collision analysis). |
| `ai_player/mock_game.py` | A three-lane dodger rendered with OpenCV, plus drop-in capture/audio/control backends. Used by `--mock` and the tests. |
| `train.py` / `play.py` / `calibrate.py` | Training loop, evaluation/recording, and region calibration (pick / check / live preview). |
| `assets/templates/` | Where the game-over template crop lives. |

The library modules sit in the `ai_player/` package rather than at the repo
root, so that generic names like `config` and `controls` cannot shadow
unrelated top-level modules on `sys.path`. Only the entry-point scripts
(`train.py`, `play.py`, `calibrate.py`) live at the root.

## Install

Python 3.10+.

```bash
python -m venv .venv
.venv\Scripts\activate               # Windows
source .venv/bin/activate            # Linux / macOS
pip install -r requirements.txt
```

Notes:

- `pydirectinput` is Windows-only, so `requirements.txt` installs it only
  there. Everything except real key injection works on Linux/macOS, which is
  enough for `--mock` runs and the test suite.
- Score OCR needs the [Tesseract binary](https://github.com/UB-Mannheim/tesseract/wiki)
  on `PATH`. Without it the observer automatically falls back to the
  pixel-signature score detector and warns once.
- Some games ignore synthetic input unless the sending process is elevated. If
  keys appear to do nothing, run your terminal as Administrator.

## Quickstart against the simulated game

```bash
python train.py --mock --timesteps 50000 --run-name mock-ppo
python play.py models/mock-ppo/final.zip --mock --episodes 5 --record play.mp4
tensorboard --logdir logs
```

Watch `rollout/ep_rew_mean` and `rollout/ep_len_mean`: both should climb as the
agent learns to survive longer.

## Results on the simulated game

Two 150k-step PPO runs (identical seed, 84×84 frames + mel spectrogram,
~250 env steps/s on four CPU cores), differing only in whether rewards were
normalised:

| | survival at 10k steps | at 150k steps | eval over 20 episodes |
| --- | --- | --- | --- |
| raw rewards (`--no-reward-norm`) | 24.1 steps | 24.5 steps | 25.7 steps, 1.2 points |
| normalised rewards (default) | 22.5 steps | 288 steps | 966 steps, 146 points |

The raw-reward run never leaves its starting policy. PPO shares one feature
extractor between the actor and the critic, and with a −100 terminal penalty
next to a +0.1 drip the value loss dominates the gradient, so the trunk is
optimised to predict the return rather than to see obstacles. Scaling the
reward is what makes the run learn at all, not a marginal tuning win.

Reproduce with:

```bash
python train.py --mock --timesteps 150000 --run-name ab-norm --seed 0
python train.py --mock --timesteps 150000 --run-name ab-raw --seed 0 --no-reward-norm
python play.py models/ab-norm/final.zip --mock --episodes 20
```

Both runs are flat for the first ~40k steps and then diverge sharply, which is
what the reward-scale argument below predicts. The better agent weaves between
lanes and finishes episodes with a positive return; it has not mastered the
game — pixel-based control needs millions of steps for that, and the simulator
exists to validate the pipeline rather than to serve as a benchmark — but the
plumbing clearly produces a learning signal the policy can exploit.

## Pointing it at a real game

Capture follows the focused game window by default. You do **not** have to
re-draw a rectangle when you alt-tab to another title: the agent retargets,
forgets the previous game's HUD/avatar, and starts a fresh episode. Clicking
the telemetry overlay or a terminal is ignored (sticky lock). Pin a title with
`--window-title Celeste` if you want to look away without losing the game.

1. Open a game and leave it visible (windowed, not exclusive fullscreen).
2. Optional calibrate — this now writes a follow-foreground config, not a
   one-off crop:

   ```bash
   python calibrate.py --output config.json
   python calibrate.py --output config.json --window-title "Celeste"
   python calibrate.py --output config.json --static-region   # old 3-box crop
   ```

   Score and game-over boxes are optional: cancel them and autonomous HUD
   detection fills them in at train time.

3. Check what the agent sees. This needs no GUI, so it also works over remote
   sessions and when you edited `config.json` by hand:

   ```bash
   python calibrate.py --config config.json --check   # annotated screenshot + readings
   python calibrate.py --config config.json --live    # the 84x84 view, upscaled
   ```

   `--check` writes `assets/captures/calibration.png` with the score and
   game-over boxes drawn on it, and prints whether the game-over detector
   fires, which score reader is active, and what it currently reads.

4. Set the key map in `config.json` if the defaults are wrong. `action_keys`
   maps action index → key, and `null` is the mandatory "do nothing" action.

   **Automatic Combination Discovery (`auto_combos`)**:
   Instead of hardcoding every possible combination manually, you only list your game's basic atomic keys (e.g. `["d", "a", "w", "s", "j", "space"]`) and enable `"auto_combos": true` (or `--auto-combos` CLI flag). The agent automatically generates the entire combination space up to `max_combo_size` (e.g. running + shooting `["d", "j"]`, jumping + shooting `["space", "j"]`, running + jumping `["d", "space"]`), and the RL policy learns via trial and error which combinations yield higher rewards:

   ```json
   "control": {
     "action_keys": [
       null,
       "d",
       "a",
       "w",
       "s",
       "j",
       "space"
     ],
     "auto_combos": true,
     "max_combo_size": 2,
     "restart_keys": ["space"],
     "hold_keys": false
   }
   ```

   - Single keys: `"d"`, `"space"`, etc.
   - When `auto_combos: false`, you can still specify custom explicit combos: `["d", "space"]` or `"d+space"`.
   - Use `hold_keys: true` for continuous movement (keys stay down until the
     agent picks a different action) and `false` for discrete inputs like jumping.
   - Test and verify all keys in your game window:
     ```bash
     python calibrate.py --config config.json --test-keys
     ```
     This validates the key names against DirectX scan codes and triggers each action in sequence so you can confirm in-game reactions.

5. Verify the environment contract and the reward signal, without sending any
   input:

   ```bash
   python train.py --config config.json --check-env      # API + shapes, seconds
   python train.py --config config.json --dry-run --timesteps 2000
   ```

   `--dry-run` captures the real screen but swallows key presses. Play by hand
   and confirm in the console that rewards jump when you score and that
   episodes end when you die. Fix the regions before training for real.

6. Train. Focus whichever game you want to play; switch titles whenever you
   like — each switch ends the episode and re-detects HUD/avatar:

   ```bash
   python train.py --config config.json --timesteps 500000 --render
   python train.py --config config.json --window-title "Celeste" --infinite --render
   ```

   `--timesteps` must be positive. To train open-endedly, add `--infinite`:
   training then runs in repeated rounds of `--timesteps` steps until Ctrl+C,
   saving `round_NNNN.zip` after each round. Because every round is a full
   `learn()` call, per-run schedules (DQN epsilon decay, linear learning
   rates) complete inside each round instead of being stretched over a
   fictitious horizon; the timestep counter and TensorBoard logs keep
   counting up across rounds.

   The `--render` flag opens a real-time HUD dashboard featuring:
   - **Real Measured FPS** calculated from actual step intervals vs target FPS.
   - **Real-Time Latency Breakdown Profiler**: Exact milliseconds spent per step across screen capture (`grab`), keyboard action (`act`), image preprocessing (`proc`), audio/obs extraction (`audio/obs`), and HUD drawing (`render`).
   - **Active Action & Full Action Palette** highlighting the current action and multi-key combos.
   - **CNN Real Input (84x84 Grayscale)**: A picture-in-picture inset in the corner showing the exact downscaled observation entering the CNN, along with the 4-frame temporal filmstrip showing how motion and velocity are perceived.
   - **Motion & Stagnation Diff**: Live visual diff percentage and idle state.

   **Benchmarking Pipeline Latency & Bottlenecks**:
   To diagnose FPS limits before training:
   ```bash
   python calibrate.py --config config.json --profile
   ```
   This isolates each pipeline stage (screen capture, vision processing, audio FFT, direct input, and OpenCV windowing) and prints exact latencies and the theoretical maximum achievable FPS.

   Ctrl+C saves `models/<run>/interrupted.zip`; checkpoints are written every
   `train.checkpoint_every` steps. Resume with
   `python train.py --config config.json --resume models/<run>/interrupted.zip`.

## How the reward is derived from pixels

There is no score API, so the critic infers everything from the frame:

```
reward = step_reward                    # +0.1 survival drip, dense signal
       + score_reward * points_gained   # +10 per detected point
       + idle_penalty if is_idle        # negative penalty when the scene is standing still
       + movement_reward if moving      # positive reward when screen pixels change
       + game_over_penalty  if dead     # -100, dominates the return
```

## Universal Autonomous Game Agent Architecture

The agent supports fully autonomous zero-shot adaptation across four core pillars:

1. **Modern High-Capacity Vision Backbone (ConvNeXt / ResNet / Nature-CNN)**:
   - Configurable resolutions: 84×84, 128×128, 256×256 RGB or grayscale.
   - Preserves 3 color channels per frame stacked into `3 * frame_stack` channels.
   - Built-in `ConvNeXtBackbone` (7×7 depthwise convs, LayerNorm, GELU) and `ResNetBackbone` (residual skip blocks) with adaptive average pooling projecting to arbitrary `features_dim` (e.g. 512, 1024).
   - CLI flags: `--backbone convnext --rgb --resolution 256 --features-dim 1024`.

2. **Controllability Probe (Autonomous Player Avatar Discovery)**:
   - Evaluates differential optical flow (`cv2.calcOpticalFlowFarneback`) or temporal differencing during initial action probes.
   - Discovers which visual component moves in direct correlation with directional inputs (`left`, `right`, `up`, `down`, `jump`).
   - Automatically outputs and tracks the player avatar bounding box over time.
   - CLI verification: `python calibrate.py --probe-avatar`.

3. **Autonomous Dynamic Observer & Zero-Shot HUD Detection**:
   - Zero-shot automatic HUD extraction (`AutonomousHUDDetector`).
   - Saliency, edge gradient, and character glyph clustering automatically identifies Score, Lives, and Game Over regions without manual calibration.
   - Optional local Vision-Language Model (`VLMObserverInterface`) queryable via Ollama or OpenAI-compatible endpoints (e.g. Qwen2.5-VL, MiniCPM).
   - CLI verification: `python calibrate.py --auto-hud`.

4. **Threat, Projectile & Entity Tracker**:
   - Dynamic foreground object segmentation outside the player bounding box.
   - **Camera Scroll & Parallax Protection**: Computes global camera motion delta. When camera scrolling or parallax background shifts occur, entity extraction is automatically throttled and capped (`scroll_threshold`, `max_active_entities`), preventing CPU spikes, FPS drops, and false-target explosions.
   - Classifies detected dynamic entities into:
     - `PROJECTILE`: Fast linear velocity (>7 px/step), compact bounding box.
     - `THREAT`: Movement oriented toward the player or platform patrolling.
     - `COLLECTIBLE`: Floating or falling items, slow vertical drift.
   - **Causal collision analysis**: Correlates entity intersections with immediate game feedback (life loss / death -> penalty; score gain -> reward).
   - Real-time HUD telemetry overlays player avatar box, entity bounding boxes, and velocity threat vectors.

- **Survival drip** gives PPO a gradient before it has ever scored. Without it
  the reward is sparse enough that early learning stalls.
- **Stagnation detection** measures consecutive full-frame pixel changes. If the
  screen changes by less than `idle_diff_threshold` (e.g. 0.05 = 95% similarity),
  the player is stationary and `idle_penalty` is applied; otherwise `movement_reward`
  encourages forward movement and exploration.
- **Score detection** prefers OCR (`--psm 7`, digits whitelisted) because it
  yields a real delta. Implausible jumps (`7 → 771`) and counter resets are
  ignored rather than rewarded or punished. The fallback thresholds the score
  box and rewards *changes* to it: no digits recognised, but the event is
  still correlated with scoring, which is all the policy gradient needs. It is
  measured against the glyph pixels rather than the whole crop — a `5 → 6`
  repaint moves under 1% of a generous score box but a large share of its ink,
  and a box-relative threshold drops most increments. It still undercounts
  when a game awards several points in one frame, which is what OCR is for.
  Either path thresholds the box so the digits are the non-zero pixels, so a
  dark-on-light HUD reads the same as a light-on-dark one.
- **Autonomous Game Over Detection**:
  When no template is supplied, `AutonomousGameOverDetector` automatically runs:
  1. Multi-scale Canny edge template correlation against synthetic text masks ("GAME OVER", "CONTINUE"), invariant to font color and style.
  2. Fast text & OCR keyword scanning across high-contrast luminance and chroma channels ("GAME OVER", "CONTINUE", "YOU DIED", "DEFEAT", "MISSION FAILED", "TRY AGAIN").
  3. Center screen blackout and fade detection with stationary low-delta motion.
  4. Color matching fallback for calibrated solid banners.
- **Template matching** uses normalised correlation, except when the saved
  template has no variance at all. Correlation is undefined for a flat image:
  OpenCV returns 0.0 for a perfect match on a solid colour block exactly as it
  does for a total mismatch, so a banner calibrated that way could never fire.
  Those templates are matched by squared difference instead.
- **Termination is also debounced** over `detection_patience` consecutive
  frames. A one-frame false positive would end the episode and poison the
  return, so this matters more than it looks.

## Universal Input & Action Space (Keyboard + Mouse Hybrid)

To support 3D FPS, RTS, and racing titles, `ControlConfig` supports both keyboard and mouse:
- **Mouse Modes**: `"relative"` for 3D camera turns (`aim_step`, `mouse_sensitivity`), `"absolute"` for cursor clicking and dragging.
- **Compound Actions**: e.g. `"w+mouse_left"`, `"aim_left"`, `"aim_up+fire"`, `["w", "space", "mouse_left"]`.
- **UnifiedInputController**: Seamlessly handles tap and hold modes across keys and mouse buttons, releasing all inputs on reset or termination.
- **OpenAI & LM Studio VLM Support**: `VLMObserverInterface` supports `/v1/chat/completions` payload format with base64 `image_url` for seamless integration with local models (Qwen2.5-VL 8B) loaded in LM Studio.

## Asynchronous VLM Sentinel & Cognitive Game Supervisor

A dedicated non-blocking background daemon thread (`CognitiveSupervisor` / `AsyncVLMSentinel` in `ai_player/cognitive.py`) continuously monitors live gameplay with Vision-Language Models:
- **Hierarchical Game State Recognition**: Identifies `"gameplay"`, `"game_over"`, `"menu"`, `"cutscene"`, `"loading"`.
- **Autonomous Menu Navigation (`auto_menu_nav`, default `false`)**: When enabled, taps the restart keys once per fresh `"menu"` verdict, rate-limited by `menu_nav_cooldown` (1.5 s). It injects input on its own, so it is opt-in.
- **Continuous Game-Over Guidance**: Fuses the VLM verdict into `AutonomousGameOverDetector` and `GameObserver`. A verdict of `game_over` with confidence ≥ `vlm_game_over_confidence` (0.75) ends the episode immediately, bypassing `detection_patience`; lower-confidence verdicts join the normal debounced detector pool. Only verdicts newer than `vlm_max_age` (6 s) count, and the verdict is cleared at every `reset()` so the terminal screenshot can never end the *next* episode.
- **Dynamic HUD Adaptation**: Frames are downscaled to `vlm_max_image_dim` (640 px) before upload; any `hud_layout` boxes the model returns are mapped back to full-frame coordinates and only re-applied to the score/game-over detectors when they actually change.
- **Tactical Guidance & Commentary**: Supplies action suggestions (`suggested_action`, e.g. `"press_start"`, `"dodge_left"`) and scene descriptions (`vlm_desc`), rendered onto the real-time HUD telemetry overlay (`[STALE]` when the last verdict is older than `vlm_max_age`).
- **Semantic Cognitive Observation (`cognitive_obs`)**: Optional 8-float observation key `[state_code, hp_ratio, ammo_ratio, lives_ratio, danger_level, action_code, vlm_confidence, game_over]`; `lives` is normalised by `cognitive_max_lives` (5). Stale verdicts produce the neutral default vector.
- **Bounded Overhead**: The sentinel runs on a daemon thread and starts at most one VLM query per `vlm_sentinel_interval` (2 s), backing off exponentially (up to 30 s) while the endpoint fails. The step loop only reads its latest verdict and never waits on it.

## Tuning that actually moves the needle

- `env.target_fps` — the agent's decision rate. 10 Hz is a good start; frame
  stacking encodes velocity as pixels-per-step, so keep it *stable* rather than
  maximal. Raising it makes episodes longer in steps and credit assignment
  harder.
- `vision.frame_stack` — 4 frames gives the policy velocity. Drop to 2 for
  slow puzzle games, raise for fast ones.
- `audio.enabled` — turn it off (`--no-audio`) when the game's sound carries no
  state; the audio branch disappears from the network entirely.
- `reward.game_over_penalty` vs `reward.step_reward` — if the agent suicides
  early, the penalty is too small relative to the drip it is forfeiting.
- `train.normalize_reward` — on by default for PPO, and the single biggest
  difference measured so far (see the table above: 966 versus 26 steps of
  survival). The reward weights are hand-picked magnitudes (+0.1 against −100)
  and PPO shares one feature extractor between actor and critic, so raw
  rewards produce a value loss in the tens against a policy gradient in the
  thousandths, and the shared CNN ends up being trained almost entirely by the
  critic. `--no-reward-norm` turns it off. It is skipped for DQN, whose replay
  buffer would otherwise mix transitions scaled by different running
  statistics.
- `train.n_steps` — real-time games produce samples slowly. 512 keeps PPO
  updating often; large rollouts mean very long waits between improvements.

## Tests

```bash
pip install -r requirements-dev.txt
python -m pytest        # 56 tests, ~11s, no display or sound card needed
python -m ruff check .
```

The suite runs the Gymnasium API checker against the environment, verifies the
perception shapes, covers the reward/termination logic (including the OCR
delta rules with a stubbed Tesseract), and trains PPO and DQN for a few dozen
real steps against the simulated game.

## Limitations

- Real key injection is Windows-only, by design of `pydirectinput`.
- One environment instance per machine: the agent drives the actual desktop, so
  vectorised training across parallel game instances is not supported here.
- Capture regions are in physical pixels. With Windows display scaling, always
  use `calibrate.py` rather than measuring coordinates by hand.
- Anti-cheat protected games may block synthetic input or screen capture
  outright. Stick to browser games, emulators, and offline arcade titles.
