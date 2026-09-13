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
| `ai_player/observer.py` | The critic. Detects game over (`cv2.matchTemplate` or a mean-colour check) and score gains (Tesseract OCR or a pixel-signature fallback), then shapes the reward. |
| `ai_player/environment.py` | The `gymnasium.Env`: Dict observation space (image + audio), `Discrete` actions, fixed-rate stepping, episode resets. |
| `ai_player/policies.py` | `MultiModalExtractor`: CNN branch + audio MLP branch → fused feature vector for SB3. |
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
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

Notes:

- `pydirectinput` is Windows-only (it is skipped by an environment marker
  elsewhere). Everything except real key injection works on Linux/macOS, which
  is enough for `--mock` runs and the test suite.
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

## Pointing it at a real game

1. Open the game in a window and leave it visible.
2. Calibrate:

   ```bash
   python calibrate.py --output config.json
   ```

   You drag three boxes — the game window, the score digits, and the area where
   the game-over screen appears — and then let a countdown capture a game-over
   template while the game sits on that screen. Everything is written to
   `config.json`.

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
   maps action index → key, and `null` is the mandatory "do nothing" action:

   ```json
   "control": {
     "action_keys": ["up", "down", null],
     "restart_keys": ["space"],
     "hold_keys": false
   }
   ```

   Use `hold_keys: true` for continuous movement (the key stays down until the
   agent picks a different action) and `false` for discrete inputs like jumping.

5. Verify the environment contract and the reward signal, without sending any
   input:

   ```bash
   python train.py --config config.json --check-env      # API + shapes, seconds
   python train.py --config config.json --dry-run --timesteps 2000
   ```

   `--dry-run` captures the real screen but swallows key presses. Play by hand
   and confirm in the console that rewards jump when you score and that
   episodes end when you die. Fix the regions before training for real.

6. Train, giving yourself time to focus the game window:

   ```bash
   python train.py --config config.json --timesteps 500000 --countdown 5
   ```

   Ctrl+C saves `models/<run>/interrupted.zip`; checkpoints are written every
   `train.checkpoint_every` steps. Resume with
   `python train.py --config config.json --resume models/<run>/interrupted.zip`.

## How the reward is derived from pixels

There is no score API, so the critic infers everything from the frame:

```
reward = step_reward                    # +0.1 survival drip, dense signal
       + score_reward * points_gained   # +10 per detected point
       + game_over_penalty  if dead     # -100, dominates the return
```

- **Survival drip** gives PPO a gradient before it has ever scored. Without it
  the reward is sparse enough that early learning stalls.
- **Score detection** prefers OCR (`--psm 7`, digits whitelisted) because it
  yields a real delta. The fallback reads the score box as a binary mask and
  rewards *changes* to it — no digits recognised, but the event is still
  correlated with scoring, which is all the policy gradient needs. Implausible
  OCR jumps (`7 → 771`) and counter resets are ignored rather than rewarded or
  punished.
- **Termination** is debounced over `detection_patience` consecutive frames.
  A one-frame false positive would end the episode and poison the return, so
  this matters more than it looks.

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
- `train.normalize_reward` — on by default for PPO. The reward weights are
  hand-picked magnitudes (+0.1 against −100) and PPO shares one feature
  extractor between actor and critic, so raw rewards produce a value loss in
  the tens against a policy gradient in the thousandths, and the shared CNN
  ends up being trained almost entirely by the critic. `--no-reward-norm`
  turns it off. It is skipped for DQN, whose replay buffer would otherwise mix
  transitions scaled by different running statistics.
- `train.n_steps` — real-time games produce samples slowly. 512 keeps PPO
  updating often; large rollouts mean very long waits between improvements.

## Tests

```bash
pip install -r requirements-dev.txt
python -m pytest        # 44 tests, ~6s, no display or sound card needed
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
