# Windows Game RL Agent MVP

A real-time reinforcement-learning pipeline that observes a windowed game through
screen pixels and loopback audio, sends DirectInput keyboard actions, and trains
a PPO policy with Gymnasium and Stable-Baselines3.

## Architecture

```text
config.py       Capture regions, feature sizes, keys, rewards, and timing
perception.py   MSS screen capture, frame stacking, and mel audio features
controls.py     Safe discrete-action wrapper around PyDirectInput
observer.py     Template/color game-over detection and pixel score rewards
environment.py Gymnasium Dict-observation environment
train.py        PPO training, checkpoints, resume, and graceful shutdown
tests/          Hardware-independent environment and observer tests
```

Each observation is a dictionary:

- `image`: four grayscale `84 x 84` frames in channel-first `uint8` format.
- `audio`: normalized mean and standard deviation for 32 mel bands.

Stable-Baselines3's `MultiInputPolicy` gives the image a CNN encoder and the
audio vector a feature encoder, then trains them jointly.

## Windows setup

Use 64-bit Python 3.10 or newer. Open PowerShell in this directory:

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Keep the game window at a fixed position and use windowed or borderless mode.
Run the game and terminal at the same privilege level; DirectInput may not
reach an elevated game from a non-elevated terminal.

## Calibrate the target game

Edit `GameConfig` in `config.py`:

1. Set `capture_region` to the game's desktop coordinates.
2. Set `score_region` and `game_over_region` relative to the capture region.
3. Set `action_keys` and `restart_keys` for the game.
4. Tune score and game-over thresholds from actual captures.

For reliable terminal detection, save a tightly cropped grayscale-compatible
screenshot of the stable "Game Over" label as `assets/game_over.png`. When that
file is absent, the observer falls back to checking the proportion of pixels
near `game_over_bgr`. Set that BGR color to a distinctive color in the
game-over panel.

The MVP score detector rewards the leading edge of a visual change in the score
region. Keep blinking icons and timers outside that region. For games with
complex score animation, replace `_detect_score_change` with OCR or a
game-specific detector before long training runs.

## Train

Focus the game window, then run:

```powershell
python train.py --timesteps 500000
```

Press `Ctrl+C` once to stop safely. The current model is written to
`models/game_agent.zip`; periodic checkpoints are stored under
`models/checkpoints/`.

Resume a checkpoint with:

```powershell
python train.py --timesteps 250000 --resume models/game_agent.zip
```

TensorBoard logs are available with:

```powershell
tensorboard --logdir runs
```

## Validation

Hardware-independent tests can run on any supported OS:

```powershell
pip install -r requirements-dev.txt
pytest
ruff check .
```

Real game capture, loopback audio, and key injection must be validated on the
Windows machine running the game. Start with a short run and verify reward
values in logs before committing to long training. Keep PyDirectInput's
failsafe enabled, and be ready to refocus or close the game if controls are
misconfigured.

## MVP limitations

- Screen coordinates assume the window does not move or resize.
- A short audio recording is made synchronously each step; a background ring
  buffer is the next optimization for high-frame-rate games.
- Visual heuristics are game-specific and require calibration.
- Training from pixels on a live game is sample-intensive; begin with a simple
  action space and dense score feedback.
