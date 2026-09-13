# Templates

Drop `game_over.png` here: a tight grayscale-or-colour crop of the "Game Over"
banner taken from the game at the **same window size** used for training.
`observer.py` matches it against every frame with `cv2.matchTemplate`
(`TM_CCOEFF_NORMED`, threshold `RewardConfig.game_over_match_threshold`).

Quickest way to create it: bring up the game-over screen, then run

```
python tools/calibrate.py --template
```

which crops `RewardConfig.game_over_region` into this file. If the file is
missing the observer falls back to the mean-colour check on that region.
