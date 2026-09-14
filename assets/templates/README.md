# Game-over templates

`game_over.png` belongs here: a tight grayscale crop of the game-over banner,
taken at the **same window size** you train at. `ai_player/observer.py` matches
it against every frame with `cv2.matchTemplate` (`TM_CCOEFF_NORMED`, threshold
`reward.template_threshold`).

The easiest way to produce it is to bring up the game-over screen and run:

```bash
python calibrate.py --output config.json
```

The interactive flow ends with a countdown that crops
`reward.game_over_region` into this folder and writes the path into
`config.json` as `reward.game_over_template`.

Without a template, the observer falls back to a mean-colour check on
`reward.game_over_region`, which works whenever the game paints a solid banner
or dims the playfield on death, but is easier to fool than template matching.

Templates are machine- and resolution-specific, so `*.png` in here is
gitignored.
