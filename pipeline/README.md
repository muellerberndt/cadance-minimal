# The transcription pipeline, as a skeleton

The brains learn from a score: per recording a list of bars, each a row of drum steps, line
steps, sounds and levels. [docs/score.md](../docs/score.md) gives the format, the pipeline as it
was built, and what was measured about each stage. This folder holds the stages as small modules.
The generic parts are implemented and run on synthetic input. The parts that need a trained
model, a separation tool or audio raise `NotImplementedError` with what to implement.

Nothing here reads or names a recording. The material the kept brains learned from, its stems,
the readers' models and the benchmark clips are not in the repository.

## The stages, in order

| Stage | Module | Implemented | Left to you |
| --- | --- | --- | --- |
| 1 Separation | `separate.py` | | `separate`: four stems from a recording |
| 2 Grid | `grid.py` | `bar_starts`, `sixteenth_lattice`, `quantise_onsets` | `estimate_tempo_and_beats`, `find_bar_start` |
| 3 Drum rows | `drums.py` | `rows_from_onsets`, `silence_rule` | `step_features`, `read_drum_rows` (the classifier) |
| 4 Lines | `lines.py` | `encode_line`, `decode_line` | `read_line` |
| 5 Pooled reading | `pooling.py` | `loop_lengths`, `pool`, `read` with open marks | the parameters, chosen on clips you know |
| 6 Sounds | `sounds.py` | `synth_token`, `kit_token`, `pool_of` | `fit_sound`, `match_bars` (the engine's fitters are in `engine/`) |
| 7 Mood | `mood.py` | `mood_rows` | |
| 8 Split and checks | `score_format.py`, `returns.py` | `write_score`, `validate_score`, `validate_pool`, `split_held_out`, `lag_returns` | |

## Run it

```
python -m pipeline.selftest                      # every implemented part once, NumPy alone
python -m pipeline.toy_score OUT --check         # a synthetic score and pool, loaded as the trainer loads them
python -m pipeline.returns OUT/toy_score.json    # how many changes of a score are returns
```

Run them from the repository's root. `toy_score --check` imports `brain/common.py` and
`brain/timbre.py`, composes no brain, and prints what the trainer sees: 213 slots (4 sounds, 8
levels, 48 drum steps, 48 line steps, 3 x 35 sound controls), the recordings and the lesson
windows.

## Order of work for a score of your own

1. Build a truth first: clips whose rows you know because you rendered them, split into clips
   you tune on and clips you score on, over different backings. A reader without a score is a
   guess, and the brain learns its mistakes as music.
2. Get the grid right before any reader. Check the bar start by ear on every recording you keep.
3. Read drums, then lines. Keep the probabilities.
4. Pool over the loop and leave open what stays undecided.
5. Fit a small pool of sounds with the engine, match every bar to one, and mark what was not
   read.
6. Hold out whole recordings, validate, and count the returns before the first lesson.
