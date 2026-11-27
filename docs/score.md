# Building your own score

The brains in this repository learn from a score: a symbolic transcription of recordings, one row
per bar. The score, the recordings it was read from, the readers' trained models and the sound
pool fitted to the recordings are not in the repository. This page gives the format the trainer
reads, the pipeline that produced the score as it was built, what was measured about each stage,
and what cost time. The folder [`pipeline/`](../pipeline/README.md) holds a skeleton of the stages
and a generator of a small synthetic score in the real format.

## What a score is

One JSON file. Per recording (or per passage of a long recording) it holds a list of bars. A bar
says which sounds play, where the drums hit on a grid of sixteen steps, which notes the three
synth lines play on the same grid, and how each layer's filter and space are set against the
sound's own setting. The trainer turns each bar into a row of slot values. With the settings of
the kept brains a bar has 213 slots:

| Group | Slots | Values per slot | From the score's field |
| --- | --- | --- | --- |
| Sounds | 4 | the kit id; on or off for bass, lead, air | `bars` |
| Filter and space levels | 8 | 7 levels | `mods` |
| Drum steps | 3 x 16 | none, hit, accent | `drums` |
| Line steps | 3 x 16 | rest, tie, or one of 12 pitch classes | `synth` |
| Sound controls | 3 x 35 | 2 to 13 levels each | the pool sound named in `bars`, and `synth_oct` |

The 35 controls of a synth layer are its voice, its register, its gain, 28 controls of the
oscillator, filter, envelopes, level and sends, and four velocities (one per sixteenth position
inside a beat). They are not read per bar from the recording: a bar is matched to one fitted
sound of the pool, and that sound's controls are the bar's labels.

## The file the trainer reads

`brain/common.py` (`load_streams`) reads the file, `pipeline/score_format.py` validates it.

```json
{
  "tokens_per_layer": 5,
  "mod_levels": 7,
  "tracks": [
    {
      "id": "recording_01",
      "held_out": false,
      "group": "reference",
      "style": "reference",
      "bars":      [[2, 1, 0, 0], [2, 1, 3, 0]],
      "mood":      [[0.56, 0.37], [0.74, 0.52]],
      "mods":      [[0, 0, 1, 0, 0, 0, 0, 0], [0, 0, 2, 0, 0, 0, 0, 0]],
      "drums":     [[[2,0,0,0, 1,0,0,0, 1,0,0,0, 1,0,0,0], [0,0,0,0, 1,0,0,0, 0,0,0,0, 1,0,0,0],
                     [0,0,2,0, 0,0,2,0, 0,0,2,0, 0,0,2,0]], "... one block of 3 x 16 per bar"],
      "synth":     [[[2,1,0,5, 0,0,9,1, 1,0,0,0, 2,0,0,0], [0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0],
                     [0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0]], "... one block of 3 x 16 per bar"],
      "synth_oct": [[1, -1, -1], [1, 4, -1]]
    }
  ]
}
```

The example is abbreviated: `drums` and `synth` hold one block of 3 x 16 for every bar.

| Field | Shape | Meaning |
| --- | --- | --- |
| `tokens_per_layer` | number | how many kit ids there are, silence (0) included |
| `mod_levels` | number | levels of filter and space, odd; 7 means -3 to 3 |
| `id` | string | unique in the score |
| `held_out` | true or false | the recording is kept for testing; no lesson comes from it |
| `bars` | n x 4 | the sound of beat, bass, lead, air; 0 means the layer is silent in the bar |
| `mood` | n x 2 | energy and brightness, each 0 to 1 |
| `mods` | n x 8 | optional; filter then space per layer, 0 is the sound's own setting |
| `drums` | n x 3 x 16 | kick, snare or clap, hats; 0 none, 1 hit, 2 accent |
| `synth` | n x 3 x 16 | bass, lead, air; 0 rest, 1 tie, 2 to 13 a new note at pitch class 0 to 11 |
| `synth_oct` | n x 3 | optional; the octave of each line in the bar (MIDI // 12 - 1), -1 unknown |
| `group`, `style` | string | optional; see below |
| `drums_sure`, `synth_sure` | n x 3 | optional; 1 where the reader was sure of the row or line |
| `drums_open`, `synth_open` | n x 3 x 16 | optional; 1 on a step the reader could not decide |
| `controls_open` | n x 3 | optional; 1 where a layer's sound was not read |
| `section` | n | optional; 0 normal, 1 breakdown, 2 build |

- **Open marks are lessons not given.** A step marked open is taken out of the lesson for that
  bar. Leaving a doubtful step open cost less than teaching it wrong: the brain learns the
  reader's mistakes as music.
- **A tie makes the note before it longer**, across the bar line too: a note of length L is one
  new note and L - 1 ties. A slide into another pitch is a new note. Accents of a line are not in
  the score. They come with the sound, as its four velocities.
- **A silent layer.** While a synth layer is silent its line steps are zero. With the trainer's
  `--synth-fill` the line of a silent bar is taught as the last line that sounded, so that "rest"
  is not the commonest lesson.
- **Held out means whole recordings.** A passage cut from a held-out recording is held out too.
  Bars of one recording on both sides make the test a memory test.
- **`group` and `style`.** A group names a part of the material (`window_shares` at the top level
  can give each group a share of the lesson windows). A style is an input only when the score has
  more than one. Where the code prefers one part of the material (the passages free runs start
  from, the pace the play temperatures are fitted to), it prefers `common.REFERENCE_STYLE` and
  `common.REFERENCE_GROUP`. Both default to `"reference"`; set them with the environment
  (`CADANCE_REFERENCE_STYLE`, `CADANCE_REFERENCE_GROUP`) or in `site/local.json`.
- **Length.** A lesson window is 64 bars by default (`--window`). A recording shorter than the
  window gives no lesson.

The trainer reads the score this way with `--steps --synth-rows --synth-vocab steps --synth-fill
--clock --synth-controls POOL --pool POOL` (the README has the whole command).

## The sound pool

A second JSON file. `layers` is a list of four entries named `beat`, `bass`, `lead` and `air`, each
with `tokens`: entry 0 is `null` (silence), every other entry is a sound, or `null` for an id no
bar uses.

- A **kit** (`beat`): `voice` `"drums"`, `device` (`"606"`, `"808"` or `"909"`), `patch` (tune,
  decay, tone and level per drum voice, each 0 to 1), `gain`, and `pattern.hits` (per voice the 16
  levels of the bar the kit was fitted to).
- A **synth sound** (`bass`, `lead`, `air`): `voice` (`"synth"` or `"acid"`), `patch` (24 unit
  controls, 28 for the acid voice), `gain`, and `pattern` with `notes` (16 MIDI numbers or nulls),
  `vel` and `len`.

`brain/timbre.py` (`teachers`) turns the pool into the labels of the sound controls, and the
player renders with the same pool. A score and its pool belong together: every sound id in `bars`
must be a sound of the pool. The trainer and the player look a pool up by its file name.

## The pipeline as it was built

| Stage | In | Out | Skeleton |
| --- | --- | --- | --- |
| 1 Separation | a recording | four stems: drums, bass, other, vocals | `separate.py` (stub) |
| 2 Grid | the drum stem | tempo, beats, bar starts, the sixteenth lattice | `grid.py` |
| 3 Drum rows | drum stem, lattice | `drums`, step probabilities | `drums.py` |
| 4 Lines | bass and other stems, lattice | `synth`, `synth_oct` | `lines.py` |
| 5 Pooled reading | step probabilities | steadier rows, the open marks | `pooling.py` |
| 6 Sounds | stems per bar | the pool, `bars`, `mods` | `sounds.py`, `engine/*fit*.mjs` |
| 7 Mood | bar levels | `mood` | `mood.py` |
| 8 Split and checks | all tracks | `held_out`, a validated file | `score_format.py`, `returns.py` |

### A truth to score against

No stage was trusted before it had a number. The numbers below come from benchmark clips whose
rows are known because they were made: drums rendered by the engine from known step rows and
mixed over real passages that have no drums of their own, lines rendered from known notes over
real drum stems, then compressed and separated like any recording. Each clip has a condition (a plain four on the floor, a snare roll,
a quiet bass, two records overlapping, a DJ master), and the clips are split into tune and test
clips over disjoint backings. Readers and thresholds were chosen on the tune clips. The numbers
are from the test clips. They say how a reader does on rendered sounds over real backings, which
is easier than a real record.

### 1. Separation

An open source separation model (Demucs) cut every recording into four stems, long recordings in
chunks with overlap. Everything after reads one stem with another as a leakage reference.
Pitfall: the readers' thresholds are relative to a recording's own loud passages, so the stems of
one recording must be normalised as one piece.

### 2. Grid

Tempo by how well a beat lattice between 85 and 185 BPM explains the drum stem's onsets, beats by
onset phase, then the bar start from where stems enter and leave, where the spectrum changes, the
snare on 2 and 4, kick fills at a bar's end and the kick returning on a 1. Measured on the test
clips: tempo within 0.5% on every clip, beats a median 16 to 17 ms from the truth. The bar start
was right on 26 of 30 arrangement clips, and on 9 of 17 clips that are drums alone. Where the
bar-start confidence was at least 0.9 and both halves of the clip agreed, it was right on 19 of
19 arrangement clips and on 3 of 5 drum clips. Pitfall: a wrong bar start does not show in a step score. Every reader still scores well and
every bar of the score is rotated. Keep the confidence and leave unsure passages out.

### 3. Drum rows

The first reader placed detected onsets on the nearest sixteenth (`pipeline/drums.py`
`rows_from_onsets`). The reader that was kept classifies every step from features of the drum
stem (level, rise and decay per band, the bass and other stems at the same place, the neighbouring
steps) with gradient-boosted trees fitted on the tune clips. It is given no step index.

| Test clips | Kick P / R / F1 | Snare P / R / F1 | Hats P / R / F1 |
| --- | --- | --- | --- |
| Machine drums over real backings | 0.963 / 1.000 / 0.981 | 0.998 / 0.922 / 0.959 | 0.971 / 0.960 / 0.966 |
| Breakbeat clips, all | 0.926 / 0.992 / 0.958 | 0.708 / 0.763 / 0.735 | 0.875 / 0.990 / 0.929 |
| The same, snare and hats refitted | 0.926 / 0.992 / 0.958 | 0.905 / 0.789 / 0.843 | 0.951 / 0.980 / 0.965 |

Pitfalls: a reader fitted on one kind of drums fails on another (on breakbeat clips whose drums
were cut from real drum stems the first snare model read F1 0.590, the refitted one 0.780). Tom
percussion read as kicks (kick precision 0.549 on that condition). Hats under a DJ master's
compression were missed (recall 0.658). A silent drum stem reads hits in its noise unless silent
bars are zeroed (`silence_rule`).

### 4. Lines

Per step a classifier of rest, tie and new note from the stem's spectral flux, the level before
and after the step and the spectral change, with the drum stem as leakage reference. The pitch is
measured, not learned: the waveform's period for the bass, the strongest pitch class for the lead.
The air layer was read as one held note per bar; the benchmark could not render it, so it has
no measured accuracy. On the test clips the bass read onsets at
precision 0.948, recall 0.810 (F1 0.874) with the pitch class right on 0.962 of them; the lead
0.982, 0.873 (F1 0.924) and 0.931. A whole bar's rhythm was exact in 0.556 of the bass bars and
0.650 of the lead bars. Pitfall: quiet lines are mostly missed (recall 0.256 on a quiet rolling
bass), and a line read step by step flickers: in one version of the score the bass row equalled
the bar before in 5% of the bars, and in 50% of the bars where the reader was sure of both.

### 5. Pooled reading

A reader decides every step alone, so a step near the threshold flips from bar to bar while the
record plays the same loop. `pipeline/pooling.py` pools a bar's step probabilities with the bars
whole loops before and after it that read like it, keeps a step's own decision where it is
decisive, and marks what stays undecided as open. On the machine-drum test clips the hats row
equalled the bar before in 0.89 of the bars read alone and 0.97 pooled, against 1.00 in the truth,
with the F1 unchanged. The parameters were chosen per row on tune clips.

### 6. Sounds

Single bars of each layer were fitted with the engine as the model: render a candidate sound,
compare it with the bar, move the controls (`engine/fit.mjs`, `engine/kit_fit.mjs`,
`engine/acid_fit.mjs`, all in the repository). The fits were reduced to a pool by clustering
their controls: 64 kits and 64 sounds per layer at first, more later. Every sounding bar was then
matched to one pool sound by its spectrum, density and register, and its filter and space level
read against that sound. The taught loudness of every sound was corrected to the level its bars
have in the recordings (`engine/token_levels.mjs` measures the engine's side). Pitfall: the level
reading did not hold up when it was measured. Against known levels the filter level was off
by 2.1 to 2.3 of the seven levels on average, no better than always answering the middle, and the
space level was never read. A bar whose sound was not read is marked `controls_open` in the score
the kept brains learned from, and the player has a setting that holds controls no lesson taught
at a pool sound's own values.

### 7. Mood

Energy is the bar's loudness on a scale of 24 dB under the 95th percentile of all bars.
Brightness is a spectral measure between its 5th and 95th percentile (`pipeline/mood.py`). The
references are taken over the whole material, so a quiet recording reads quiet.

### 8. Split and checks

Whole recordings are held out (`split_held_out`), the file is validated (`validate_score`), and
the finished score is counted: how often a step changes from one bar to the next, and how many of
the changes are returns to the value 2, 4, 8 or 16 bars back (`pipeline/returns.py`). In the score
the kept brains learned from, 84.9% of the changed drum steps and 72.9% of the changed line steps
were such returns, which is why those brains hear the bars at those lags
([how the brains were constructed](brains.md)). Read these counts before the first lesson. A
score whose lines change in most bars and seldom return is a flickering reader, not busy music.

## What is in the repository, and what is not

| In `pipeline/` | Not in the repository |
| --- | --- |
| The format: writer, validator, held-out split | The recordings and their stems |
| The lattice and the onset quantiser | The score read from them |
| The simple drum reader from onsets, the silence rule | The classifiers' trained models and features |
| The line coding with ties and octaves | The benchmark clips and their truth |
| The pooled reading with open marks | The sound pool fitted to the recordings |
| Mood rows, the count of returns | The scripts that ran the stages on the material |
| A synthetic score and pool in the real format | |

`python -m pipeline.toy_score OUT --check` writes `toy_score.json` and `toy_pool.json` and loads
them through the trainer's own loader: 213 slots, the open marks, lesson windows of 64 bars. It
is a format example and a smoke test. A brain trained on it has learned a programmed loop and
nothing about recordings.
