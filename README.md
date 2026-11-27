# Cadance

Cadance is a minimal techno stream generator built on the [Cadence](https://github.com/muellerberndt/cadence)
library. One Cadence brain hears the bars it has played and writes the next bar, without end. A
browser player renders the bars with its own drum machines and synth voices, and an Auto DJ mixes
whole tracks the brain composed ahead.

- [How the brains were constructed](docs/brains.md): the bar, the layouts, the lessons, how a brain
  is read, the play settings, and what did not work.
- [Running the code](docs/running.md): the player's pages, the files it expects, composing tracks
  ahead, the tests, hosting.
- [The Cadence tutorial](https://github.com/muellerberndt/cadence/blob/main/docs/tutorials/minimal-techno.md):
  the general lessons of this project, told for any application with many motor slots.

## What is here, and what is not

| Here | What it is |
| --- | --- |
| `brain/` | The trainer (`train.py`), the bar and the observation (`common.py`), the layouts (`layouts.py`, `hierarchy.py`, `anatomy.py`), the readings (`gate.py`, `composition.py`, `quality.py`), the layout checks (`check.py`) and tests |
| `site/` | The player: a local server (`serve.py`) with the player page, Stats, the mixer with the Auto DJ, the repertoire curator, a tool that composes whole tracks ahead (`shelf.py`), and the play settings (`play-reading.json`) |
| `engine/` | The audio engine in JavaScript: modelled 606, 808 and 909 drums and one-shots of the three machines (`engine/samples/`, with their source), an acid voice and a general synth voice, an offline renderer and its checks |
| `tools/` | Counters for composed tracks: notes per bar by segment, repeats, held bars, kick patterns |
| `docs/` | The two documents above |

| Not here | Why |
| --- | --- |
| The recordings and the score | The brains learned from a private collection of recordings, transcribed into a symbolic score. Neither is in this repository |
| The transcription pipeline | The code that reads recordings into the score is not part of this first cut |
| The sound pool | The player's sounds were fitted to the recordings. The pool file a brain names is not included |
| The trained brains | A brain is a card (`.json`) and its weights (`.npz`, about 600 MB). None is included yet |
| Cloud training scripts | The scripts that started and watched training machines are tied to one account |

So a fresh clone holds the complete source of the trainer and the player, and nothing to train on
or to play. The server starts and serves its pages; to hear a brain you need the three files named
under [what the player expects](docs/running.md#what-the-player-expects).

## Requirements

- Python 3.11 or later with NumPy and Cadence 0.104.1: `python -m pip install "cadence-net[fast]==0.104.1" numpy`
- PyTorch with a CUDA device for training at full size (`--backend torch`). The player runs on a CPU.
- Node.js for the engine's checks and the Auto DJ's planner test. A current browser for the player.

The code was developed on Cadence 0.104.0 with three patches. Cadence 0.104.1 ships all three (a
one-way sensory projection beside a learned code, surprise lessons by slot, a faster learned code),
and brains trained during development on the patched 0.104.0 load on 0.104.1.

## Quick start

```bash
python site/serve.py                      # the player: http://127.0.0.1:8792/
CADANCE_PORT=8793 python site/serve.py    # a second instance, for tests
node site/test_autodj.mjs                 # the Auto DJ's planner, without audio or data
python brain/layouts.py                   # the slate of layouts
python brain/layouts.py inst_reads_echo2  # one layout's preset and settings, as JSON
```

A training run of the shape that produced the kept brains (file names are yours; the flags are
those recorded in the brains' cards):

```bash
python brain/train.py --name my_echo --layout inst_reads_echo2 \
    --streams corpus/SCORE.json --pool POOL.json --synth-controls pool/POOL.json \
    --steps --synth-rows --synth-vocab steps --synth-fill --clock \
    --mask-unheard-targets --teach-returns --set learning_surprise_slots=true \
    --backend torch --set 'device="cuda"' \
    --window 96 --stride 8 --batch 16 --hours 10 \
    --eval-minutes 60,120,240,480 --keep-latest --out quality/composer/my_echo
```

It needs a score file and a pool file in the formats `brain/common.py` reads (`load_streams`), which
this repository does not ship. The player finds a run's checkpoints under `quality/composer/<name>/`.

Of the 47 tests in `brain/test_composer.py`, 40 pass on Cadence 0.104.1; seven that drive a whole
training run are behind the trainer's last changes.

Settings of one computer stay out of the code: the environment, or `site/local.json` (not in the
repository). `CADANCE_BUCKET` names where training runs keep their checkpoints; without it the
player works on local files only. `CADANCE_CADENCE` names a library source to use in place of the
installed one, and `CADANCE_PLAY_READING` another play settings file.

## Status

Research code, published as it was used. Four brains were kept from the development; with the
player's settings they play a steady kick, a snare of up to four hits per bar, a bass of six to
seven and a lead of up to six notes per bar, and whole 304-bar tracks that keep changing.
Counted against the recordings on 64-bar test tracks, a brain passes on a full starting passage
and not on a thin one. Not there yet: arrangement by the brain's own plan (voices enter and leave
by the player's settings), new melodic material beyond what sampling brings, hats that stay in over
a whole track, and any measured contribution of the brain's hierarchy. Whether a track sounds good
is judged by ear. The details and the numbers are in [docs/brains.md](docs/brains.md).

## Licence

GPL-3.0, as the Cadence library ([LICENSE](LICENSE)).
