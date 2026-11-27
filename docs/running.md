# Running the code

Everything here is read from the code. A fresh clone has no brain, no sound pool and no score, so
the parts that need them are named as such.

## The player

```bash
python -m pip install "cadence-net[fast]==0.104.1" numpy
python site/serve.py                      # http://127.0.0.1:8792/
CADANCE_PORT=8793 python site/serve.py    # a second instance, for tests
```

The server listens on 127.0.0.1 only. It holds one brain in memory and answers one bar per
request: the page sends the wanted mood, the brain writes the next bar, and the page plays it with
the engine. Loading a brain takes one to two minutes and about 1.8 GB of memory; a bar takes about
2 seconds on a laptop, which is about as long as the bar lasts.

| Page | Address | What it is |
| --- | --- | --- |
| Player | `/` | one brain playing live, a mood pad, the brain list, a view of the brain's regions |
| Stats | on the player page | the training curves of the runs the server knows, with buttons to load a run's state |
| Mixer | `/mixer` | three decks with a DJ mixer and effects, and the Auto DJ |
| Curator | `python site/curator.py`, port 8794 | a separate site for marking passages of composed tracks as repertoire pieces |

The Auto DJ (`site/autodj.js`) is a planner without machine learning: it mixes whole tracks of one
brain with programmed transitions on 32-bar sections. `site/serve.py` composes its tracks one after
another (`/api/autodj/track`) and keeps the newest 24 per brain in `site/tracks/<brain>/`.

## What the player expects

| File | Where | Shipped |
| --- | --- | --- |
| A brain: its card `<name>.json` and weights `<name>.npz` | `site/pinned/` (kept by hand) or `site/checkpoints/` | no |
| The sound pool the card names (`"pool"`) | `pool/` | no |
| The score the card names (`"streams"`), for starting passages | `corpus/` | no |
| Play settings | `site/play-reading.json` | yes |
| Whole tracks composed ahead | `site/tracks/<brain>/<stamp>_<bars>.json` | no |
| Settings of this computer | `site/local.json` | no, yours to write |

The brain list is `site/brains.json`, which the server writes from the pinned brains when it is
missing. A card that names a library patch the running library does not hold is not listed.

The server picks its Cadence library in this order: the path in `CADANCE_CADENCE`, a local patched
copy in a dot-folder beside the code (development only, not in the repository), then the installed
`cadence-net` (0.103.0 or later). Install 0.104.1.

Settings of one computer come from the environment or from `site/local.json`:

| Name | Meaning |
| --- | --- |
| `CADANCE_PORT` | the server's port (8792) |
| `CADANCE_BRAINS` | another brain list file, for a test instance |
| `CADANCE_PLAY_READING` | another play settings file, to measure a setting without changing the player's |
| `CADANCE_CADENCE` | a Cadence source folder to use in place of the installed library |
| `CADANCE_BUCKET` | where training runs keep their checkpoints (`s3://bucket/prefix`). Without it the player works on local files only: `site/pinned`, `site/checkpoints`, `site/tracks` |

## Composing tracks ahead

```bash
nice -n 10 python site/shelf.py <brain> --tracks 4 --bars 304 \
    --require "bass>=3,lead>=3,kick>=3" --tries 2
```

`site/shelf.py` loads a brain in its own process, with the play settings of its entry in
`play-reading.json`, and writes whole tracks to the shelf the Auto DJ reads. A track that misses
`--require` (whole-track means per bar) is kept in `rejected/` below the brain's folder, which the
Auto DJ does not read. 304 bars is what the Auto DJ asks for.

Count what was composed before you keep it:

```bash
python tools/track_summary.py site/tracks/<brain>/*.json   # per segment: kick, snare, hats, bass, lead per bar; repeats; held bars
python tools/track_summary.py site/tracks/<brain>/*.json | python tools/track_means.py   # one line per track
python tools/kick_patterns.py site/tracks/<brain>/*.json   # the kick patterns a track plays, and their runs
```

Sixty-four bars hide what a whole track shows: a frozen kick pattern, hats that drop out, a line
bar held for a hundred bars.

## Training and checks

```bash
python brain/layouts.py                    # the slate of layouts
python brain/check.py inst_reads_echo2     # build and bench of one layout, written to quality/checks/
python brain/train.py --help
```

`brain/check.py` composes a layout at full size and teaches a small copy of its wiring to hold a
bar. Its full-size stage and `brain/train.py` need a score and a pool. The command of a full run is
in the [README](../README.md#quick-start); the settings and their reasons are in
[brains.md](brains.md#6-lessons). A run writes its curve and checkpoints to `--out`; the player
lists the runs it finds under `quality/composer/<name>/`.

## Tests

| Test | Run | Needs |
| --- | --- | --- |
| The Auto DJ's planner | `node site/test_autodj.mjs` | nothing |
| The engine | `node engine/checks.mjs --no-fit` | a pool file (`pool/pool_v0.json`), not shipped |
| The brain's wiring | `cd brain && python -m unittest test_hierarchy` | a pool file (`pool/pool_underground.json`), not shipped |
| The composer checkpoint | `cd brain && python -m unittest test_composer` | the Cadence library |
| The player's requests | `cd site && python -m unittest test_playback test_stats_live` | the Cadence library |
| The page in a browser | `python site/e2e_test.py 8793` | Playwright, and a test instance on that port with a brain |

## Hosting without a server

Live composing needs the Python server: a brain in memory (about 1.8 GB) that settles one bar in
about 2 seconds on a laptop. The rest runs in the browser. The engine, the mixer and the Auto DJ
are JavaScript and only fetch tracks and the pool, so a static page can play tracks composed
ahead. A 304-bar track is about 3 MB of JSON, about 0.5 MB gzipped. Such a static page is not in
the repository yet.
