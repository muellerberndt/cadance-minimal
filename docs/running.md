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
| The brain's wiring | `cd brain && python -m unittest test_hierarchy` | the Cadence library; without a fitted pool the tests take a synthetic one of the same shape, and the three SID tests skip (they need `pool/pool_underground_sid.json`, not shipped) |
| The composer checkpoint | `cd brain && python -m unittest test_composer` | the Cadence library |
| The player's requests | `cd site && python -m unittest test_playback test_stats_live` | the Cadence library |
| The page in a browser | `python site/e2e_test.py 8793` | Playwright, and a test instance on that port with a brain |

## The radio site

[cadance.floatingpragma.ai](https://cadance.floatingpragma.ai) is a static page. In radio mode its
Auto DJ mixes whole tracks that Cadance brains composed ahead. The tracks are MP3 files, rendered
by the engine with the player's mix settings and picked by a weighted draw; the visuals follow the
audio and the tracks' note files. No brain runs anywhere while the radio plays, and the settings
panel says so ("Cadance brain offline, radio mode active").

The visuals menu has three groups: brain (wired brain, MRI, connectome), instruments (devices,
mixer) and art (Amiga, OPH screen). A link can name the visual the page starts with:
`https://cadance.floatingpragma.ai/?visuals=glorb` opens the OPH screen. The other names are
`brain`, `mri`, `connectome`, `devices`, `mixer` and `amiga`; an unknown name starts with the
visitor's last visual.

Live composing is a different mode: a brain writes the next bar while you listen. That needs a
server.

## Rating the radio's tracks

`python tools/track_viewer.py` opens a list of the store's tracks on your own computer
(http://127.0.0.1:8795/): listen, jump by 32 bars, give 1 to 5 stars or a veto. A track nobody
rated counts as 3 stars. The ratings are kept in a file on that computer, and a quarter of a minute after the last change the viewer writes the
store's `tracks.json` again: vetoed tracks leave the list, and the others carry `stars` and a
`weight` for the radio's random pick. Publishing uses the AWS command line of the computer, so
only someone who may write the store can change what the radio plays.

## A live composer for the radio site

A live composer is this repository's player server with a brain loaded. The radio page talks to it
when you name its address in the page's settings (the gear icon, "Live brain (server URL)").

What it takes:

1. Python 3.11 or later and the library: `python -m pip install "cadence-net[fast]==0.104.1" numpy`.
2. A trained brain (its card and weights in `site/pinned/`), the sound pool its card names (in
   `pool/`) and the score it names (in `corpus/`, for starting passages). None of the three is
   shipped. [brains.md](brains.md) says how the brains were built and trained and
   [score.md](score.md) how a score is made.
3. About 2 GB of memory for the brain. A bar takes 2 to 2.5 seconds on a laptop CPU, longer than it
   lasts at the radio's 145 bpm (1.66 s). The Auto DJ then plays the brain's kept tracks (and the
   stored ones), and a track being composed comes in once enough of it lies ahead.
4. `python site/serve.py`. The server listens on 127.0.0.1, port 8792.
5. On the radio page: the gear icon, the address `http://127.0.0.1:8792`, save. The settings then
   read "Live brain at http://127.0.0.1:8792: the set takes its tracks from it". The Auto DJ takes
   its tracks from the server's track factory: the tracks the brain composes, its kept tracks and its
   repertoire, played in the browser by the instrument (`engine/engine.mjs`) on the radio's bar clock
   beside the stored tracks. When the server stops answering, the set goes on with the stored tracks
   from the next track on, and the page asks the server again every 30 seconds.

The server answers the radio site's origin and local pages (`CADANCE_CORS_ORIGINS` adds origins,
comma separated). A server on this computer: the browser asks once for access to the local network
(Chrome's local network access); allow it. A server on another machine must be reached over https;
the plain way is a tunnel that gives it an https address. It has no access control of its own: put
it behind something that has.

## Hosting without a server

Live composing needs the Python server: a brain in memory that settles one bar in about 2 seconds
on a laptop. The rest runs in the browser, which is what the radio site uses.
