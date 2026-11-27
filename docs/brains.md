# How the brains were constructed

This page describes the brains of Cadance as the code builds them and as they were measured. The
general lessons, with runnable toy examples, are in
[the Cadence tutorial](https://github.com/muellerberndt/cadence/blob/main/docs/tutorials/minimal-techno.md).
This page is the concrete case: the sizes, the settings and the numbers.

All numbers are readings of this one application on its own material, mostly from single runs.

## 1. The material and the score

The brains learned from a private collection of recordings in the reference style, transcribed
into a symbolic score. Neither the recordings nor the score are in this repository. The score
holds, per recording, a list of bars; a bar says which kit and which voices sound, the drum hits on
16 steps, the notes of three synth lines on 16 steps, and the settings of the sounds. Where the
transcription could not read a part, the score marks it, and a lesson leaves those slots open.

A track of the score may carry a style id and a material group. Where a function prefers one
(the passages free runs start from, the pace the play temperatures are calibrated to), it prefers
`common.REFERENCE_STYLE` and `common.REFERENCE_GROUP`, two settings of the computer the code runs
on (`CADANCE_REFERENCE_STYLE`, `CADANCE_REFERENCE_GROUP`; default `"reference"`). Recordings are
held out by whole recording: the errors below are read on recordings the brain was never taught.

## 2. A bar is 213 motor slots

The brain answers a whole bar in one settle. Each slot is a group of motor neurons of which one
wins (`slots=` of `Brain.compose`). The shape is set by `common.configure` and stored in every
brain's card.

| Group | Slots | Values per slot | Content |
| --- | --- | --- | --- |
| Kit | 1 | 65 | silence or one of 64 drum kits |
| Voices | 3 | 2 | whether the bass, the lead and the air sound |
| Mix controls | 8 | 7 | a filter and a space level for each of the four layers |
| Drum steps | 48 | 3 | kick, snare and hats on 16 steps: rest, hit, accent |
| Line steps | 48 | 14 | bass, lead and air on 16 steps: rest, tie, or one of 12 pitch classes |
| Sound controls | 105 | 2 to 13 | 35 controls per synth layer: oscillator, filter, envelope, mix |
| Total | 213 | 1,720 motor neurons | |

The kit slot and the three voice slots are the arrangement. Errors are reported in four groups:
arrangement, drum rows, synth lines and sound controls, and "balanced" is the mean of the four.

## 3. The observation

The observation of a bar is the bar itself, one sensory neuron per slot value (1,720), and eight
more columns: two for the mood the listener asks for (energy and brightness) and six for the bar
clock, the count of the bar being written in binary. A voice that does not sound is not heard.

The echo brains also hear earlier bars (`common.ECHO_LAGS = [1, 2, 4, 8, 16]`,
`common.lagged_observation`). Counted back from the bar being written, lag 1 is the bar just heard;
the bars 2, 4, 8 and 16 back follow as four more blocks of 1,720 columns, all zero where the stream
is younger than the lag. The whole observation is 8,608 columns wide. The trainer wraps a brain in
`common.EchoBrain`, which keeps the bars it saw acted on; the player builds the same observation
from the bars it played.

## 4. The layouts

Every brain is one apical Cadence brain built from a custom genome (`hierarchy.genome`,
`hierarchy.compose`). The settings shared by the three layouts that were kept:

- A learned code of 1,792 units reads the whole observation (`learned_code`), with attention.
- Apical feedback at 0.25 and `teaching="deep"`.
- Each pathway has a detail region and a plan region. Detail and plan exchange signals, the plan
  exchanges with the coordinator, the detail writes the pathway's own motor slots, and the
  coordinator drives each detail one way. Each detail also hears the learned code directly. The
  pathways exchange information only through the coordinator.
- The coordinator writes the arrangement slots itself.
- Four recall heads read at the coordinator (`recall_heads=4`, unit keys, recall temperature 0.25,
  128 slots). There is no route from the memory to the motor slots (`recall_tokens=None`,
  `recall_gain=0`): a remembered bar can inform a plan and cannot write a slot.
- `learning_drive_budget=4.0` and `learning_credit_window=15`.

| Layout (name in `layouts.py`) | Pathways | Detail / plan widths | Coordinator | Plastic synapses |
| --- | --- | --- | --- | --- |
| QUAD (`inst_reads_echo`) | beat, bass, lead, air | 240 / 96 each | 64 | 3,955,480 |
| Deeper QUAD (`inst_reads_echo2`) | beat, bass, lead, air | 240 / 96 each | 64, under an upper coordinator of 64 | 4,183,832 |
| DUO (`split_echo`) | rhythm, melody | 280 / 192 and 240 / 192 | 64 | 3,166,840 |

Who writes which slots (`hierarchy.owners`): the beat pathway its two mix controls and the 48 drum
steps; each synth pathway its two mix controls, its 16 line steps and its 35 sound controls. In the
DUO the rhythm pathway takes the beat's slots and the melody pathway everything of the three synths.

In the deeper QUAD the upper coordinator holds the recall heads, lies before the phrase coordinator
in the graph, and writes the arrangement beside it. It also has a return from the pathways' motor
slots, born at a tenth of a projection's scale (`upper_return=0.1`). Both choices answer a failure
of an earlier form in a small test chamber. Laid out behind the phrase coordinator, its cue reached
that coordinator only as apical modulation. With the phrase coordinator as its only way back from
the taught slots, 81 to 90% of its units sat below rest and it took 0.02% of the motor's teaching
contrast.

Before a layout is trained it passes `brain/check.py`: the brain composes at full size and saves
and loads, and a small copy of the wiring learns to hold a bar it heard.

## 5. The copy arcs and the echo arcs

A brain whose only route from the heard bar to the motor runs through hidden regions did not learn
to hold (section 10). The kept layouts have a direct one: for every motor slot a one-way projection
from the slot's sensory columns to the slot's motor neurons (`reflex=True`; Cadence 0.104.1 admits
a one-way sensory projection beside a learned code).

The arcs are born as a copy (`reflex_birth="diagonal"`). After the connectome is developed, the
synapse from heard value v of a slot to motor value v of the same slot is set to a gain and the rest
of the slot's block to zero. All of it stays plastic.

| Arc | Birth gain | Slots |
| --- | --- | --- |
| The bar heard (lag 1) | 2.0 | all 213 |
| The kit slot at lag 1 | 4.0 | the one slot of 65 values; at 2.0 the kit was wrong on 31% of staying bars at birth |
| The bars 2, 4, 8 and 16 back | 1.0 each | the 96 step slots (drum steps and line steps); the other blocks start at zero |

The lags are the ones the material returns to: of the slots that change from one bar to the next,
85% in the drum rows and 73% in the lines take a value the same slot held 2, 4, 8 or 16 bars
earlier.

The projections are placed slot by slot through small populations. Those populations are dropped
from the connectome after development: kept, they cut the graph into more than 256 blocks and the
library lays it out as one dense block (Cadence 0.104.1 warns about that fallback).

At birth, before any lesson, the deeper QUAD reads 19.5% on the held-out recordings (staying slots
8.0% wrong, changing slots 77.3%), against 17.1% for repeating the bar heard and 30.3% for always
writing the commonest bar. The vote of the further lags costs some staying slots at birth and
answers some of the changes.

## 6. Lessons

`brain/train.py` teaches on the score. A window of 96 bars of one recording is one continuing
stream; 16 windows run side by side as a batch, and windows start every 8 bars (2,785 windows over
37,671 training bars). For every bar the brain is taught the score's next bar and then acts on the
present bar, so the recall memory holds the moment as an act leaves it.

| Setting | Value | Why |
| --- | --- | --- |
| Step | 0.003, each synapse's step divided by a running RMS of its updates (forgetting 0.99) | the composed learner's own |
| `learning_surprise` with `learning_surprise_slots` | on | only the slots answered wrong are pulled; with whole rows 98.8% of rows were taught and the copy wore away |
| `learning_surprise_share` | on | the step follows the share of slots taught |
| Returns only (`--teach-returns`) | on | a changed slot whose new value none of the bars 2, 4, 8, 16 back holds is left open |
| Open labels (`--mask-unheard-targets`, the score's marks) | on | the controls of a voice that does not sound, and slots the transcription did not read, are not taught |
| `learning_drive_budget` | 4.0 | without it the coordinator saturated within the first pass |

Returns only is a choice about what the brain should do. Taught every wrong slot, a brain kept
adding notes in free play. Taught returns only, it does not learn to introduce new material: new
values arrive through sampling at play.

A run trained on a GPU at 6 to 7 bars per second. The first three brains that were kept had 62 to
71 minutes of lessons.

## 7. How a brain is read

Every reading is on held-out recordings (`brain/gate.py`).

- **Next-bar error.** The share of wrong slots when the brain hears the score's own bars
  (`prediction_error`), over the four groups, and split into slots that stay as in the bar before
  and slots that change.
- **Two baselines.** Repeating the bar heard (17.1%) and always writing the commonest bar (30.3%).
- **Rollout.** The brain hears 8 bars of the score, then plays 16 bars hearing its own answers,
  read greedily and scored against the score's continuation (`rollouts`).
- **Cuts.** The same error on a copy whose coordinator's outgoing influence is muted
  (`common.mute_coordination`), whose upper coordinator is muted (`common.mute_upper`), or whose
  arcs are muted (`common.mute_reflex`). A cut that costs nothing says the part is not shown to
  contribute.
- **Free-play density.** From 8 held-out bars the brain plays 64 bars sampled at the player's
  temperature, on four passages (`density_rollout`). Per segment (bars 1 to 8, 9 to 16, 17 to 32,
  33 to 64) it counts kick steps, drum hits, bass notes, lead notes and sounding voices per bar
  beside the same counts of the recording's own continuation. The density ratio is drum hits plus
  bass plus lead notes, brain over material. A reading on two passages of 32 bars undercounted it
  (1.38 against 1.63 for the same brain).
- **Change rates.** Per bar, the slots that changed against the bar before, by group, split into
  returns and new values (`change_rates`).
- **Whole tracks.** Tracks of 304 bars composed with the player's settings, counted with `tools/`.

A training run stops when the last segment's density passes 1.5 times the material's. The lesson
error does not show that point: it kept falling while free play turned to overplaying.

## 8. The brains that were kept

| Brain | Layout | Minutes of lessons | Error | Staying slots wrong | Rollout | Coordinator cut | Arcs cut |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `slate_inst_reads_s2_echo2_e001b002` (the Auto DJ's default) | deeper QUAD | 62 | 13.9% | 2.6% | 21.8% | 13.6% | 44.4% |
| `slate_inst_reads_s2_echor_e001b001` | QUAD | 63 | 13.4% | 1.9% | 21.7% | 13.1% | 50.4% |
| `slate_split_s2_echor_e001b003` | DUO | 71 | 13.1% | 1.4% | 21.6% | 13.1% | 57.7% |
| `slate_inst_reads_s2_echo2q_e001b008` (the alternative) | deeper QUAD | 150 | 13.4% | 1.7% | 21.5% | 13.5% | 44.1% |

Repeating the bar heard reads 17.1% on the same recordings. By group, the default reads 6.9% on the
arrangement, 9.4% on the drum rows, 22.0% on the synth lines and 17.2% on the sound controls. About
three quarters of the changing slots are still answered wrong (74.8% for the default).

What the cuts say: the arcs carry the brain. Muting the coordinator leaves the error where it was
or a little lower in all four, and muting the upper coordinator of the deeper QUAD lowers it (13.1%
against 13.9%). The hierarchy and the recall heads are part of the brains that were measured; their
contribution is not shown.

The alternative is the default after 90 more minutes at a quarter of the step. It predicts better
and plays denser. Free-play density by segment, without the player's caps:

| Brain | Bars 1 to 8 | 9 to 16 | 17 to 32 | 33 to 64 |
| --- | --- | --- | --- | --- |
| Default | 1.30 | 0.99 | 1.05 | 0.98 |
| Alternative | 1.85 | 1.60 | 1.69 | 1.63 |

The alternative is over the stop line of 1.5 (its bass about twice the material's), which is why it
is not the default. With the player's caps its whole tracks sit at 6.4 to 6.5 bass and 6.3 to 6.4
lead notes per bar, and its drums repeat a whole bar less often than the default's (17 and 22% of
bars on two whole tracks, where the default's tracks repeat 36 to 48%).

## 9. Play settings

The player reads a brain sampled, with the settings of `site/play-reading.json`. Shared settings
come first; an entry under `"brains"` replaces them for the brains whose name starts with its key.

| Setting | Value | What it does |
| --- | --- | --- |
| `temperature` | 0.11 | the sampling temperature of every slot. Read greedily, every trained brain froze into a loop of a few bars |
| `line_scale` | 2.0 | the drum rows and the synth lines are read at the temperature times this |
| `synth_line_scale` | 1.0 (DUO 0.65) | the synth lines alone, on top of `line_scale` |
| `kick_scale` | 0.5 | the kick row alone: at 0.5 it held, at 0.65 it broke out late in a track |
| `entry_scale` | 2.0 | whether a silent voice comes back is read at the temperature times this |
| `voice_dwell` | 8 bars per voice | a voice keeps a major change for that long before the next |
| `control_scale`, `loudness_scale` | 0.6, 0.3 | the sound controls, and those that set loudness |
| `row_caps` | 4, 4, 13 | the most kick, snare and hat hits per bar |
| `line_caps` | 7, 6, 1 (DUO bass 9) | the most bass, lead and air notes per bar |
| `cap_mode` | `"thin"` | a row or line over its cap is thinned in place |

The caps are the 75th percentile of the material's bars. A cap above that is where the row settles:
with a kick cap of 6 the kick sat at 5 to 6 per bar late in a track, with 4 it held four.

Thinning keeps what the brain wrote and drops hits down to the cap: the kick keeps its quarter
notes longest, snare and hats lose plain hits before accents and off-beat steps first, a dropped
line note becomes a tie. The brain hears the thinned bar. The first form of the caps replaced a
busy row by a calm copy of an earlier bar, and the brain then heard its own copy (section 10).

The player composes at full precision. Settling to a looser tolerance at play froze the lead in 5
of 6 test tracks.

## 10. What did not work

- **No copy path.** The first layouts of this design, without arcs, read 28 to 30% for an hour of
  lessons with 20 to 28% of the staying slots wrong. An earlier generation had copied through a
  route from the memory to the motor, which leaves the hidden regions without work.
- **Arcs born at random, and arcs from the learned code.** The same arcs born small and random
  left a quarter of the staying slots wrong after their first lessons. Arcs from the learned code
  onto the motor stayed at 28%, and cutting that brain's coordinator changed nothing.
- **Whole-row surprise.** With 213 slots a row is almost never right in all of them: 98.8% of rows
  were taught and every taught row pulled all its slots. A brain born at the level of repeating
  read 17.6% after its first batch and 22.5% after 38 minutes.
- **Teaching every wrong slot.** Free play piled up notes: 9 to 10 kicks per bar where the material
  has 2.9 to 3.8, although its greedy answers were 96% right on staying slots. Returns only was
  ahead after the first batch (13.2% against 13.8%, staying slots 1.2% against 2.1% wrong).
- **A sampled judge for surprise lessons.** Judging a row by a sampled answer in place of the
  greedy one overshot within six passes. It is not used and not in the library.
- **Credit transport.** `learning_credit_transport` ran for twenty minutes with lessons equal, bit
  for bit, to the run without it: in float32 at this size its reference solve never qualified.
  Cadence 0.104.1 reports that.
- **More lessons at the same step.** Past the first hour the staying-slot error rose, the kit
  drifted and free-play density climbed to 1.6 to 1.97 times the material's.
- **A quarter of the step.** Three continuations at 0.00075 improved the error by a few tenths for
  about two hours, and all three ended past the density line (1.67 at 147 minutes, 1.68 at 191,
  1.55 at 197), two of them with 9 to 10 kicks per bar.
- **The calm-copy cap.** Whole 304-bar tracks froze: the drums equalled the previous bar in 68 to
  84% of bars and one bass bar was held for 122 bars. On one 128-bar passage, thinning against the
  calm copy: drums equal to the previous bar 31% against 61%, different drum bars per 16 bars 10.1
  against 5.5, longest held bass and lead bar 10 and 9 against 27 and 32.

## 11. What is not shown

- **The hierarchy's contribution.** See the cuts in section 8.
- **Arrangement.** Voices enter and leave by the player's settings (`entry_scale`, `voice_dwell`),
  not by a plan of the brain. Next-bar error and a self-fed rollout do not establish phrases or
  the form of a track.
- **Hats over a whole track.** On whole tracks the hats drop out for 30 to 120 bars at a time. The
  DUO's four whole tracks average 1.0 to 3.6 hats per bar.
- **The kick pattern.** A first-hour brain holds the kick pattern of its starting passage for 100
  to 300 bars, whatever that pattern is. The alternative varies it in runs of 12 to 30 bars.
- **Thin openings.** Of 13 test tracks of 64 bars in thinning mode, counted against the
  recordings, 3 pass. No brain passes on both a full and a thin starting passage; on the thin one
  the default's lead stays at 2.0 to 2.5 notes per bar for the first 16 bars.
- **New material.** New values arrive through sampling alone.
- **Sound.** The counts say how close free play is to the material in density and change. Whether
  a track sounds good is judged by ear.
