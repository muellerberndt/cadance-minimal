"""Quality and diversity report for the newest checkpoint of every candidate (runs on the CPU box, overnight).

Each brain plays three long free runs (256 bars) from three passages of the corpus it learned from, with the
site's coherent sampler at its calibrated temperature. Measured against the same statistics of the corpus:
holds and changes per layer, phrase alignment of new sounds, the beat (kick, four-on-the-floor, snare on 2+4,
variation), synth line variety, diversity (distinct sounds, entropy, difference between runs, repeated loops),
copying (8-bar stretches found verbatim in the training data), silence, and mood following. The best brain of
each round gets a 2-minute render whose audio is measured too.

    python3 brain/quality.py --cards ~/minimal/quality/cards --out ~/minimal/quality [--render]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common  # noqa: E402
from common import L, LAYERS, observe  # noqa: E402

CODE = HERE.parent
CORPUS = Path.home() / "minimal" / "corpus_out"


def configure(card):
    common.ONOFF = bool(card.get("onoff"))
    common.shape_from(card)
    common.configure(card["tokens_per_layer"], card["mod_levels"], steps=card.get("steps", False), styles=card.get("styles"),
                     rows=card.get("beat_rows"), clock=card.get("clock", False), synth=card.get("synth_rows"), patch_book=card.get("patch_book"))
    path = CODE / "pool" / card.get("pool", "pool.json")
    if card.get("streams_sha"):
        frozen = CORPUS / card["streams_sha"] / path.name
        if frozen.is_file():
            path = frozen
    raw = path.read_bytes()
    if card.get("streams_sha") and hashlib.sha256(raw).hexdigest()[:12] != card.get("pool_sha"):
        raise ValueError("QA instrument differs from the training checkpoint")
    pool = json.loads(raw)
    common.set_content(pool if card.get("content") else None)
    return pool


def streams_path(card):
    name = card.get("streams") or "streams.json"
    if card.get("streams_sha"):
        for path in (CORPUS / card["streams_sha"] / name, CODE / "corpus" / name, CORPUS / name):
            if path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest() == card["streams_sha"]:
                return path
        raise ValueError("QA needs the training score whose digest is in the checkpoint")
    p = CORPUS / name
    if not p.exists() and (CODE / "corpus" / name).is_file():
        return CODE / "corpus" / name
    if not p.exists():                                  # derived files: denoised drums of a base file
        base = CORPUS / name.replace("_smooth", "").replace("_synth", "")
        if base.exists():
            subprocess.run([sys.executable, str(HERE / "denoise_drums.py"), str(base), str(p)], check=True)
    return p


def sample_bar(brain, x, state, temperature, rng):
    """The site's coherent sampler (common.sample_slots); temperature: one number or one per slot."""
    common.answer(brain, x)
    return common.sample_slots(common.motor_activity(brain)[0], state, temperature, rng)


def passages(tracks, k, n, seed):
    """Passages in full flow with a simple beat, the reference style first (common.flow_passages)."""
    return common.flow_passages(tracks, k, n, seed)


def free_runs(brain, tracks, temperature, n_bars=256, k=3, seed_bars=8, seed=11):
    runs = []
    for j, (tr, s) in enumerate(passages(tracks, k, seed_bars, seed)):
        rng = np.random.default_rng(seed + j)
        brain.reset()
        b, m = tr["bars"], tr["mood"]
        for t in range(s, s + seed_bars - 1):
            common.answer(brain, observe(b[t:t + 1], m[t + 1:t + 2]))
        cur = b[s + seed_bars - 1].copy()
        pos = s + seed_bars
        path = np.clip(0.5 + 0.35 * np.sin(np.linspace(0, 3 * np.pi, n_bars)), 0, 1)   # mood path: energy rises and falls
        out = []
        for k2 in range(n_bars):
            mood = common.mood_rows([[path[k2], 0.5]], tr.get("style"), bars=[pos])
            cur = sample_bar(brain, observe(cur[None], mood), cur, temperature, rng)
            out.append(cur.copy()); pos += 1
        runs.append({"track": tr["title"], "start": int(s), "bars": np.asarray(out), "energy": path})
    return runs


def breakdown_passages(tracks, k=6, seed_bars=8, min_len=16, seed=5):
    """Kickless stretches INSIDE corpus tracks (at least min_len bars, the kick comes back after them), the reference
    style first: (track, start of the prime = start of the stretch, bars the track itself waits after the prime)."""
    found = []
    for tr in tracks:
        b = np.asarray(tr["bars"]); n = len(b)
        if n < 64:
            continue
        snd = common.sounds(b)
        kick = np.array([(common.decode_beat(x)[0] > 0).any() and snd[i, 0] > 0 for i, x in enumerate(b)])
        i = 1
        while i < n:
            if not kick[i] and kick[i - 1]:
                j = i
                while j < n and not kick[j]:
                    j += 1
                if j < n and j - i >= min_len and kick[j:j + 4].sum() >= 3:
                    found.append((tr, i, j - (i + seed_bars)))
                i = j
            else:
                i += 1
    rng = np.random.default_rng(seed)
    pk = [f for f in found if f[0].get("style") == common.REFERENCE_STYLE]
    rest = [f for f in found if f[0].get("style") != common.REFERENCE_STYLE]
    pick = [pk[i] for i in rng.permutation(len(pk))[:k]]
    pick += [rest[i] for i in rng.permutation(len(rest))[:k - len(pick)]]
    return pick


def breakdown_test(brain, tracks, temperature, k=6, seed_bars=8, slack=16, seed=7):
    """Does the kick come back after a breakdown? Prime with 8 kickless bars from inside a corpus breakdown and let
    the brain play on: it passes when a sustained kick (3 of 4 bars) returns within the track's own wait + slack
    bars (at least 32 + slack). The failure behind the 11-23 false flag: primed kickless, a brain stayed kickless."""
    res = []
    for j, (tr, s, wait) in enumerate(breakdown_passages(tracks, k, seed_bars)):
        rng = np.random.default_rng(seed + j)
        brain.reset()
        b, m = tr["bars"], tr["mood"]
        for t in range(s, s + seed_bars - 1):
            common.answer(brain, observe(b[t:t + 1], m[t + 1:t + 2]))
        cur = np.asarray(b[s + seed_bars - 1]).copy(); pos = s + seed_bars
        limit = max(wait, 32) + slack
        kicks, back = [], None
        for t in range(limit + 4):
            cur = sample_bar(brain, observe(cur[None], common.mood_rows([[0.5, 0.5]], tr.get("style"), bars=[pos])), cur, temperature, rng)
            pos += 1
            kicks.append(bool((common.decode_beat(cur)[0] > 0).any() and common.sounds(cur)[0] > 0))
            if len(kicks) >= 4 and sum(kicks[-4:]) >= 3:
                back = len(kicks) - 4
                break
        res.append({"track": tr["title"], "start": int(s), "corpus_wait": int(wait), "returned_after": back,
                    "pass": back is not None and back <= limit})
    return res


def entropy(values):
    _, c = np.unique(values, return_counts=True)
    p = c / c.sum()
    return float(-(p * np.log2(p)).sum())


def stats(bars, offset=0):
    """Musical statistics of one state stream (bars, slots)."""
    tok = common.sound_identities(bars)
    bars = common.external(bars)
    n = len(bars)
    out = {}
    ch = tok[1:] != tok[:-1]
    for l, name in enumerate(LAYERS):
        col = tok[:, l]
        runs, r = [], 1
        for i in range(1, n):
            if col[i] == col[i - 1]:
                r += 1
            else:
                runs.append(r); r = 1
        runs.append(r)
        out[name] = {"change": float(ch[:, l].mean()), "hold": float(np.mean(runs)), "distinct": int(len(set(col[col > 0]))),
                     "entropy": entropy(col), "silence": float((col == 0).mean())}
    nov = [t for t in range(4, n) if any(tok[t, l] not in tok[t - 4:t, l] for l in range(L))]
    out["novel_per_100"] = 100.0 * len(nov) / n
    out["novel_on_8"] = float(np.mean([(t + offset) % 8 == 0 for t in nov])) if nov else 0.0
    still, longest = 0, 0
    for i in range(1, n):
        still = still + 1 if (bars[i] == bars[i - 1]).all() else 0
        longest = max(longest, still)
    out["longest_frozen"] = int(longest)
    blocks = [tuple(map(tuple, tok[i:i + 4])) for i in range(0, n - 3, 4)]
    out["distinct_4bar_blocks"] = float(len(set(blocks)) / max(len(blocks), 1))
    out["all_silent"] = float((tok == 0).all(axis=1).mean())
    if common.STEPS:
        d = np.stack([common.decode_beat(s) for s in bars])
        on = (d.sum(axis=(1, 2)) > 0) & (tok[:, 0] > 0)
        out["drums"] = {"kick_bars": float(((d[:, 0] > 0).any(axis=1) & on).mean()),
                       "four_floor": float(((d[:, 0, [0, 4, 8, 12]] > 0).all(axis=1) & on).sum() / max(on.sum(), 1)),
                       "snare_2_4": float(((d[:, 1, [4, 12]] > 0).all(axis=1) & on).sum() / max(on.sum(), 1)),
                       "change": float((d[1:] != d[:-1]).any(axis=(1, 2)).mean()),
                       "distinct_beats": int(len({d[i].tobytes() for i in range(n) if on[i]})),
                       "kick_quarters": float(((d[:, 0, [0, 4, 8, 12]] > 0).sum(axis=1) >= 2)[on].mean()) if on.any() else 0.0,
                       "busy_snare": float(((d[:, 1] > 0).sum(axis=1) > 8)[on].mean()) if on.any() else 0.0,
                       "busy_hats": float(((d[:, 2] > 0).sum(axis=1) > 12)[on].mean()) if on.any() else 0.0,
                       "hat_bars": float((d[:, 2] > 0).any(axis=1)[on].mean()) if on.any() else 0.0,
                       "snare_bars": float((d[:, 1] > 0).any(axis=1)[on].mean()) if on.any() else 0.0,
                       "more_than_kick": float(((d[:, 1] > 0).any(axis=1) | (d[:, 2] > 0).any(axis=1))[on].mean()) if on.any() else 0.0,
                       "longest_kickless": int(max((len(r) for r in "".join("k" if x else "_" for x in ((d[:, 0] > 0).any(axis=1) & (tok[:, 0] > 0))).split("k")), default=0))}
    if common.MODS:                                                    # FILTER/SPACE per layer, -1..1 (0 = the sound as fitted)
        mo = np.array([common.mods_out(b) for b in bars])                # bars x 4 layers x [filter, space]
        out["mods"] = {**{f"{name}.filter": float(mo[:, l, 0].mean()) for l, name in enumerate(LAYERS)},
                       **{f"{name}.space": float(mo[:, l, 1].mean()) for l, name in enumerate(LAYERS)},
                       "drum_highpass": float((mo[:, 0, 0] >= 0.6).mean()),       # drum bus highpass >= ~65 Hz: the kick thins out
                       "drum_lowpass": float((mo[:, 0, 0] <= -0.6).mean())}
    if common.SYNTH:                                                   # per layer from the decoded note rows (any synth code)
        rows = np.stack([common.decode_synth(x) for x in bars]) * (tok[:, 1:, None] > 0)          # bars x 3 x 16
        out["synth"] = {}
        for i, name in enumerate(common.SYNTH_LAYERS):
            r = rows[:, i]; on = r >= 2; keys = [x.tobytes() for x in r]
            snd = on.any(axis=1)
            pcs = [len(set((x[x >= 2] - 2).tolist())) for x in r[snd]]
            out["synth"][name] = {"distinct": int(len(set(keys))), "repeat": float(np.mean([a == b for a, b in zip(keys, keys[1:])])) if len(keys) > 1 else 0.0,
                                  "onsets": float(on.sum(axis=1)[snd].mean()) if snd.any() else 0.0,
                                  "pitch_classes": float(np.mean(pcs)) if pcs else 0.0, "sounding": float(snd.mean())}
    return out


def music_keys(bars):
    """Per-bar identities for the copying check. DSP gates alone discard the melody."""
    snd = common.sounds(bars)
    if not common.PATCH_BOOK:
        return [row.tobytes() for row in snd]
    keys = []
    for state, active in zip(bars, snd > 0):
        beat = common.decode_beat(state) * active[0]
        notes = common.decode_synth(state) * active[1:, None]
        keys.append(np.concatenate((beat, notes)).astype(np.uint8).tobytes())
    return keys


def copied_share(run_keys, corpus_windows, w=8):
    windows = [tuple(run_keys[i:i+w]) for i in range(len(run_keys) - w + 1)]
    changing = [row for row in windows if len(set(row)) > 1]
    return sum(row in corpus_windows for row in changing) / max(len(changing), 1)


def corpus_windows(tracks, w=8):
    out = set()
    for tr in tracks:
        if not tr["held_out"]:
            keys = music_keys(tr["bars"])
            out.update(tuple(keys[i:i+w]) for i in range(len(keys) - w + 1))
    return out


def corpus_stats(tracks):
    per = [stats(np.asarray(t["bars"])) for t in tracks if not t["held_out"] and len(t["bars"]) > 16]
    def mean(path):
        vals = []
        for p in per:
            v = p
            for k in path:
                v = v.get(k) if isinstance(v, dict) else None
            if isinstance(v, (int, float)):
                vals.append(v)
        return float(np.mean(vals)) if vals else None
    keys = [(n, f) for n in LAYERS for f in ("change", "hold", "distinct", "entropy", "silence")]
    ref = {f"{n}.{f}": mean((n, f)) for n, f in keys}
    for k in ("novel_per_100", "novel_on_8", "longest_frozen", "distinct_4bar_blocks", "all_silent"):
        ref[k] = mean((k,))
    for k in ("kick_bars", "four_floor", "snare_2_4", "change", "distinct_beats", "kick_quarters", "busy_snare", "busy_hats", "hat_bars", "snare_bars", "more_than_kick"):
        ref[f"drums.{k}"] = mean(("drums", k))
    for k in ("drum_highpass", "drum_lowpass") + tuple(f"{n}.{f}" for n in LAYERS for f in ("filter", "space")):
        ref[f"mods.{k}"] = mean(("mods", k))
    return ref


def flatten(s, prefix=""):
    out = {}
    for k, v in s.items():
        if isinstance(v, dict):
            out.update(flatten(v, f"{prefix}{k}."))
        else:
            out[f"{prefix}{k}"] = v
    return out


def render(card, pool, run, out_wav, bars=64):
    """A 2-minute render of a free run (sounds, FILTER/SPACE, the brain's beat and synth lines)."""
    held, previous = [None] * 3, [None] * L
    score_bars = [common.score_bar(st, pool, held, previous) for st in run["bars"][:bars]]
    score = {"pool": pool, "bpm": pool.get("bpm", 120), "bars": score_bars}
    sp = out_wav.with_suffix(".score.json"); sp.write_text(json.dumps(score))
    renderer = (CODE / "engine/render.mjs").resolve()
    r = subprocess.run([(shutil.which("node") or str(Path.home() / "node/bin/node")), str(renderer), str(sp), str(out_wav)], capture_output=True, text=True)
    return r.returncode == 0 and out_wav.is_file(), (r.stdout or r.stderr or "No WAV produced").strip()[-200:]


def audio_check(wav, bpm, bars=None):
    """Loudness, silence, and whether a kick is AUDIBLE: at each quarter note, the 40-120 Hz band must jump
    clearly (>= 6 dB over the band's level just before the quarter)."""
    import wave
    w = wave.open(str(wav)); n = w.getnframes(); sr = w.getframerate(); ch = w.getnchannels()
    x = np.frombuffer(w.readframes(n), dtype=np.int16).astype(float).reshape(-1, ch).mean(axis=1) / 32768
    seg = x[: len(x) // sr * sr].reshape(-1, sr)
    rms = np.sqrt((seg ** 2).mean(axis=1))
    hop = 256
    X = np.fft.rfft(x); f = np.fft.rfftfreq(len(x), 1 / sr)
    X[(f < 40) | (f > 120)] = 0                                      # the kick's band, 40-120 Hz
    band = np.fft.irfft(X, n=len(x))
    frames = band[: len(band) // hop * hop].reshape(-1, hop)
    db = 10 * np.log10((frames ** 2).mean(axis=1) + 1e-12)          # band energy per 5.8 ms frame
    q = 60.0 / bpm * sr / hop                                        # frames per quarter note
    hits = 0; total = 0
    end = len(db) - q if bars is None else min(len(db) - q, bars * 4 * q - 1)   # only the quarters inside the bars: the render's tail has none
    for i in np.arange(q, end, q):
        i = int(i); total += 1
        before = db[max(0, i - int(q * 0.4)):max(1, i - 2)].mean()
        after = db[i:i + max(3, int(q * 0.15))].max()
        hits += int(after - before >= 6.0)
    return {"rms": float(np.sqrt((x ** 2).mean())), "peak": float(np.abs(x).max()), "silent_seconds": int((rms < 0.005).sum()),
            "seconds": int(len(seg)), "kick_audible_quarters": round(hits / max(total, 1), 3)}


RENDER_TOP = 8                     # renders per round and pass (kick audibility needs the audio)


def flags(m, ref, render=None):
    """What a listener would hear as broken."""
    out = []
    if m.get("drums.kick_bars", 1) < 0.3:
        out.append("no kick in most bars")
    elif m.get("drums.kick_bars_min", 1) < 0.3:
        out.append("a run without kick")
    if m.get("drums.longest_kickless_max", 0) > 64:
        out.append("kickless > 64 bars")
    if m.get("breakdown.pass", 1) < 0.67:
        out.append("kick does not return after breakdowns")
    elif m.get("drums.kick_quarters", 1) < 0.5 * max(ref.get("drums.kick_quarters") or 0.3, 0.2):
        out.append("kick rarely on the quarters")
    if m.get("drums.busy_snare", 0) > 0.25:
        out.append("machine-gun snare")
    hp, ref_hp = m.get("mods.drum_highpass", 0), ref.get("mods.drum_highpass") or 0
    if hp > max(2 * ref_hp, ref_hp + 0.15):
        out.append("drums highpassed")
    if m.get("longest_frozen", 0) > 64:
        out.append("frozen > 64 bars")
    if m.get("copied", 0) > 0.5:
        out.append("copying")
    if m.get("all_silent", 0) > 0.1:
        out.append("silent bars")
    if render:
        if render.get("silent_seconds", 0) > 10:
            out.append("silent audio")
        kb = render.get("kick_bars")
        kb = m.get("drums.kick_bars", 0) if kb is None else kb
        if (kb or 0) >= 0.5 and (render.get("kick_audible_quarters") if render.get("kick_audible_quarters") is not None else 1) < 0.25:
            out.append("kick written but not audible")   # compares the render with the kicks of the SAME stretch
    return out


PENALTY = {"no kick in most bars": 0.5, "a run without kick": 0.4, "kickless > 64 bars": 0.3, "kick does not return after breakdowns": 0.5, "kick rarely on the quarters": 0.3, "kick written but not audible": 0.5,
           "silent audio": 1.0, "drums highpassed": 0.3, "machine-gun snare": 0.2, "frozen > 64 bars": 0.2, "copying": 0.3, "silent bars": 0.3}


def penalized(row):
    """The guide score minus what a listener would hear as broken (flags), so a flagged brain never ranks on top."""
    return round(row["guide"] - sum(PENALTY.get(f, 0.2) for f in row.get("flags") or []), 3)


def score(flat, ref):
    """A single guide number (higher = closer to the corpus, a beat, not copying); the owner's ears decide."""
    close = []
    for k, rv in ref.items():
        if k.startswith("mods."):                                        # reported and flagged, not scored (keeps scores comparable)
            continue
        v = flat.get(k)
        if v is None or rv is None:
            continue
        close.append(min(abs(v - rv) / (abs(rv) + 0.05), 2.0))
    s = 1.0 - (float(np.mean(close)) / 2.0 if close else 0.5)
    s += 0.3 * flat.get("drums.kick_bars", 0) - 0.5 * max(0.0, flat.get("copied", 0) - 0.3) - 0.5 * flat.get("all_silent", 0)
    return round(s, 3)


# THE QA PROTOCOL (owner 11-23: "a strict quality assurance protocol"). A checkpoint HOLDS A BEAT when every line of
# QA_BEAT holds; it PASSES when every line of QA_FULL holds too. Only these reach the web app's default (serve.py).
QA_BEAT = [
    ("kick in >= 70% of the bars of EVERY run", lambda m, rd: m.get("drums.kick_bars_min", 0) >= 0.7),
    ("kick on the quarters in >= 60% of beat bars", lambda m, rd: m.get("drums.kick_quarters", 0) >= 0.6),
    ("no kickless stretch over 32 bars in any run", lambda m, rd: m.get("drums.longest_kickless_max", 999) <= 32),
    ("machine-gun snare in <= 15% of bars", lambda m, rd: m.get("drums.busy_snare", 1) <= 0.15),
    ("never 64 identical bars", lambda m, rd: m.get("longest_frozen", 999) <= 64),
    ("all layers silent in <= 5% of bars", lambda m, rd: m.get("all_silent", 1) <= 0.05),
    ("more than a kick: hats or snare/clap in >= 50% of beat bars", lambda m, rd: m.get("drums.more_than_kick", 0) >= 0.5),
    ("kick AUDIBLE on >= 50% of quarters in the render, <= 10 s silence", lambda m, rd: bool(rd) and rd.get("ok") and rd.get("kick_audible_quarters", 0) >= 0.5 and rd.get("silent_seconds", 99) <= 10),
]
def _heard(rd, name, within):
    """In the render: the layer plays in at least a quarter of the bars and sits within `within` dB of the drums."""
    if not rd or not rd.get("layer_db"):
        return False
    lv, pl = rd["layer_db"], rd.get("layer_plays", {})
    return pl.get(name, 0) >= 0.25 and name in lv and "beat" in lv and lv[name] >= lv["beat"] - within


def _evolves(m, ref):
    """beat, bass and lead each change their sound between half and three times as often as in the tracks."""
    for n in ("beat", "bass", "lead"):
        r, c = (ref or {}).get(f"{n}.change"), m.get(f"{n}.change")
        if r is None or c is None or not (0.5 * r <= c <= 3.0 * r):
            return False
    return True


QA_FULL = QA_BEAT + [
    ("the bass is HEARD: plays in >= 25% of the rendered bars, within 14 dB of the drums", lambda m, rd: _heard(rd, "bass", 14)),
    ("a lead or air sound is HEARD: plays in >= 25% of the rendered bars, within 20 dB of the drums", lambda m, rd: _heard(rd, "lead", 20) or _heard(rd, "air", 20)),
    ("it EVOLVES: beat, bass and lead each change sounds between half and three times as often as the tracks", lambda m, rd: _evolves(m, m.get("_ref"))),
    ("it follows the mood pad: energy correlation >= 0.2", lambda m, rd: m.get("mood.energy_corr", 0) >= 0.2),
    ("kick comes back after >= 4 of 6 corpus breakdowns", lambda m, rd: m.get("breakdown.pass", 0) >= 0.67),
    ("<= 50% of changing 8-bar stretches copied from the corpus", lambda m, rd: m.get("copied", 1) <= 0.5),
]


def qa(row):
    m, rd = dict(row.get("metrics") or {}), row.get("render")
    m["_ref"] = row.get("corpus") or {}
    failed = [name for name, test in QA_FULL if not test(m, rd)]
    beat = not any(name in failed for name, _ in QA_BEAT)
    return {"beat": beat, "pass": not failed, "failed": failed}


_CACHE = {}


def eval_card(cj):
    """Free runs, breakdown test and statistics of one checkpoint (runs in a worker process)."""
    from cadence import Brain
    cj = Path(cj); name = cj.stem
    card = json.loads(cj.read_text())
    try:
        if card.get("patch_book"):
            import cadence
            if tuple(map(int, cadence.__version__.split("."))) < (0, 103, 0):
                raise ValueError("composer QA requires a post-audit Cadence installation")
        pool = configure(card)
        common.SYNTH_FILL = bool(card.get("synth_fill"))
        common.ENTRY_SCALE = card.get("entry_scale") if card.get("entry_scale") is not None else 2.0
        common.LINE_SCALE = card.get("line_scale") if card.get("line_scale") is not None else 1.2
        sp = streams_path(card)
        key = common.layout_key(card)
        if key not in _CACHE:
            tracks = common.load_streams(sp, steps=card.get("steps", False), rows=card.get("beat_rows"), clock=card.get("clock", False), synth=card.get("synth_rows"), patch_book=card.get("patch_book"))
            configure(card)
            _CACHE[key] = (tracks, corpus_stats(tracks), corpus_windows(tracks), common.corpus_reference(tracks)["token_mood"])
        else:
            configure(card)
        tracks, ref, win, tm = _CACHE[key]
        brain = Brain.load(cj.with_suffix(".npz"), backend="cpu")
        # every voice read at its own temperature (common.calibrate_slots), as the player reads it
        temp = common.slot_temperatures(brain, tracks, card, cache=cj.with_suffix(".slots.json"))
        runs = free_runs(brain, tracks, temp)
        per = []
        for r in runs:
            st = flatten(stats(r["bars"]))
            st["copied"] = copied_share(music_keys(r["bars"]), win)
            per.append(st)
        flat = {k: float(np.mean([p[k] for p in per if k in p])) for k in per[0]}
        if "drums.kick_bars" in per[0]:                            # the worst run counts too: an average hides a kickless run
            flat["drums.kick_bars_min"] = float(min(p["drums.kick_bars"] for p in per))
            flat["drums.longest_kickless_max"] = float(max(p["drums.longest_kickless"] for p in per))
        if common.STEPS:
            bd = breakdown_test(brain, tracks, temp)
            if bd:
                flat["breakdown.pass"] = float(np.mean([r["pass"] for r in bd]))
                flat["breakdown.n"] = float(len(bd))
                back = [r["returned_after"] for r in bd if r["returned_after"] is not None]
                flat["breakdown.median_return"] = float(np.median(back)) if back else -1.0
        corr = []                                                  # the energy of what it plays against the pad's path
        for r in runs:
            read = common.read_mood(np.asarray(r["bars"]), tm)[:, 0]
            if read.std() > 0 and np.asarray(r["energy"]).std() > 0:
                corr.append(float(np.corrcoef(np.asarray(r["energy"]), read)[0, 1]))
            else:
                corr.append(0.0)
        flat["mood.energy_corr"] = float(np.mean(corr))
        sets = [set(map(tuple, common.sounds(r["bars"]).tolist())) for r in runs]
        pairs = [len(a & b) / max(len(a | b), 1) for i, a in enumerate(sets) for b in sets[i + 1:]]
        flat["between_runs_overlap"] = float(np.mean(pairs)) if pairs else 0.0
        row = {"name": name, "candidate": card.get("candidate"), "epoch": card.get("epoch"), "streams": sp.name,
               "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "guide": score(flat, ref), "metrics": flat,
               "corpus": ref, "runs": [{"track": r["track"], "start": r["start"]} for r in runs]}
        row["_run0"] = {"track": runs[0]["track"], "start": runs[0]["start"], "bars": np.asarray(runs[0]["bars"]).tolist()}
        if common.STEPS:                                            # what the rendered stretch (run 0, 64 bars) itself writes
            excerpt = runs[0]["bars"][:max(64, int(np.ceil(pool.get("bpm", 120) / 2)))]
            d0 = np.stack([common.decode_beat(b) for b in excerpt]); t0 = common.sounds(excerpt)[:, 0]
            row["render_kick_bars"] = float(((d0[:, 0] > 0).any(axis=1) & (t0 > 0)).mean())
        row["flags"] = flags(flat, ref)
        row["score"] = penalized(row)
        row["qa"] = qa(row)
        return row
    except Exception as e:
        return {"name": name, "failed": f"{type(e).__name__}: {e}"}


def bar_levels(wav, bpm, bars):
    """Stereo RMS in dBFS for each bar of a rendered stem."""
    import wave
    with wave.open(str(wav)) as reader:
        frames_per_bar = reader.getframerate() * 240 / bpm
        x = np.frombuffer(reader.readframes(reader.getnframes()), dtype=np.int16)
        x = x.reshape(-1, reader.getnchannels()).astype(float) / 32768
    db = np.full(bars, -180.)
    for t in range(bars):
        segment = x[round(t * frames_per_bar):round((t + 1) * frames_per_bar)]
        if len(segment):
            db[t] = 20 * np.log10(np.sqrt(np.mean(segment ** 2)) + 1e-9)
    return db


def heard_layers(db):
    return (db >= -50.) & (db >= db.max(axis=1, keepdims=True) - 20.)


def render_row(args):
    """The 2-minute render of run 0 and its audio check (runs in a worker process)."""
    row, cards, out = args
    card = json.loads((Path(cards) / f"{row['name']}.json").read_text())
    pool = configure(card)
    wav = Path(out) / "samples" / f"quality_{row['name']}.wav"; wav.parent.mkdir(exist_ok=True)
    run = dict(row["_run0"]); run["bars"] = np.asarray(run["bars"])
    count = min(len(run["bars"]), max(64, int(np.ceil(pool.get("bpm", 120) / 2))))
    ok, msg = render(card, pool, run, wav, bars=count)
    rd = {"file": wav.name, "ok": ok, **(audio_check(wav, pool.get("bpm", 120), bars=count) if ok else {"error": msg})}
    if ok:                                                             # each layer alone: its loudness and whether it plays at all
        score = wav.with_suffix(".score.json")
        db = np.full((count, L), -180.)
        for l, name in enumerate(LAYERS):
            solo = wav.with_name(f"solo_{row['name']}_{name}.wav")
            r = subprocess.run([(shutil.which("node") or str(Path.home() / "node/bin/node")), str(CODE / "engine/render.mjs"), str(score), str(solo), "--layer", str(l), "--tail", "0"], capture_output=True, text=True)
            if r.returncode:
                rd.update(ok=False, error=f"{name} render: {r.stderr[-200:]}")
                break
            db[:, l] = bar_levels(solo, pool.get("bpm", 120), count)
            solo.unlink(missing_ok=True)
        rd["layer_db"] = {name: round(10 * np.log10(np.mean(10 ** (db[:, l] / 10))), 1) for l, name in enumerate(LAYERS)}
        rd["layer_plays"] = dict(zip(LAYERS, heard_layers(db).mean(axis=0).round(2).tolist()))
    return row["name"], rd


def main():
    from concurrent.futures import ProcessPoolExecutor
    ap = argparse.ArgumentParser()
    ap.add_argument("--cards", required=True, help="folder with the newest card + npz per candidate")
    ap.add_argument("--out", required=True)
    ap.add_argument("--render", action="store_true")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--max", type=int, default=24, help="checkpoints per pass (the newest round first)")
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    state_f = out / "done.json"
    done = json.loads(state_f.read_text()) if state_f.exists() else {}
    done = {k: v for k, v in done.items() if v != "failed"}          # failed brains are retried
    todo = [cj for cj in Path(args.cards).glob("*.json") if cj.stem not in done and cj.with_suffix(".npz").exists()
            and not cj.stem.startswith(("r3", "g_", "imit"))]        # rounds 4 and later only (round 3: old labels, superseded)
    rnd = lambda n: int("".join(ch for ch in n.split("_")[0] if ch.isdigit()) or 0)
    todo.sort(key=lambda cj: (-rnd(cj.stem), -cj.stat().st_mtime))   # round 4 before round 3, newest first
    left = max(0, len(todo) - args.max); todo = todo[:args.max]
    rows = []
    with ProcessPoolExecutor(max(1, min(args.workers, len(todo) or 1))) as ex:
        for row in ex.map(eval_card, [str(c) for c in todo]):
            if row.get("failed"):
                print(f"{row['name']}: failed {row['failed']}", flush=True); done[row["name"]] = "failed"; continue
            rows.append(row); done[row["name"]] = row["time"]
            m = row["metrics"]
            print(f"{row['name']}: score {row['score']} kick min {m.get('drums.kick_bars_min', 0):.2f} kickless max {m.get('drums.longest_kickless_max', 0):.0f} "
                  f"breakdown {m.get('breakdown.pass', '')} beat-QA (before audio) {not [f for f in row['qa']['failed'] if f.startswith(('kick in', 'kick on', 'no kick', 'machine', 'never', 'all layers'))]}", flush=True)
        if args.render and rows:
            by_round = {}
            for r in rows:
                by_round.setdefault(r["name"].split("_")[0], []).append(r)
            written_ok = lambda r: all(test(r["metrics"], {"ok": True, "kick_audible_quarters": 1, "silent_seconds": 0}) for _, test in QA_BEAT)
            chosen = {r["name"]: r for rs in by_round.values() for r in sorted(rs, key=lambda r: -r["score"])[:RENDER_TOP]}
            chosen.update({r["name"]: r for r in rows if written_ok(r)})   # every brain that passes the written checks gets the audio check
            for name, rd in ex.map(render_row, [(r, args.cards, str(out)) for r in chosen.values()]):
                r = chosen[name]
                rd["kick_bars"] = r.get("render_kick_bars")
                r["render"] = rd
                r["flags"] = flags(r["metrics"], r["corpus"], rd if rd.get("ok") else None)
                r["score"] = penalized(r); r["qa"] = qa(r)
                print(f"rendered {name}: audible {rd.get('kick_audible_quarters')} QA {'PASS' if r['qa']['pass'] else ('HOLDS A BEAT' if r['qa']['beat'] else 'fail')} {r['qa']['failed']}", flush=True)
    with open(out / "history.jsonl", "a") as f:
        for r in rows:
            r.pop("_run0", None); r.pop("render_kick_bars", None)
            if "qa" not in r: r["qa"] = qa(r)
            f.write(json.dumps(r) + "\n")
    state_f.write_text(json.dumps(done))
    hist = [json.loads(l) for l in (out / "history.jsonl").read_text().splitlines()] if (out / "history.jsonl").exists() else []
    latest = {}
    for h in hist:
        h.setdefault("qa", qa(h))
        latest[h["candidate"]] = h
    level = lambda h: 2 if h["qa"]["pass"] else (1 if h["qa"]["beat"] else 0)
    lines = ["| QA | score | brain | epoch | kick bars min | kick on quarters | kickless max | kick audible | more than a kick | layers dB (beat/bass/lead/air) | changes per 100 bars beat/bass/lead/air (tracks) | pad | kick back after breakdown | busy snare | copied | QA failures |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for h in sorted(latest.values(), key=lambda h: (-level(h), -h["score"]))[:40]:
        m = h["metrics"]; rd = h.get("render") or {}
        lines.append(f"| {['fail', 'HOLDS A BEAT', 'PASS'][level(h)]} | {h['score']} | {h['name']} | {h['epoch']} | {m.get('drums.kick_bars_min', 0):.2f} | {m.get('drums.kick_quarters', 0):.2f} | {m.get('drums.longest_kickless_max', 0):.0f} | "
                     f"{rd.get('kick_audible_quarters', '')} | {m.get('drums.more_than_kick', 0):.2f} | " + "/".join(str((rd.get('layer_db') or {}).get(n, '-')) for n in LAYERS) + " | "
                     + "/".join(f"{100 * m.get(f'{n}.change', 0):.1f}" for n in LAYERS) + " (" + "/".join(f"{100 * ((h.get('corpus') or {}).get(f'{n}.change') or 0):.1f}" for n in LAYERS) + ") | "
                     + f"{m.get('mood.energy_corr', 0):.2f} | {m.get('breakdown.pass', '')} | {m.get('drums.busy_snare', 0):.2f} | {m.get('copied', 0):.2f} | {'; '.join(h['qa']['failed'])} |")
    flagged = [f"{h['name']}: {', '.join(h['flags'])}" for h in latest.values() if h.get("flags")]
    (out / "flags.txt").write_text("\n".join(flagged) + ("\n" if flagged else ""))
    proto = "\n".join(f"- {n}" for n, _ in QA_BEAT) + "\n\nPASS also needs:\n" + "\n".join(f"- {n}" for n, _ in QA_FULL[len(QA_BEAT):])
    (out / "report.md").write_text("# Quality assurance (newest checkpoint per candidate)\n\nHOLDS A BEAT = every line of:\n" + proto + "\n\n" + "\n".join(lines) + "\n")
    (out / "left.txt").write_text(str(left))
    print(f"report: {len(rows)} new, {len(latest)} candidates, {left} left for the next pass", flush=True)


if __name__ == "__main__":
    main()
