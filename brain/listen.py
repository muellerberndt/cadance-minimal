"""Listening check: long compositions by a brain, rendered on the instrument and measured bar by bar, beside the
real tracks (their labels) played on the same instrument from the same kind of starting point.

Per brain: RUNS compositions of BARS bars (quality.free_runs: each continues a passage in full flow from a training
track, the mood rising and falling), every layer rendered alone. Per bar and layer the loudness (dBFS); a layer is
HEARD in a bar when it is at least -50 dBFS and within 20 dB of the loudest layer of that bar. From that and what the
bars hold:
  kick only    the drums play a kick and nothing else, and no synth is heard (the owner: such stretches should not be)
  drums only   the drums play and no synth is heard
  transitions  bars where the heard texture changes (a layer comes in, drops out, or changes its sound), where they
               fall in 8- and 16-bar phrases (bar clock), and how long a texture lasts between them
  patterns     per heard synth bar: onsets, pitch classes; per 16 bars: distinct beat and synth bars
The reference: the real continuation of passages from TEST tracks (never trained on), same length, same render.

    python brain/listen.py --names r7_n512_flow_kw60_e001b025 ... --out ~/minimal/quality/listen [--runs 3] [--bars 256]
Writes <out>/<name>.json and <out>/ref_<streams>.json (per-bar arrays + summary), and a row per brain in
<out>/listen.jsonl. brain/listen_sheet.py draws the sheets.
"""
import argparse, json, os, subprocess, sys, tempfile, time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numpy as np  # noqa: E402
import common  # noqa: E402
import quality  # noqa: E402
from common import L, LAYERS  # noqa: E402

# where training runs keep their brains ("s3://bucket/prefix"; common.local_setting): empty names none
BUCKET = common.local_setting("CADANCE_BUCKET") + "/brains_r4"
NODE = quality.shutil.which("node") or str(Path.home() / "node/bin/node")
_TRACKS = {}


def fetch(name, d):
    for ext in ("json", "npz"):
        local = Path.home() / "minimal" / "brains_r4" / f"{name}.{ext}"
        if local.exists():
            continue
        if subprocess.run(["aws", "s3", "cp", f"{BUCKET}/{name}.{ext}", f"{d}/{name}.{ext}", "--only-show-errors"]).returncode:
            return None
    pick = lambda ext: next(p for p in (Path.home() / "minimal" / "brains_r4" / f"{name}.{ext}", Path(d) / f"{name}.{ext}") if p.exists())
    return pick("json"), pick("npz")


def tracks_for(card):
    sp = quality.streams_path(card)
    key = common.layout_key(card)
    if key not in _TRACKS:
        common.shape_from(card)
        _TRACKS[key] = common.load_streams(sp, steps=card.get("steps", False), rows=card.get("beat_rows"),
                                           clock=card.get("clock", False), synth=card.get("synth_rows"), patch_book=card.get("patch_book"))
    return _TRACKS[key]


def stems(card, pool, bars, wav):
    """Render the mix and each layer alone; per bar and layer the loudness in dBFS (bars x 4)."""
    ok, msg = quality.render(card, pool, {"bars": np.asarray(bars)}, wav, bars=len(bars))
    if not ok:
        raise RuntimeError(f"render: {msg}")
    score = wav.with_suffix(".score.json")
    renderer = (quality.CODE / "engine/render.mjs").resolve()
    bpm = float(pool.get("bpm", 120))
    def levels(layer):
        solo = wav.with_name(wav.stem + f"_{LAYERS[layer]}.wav")
        r = subprocess.run([NODE, str(renderer), str(score), str(solo), "--layer", str(layer), "--tail", "0"], capture_output=True, text=True)
        if r.returncode or not solo.is_file():
            raise RuntimeError(f"stem {LAYERS[layer]}: {(r.stderr or r.stdout or 'No WAV produced')[-200:]}")
        db = quality.bar_levels(solo, bpm, len(bars))
        solo.unlink(missing_ok=True)
        return db

    workers = max(1, min(L, int(os.environ.get("CADANCE_RENDER_JOBS", "1"))))
    if workers == 1:
        rendered = list(map(levels, range(L)))
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            rendered = list(executor.map(levels, range(L)))
    return np.column_stack(rendered)


def longest(mask):
    best = cur = 0
    for v in mask:
        cur = cur + 1 if v else 0
        best = max(best, cur)
    return int(best)


def measure(bars, db, pos0):
    """Per-bar arrays and the summary of one composition (bars: states; db: bars x 4; pos0: bar clock of bar 0)."""
    bars = common.external(bars); n = len(bars)
    heard = quality.heard_layers(db)
    beat = np.stack([common.decode_beat(b) for b in bars]) if common.STEPS else None          # n x 3 voices x 16
    kick_w = (beat[:, 0] > 0).any(axis=1) if beat is not None else bars[:, 0] > 0
    other_w = (beat[:, 1:] > 0).any(axis=(1, 2)) if beat is not None else np.zeros(n, bool)
    synth_heard = heard[:, 1:].any(axis=1)
    kick_only = heard[:, 0] & kick_w & ~other_w & ~synth_heard
    drums_only = heard[:, 0] & ~synth_heard
    dead = ~heard.any(axis=1)
    tex = [tuple(int(bars[t, l]) if heard[t, l] else 0 for l in range(L)) for t in range(n)]
    changes = [t for t in range(1, n) if tex[t] != tex[t - 1]]
    pos = pos0 + np.arange(n)
    sec, last = [], 0
    for t in changes + [n]:
        sec.append(t - last); last = t
    entries = {LAYERS[l]: [t for t in range(1, n) if heard[t, l] and not heard[t - 1, l]] for l in range(L)}
    rows = np.stack([common.decode_synth(b) for b in bars]) if common.SYNTH else None           # n x 3 x 16 (0 rest, 1 tie, 2+pc)
    syn = {}
    if rows is not None:
        for i, name in enumerate(common.SYNTH_LAYERS):
            h = heard[:, i + 1]
            on = rows[:, i] >= 2
            pcs = [len(set((r[r >= 2] - 2).tolist())) for r in rows[h, i]]
            win = [len({rows[t, i].tobytes() for t in range(s, min(n, s + 16)) if h[t]}) for s in range(0, n, 16)]
            win = [w for w in win if w]
            syn[name] = {"heard": float(h.mean()), "onsets": float(on[h].sum(axis=1).mean()) if h.any() else 0.0,
                         "fast": float((on[h].sum(axis=1) >= 8).mean()) if h.any() else 0.0,
                         "pitch_classes": float(np.mean(pcs)) if pcs else 0.0,
                         "distinct_per_16": float(np.mean(win)) if win else 0.0, "longest_gap": longest(~h)}
    bwin = [len({beat[t].tobytes() for t in range(s, min(n, s + 16)) if heard[t, 0]}) for s in range(0, n, 16)] if beat is not None else []
    bwin = [w for w in bwin if w]
    summary = {
        "bars": n,
        "heard": {LAYERS[l]: float(heard[:, l].mean()) for l in range(L)},
        "layers_heard_mean": float(heard.sum(axis=1).mean()),
        "kick_only": float(kick_only.mean()), "kick_only_longest": longest(kick_only),
        "drums_only": float(drums_only.mean()), "drums_only_longest": longest(drums_only),
        "dead": float(dead.mean()), "dead_longest": longest(dead),
        "changes_per_100": 100.0 * len(changes) / max(n - 1, 1),
        "changes_on_8": float(np.mean([pos[t] % 8 == 0 for t in changes])) if changes else 0.0,
        "changes_on_16": float(np.mean([pos[t] % 16 == 0 for t in changes])) if changes else 0.0,
        "section_median": float(np.median(sec)), "section_longest": int(max(sec)),
        "entries_per_100": {k: 100.0 * len(v) / n for k, v in entries.items()},
        "entries_on_8": float(np.mean([pos[t] % 8 == 0 for v in entries.values() for t in v])) if any(entries.values()) else 0.0,
        "beat_distinct_per_16": float(np.mean(bwin)) if bwin else 0.0,
        "hats_or_snare": float(other_w[heard[:, 0]].mean()) if heard[:, 0].any() else 0.0,
        "synth": syn,
    }
    per_bar = {"db": np.round(db, 1).tolist(), "heard": heard.astype(int).tolist(), "kick_only": kick_only.astype(int).tolist(),
               "drums_only": drums_only.astype(int).tolist(), "changes": changes, "pos0": int(pos0),
               "tokens": bars[:, :L].tolist(), "notes": rows.tolist() if rows is not None else None,
               "beat": beat.astype(int).tolist() if beat is not None else None}
    return summary, per_bar


def merge(summaries):
    """The mean of the runs' summaries; the worst run for the longest stretches."""
    out = {}
    for k, v in summaries[0].items():
        if isinstance(v, dict):
            out[k] = merge([s[k] for s in summaries])
        elif k.endswith("_longest") or k == "longest_gap":
            out[k] = int(max(s[k] for s in summaries))
        else:
            out[k] = float(np.mean([s[k] for s in summaries]))
    return out


def listen_brain(args):
    name, out, runs, n_bars, entry, line = args
    common.ENTRY_SCALE, common.LINE_SCALE = entry, line
    tag = name + (f"@entry{entry:g}" if entry is not None else "") + (f"@line{line:g}" if line is not None else "")
    import warnings
    warnings.simplefilter("ignore")
    from cadence import Brain
    try:
        with tempfile.TemporaryDirectory() as d:
            got = fetch(name, d)
            if got is None:
                return {"name": name, "failed": "download"}
            cj, npz = got
            card = json.loads(cj.read_text())
            common.SYNTH_FILL = bool(card.get("synth_fill"))
            pool = quality.configure(card)
            tracks = tracks_for(card)
            quality.configure(card)
            brain = Brain.load(npz, backend="cpu")
            temp = common.slot_temperatures(brain, tracks, card)
            comps = quality.free_runs(brain, tracks, temp, n_bars=n_bars, k=runs)
            sums, pers = [], []
            for j, c in enumerate(comps):
                wav = Path(d) / f"{name}_{j}.wav"
                db = stems(card, pool, c["bars"], wav)
                s, p = measure(c["bars"], db, c["start"] + 8)
                p.update(track=c["track"], start=c["start"]); sums.append(s); pers.append(p)
                if j == 0:                                     # keep one mix to listen to
                    (Path(out) / "wav").mkdir(exist_ok=True)
                    subprocess.run(["cp", str(wav), str(Path(out) / "wav" / f"{tag}.wav")])
        row = {"name": tag, "brain": name, "entry_scale": entry, "line_scale": line, "candidate": card.get("candidate"), "epoch": card.get("epoch"), "streams": card.get("streams"),
               "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "summary": merge(sums), "runs": [s for s in sums]}
        (Path(out) / f"{tag}.json").write_text(json.dumps({**row, "per_bar": pers}))
        return row
    except Exception as e:
        import traceback
        return {"name": name, "failed": f"{type(e).__name__}: {e}", "trace": traceback.format_exc()[-800:]}


def listen_ref(args):
    """The real continuation of passages from test tracks, on the same instrument (labels -> render)."""
    card_name, out, runs, n_bars = args
    import warnings
    warnings.simplefilter("ignore")
    with tempfile.TemporaryDirectory() as d:
        cj, _ = fetch(card_name, d)
        card = json.loads(cj.read_text())
    common.SYNTH_FILL = bool(card.get("synth_fill"))
    pool = quality.configure(card)
    tracks = tracks_for(card)
    quality.configure(card)
    test = [dict(t, held_out=False) for t in tracks if t["held_out"] and len(t["bars"]) >= n_bars // 2 + 16]
    picks = common.flow_passages(test, runs, 8, 23)
    sums, pers = [], []
    with tempfile.TemporaryDirectory() as d:
        for j, (tr, s) in enumerate(picks):
            bars = np.asarray(tr["bars"])[s + 8:s + 8 + n_bars]
            wav = Path(d) / f"ref_{j}.wav"
            db = stems(card, pool, bars, wav)
            m, p = measure(bars, db, s + 8)
            p.update(track=tr["title"], start=int(s)); sums.append(m); pers.append(p)
            if j == 0:
                (Path(out) / "wav").mkdir(exist_ok=True)
                subprocess.run(["cp", str(wav), str(Path(out) / "wav" / f"ref_{Path(card['streams']).stem}.wav")])
    row = {"name": f"ref_{Path(card['streams']).stem}", "streams": card.get("streams"), "tracks": [p["track"] for p in pers],
           "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "summary": merge(sums), "runs": sums}
    (Path(out) / f"{row['name']}.json").write_text(json.dumps({**row, "per_bar": pers}))
    return row


def listen_error(s, ref):
    """How far a brain's free play is from the tracks' (0 = the same), from what is heard and how the lines are built:
    drums alone (counted double), each synth layer heard, the beat's variety, and per bass and lead the notes per bar,
    pitch classes and different lines per 16 bars (relative errors, capped at 1)."""
    rel = lambda x, r: min(1.0, abs(x - r) / max(abs(r), 1e-6))
    e = [min(1.0, 2 * abs(s["drums_only"] - ref["drums_only"]))]
    e += [abs(s["heard"][n] - ref["heard"][n]) for n in ("bass", "lead", "air")]
    e += [rel(s["beat_distinct_per_16"], ref["beat_distinct_per_16"])]
    for n in ("bass", "lead"):
        e += [rel(s["synth"][n][k], ref["synth"][n][k]) for k in ("onsets", "pitch_classes", "distinct_per_16")]
    return round(float(np.mean(e)), 4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--names", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--bars", type=int, default=256)
    ap.add_argument("--ref-runs", type=int, default=6)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--entry", type=float, nargs="*", default=[None], help="ENTRY_SCALE values to try (common.sample_slots)")
    ap.add_argument("--line", type=float, nargs="*", default=[None], help="LINE_SCALE values to try (common.sample_slots)")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    jobs = [(n, str(out), a.runs, a.bars, e, ln) for n in a.names for e in a.entry for ln in a.line]
    refs = {}                                              # one reference per label file
    for n in a.names:
        with tempfile.TemporaryDirectory() as d:
            got = fetch(n, d)
            if got:
                s = json.loads(got[0].read_text()).get("streams")
                if s and not (out / f"ref_{Path(s).stem}.json").exists():
                    refs.setdefault(s, n)
    with ProcessPoolExecutor(min(a.workers, len(jobs) + len(refs))) as ex, open(out / "listen.jsonl", "a") as w:
        futs = [ex.submit(listen_ref, (n, str(out), a.ref_runs, a.bars)) for n in refs.values()]
        futs += [ex.submit(listen_brain, j) for j in jobs]
        for f in futs:
            r = f.result()
            rf = out / f"ref_{Path(r.get('streams') or '').stem}.json"
            if r.get("summary") and not r["name"].startswith("ref_") and rf.exists():
                r["listen_error"] = listen_error(r["summary"], json.loads(rf.read_text())["summary"])
            w.write(json.dumps(r) + "\n"); w.flush()
            s = r.get("summary") or {}
            print("listen:", r["name"], r.get("failed") or
                  f"kick-only {s['kick_only']:.2f} (longest {s['kick_only_longest']}) drums-only {s['drums_only']:.2f} "
                  f"heard {' '.join(f'{k} {v:.2f}' for k, v in s['heard'].items())} changes/100 {s['changes_per_100']:.1f} "
                  f"on 8 {s['changes_on_8']:.2f} section median {s['section_median']:.0f}" + (f" | listen error {r['listen_error']:.3f}" if "listen_error" in r else ""), flush=True)
            if r.get("trace"):
                print(r["trace"], flush=True)


if __name__ == "__main__":
    main()
