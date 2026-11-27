"""Learning curves from published cards and the watcher's telemetry cache."""
import json
import math
from pathlib import Path


def records(path):
    try:
        with path.open() as file:
            for line in file:
                try:
                    row = json.loads(line)
                except ValueError:
                    continue
                if isinstance(row, dict):
                    yield row
    except FileNotFoundError:
        return


def document(path):
    try:
        row = json.loads(path.read_bytes())
    except (FileNotFoundError, ValueError):
        return {}
    return row if isinstance(row, dict) else {}


def number(value):
    return value if type(value) in (int, float) and math.isfinite(value) else None


_BARS = {}


def training_bars(root, streams, every_track=False):
    """The bars a run is taught from: the score's tracks that are not held out (all of them when every track is
    taught). Read once for each state of the score file."""
    path = root / "corpus" / str(streams)
    try:
        state = path.stat()
    except OSError:
        return None
    key = (str(path), state.st_mtime_ns, state.st_size, bool(every_track))
    if key not in _BARS:
        try:
            score = json.loads(path.read_bytes())
            tracks = score["tracks"] if isinstance(score, dict) else score
            _BARS[key] = sum(len(track["bars"]) for track in tracks if every_track or not track.get("held_out"))
        except (ValueError, KeyError, TypeError):
            _BARS[key] = None
    return _BARS[key]


def training(root, now=None):
    """Which runs have a box right now, from the mirrors' reports (quality/composer/training-status-*.json).

    Returns the state of every run a report names and the time of the newest report. A run's state is its box
    (instance, type, launch time) while it trains, and None while it is unknown: a mirror that stopped reporting
    with boxes alive cannot say. A run no report names is not training."""
    import time

    now = time.time() if now is None else now
    states, checked = {}, None
    # runs stopped by order (quality/stage1/stopped.txt) stay registered on their box but no longer train
    try:
        stopped = set((root / "quality/stage1/stopped.txt").read_text().split())
    except OSError:
        stopped = set()
    for file in sorted((root / "quality/composer").glob("training-status-*.json")):
        report = document(file)
        at = number(report.get("at"))
        if at is None:
            continue
        checked = at if checked is None else max(checked, at)
        current = bool(report.get("done")) or now - at < 300
        for run, box in (report.get("running") or {}).items():
            if isinstance(box, dict) and run not in stopped:
                states[run] = box if current else None
    return states, checked


def slate(root, layouts, code=None, judge=None):
    """The layouts under comparison with the checks each has passed (brain/check.py, quality/checks).

    ``layouts`` is the slate (brain/layouts.py) and ``code`` the present builder's mark: a check made on
    another builder does not count. ``judge`` reads a record by the present rules (brain/check.py). Every
    entry says whether the layout may be trained (``ready``) and gives each stage as passed, failed or not
    yet run, with the readings a reader needs to see why."""
    out = []
    for name, entry in layouts.items():
        # the stage-1 shape (reserved inputs) has its own record; the newest record made on the present builder counts
        kept = {}
        for record in (f"{name}.inputs.json", f"{name}.json"):
            found = document(root / "quality/checks" / record)
            if found and (code is None or found.get("code") == code):
                kept = found
                break
        verdict = judge(kept) if judge else {"ready": bool(kept.get("ready")), "rule": None}
        item = {"layout": name, "title": entry["title"], "pathways": entry["pathways"], "wiring": entry["wiring"],
                "ready": bool(verdict["ready"]), "stages": {}}
        build, bench, listen = (kept.get(stage) or {} for stage in ("build", "bench", "listen"))
        if build:
            item["stages"]["build"] = {key: build.get(key) for key in ("passed", "problems", "neurons", "synapses", "at")}
        if bench:
            stage = {"passed": bench.get("passed"), "problems": bench.get("problems"), "at": bench.get("at")}
            for task in ("hold", "rule"):
                if isinstance(bench.get(task), dict):
                    stage[task] = {key: bench[task].get(key) for key in ("error", "commonest", "repeat", "bar", "lessons")}
            if "rule" in stage:
                stage["rule"]["reading"] = verdict["rule"]
            item["stages"]["bench"] = stage
        if listen:
            curve = listen.get("curve") or []
            item["stages"]["listen"] = {
                "passed": listen.get("passed"), "problems": listen.get("problems"), "running": bool(listen.get("running")),
                "where": listen.get("where"), "levels": listen.get("levels"), "at": listen.get("at"),
                "curve": [{key: row.get(key) for key in ("batches", "minutes", "train", "test")} for row in curve]}
        out.append(item)
    return out


def lessons(root, card):
    """How much a run hears in one epoch, from its card: windows, their length, and the bars of its training set."""
    training = card.get("training") or {}
    window, batch = number(training.get("window")), number(training.get("batch"))
    groups = training.get("windows_per_group")
    unique = number(training.get("unique_windows"))
    extra = number(training.get("extra_focus_windows")) or 0
    heard = sum(groups.values()) if isinstance(groups, dict) and groups else (unique + extra if unique else None)
    if not window or not heard:
        return None
    return {"window": window, "batch": batch, "epoch_windows": heard, "unique_windows": unique,
            "training_bars": training_bars(root, card.get("streams") or Path(str(training.get("streams"))).name,
                                           training.get("all_tracks"))}


def point(row, name, card=None, playable=False):
    card = card or {}
    train, test = row.get("train_error") or {}, row.get("test_error") or {}
    settings = card.get("settings") or {}
    modules = settings.get("modules")
    source = card.get("source_checkpoint")
    return {"name": name, "epoch": number(card.get("epoch")), "minutes": number(row.get("minutes", card.get("minutes"))),
            "created": card.get("created") if isinstance(card.get("created"), str) else None,
            "train": number(train.get("balanced")), "test": number(test.get("balanced")),
            "train_parts": train, "test_parts": test, "eta": number(row.get("eta")), "next_eta": number(row.get("next_eta")),
            "temperature": number(row.get("temperature", card.get("temperature"))),
            "copy_baseline": row.get("copy_baseline"),
            "gains": row.get("gains") if isinstance(row.get("gains"), list) else None, "light": bool(row.get("light")),
            "published": bool(card), "playable": playable, "initial": row.get("tag") == "untrained",
            "starting_checkpoint": row.get("starting_checkpoint"),
            "source_checkpoint": source, "playback_review": card.get("playback_review"),
            "gate_pass": None if source else (row.get("gate") or card.get("gate") or {}).get("pass"),
            "gate_checks": None if source else (row.get("gate") or card.get("gate") or {}).get("checks"),
            "composition": None if source else number(row.get("composition", (card.get("composition") or {}).get("selection_score"))),
            "symbolic": None if source else number(row.get("symbolic", (card.get("composition") or {}).get("score"))),
            "hold": None if source else number(row.get("hold", (card.get("composition") or {}).get("hold_score"))),
            "phrase_events": None if source else row.get("phrase_events", (card.get("composition") or {}).get("phrase_events")),
            "development": None if source else row.get("development", (card.get("composition") or {}).get("development")),
            "prompt_protocol": number(row.get("prompt_protocol", (card.get("composition") or {}).get("prompt_protocol"))),
            "modules": modules if isinstance(modules, list) and all(number(n) for n in modules) else None,
            "apical": number(settings.get("apical")), "recall_heads": number(settings.get("recall_heads")),
            "architecture": settings.get("architecture"), "coordinator_depth": number(settings.get("coordinator_depth")),
            "learned_code": number(settings.get("learned_code")), "recall_probe": card.get("recall_probe"),
            "recall_attention": (card.get("composition") or {}).get("recall_attention"),
            # does the coordinator contribute (generation 2): the teaching contrast each region gets (Cadence's
            # credit report, mean over the lessons since the last evaluation), the evaluation with the coordinator
            # cut, and the free rollout; a row without them gives None
            "hidden_credit": row.get("hidden_credit") if isinstance(row.get("hidden_credit"), dict) else None,
            "credit_warnings": row.get("credit_warnings") if isinstance(row.get("credit_warnings"), list) else None,
            "coordinator_cut": row.get("coordinator_cut") if isinstance(row.get("coordinator_cut"), dict) else None,
            "rollout": row.get("rollout") if isinstance(row.get("rollout"), dict) else None}


def signature(root, checkpoints, pinned):
    """What the curves are read from, as names, sizes and change times: equal signatures, equal curves."""
    files = [file for folder in (root / "brains", checkpoints, pinned) for file in folder.glob("*.json")]
    files += [file for folder in (root / "quality/composer").glob("*") if folder.is_dir()
              for pattern in ("*.json", "metrics.jsonl", "checkpoints/*.json") for file in folder.glob(pattern)]
    files.append(root / "quality/curves.jsonl")
    files += list((root / "quality/checks").glob("*.json"))      # the checks of the layout slate
    files += [file for folder in (checkpoints, pinned) for file in folder.glob("*.npz")]   # what the player can load
    seen = []
    for file in sorted(files):
        try:
            state = file.stat()
        except OSError:
            continue
        seen.append((str(file), state.st_mtime_ns, state.st_size))
    return tuple(seen)


def curves(root, checkpoints, pinned, composer_only=False, runtime_patch=None, *, installed_names=None,
           describe=None):
    """``describe`` turns a checkpoint card into the brain's layout (brain/anatomy.py); every candidate
    carries the layout of its newest card."""
    installed = (set(installed_names) if installed_names is not None else
                 {p.stem for folder in (checkpoints, pinned) for p in folder.glob("*.npz")})
    cards = {}
    for folder in (root / "brains", checkpoints, pinned):
        for file in folder.glob("*.json"):
            card = document(file)
            if card.get("name") == file.stem and card.get("candidate"):
                if not composer_only or (card.get("patch_book") and card.get("streams_sha")):
                    cards[file.stem] = card
    groups = {}
    # runtime_patch: one stamp, or every stamp the runtime reads (its own and the patches it contains)
    readable = runtime_patch if isinstance(runtime_patch, (set, frozenset, tuple, list)) else {runtime_patch}
    installed -= {name for name, card in cards.items() if card.get("cadence_patch_sha256")
                  and card["cadence_patch_sha256"] not in readable}
    for name, card in cards.items():
        if card.get("errors") or (card.get("patch_book") and card.get("streams_sha")):
            candidate = card["candidate"]
            group = groups.setdefault(candidate, {"candidate": candidate, "kind": "composer", "points": {}})
            group["points"][name] = point(card.get("errors") or {}, name, card, name in installed)
    for folder in (root / "quality/composer").glob("*"):
        if not folder.is_dir():
            continue
        candidate = folder.name
        group = groups.setdefault(candidate, {"candidate": candidate, "kind": "composer", "points": {}})
        group["status"] = document(folder / "status.json")
        monitor = document(folder / "monitor.json")
        if monitor:
            events = list((monitor.get("events") or {}).values())
            group["monitor"] = {"groove": monitor.get("groove"), "last_review": monitor.get("last_review"),
                                "events": [{"message": e.get("message"), "status": e.get("status")}
                                           for e in events[-4:]]}
        copy = document(folder / "copy_baseline.json")
        blind = document(folder / "blind_baseline.json")      # measured afterwards for a run that began without it
        if blind.get("train") and blind.get("test"):
            group["blind"] = {"train": blind["train"], "test": blind["test"]}
        source = next((cards[name].get("streams_sha") for name in group["points"]), None)
        for name, checkpoint in group["points"].items():
            if copy and not checkpoint["copy_baseline"] and cards[name].get("streams_sha") == copy.get("streams_sha"):
                checkpoint["copy_baseline"] = copy
        initial = document(folder / "untrained.json")
        if initial.get("streams_sha") == source and initial.get("tag") == "untrained":
            result = point(initial, f"{candidate}_e000")
            if copy and source == copy.get("streams_sha"):
                result["copy_baseline"] = copy
            group["points"][result["name"]] = result
        for row in records(folder / "metrics.jsonl"):
            tag = row.get("checkpoint")
            if not isinstance(tag, str):
                continue
            if isinstance(row.get("anatomy"), dict):      # the layout the trainer wrote with its first evaluation
                group["layout"] = row["anatomy"]
            if isinstance(row.get("lessons"), dict):      # and what one of its epochs holds
                group["lessons"] = row["lessons"]
            if isinstance(row.get("blind_baseline"), dict):      # the error of the commonest bar, whatever was heard
                group["blind"] = row["blind_baseline"]
            if isinstance(row.get("layout"), str):      # the layout of the slate the run trains
                group["slate"] = row["layout"]
            name = f"{candidate}_e{tag}"
            previous = group["points"].get(name)
            fresh = point(row, name, cards.get(name), name in installed)
            if not fresh["copy_baseline"]:
                fresh["copy_baseline"] = previous.get("copy_baseline") if previous else (
                    copy if copy and source == copy.get("streams_sha") else None)
            group["points"][name] = fresh
        for file in folder.glob("*.recall.json"):
            trace = document(file)
            name = trace.get("checkpoint")
            if not isinstance(name, str) or name not in cards or name not in group["points"]:
                continue
            if trace.get("streams_sha") == cards[name].get("streams_sha"):
                checkpoint = group["points"][name]
                if trace.get("heads"):
                    checkpoint["recall_attention"] = trace["heads"]
                if trace.get("phrase_events"):
                    checkpoint["phrase_events"] = trace["phrase_events"]
                if trace.get("development"):
                    checkpoint["development"] = trace["development"]
    if not composer_only:
        for row in records(root / "quality/curves.jsonl"):
            train, test, epoch = map(number, (row.get("train_score"), row.get("test_score"), row.get("epoch")))
            name, candidate = row.get("name"), row.get("candidate")
            if row.get("failed") or None in (train, test, epoch, name, candidate) or candidate in groups and groups[candidate]["kind"] == "composer":
                continue
            group = groups.setdefault(candidate, {"candidate": candidate, "kind": "legacy", "points": {}})
            group["points"][name] = {"name": name, "epoch": epoch, "minutes": None, "train": 1-train, "test": 1-test,
                                     "published": True, "playable": name in installed,
                                     "test_change": {voice: values.get("change") for voice, values in (row.get("test") or {}).items()}}
    for group in groups.values():
        if group["kind"] != "composer" or (group.get("lessons") and (group.get("layout") or describe is None)):
            continue
        named = sorted(name for name in group["points"] if name in cards)
        card = cards[named[-1]] if named else None
        if card is None:      # a run whose checkpoints were never installed keeps its cards beside its curve
            folder = root / "quality/composer" / group["candidate"]
            local = sorted(file for part in (folder, folder / "checkpoints")
                           for file in part.glob(f"{group['candidate']}_e*.json")
                           if not file.name.endswith(".recall.json"))
            card = document(local[-1]) if local else None
        if not card:
            continue
        if not group.get("lessons"):
            heard = lessons(root, card)
            if heard:
                group["lessons"] = heard
        if describe is not None and not group.get("layout"):
            layout = describe(card)
            if layout:
                group["layout"] = layout
    out = []
    for group in groups.values():
        axis = "minutes" if group["kind"] == "composer" else "epoch"
        points = sorted(group["points"].values(), key=lambda p: (p[axis] is None, p[axis] or 0, p["name"]))
        out.append({**group, "points": points})
    return sorted(out, key=lambda group: (group["kind"] == "composer", group["candidate"]), reverse=True)
