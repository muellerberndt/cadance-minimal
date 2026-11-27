"""A brain's design as plain data, for the layout drawing on the Stats page.

The description is read from the genome the trainer develops, so the drawing shows the wiring
the brain has: its regions and their sizes, which region drives which, where feedback returns,
where the recall heads land and which motor slots every pathway writes. Nothing here touches
the configured layout in ``common``; a checkpoint card is enough.
"""
import hashlib
import json

import hierarchy
import timbre

VERSION = 1
_CACHE = {}
# Settings that tune a brain without shaping its layout: checkpoints that differ in these share a drawing.
_TUNING = ("eta", "eta_scale", "init", "backend", "device", "precision", "migrated_recall_patch_sha256",
           "recall_amplitude", "recall_temperature", "recall_gain_born", "recall_lags_born", "recall_lag_gains_born")

# What a region is for, by its name in the genome.
_ROLES = {"sensory": "hears", "code": "code", "association": "coordinator", "phrase_context": "context",
          "prefrontal": "trace", "motor": "motor", "efference": "sense", "history": "sense",
          "observation": "sense", "message": "sense", "message_read": "sense"}


def slot_kinds(card):
    """What each motor slot holds, from the card alone: kit, gate or sound, mix, drum, line, control."""
    sizes = list(card["settings"]["slots"])
    book = card.get("patch_book")
    patch = timbre.width(book) if book else 0
    drum = 48 if card.get("steps") else 3 if card.get("beat_rows") else 0
    onoff = hierarchy.LAYERS if card.get("onoff") else 0
    # the SID blocks of the stage-1 shape, per synth (bass, lead, air): all three, or the lead's alone
    reserve = card.get("reserve")
    sid = tuple(timbre.SID_WIDTH if reserve in (True, "sid") or (reserve == "sidlead" and i == 1) else 0
                for i in range(3))
    import common
    plan = (len(common.PLAN_SIZES) if card.get("lookahead") else 0) + (1 if card.get("section") else 0)  # plan, section
    lines = len(sizes) - hierarchy.LAYERS - hierarchy.MIX - drum - 3 * patch - onoff - sum(sid) - plan
    if lines < 0 or lines % 3:
        raise ValueError("the card's slots do not match its layout")
    sounds = ["kit", "gate", "gate", "gate"] if book else ["sound"] * hierarchy.LAYERS
    kinds = (sounds + ["mix"] * hierarchy.MIX + ["drum"] * drum + ["line"] * lines + ["control"] * (3 * patch)
             + ["onoff"] * onoff + ["reserved"] * sum(sid) + ["plan"] * plan)
    return kinds, drum, lines // 3, patch, sid, plan


def _role(name):
    if name in _ROLES:
        return _ROLES[name], None
    if name.startswith("module_"):
        return "stage", None
    if name.startswith("observer_"):
        return "observer", None
    if name.startswith(("recall", "next")):
        return "memory", None
    for suffix in ("_detail", "_plan"):
        if name.endswith(suffix):
            return suffix[1:], name[:-len(suffix)]
    return "sense", None


def _genome(card, drum, line, patch, sid=0, plan=0):
    from cadence import Genome

    settings = dict(card["settings"])
    seed = int(card.get("seed") or 0)
    architecture = settings.get("architecture")
    for name in ("architecture", "plastic_synapses", "init", "recall_gain_born", "recall_lags_born", "recall_lag_gains_born"):
        settings.pop(name, None)
    if architecture:
        parts = hierarchy.owners(architecture, drum, line, patch, sid, plan)
        return hierarchy.genome(settings, parts, seed)[0], parts
    inputs, actions = settings.pop("inputs"), settings.pop("actions")
    settings.pop("coordinator_depth", None)
    if settings.get("learned_code") is None:
        settings.pop("learned_code", None)
    if settings.get("attention") or settings.get("recall_heads"):
        # Brain.compose turns the recall memory on for these; the bare genome needs it said.
        settings.setdefault("recall", True)
        if settings.get("attention"):
            settings.setdefault("recall_next", True)
    return Genome.composed(inputs, actions, seed=seed, **settings), None


def _describe(card):
    kinds, drum, line, patch, sid, plan = slot_kinds(card)
    settings = card["settings"]
    genome, parts = _genome(card, drum, line, patch, sid, plan)
    order = {region.name: index for index, region in enumerate(genome.regions)}
    apical = bool(settings.get("apical"))

    def size(end):
        name, _, population = end.partition("/")
        region = genome.region(name)
        return len(region.neurons(population)) if population else region.size

    regions = []
    for region in genome.regions:
        role, pathway = _role(region.name)
        regions.append({"name": region.name, "size": region.size, "role": role, "pathway": pathway})

    links, heads, synapses = [], {}, 0
    for edge in genome.projections:
        pre, post = edge.pre.partition("/")[0], edge.post.partition("/")[0]
        count = round(size(edge.pre) * size(edge.post) * edge.density)
        if post != "code":
            synapses += count * (2 if edge.reciprocal else 1)
        if _role(pre)[0] == "memory":
            heads.setdefault(hierarchy._head(pre), post)
            continue
        if post == "code" or pre == "sensory":
            continue      # the learned code's own tables: sensory to code, trace to code
        first, second = (edge.pre, edge.post) if order[pre] <= order[post] else (edge.post, edge.pre)
        links.append({"from": first, "to": second,
                      "kind": "one-way" if not edge.reciprocal else "feedback" if apical else "mutual"})
    for region in genome.regions:
        if region.name != "code" and region.circuit is not None:
            synapses += int(region.circuit.synapses)

    outputs = []
    if parts:
        for name, slots in parts.items():
            count = {}
            for slot in slots:
                count[kinds[slot]] = count.get(kinds[slot], 0) + 1
            if name == hierarchy.COORDINATOR:
                errors = [["Arrangement", "arrangement"]]
            elif name == "rhythm":
                errors = [["Drum patterns", "drum_rows"]]
            elif name == "melody":
                errors = [["Synth lines", "synth_line"], ["Synth controls", "synth_controls"]]
            else:
                errors = [[name.capitalize(), f"instruments.{name}"]]
            outputs.append({"name": name, "slots": len(slots), "parts": count, "errors": errors})
    else:
        count = {}
        for kind in kinds:
            count[kind] = count.get(kind, 0) + 1
        outputs.append({"name": "all", "slots": len(kinds), "parts": count, "errors": [["All parts", "balanced"]]})

    recall = None
    if heads:
        recall = {"heads": len(heads), "bars": settings.get("recall_capacity"),
                  "lands": [heads[head] for head in sorted(heads)]}
    return {
        "version": VERSION, "architecture": settings.get("architecture") or "chain",
        "neurons": sum(region.size for region in genome.regions),
        "synapses": int(settings.get("plastic_synapses") or synapses),
        "inputs": int(settings["inputs"]), "apical": float(settings.get("apical") or 0),
        "regions": regions, "links": links, "outputs": outputs, "recall": recall,
        "reads": bool(settings.get("pathway_reads")),
    }


def describe(card):
    """The layout of the brain a checkpoint card names, or None when the card cannot say."""
    settings = card.get("settings")
    if not isinstance(settings, dict) or not settings.get("slots") or not settings.get("modules"):
        return None
    shaping = {name: value for name, value in settings.items() if name not in _TUNING}
    fields = [shaping, card.get("seed"), card.get("steps"), bool(card.get("beat_rows")), card.get("onoff"),
              (card.get("patch_book") or {}).get("version"), len((card.get("patch_book") or {}).get("samples") or [])]
    if card.get("reserve"):
        fields.append(f"reserve {card['reserve']}")
    key = hashlib.sha256(json.dumps(fields, sort_keys=True, default=str).encode()).hexdigest()
    if key not in _CACHE:
        try:
            _CACHE[key] = _describe(card)
        except Exception:      # an old card with retired settings still gets its curve
            _CACHE[key] = None
    return _CACHE[key]
