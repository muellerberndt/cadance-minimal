"""What the page draws of the creature's brain: a static layout once per checkpoint, a snapshot per bar.

layout(brain): regions in signal order (name, neurons, role), the apical pools and the regions their
neurons sit in, the recall heads and the notebook's capacity.
snapshot(brain, x): per region the settled activation (downsampled to at most WIDTH values), its mean and its
mean change since the last snapshot, per recall head the slots it read (age in bars, share), the apical gain per
pool, the settle's steps, residual, tolerance and budget; with neurons=True also every drawn neuron's activation,
one byte each (the 3D views). The compiled act keeps no per-sweep trace of its settle (cadence's
record_settlements would switch the compiled kernel off and copy three full states per sweep), so the MRI view's
brain waves are made of these per-bar values.
Everything is read from the live brain after its `live` moment; nothing here changes it.

brain3d(brain, name): the 3D view of one checkpoint (served at /api/brain3d, cached in site/.brain3d/): a
position for every drawn neuron, placed by region along the signal path (hearing at one end, motor at the
other, the recall heads and the notebook below), and a sample of the synapses: per pair of connected regions
its strongest, the rest of the budget by strength. Binary, see pack().
"""
from __future__ import annotations

import base64
import hashlib
import json
import struct
import threading
import time
from pathlib import Path

import numpy as np

WIDTH = 48
ROLES = {"sensory": "hears", "code": "meaning", "association": "association", "prefrontal": "prefrontal",
         "motor": "plays", "rhythm_detail": "drum detail", "rhythm_plan": "rhythm plan",
         "melody_detail": "notes and patches", "melody_plan": "melody plan", "phrase_context": "phrase feedback"}


def _regions(brain):
    pops = brain.connectome.populations
    names = []
    for name in pops:
        if "/" in name or name.startswith("next") or name.endswith("_action"):
            continue
        names.append(name)
    order = {"sensory": 0, "code": 1, "rhythm_detail": 10, "rhythm_plan": 11, "melody_detail": 12,
             "melody_plan": 13, "association": 50, "phrase_context": 51, "prefrontal": 60, "motor": 70}
    def key(n):
        if n.startswith("module_"):
            return 10 + int(n.split("_")[1])
        if n.startswith("recall"):
            head = n.split("_")[1:2]
            return 80 + (int(head[0]) if head and head[0].isdigit() else 0)
        return order.get(n, 40)
    return sorted(names, key=key)


def layout(brain) -> dict:
    pops = brain.connectome.populations
    regions = []
    for name in _regions(brain):
        idx = np.asarray(pops[name])
        role = ("coordinates rhythm and melody" if name == "association" and "rhythm_plan" in pops else ROLES.get(name))
        role = role or ("module" if name.startswith("module_") else ("memory read" if name.startswith("recall") else name))
        regions.append({"name": name, "neurons": int(idx.size), "role": role})
    apical = []
    ap = getattr(brain.brain, "apical", None)
    if ap is not None:
        groups = np.asarray(ap.groups)
        for pool in sorted(set(groups[groups >= 0].tolist())):
            members = np.flatnonzero(groups == pool)
            where = {}
            for name in _regions(brain):
                n = int(np.isin(members, np.asarray(pops[name])).sum())
                if n:
                    where[name] = n
            apical.append({"pool": int(pool), "neurons": int(members.size), "regions": where})
    heads = int(brain.recall.heads) if brain.recall is not None else 0
    return {"regions": regions, "apical": apical, "heads": heads,
            "capacity": int(brain.recall.capacity) if brain.recall is not None else 0}


def _down(v: np.ndarray, width: int = WIDTH) -> list:
    v = np.asarray(v, dtype=float)
    if v.size > width:
        edges = np.linspace(0, v.size, width + 1).astype(int)
        v = np.array([v[a:b].mean() for a, b in zip(edges[:-1], edges[1:])])
    return [round(float(a), 3) for a in v]


_BEFORE: dict = {}       # per brain its activation at the last snapshot: each region's change per bar
_INDEX: dict = {}        # per brain its regions' neuron indices


def snapshot(brain, x, neurons=False) -> dict:
    out = {}
    try:
        state = brain._activity()
        act = np.asarray(state.activation)[0]
        pops = brain.connectome.populations
        before = _BEFORE.get(id(brain))
        before = before if before is not None and before.shape == act.shape else None
        held = _INDEX.get(id(brain))
        if held is None or held[0] is not brain.connectome:     # each region's neurons as an index array, once per brain
            held = (brain.connectome, {n: np.asarray(pops[n], dtype=np.int64) for n in _regions(brain)})
            if len(_INDEX) > 8:
                _INDEX.clear()
            _INDEX[id(brain)] = held
        out["regions"] = []
        for n, idx in held[1].items():
            a = act[idx]
            out["regions"].append({"name": n, "mean": round(float(a.mean()), 3), "values": _down(a),
                                   "change": round(float(np.abs(a - before[idx]).mean()), 5) if before is not None else None})
        if len(_BEFORE) > 8:
            _BEFORE.clear()
        _BEFORE[id(brain)] = np.array(act, copy=True)
        if neurons:      # the 3D view: every drawn neuron in brain3d's order, byte = 255 * sqrt(activation clipped to 0..1)
            q = np.sqrt(np.clip(act[drawn(brain)[2]], 0.0, 1.0)) * 255.0 + 0.5
            out["neurons"] = base64.b64encode(q.astype(np.uint8).tobytes()).decode("ascii")
    except Exception as e:  # a view must never stop the music
        out["regions_error"] = str(e)
        state = None
    try:
        heads = []
        ages = np.asarray(brain.recall.ages(), dtype=float).reshape(-1)
        for h in brain.recall.attended_heads or ():
            w = np.asarray(h, dtype=float).reshape(-1)
            top = np.argsort(w)[::-1][:5]
            heads.append([{"age": int(ages[i]) if i < ages.size and np.isfinite(ages[i]) else None, "w": round(float(w[i]), 3)}
                          for i in top if w[i] > 0.02])
        out["heads"] = heads
    except Exception as e:
        out["heads_error"] = str(e)
    try:
        if getattr(brain.brain, "apical", None) is not None and state is not None:
            drive = None
            if getattr(brain, "learned_code", None) is not None:
                # the drive the bar was settled with, as the act keeps it: brain.stimulus(x) would settle the learned
                # code three more times (an echo brain: 2.5 to 4 s a bar); without it no apical readout this bar
                held = getattr(getattr(brain, "basal_ganglia", None), "_drive", None)
                if held is not None and np.shape(held)[-1] == brain.brain.connectome.n:
                    drive = held
            else:
                drive = brain.stimulus(x)
            gain = brain.brain.apical_gain(drive, state) if drive is not None else None
            out["apical"] = [round(float(g), 3) for g in np.asarray(gain).reshape(-1)] if gain is not None else []
    except Exception as e:
        out["apical_error"] = str(e)
    s = brain.last_settlement or {}
    get = s.get if hasattr(s, "get") else (lambda k, d=None: getattr(s, k, d))
    out["settle"] = {"steps": int(np.max(get("steps", 0) or 0)), "residual": float(np.max(get("residual", 0.0) or 0.0)),
                     "budget": int(get("budget", 0) or 0), "tolerance": float(get("tolerance", 0.0) or 0.0),
                     "checks": int(get("residual_checks", 0) or 0)}
    return out


# ======================= the 3D view (/api/brain3d) =======================
MAX_NEURONS = 40000          # every neuron up to this many; above it a seeded sample per region
SYNAPSES = 48000             # synapses drawn
DENSE = 160000               # synapses drawn by the compact view: every pathway a thick bundle
PER_PAIR = 48                # every connected pair of regions keeps at least this many of its strongest synapses
FIRST = 12                   # the first FIRST of every pair lead the list: a page that draws fewer keeps every pathway
CACHE = Path(__file__).resolve().parent / ".brain3d"
VERSION = 3                  # 2: with the MRI view (anatomy, MRI positions); 3: the compact brain and the dense sample
GOLDEN = np.pi * (3.0 - np.sqrt(5.0))

# the signal path: where a region sits on it (0 hears .. 0.68 plays; the memory 0.88), which sets its place on the
# long axis of the layout and the moment in the bar the page's sweep passes it
FLOW = {"sensory": 0.0, "code": 0.12, "association": 0.38, "phrase_context": 0.4, "prefrontal": 0.42,
        "rhythm_plan": 0.5, "melody_plan": 0.5, "rhythm_detail": 0.58, "melody_detail": 0.58, "motor": 0.68}
ANCHOR = {"sensory": (-2.25, 0.05, 0.0), "code": (-1.2, 0.12, 0.0), "association": (0.0, 0.18, 0.0),
          "phrase_context": (0.05, -0.32, 0.3), "prefrontal": (0.12, 0.9, 0.0), "rhythm_plan": (0.62, 0.12, 0.55),
          "melody_plan": (0.62, 0.12, -0.55), "rhythm_detail": (1.08, 0.05, 0.62), "melody_detail": (1.08, 0.05, -0.62),
          "motor": (1.8, 0.05, 0.0)}
NOTEBOOK = {"centre": [0.0, -2.1, 0.0], "radius": 1.3}     # the notebook's slots, a ring the page draws below the heads


def _head(name):
    parts = name.split("_")
    if parts[0] not in ("recall", "next"):
        return -1
    return int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 0


def _flow(name, modules):
    if name in FLOW:
        return FLOW[name]
    if name in modules:
        return 0.2 + 0.14 * modules.index(name) / max(1, len(modules) - 1)
    if _head(name) >= 0:
        return 0.88
    return 0.45


def _role(name):
    h = _head(name)
    if h >= 0:
        what = "memory read" if name.startswith("recall") else "memory next bar"
        return f"{what} time {h + 1}" if name.endswith("_when") else f"{what} {h + 1}"
    if name.startswith("module_"):
        return f"module {name.split('_')[1]}"
    return ROLES.get(name, name)


def _seed(c):
    key = f"{c.n}:{c.synapses}:{','.join(sorted(c.populations))}"
    return int(hashlib.sha256(key.encode()).hexdigest()[:8], 16)


_DRAWN: dict = {}


def drawn(brain):
    """(regions, members, index) of the 3D view: the regions in its order, per region the positions in the region's
    own order of its drawn neurons, and the neuron index of every drawn neuron (region by region). A neuron in two
    populations is drawn once, with the first; neurons in none form a region "other". Every neuron up to
    MAX_NEURONS, else a sample per region, seeded by the brain's shape, so the same checkpoint draws the same."""
    c = brain.connectome
    hit = _DRAWN.get(id(brain))
    if hit is not None and hit[0] is c:
        return hit[1]
    pops = c.populations
    modules = sorted((n for n in pops if n.startswith("module_") and "/" not in n), key=lambda n: (len(n), n))
    names = sorted((n for n in pops if "/" not in n), key=lambda n: (_flow(n, modules), max(_head(n), 0), n.endswith("_when"), n))
    taken = np.zeros(c.n, bool)
    regions = []
    for n in names:
        idx = np.asarray(pops[n], dtype=np.int64)
        pos = np.flatnonzero(~taken[idx])
        taken[idx[pos]] = True
        if pos.size:
            regions.append([n, idx, pos])
    rest = np.flatnonzero(~taken)
    if rest.size:
        regions.append(["other", rest, np.arange(rest.size)])
    total = sum(r[2].size for r in regions)
    if total > MAX_NEURONS:
        rng = np.random.default_rng(_seed(c))
        for r in regions:
            k = max(min(r[2].size, 64), int(r[2].size * MAX_NEURONS / total))
            if k < r[2].size:
                r[2] = np.sort(rng.choice(r[2], k, replace=False))
    out = ([r[0] for r in regions], [(r[1], r[2]) for r in regions], np.concatenate([r[1][r[2]] for r in regions]))
    if len(_DRAWN) > 8:
        _DRAWN.clear()
    _DRAWN[id(brain)] = (c, out)
    return out


# ---------- shapes: unit point sets, the region's size shows in its radius and density ----------
def _fib(n, turn=0.0):
    i = np.arange(n) + 0.5
    y = 1.0 - 2.0 * i / n
    r = np.sqrt(np.maximum(0.0, 1.0 - y * y))
    th = GOLDEN * i + turn
    return np.stack([np.cos(th) * r, y, np.sin(th) * r], 1)


def _ball(n, R):            # a filled ball: directions on a Fibonacci sphere, radii from a low-discrepancy sequence
    r = R * np.cbrt((np.arange(n) * 0.6180339887 + 0.5) % 1.0)
    return _fib(n) * r[:, None]


def _shells(n, R):          # concentric Fibonacci shells, as many as the region's size asks for, each turned
    L = int(np.clip(round(np.cbrt(n) / 4), 1, 4))
    radii = R * (0.45 + 0.55 * (np.arange(L) + 1) / L)
    counts = np.floor(n * radii ** 2 / (radii ** 2).sum()).astype(int)
    counts[-1] += n - counts.sum()
    return np.concatenate([_fib(k, 2.1 * l) * radii[l] for l, k in enumerate(counts) if k > 0])


def _cochlea(n, R):         # the hearing: a coiled tube along the long axis, tightening toward the code
    t = (np.arange(n) + 0.5) / n
    th = 2 * np.pi * 2.6 * t
    rho = R * (1.0 - 0.72 * t)
    tube = 0.17 * R * (1.0 - 0.5 * t)
    ph = GOLDEN * np.arange(n)
    ny, nz = np.cos(th), np.sin(th)
    return np.stack([0.5 * R * t + tube * np.sin(ph), (rho + tube * np.cos(ph)) * ny, (rho + tube * np.cos(ph)) * nz], 1)


def _torus(n, R, tube, u0, u1):   # a ring around the long axis, u from u0 to u1
    u = u0 + (u1 - u0) * (np.arange(n) + 0.5) / n
    v = GOLDEN * np.arange(n) * 3.0
    return np.stack([tube * np.sin(v), (R + tube * np.cos(v)) * np.cos(u), (R + tube * np.cos(v)) * np.sin(u)], 1)


def _helix(n, height, R, phase):  # one strand of a memory head's double helix, top down
    t = (np.arange(n) + 0.5) / n
    th = 2 * np.pi * 3.5 * t + phase
    ph = GOLDEN * np.arange(n)
    tube = 0.22 * R
    return np.stack([(R + tube * np.cos(ph)) * np.cos(th), -height * t + tube * np.sin(ph), (R + tube * np.cos(ph)) * np.sin(th)], 1)


def _positions(brain, regions, members, heads):
    """Every drawn neuron's position, region by region; per region its centre and radius."""
    c = brain.connectome
    rng = np.random.default_rng(_seed(c))
    modules = [n for n in regions if n.startswith("module_")]
    pops = c.populations
    out, meta = [], []
    for name, (idx, pos) in zip(regions, members):
        n_all, n = idx.size, pos.size
        R = float(np.clip(0.042 * np.cbrt(n_all), 0.05, 0.75))
        h = _head(name)
        if h >= 0:                                       # memory: one double helix per head, recall and next intertwined
            x = -1.0 + 2.0 * h / max(1, heads - 1) if heads > 1 else 0.0
            centre = np.array([x, -0.72, 0.0])
            if name.endswith("_when"):
                p = _ball(n_all, 0.035) + [0.0, 0.1, 0.0]
            else:
                p = _helix(n_all, 0.95, 0.17, 0.0 if name.startswith("recall") else np.pi)
            R = 0.2
        elif name == "sensory":
            centre = np.array(ANCHOR[name])
            p = _cochlea(n_all, 0.56)
        elif name == "motor":
            centre = np.array(ANCHOR[name])
            p = np.zeros((n_all, 3))
            rhythm = np.isin(idx, np.asarray(pops.get("motor/rhythm", ()), dtype=np.int64))
            melody = np.isin(idx, np.asarray(pops.get("motor/melody", ()), dtype=np.int64)) & ~rhythm
            other = ~(rhythm | melody)
            # the rhythm half of the plays region faces the rhythm pathway (+z), the melody half the melody one
            for mask, (u0, u1) in ((rhythm, (0.08 * np.pi, 0.92 * np.pi)), (melody, (1.08 * np.pi, 1.92 * np.pi)), (other, (0.0, 2 * np.pi))):
                if mask.any():
                    p[mask] = _torus(int(mask.sum()), 0.5, 0.12, u0, u1)
        else:
            if name in ANCHOR:
                centre = np.array(ANCHOR[name])
            elif name in modules:
                k = modules.index(name)
                f = k / max(1, len(modules) - 1)
                centre = np.array([-0.62 + 0.42 * f, 0.22 * (1 if k % 2 else -1), 0.42 * (1 if k % 2 == 0 else -1)])
            else:                                        # a region the table does not know: a ring around the hub
                j = sum(1 for m in meta if m.get("free"))
                centre = np.array([0.3 * np.cos(j * 2.4), -0.55, 0.9 * np.sin(j * 2.4)])
            p = _shells(n_all, R) if n_all > 400 else _ball(n_all, R)
        p = p[pos] + rng.normal(0.0, 0.01 * max(R, 0.1) / 0.5, (n, 3))
        out.append(p + centre)
        meta.append({"centre": [round(float(v), 3) for v in centre + (p.mean(0) if h >= 0 else 0)], "radius": round(R, 3),
                     "free": name not in ANCHOR and name not in modules and h < 0})
    return np.concatenate(out).astype(np.float32), meta


def _weights(brain):
    """One weight per synapse: the graph's, with the learned code's tables in place of the code region's synapses
    (the graph holds those at zero; the code's dictionary, lateral weights and predictor act in its own phase)."""
    w = np.array(brain.brain.weights, dtype=np.float32)
    code = getattr(brain, "learned_code", None)
    if code is None:
        return w
    try:
        c = brain.connectome
        pre, post = np.asarray(c.pre), np.asarray(c.post)

        def local(index):
            out = np.full(c.n, -1, np.int64)
            out[np.asarray(index, dtype=np.int64)] = np.arange(len(index))
            return out
        s, k, p = local(brain.sensory_index), local(brain.code_index), local(brain.prefrontal_index)
        into = k[post] >= 0
        m = into & (s[pre] >= 0)
        w[m] = np.asarray(code.dictionary)[s[pre[m]], k[post[m]]]
        m = into & (k[pre] >= 0)
        w[m] = np.asarray(code.lateral())[k[pre[m]], k[post[m]]]
        m = into & (p[pre] >= 0)
        w[m] = np.asarray(code.weights)[k[post[m]], p[pre[m]]]
    except Exception as e:  # the graph's weights alone
        print(f"brain3d: learned code weights: {type(e).__name__}: {e}", flush=True)
    return w


def _sample(brain, index, labels, budgets=(SYNAPSES,)):
    """The synapses to draw among the drawn neurons, one sample per budget: per pair of regions with a nonzero
    synapse its strongest (at least PER_PAIR, more for a big pathway: half the budget shared by the square root of
    the pair's size), the rest of the budget by |weight| (no pathway between two regions more than a sixth, none
    inside one region more than a sixteenth: the code's lateral inhibition is dense). Ordered so every prefix keeps
    every pathway. Also the pathways' strengths: summed |weight| per (pre region, post region), all synapses."""
    c = brain.connectome
    where = np.full(c.n, -1, np.int64)
    where[index] = np.arange(index.size)
    w = _weights(brain)
    pre, post = where[np.asarray(c.pre)], where[np.asarray(c.post)]
    syn = np.flatnonzero((pre >= 0) & (post >= 0) & (w != 0))
    aw = np.abs(w[syn])
    R = int(labels.max()) + 1
    pair = labels[pre[syn]].astype(np.int64) * R + labels[post[syn]]
    strength = np.bincount(pair, weights=aw, minlength=R * R).reshape(R, R)
    order = np.argsort(pair, kind="stable")
    groups, starts, counts = np.unique(pair[order], return_index=True, return_counts=True)
    roots = np.sqrt(counts.astype(float))
    inside = groups // R == groups % R
    caps = [np.minimum(counts, np.where(inside, max(PER_PAIR, b // 16), max(PER_PAIR, b // 6))) for b in budgets]
    top = np.max(caps, axis=0)
    ranked = []                                   # per pathway its strongest, strongest first (as many as any budget takes)
    for g, (a, k) in enumerate(zip(starts, counts)):
        seg = order[a:a + k]
        part = seg if top[g] >= k else seg[np.argpartition(-aw[seg], top[g] - 1)[:top[g]]]
        ranked.append(part[np.argsort(-aw[part], kind="stable")])
    kind = np.uint16 if index.size < 65536 else np.uint32
    out = []
    for budget, cap in zip(budgets, caps):
        quota = np.minimum(counts, np.maximum(PER_PAIR, (0.5 * budget * roots / roots.sum()).astype(int)))
        must, rank_must, extra = [], [], []
        for g, part in enumerate(ranked):
            part = part[:cap[g]]
            must.append(part[:quota[g]])
            rank_must.append(np.arange(min(quota[g], part.size)))
            extra.append(part[quota[g]:])
        must = np.concatenate(must)
        rank = np.concatenate(rank_must)
        extra = np.concatenate(extra) if extra else np.zeros(0, np.int64)
        room = max(0, budget - must.size)
        if extra.size > room:
            extra = extra[np.argpartition(-aw[extra], room - 1)[:room]] if room else extra[:0]
        chosen = np.concatenate([must, extra])
        rank = np.concatenate([rank, np.full(extra.size, FIRST)])
        lead = np.lexsort((-aw[chosen], rank >= FIRST))
        chosen, picked = syn[chosen[lead]], aw[chosen[lead]]
        wmax = float(picked.max()) if picked.size else 1.0
        q = np.sign(w[chosen]) * np.maximum(1.0, np.round(127.0 * np.sqrt(np.abs(w[chosen]) / wmax)))
        out.append((pre[chosen].astype(kind), post[chosen].astype(kind), q.astype(np.int8), wmax, len(groups)))
    return out, strength


def brain3d(brain, name=None):
    """(header, arrays) of the 3D view of a loaded brain; pack() makes the bytes the route serves."""
    c = brain.connectome
    regions, members, index = drawn(brain)
    heads = int(brain.recall.heads) if getattr(brain, "recall", None) is not None else 0
    pos, meta = _positions(brain, regions, members, heads)
    labels = np.concatenate([np.full(m[1].size, min(i, 255), np.uint8) for i, m in enumerate(members)])
    snap = set(_regions(brain))
    bins = []
    for rname, (idx, p) in zip(regions, members):
        if rname not in snap:
            bins.append(np.full(p.size, 255, np.uint8))
        elif idx.size > WIDTH:                         # the bin of snapshot()'s downsampled values the neuron falls in
            edges = np.linspace(0, idx.size, WIDTH + 1).astype(int)
            bins.append((np.searchsorted(edges, p, side="right") - 1).astype(np.uint8))
        else:
            bins.append(p.astype(np.uint8))
    samples, strength = _sample(brain, index, labels, (SYNAPSES, DENSE))
    pre, post, weight, wmax, pairs = samples[0]
    modules = [n for n in regions if n.startswith("module_")]
    spans, start = [], 0
    for rname, (idx, p), m in zip(regions, members, meta):
        spans.append({"name": rname, "role": _role(rname), "start": start, "count": int(p.size), "neurons": int(idx.size),
                      "centre": m["centre"], "radius": m["radius"], "flow": _flow(rname, modules), "head": _head(rname),
                      "snapshot": rname in snap})
        start += int(p.size)
    lo, hi = pos.min(0), pos.max(0)
    mid = (lo + hi) / 2
    header = {"version": VERSION, "checkpoint": name, "neurons": int(c.n), "synapses_total": int(c.synapses),
              "drawn": int(index.size), "drawn_synapses": int(pre.size), "pathways": int(pairs), "first": FIRST,
              "weight_max": round(wmax, 5), "weight_code": "sign * 127 * sqrt(|w| / weight_max)",
              "activation_code": "255 * sqrt(clip(activation, 0, 1))", "regions": spans, "heads": heads,
              "capacity": int(brain.recall.capacity) if heads else 0, "notebook": NOTEBOOK if heads else None,
              "bounds": {"centre": [round(float(v), 3) for v in mid],
                         "radius": round(float(max(np.linalg.norm(pos - mid, axis=1).max(), 1e-3)), 3)},
              "made": time.strftime("%Y-%m-%dT%H:%M:%S")}
    arrays = [("position", pos), ("region", labels), ("bin", np.concatenate(bins)), ("pre", pre), ("post", post), ("weight", weight)]
    try:                                           # the compact brain (modes compact and mri): its layout, tissue and dense sample
        dense = samples[1]
        comp, more = compact_brain(brain, regions, members, heads, strength, dense, index)
        header["compact"] = {**comp, "dense_synapses": int(dense[0].size), "dense_weight_max": round(dense[3], 5)}
        arrays += more + [("dense_pre", dense[0]), ("dense_post", dense[1]), ("dense_weight", dense[2])]
    except Exception as e:  # the other views do without it
        header["compact"] = None
        print(f"brain3d {name}: no compact view: {type(e).__name__}: {e}", flush=True)
    try:
        mri, more = mri_arrays(brain, regions, members, heads)
        header["mri"] = mri
        arrays += more
    except Exception as e:  # the other views do without it
        header["mri"] = None
        print(f"brain3d {name}: no MRI view: {type(e).__name__}: {e}", flush=True)
    return header, arrays


def pack(header, arrays) -> bytes:
    """b"B3D1", the header's length (uint32, little endian), the header (JSON, padded to 4 bytes), then the arrays,
    each at a 4-byte boundary; header["arrays"][name] = [offset after the header, bytes, numpy dtype]."""
    blobs, table, at = [], {}, 0
    for name, a in arrays:
        b = np.ascontiguousarray(a).astype(a.dtype.newbyteorder("<"), copy=False).tobytes()
        table[name] = [at, len(b), np.dtype(a.dtype).str]
        pad = (-len(b)) % 4
        blobs.append(b + b"\0" * pad)
        at += len(b) + pad
    head = json.dumps({**header, "arrays": table}, separators=(",", ":")).encode()
    head += b" " * ((-len(head)) % 4)
    return b"B3D1" + struct.pack("<I", len(head)) + head + b"".join(blobs)


_BUILD = {"lock": threading.Lock(), "busy": set(), "failed": {}}


def get3d(name, brain=None, npz=None):
    """For the route: (200, bytes) when the 3D view of checkpoint `name` is in the cache; (202, {"building": name})
    while a thread builds it, from the loaded brain or read from its file; (404, ..) with neither. A name of None
    serves the newest view in the cache (the instrument check shows the last brain)."""
    if name is None:
        found = sorted(CACHE.glob(f"*.v{VERSION}.b3d"), key=lambda f: f.stat().st_mtime) if CACHE.exists() else []
        return (200, found[-1].read_bytes()) if found else (404, {"error": "no brain and no 3D view kept"})
    f = CACHE / f"{name}.v{VERSION}.b3d"
    if f.is_file() and (npz is None or not Path(npz).is_file() or f.stat().st_mtime >= Path(npz).stat().st_mtime):
        return 200, f.read_bytes()
    with _BUILD["lock"]:
        if name in _BUILD["busy"]:
            return 202, {"building": name}
        failed = _BUILD["failed"].get(name)
        if failed and time.time() - failed[0] < 120:
            return 500, {"error": failed[1]}
        if brain is None and (npz is None or not Path(npz).is_file()):
            return 404, {"error": f"no brain {name}"}
        _BUILD["busy"].add(name)
    threading.Thread(target=_build, args=(name, brain, npz, f), daemon=True, name=f"brain3d {name}").start()
    return 202, {"building": name}


def _build(name, brain, npz, f):
    try:
        started = time.time()
        if brain is None:                          # a checkpoint no player holds: read from its file, then let go
            import warnings
            from cadence import Brain
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                brain = Brain.load(npz, backend="cpu")
        header, arrays = brain3d(brain, name)
        data = pack(header, arrays)
        CACHE.mkdir(exist_ok=True)
        part = f.with_suffix(".part")
        part.write_bytes(data)
        part.replace(f)
        print(f"brain3d {name}: {header['drawn']} neurons, {header['drawn_synapses']} synapses of {header['synapses_total']} "
              f"({header['pathways']} pathways), {len(data) // 1024} KB, {time.time() - started:.1f} s", flush=True)
    except Exception as e:
        _BUILD["failed"][name] = (time.time(), f"{type(e).__name__}: {e}")
        print(f"brain3d {name}: {type(e).__name__}: {e}", flush=True)
    finally:
        with _BUILD["lock"]:
            _BUILD["busy"].discard(name)


# ======================= the MRI view: the regions on a human brain's anatomy =======================
# A T1-like volume of a human brain (two hemispheres with sulci, the temporal lobes, cerebellum, brain stem,
# ventricles, deep nuclei, hippocampus and entorhinal cortex, skull and scalp), the same for every checkpoint, and
# every drawn neuron placed in the zone of its region: hearing on the superior temporal gyrus, the learned code on
# the temporo-parietal cortex, association across the parietal lobe, prefrontal in the frontal lobe, plays on the
# precentral gyrus (its rhythm part medially, its melody part laterally), the rhythm pathways in the basal ganglia
# (plan) and the cerebellum (detail), the melody pathways in the right temporal lobe, the memory heads in the
# hippocampus (the first half of the heads left, the second right, each a segment along it), the notebook's slots
# in the entorhinal cortex beside it. Coordinates: x left to right, y down to up, z back to front; one unit is
# about 82 mm. Inside a zone a region's neurons follow the zone's long axis in their own order (a topographic map).
MRI_VOXEL = 0.02
MRI_BOX = ((-0.94, 0.94), (-1.06, 0.9), (-1.04, 1.04))
_ANATOMY: dict = {}


def _ell(x, y, z, c, r):
    return np.sqrt(((x - c[0]) / r[0]) ** 2 + ((y - c[1]) / r[1]) ** 2 + ((z - c[2]) / r[2]) ** 2)


def _tube(x, y, z, a, b, r):
    p = np.stack([x - a[0], y - a[1], z - a[2]], -1)
    d = np.asarray(b, float) - np.asarray(a, float)
    t = np.clip((p @ d) / (d @ d), 0.0, 1.0)
    return np.linalg.norm(p - t[..., None] * d, axis=-1) / r


def anatomy():
    """(intensity, zones, grid): the T1-like volume as uint8 [z, y, x] (x fastest), the zone masks the neurons
    are placed in, and the voxel centres. Built once per process (about a second), the same for every brain."""
    if _ANATOMY:
        return _ANATOMY["out"]
    from scipy import ndimage
    v = MRI_VOXEL
    xs, ys, zs = (np.arange(a + v / 2, b, v) for a, b in MRI_BOX)
    Z, Y, X = np.meshgrid(zs, ys, xs, indexing="ij")
    ax = np.abs(X)
    # the cerebrum: per hemisphere a main body, the frontal and occipital poles and the temporal lobe
    d_temporal = _ell(ax, Y, Z, (0.5, -0.3, 0.12), (0.28, 0.25, 0.48))
    d_main = np.minimum.reduce([_ell(ax, Y, Z, (0.38, 0.1, -0.02), (0.42, 0.64, 0.92)),
                                _ell(ax, Y, Z, (0.33, 0.02, 0.42), (0.37, 0.56, 0.56)),
                                _ell(ax, Y, Z, (0.30, 0.12, -0.6), (0.32, 0.46, 0.40))])
    bridge = (Y < 0.25) & (Y > -0.22) & (np.abs(Z) < 0.5)                     # callosum and diencephalon join the halves
    cerebrum = (np.minimum(d_main, d_temporal) < 1) & ((ax > 0.016) | bridge) & ~((Z < -0.36) & (Y < -0.25))
    lobes = np.minimum.reduce([_ell(ax, Y, Z, (0.24, -0.5, -0.62), (0.3, 0.2, 0.3)), _ell(X, Y, Z, (0, -0.48, -0.62), (0.12, 0.2, 0.27))])
    cerebellum = (lobes < 1) & ~cerebrum
    stem = ((_tube(X, Y, Z, (0, -0.12, -0.26), (0, -1.0, -0.42), 0.105) < 1) | (_ell(X, Y, Z, (0, -0.45, -0.3), (0.15, 0.13, 0.14)) < 1)) & ~cerebrum & ~cerebellum
    # deep structures (inside the cerebrum)
    ventricle = ((_ell(ax, Y, Z, (0.11, 0.14, -0.02), (0.05, 0.07, 0.36)) < 1) | (_ell(ax, Y, Z, (0.1, 0.12, 0.25), (0.06, 0.08, 0.12)) < 1)
                 | ((ax < 0.012) & (_ell(X, Y, Z, (0, -0.02, -0.02), (0.02, 0.1, 0.14)) < 1)))
    callosum = (_ell(X, Y, Z, (0, 0.235, -0.02), (0.4, 0.045, 0.5)) < 1) & ~ventricle
    basal = ((_ell(ax, Y, Z, (0.21, 0.03, 0.1), (0.07, 0.11, 0.2)) < 1) | (_ell(ax, Y, Z, (0.13, 0.12, 0.22), (0.045, 0.07, 0.1)) < 1)) & ~ventricle
    thalamus = (_ell(ax, Y, Z, (0.075, 0.0, -0.1), (0.07, 0.08, 0.12)) < 1) & ~ventricle
    hippocampus = _tube(ax, Y, Z, (0.3, -0.31, 0.13), (0.25, -0.16, -0.27), 0.05) < 1
    entorhinal = (_ell(ax, Y, Z, (0.27, -0.4, 0.24), (0.06, 0.05, 0.09)) < 1) & ~hippocampus
    deep = basal | thalamus | hippocampus | entorhinal | ventricle | callosum
    cerebrum |= deep
    brain = cerebrum | cerebellum | stem
    depth = ndimage.distance_transform_edt(cerebrum) * v                     # below the cerebrum's surface
    outside = ndimage.distance_transform_edt(~brain) * v                     # out from the brain
    # gyri and sulci: thin sheets where two warped wave fields cross zero, cut from the surface inward and lined
    # with cortex; the lateral fissure between the temporal lobe and the frontal and parietal lobes
    fold = (np.sin(29 * Y + 6 * np.sin(5 * Z) + 3 * X) + np.sin(27 * Z + 5 * np.sin(6 * Y) - 2 * X) + np.sin(17 * (Y + Z) + 4 * np.sin(7 * X))) / 3
    fold2 = np.sin(37 * Z + 9 * np.sin(3 * Y + 2 * X) + 5 * np.sin(5 * Y))
    fissure = (np.abs(Y - (-0.13 + 0.22 * Z + 0.12 * (ax - 0.6))) < 0.02) & (ax > 0.42) & (Z > -0.3) & (Z < 0.52) & (depth < 0.13)
    sulcus = cerebrum & ~deep & (ax > 0.05) & (((depth < 0.13) & (np.abs(fold) < 0.13)) | ((depth < 0.08) & (np.abs(fold2) < 0.09)) | fissure)
    gray = cerebrum & ~deep & ~sulcus & ((depth < 0.065) | ((np.abs(fold) < 0.42) & (depth < 0.16)) | ((np.abs(fold2) < 0.3) & (depth < 0.1)))
    white = cerebrum & ~deep & ~sulcus & ~gray
    t1 = np.zeros(X.shape, np.float32)
    shell = ~brain & (Y > -0.7)
    t1[shell & (outside < 0.045)] = 22                                       # cerebrospinal fluid
    t1[shell & (outside >= 0.045) & (outside < 0.09)] = 38                   # bone
    t1[shell & (outside >= 0.062) & (outside < 0.074)] = 150                 # its marrow
    t1[shell & (outside >= 0.09) & (outside < 0.13)] = 120                   # scalp
    t1[sulcus] = 26
    t1[gray] = 104
    t1[white] = (176 + 9 * np.sin(3 * X + 1) * np.cos(2.5 * Z) + 6 * np.sin(4 * Y))[white]       # a gentle bias field
    folia = np.sin(70 * Y + 25 * Z + 10 * X) > 0.1
    t1[cerebellum] = np.where(folia[cerebellum] & (ndimage.distance_transform_edt(cerebellum)[cerebellum] * v > 0.03), 160, 100)
    t1[stem] = 156
    t1[basal] = 128
    t1[thalamus] = 138
    t1[hippocampus | entorhinal] = 112
    t1[callosum] = 196
    t1[ventricle] = 16
    t1 += np.random.default_rng(7).normal(0.0, 4.0, t1.shape).astype(np.float32) * (t1 > 0)
    t1 = ndimage.gaussian_filter(t1, 0.55)
    side = np.sign(X)
    cortex = gray | (cerebrum & ~deep & ~white & ~sulcus)
    temporal = (d_temporal < 1) & (Y < -0.08)
    motor_band = cortex & (Y > 0.05) & (np.abs(Z - (0.1 - 0.14 * (Y - 0.35))) < 0.06)
    zones = {
        "stg": cortex & temporal & (Y > -0.34) & (ax > 0.42) & (Z > -0.25) & (Z < 0.45),
        "tpj": cortex & ~temporal & (Z > -0.62) & (Z < -0.12) & (Y > -0.2) & (Y < 0.34) & (ax > 0.34),
        "parietal": cortex & (Z > -0.58) & (Z < -0.05) & (Y > 0.38),
        "frontal": cortex & (Z > 0.42),
        "m1": motor_band, "m1_medial": motor_band & (ax < 0.3) & (Y > 0.5), "m1_lateral": motor_band & ~((ax < 0.3) & (Y > 0.5)),
        "bg": basal, "cerebellum": cerebellum, "cingulate": cortex & (ax < 0.1) & (Y > 0.05) & (Y < 0.36) & (np.abs(Z) < 0.45),
        "rtl_anterior": cortex & temporal & (X > 0) & (Y <= -0.34) & (Z > 0.12),
        "rtl_posterior": cortex & temporal & (X > 0) & (Y <= -0.34) & (Z <= 0.12),
        "modules": cortex & (X < 0) & (Z > -0.5) & (Z < 0.05) & (Y > -0.3) & (Y < 0.15),
        "hippocampus": hippocampus, "entorhinal": entorhinal,
        "insula": cerebrum & ~deep & (ax > 0.3) & (ax < 0.42) & (Y > -0.1) & (Y < 0.12) & (Z > -0.1) & (Z < 0.3),
    }
    surface = brain & (ndimage.binary_erosion(brain) == 0)
    out = {"t1": np.clip(t1, 0, 255).astype(np.uint8), "zones": zones, "grid": (X, Y, Z), "side": side, "surface": surface}
    _ANATOMY["out"] = out
    return out


def _zone_points(mask, X, Y, Z, side=None):
    """The voxel centres of a zone (one side when asked), ordered along its longest axis."""
    m = mask if side is None else mask & ((X < 0) if side < 0 else (X > 0))
    p = np.stack([X[m], Y[m], Z[m]], 1)
    if len(p) < 2:
        return p
    c = p - p.mean(0)
    axis = np.linalg.svd(c, full_matrices=False)[2][0]
    return p[np.argsort(c @ axis, kind="stable")]


def _patch(pts, n):
    """A region much smaller than its zone keeps to a patch around the zone's centre (about six voxels per
    neuron), still ordered along the zone's axis: its activity reads as one blob, as in an fMRI."""
    k = int(np.clip(6 * n, 60, len(pts)))
    if k >= len(pts):
        return pts
    near = np.argsort(np.linalg.norm(pts - pts.mean(0), axis=1), kind="stable")[:k]
    return pts[np.sort(near)]


def _spread(pts, n, rng, start=0.0, end=1.0):
    """n positions along an ordered point list (between the fractions start and end), jittered in their voxel."""
    if not len(pts) or not n:
        return np.zeros((n, 3))
    k = np.clip(((start + (end - start) * (np.arange(n) + 0.5) / n) * len(pts)).astype(int), 0, len(pts) - 1)
    return pts[k] + rng.uniform(-0.5, 0.5, (n, 3)) * MRI_VOXEL


def mri_positions(brain, regions, members, heads):
    """The MRI position of every drawn neuron (in brain3d's order) and of the notebook's slots."""
    A = anatomy()
    X, Y, Z = A["grid"]
    zones = A["zones"]
    rng = np.random.default_rng(_seed(brain.connectome) + 1)
    pops = brain.connectome.populations
    modules = [n for n in regions if n.startswith("module_")]
    cache = {}

    def pts(zone, side=None):
        key = (zone, side)
        if key not in cache:
            cache[key] = _zone_points(zones[zone], X, Y, Z, side)
        return cache[key]

    def both(zone, n, start=0.0, end=1.0):          # interleaved over the hemispheres: both carry the whole map
        out = np.zeros((n, 3))
        for s, sl in ((-1, slice(0, None, 2)), (1, slice(1, None, 2))):
            k = len(out[sl])
            out[sl] = _spread(_patch(pts(zone, s), k), k, rng, start, end)
        return out
    out = []
    for name, (idx, pos) in zip(regions, members):
        n = pos.size
        h = _head(name)
        if h >= 0:                                   # the hippocampus: half the heads left, half right, a segment each
            half = max(1, heads // 2)
            s = -1 if h < half else 1
            seg, of = (h % half), half
            if name.endswith("_when"):
                p = _spread(pts("hippocampus", s), n, rng, seg / of, seg / of + 0.02)
            else:
                p = _spread(pts("hippocampus", s), n, rng, seg / of, (seg + 1) / of)
        elif name == "motor":
            p = np.zeros((n, 3))
            ids = idx[pos]
            rhythm = np.isin(ids, np.asarray(pops.get("motor/rhythm", ()), dtype=np.int64))
            melody = np.isin(ids, np.asarray(pops.get("motor/melody", ()), dtype=np.int64)) & ~rhythm
            for mask, zone in ((rhythm, "m1_medial"), (melody, "m1_lateral"), (~(rhythm | melody), "m1")):
                if mask.any():
                    p[mask] = both(zone, int(mask.sum()))
        elif name in modules:
            k = modules.index(name)
            p = _spread(pts("modules", -1), n, rng, k / len(modules), (k + 1) / len(modules))
        else:
            zone, bilateral = {"sensory": ("stg", True), "code": ("tpj", True), "association": ("parietal", True),
                               "prefrontal": ("frontal", True), "rhythm_plan": ("bg", True), "rhythm_detail": ("cerebellum", True),
                               "melody_plan": ("rtl_anterior", False), "melody_detail": ("rtl_posterior", False),
                               "phrase_context": ("cingulate", True)}.get(name, ("insula", True))
            p = both(zone, n) if bilateral else _spread(_patch(pts(zone), n), n, rng)
        out.append(p)
    cap = int(brain.recall.capacity) if heads else 0
    book = np.zeros((cap, 3))
    for s, sl in ((-1, slice(0, None, 2)), (1, slice(1, None, 2))):
        book[sl] = _spread(pts("entorhinal", s), len(book[sl]), rng)
    return np.concatenate(out).astype(np.float32), book.astype(np.float32)


def mri_arrays(brain, regions, members, heads):
    """The MRI part of the 3D view: header fields and arrays."""
    A = anatomy()
    t1 = A["t1"]
    X, Y, Z = A["grid"]
    pos, book = mri_positions(brain, regions, members, heads)
    surf = np.stack([X[A["surface"]], Y[A["surface"]], Z[A["surface"]]], 1)
    keep = np.random.default_rng(3).permutation(len(surf))[:24000]
    head = {"shape": [int(t1.shape[2]), int(t1.shape[1]), int(t1.shape[0])], "box": [list(b) for b in MRI_BOX], "voxel": MRI_VOXEL,
            "mm_per_unit": 82}
    return head, [("mri_position", pos), ("mri_notebook", book), ("anatomy", t1), ("surface", surf[np.sort(keep)].astype(np.float32))]


# ======================= the compact brain: the Cadence brain itself as one brain-shaped volume =======================
# Every region a solid lobe of one brain-shaped volume (two hemispheres, a cerebellum, a stem), its volume in
# proportion to its neurons (at least MIN_SHARE of the brain, so a 64-neuron region still reads), its neighbours by
# how strongly they are wired: seeds placed along the signal path (hearing at the back, plays at the front, the memory
# heads below, the notebook under them), pulled together by springs as strong as their pathways' summed |weight|,
# then the volume split among them by a capacity-constrained power diagram. Its MRI tissue is made of the network:
# grey = drawn neurons per voxel, white = the dense sample's synapses along their bundled paths (summed |weight|),
# dark lines = the borders between lobes. The compact mode draws the same layout with the dense sample.
CB_VOXEL = 0.025
CB_BOX = ((-0.96, 0.96), (-1.08, 0.8), (-1.12, 1.12))
MIN_SHARE = 0.006
NOTEBOOK_SHARE = 0.008
BUNDLE, BEND = 0.85, 0.28       # the strands' path (the page draws the same): pulled toward the pathway's axis


def _envelope(X, Y, Z):
    ax = np.abs(X)
    cerebrum = (_ell(ax, Y, Z, (0.4, 0.06, 0.0), (0.46, 0.62, 1.0)) < 1) & ((ax > 0.015) | (Y < 0.22))
    cerebellum = _ell(X, Y, Z, (0.0, -0.5, -0.62), (0.55, 0.24, 0.32)) < 1
    stem = _tube(X, Y, Z, (0, -0.3, -0.22), (0, -0.98, -0.34), 0.1) < 1
    return cerebrum | cerebellum | stem


def _seed_of(name, modules, heads):
    fixed = {"sensory": (0, 0.2, -0.78), "code": (0, 0.25, -0.42), "association": (0, 0.2, 0.0), "phrase_context": (0, -0.05, 0.05),
             "prefrontal": (0, 0.5, 0.15), "rhythm_plan": (0.35, 0.25, 0.32), "melody_plan": (-0.35, 0.25, 0.32),
             "rhythm_detail": (0.4, 0.15, 0.58), "melody_detail": (-0.4, 0.15, 0.58), "motor": (0, 0.2, 0.82), "notebook": (0, -0.66, 0.0)}
    if name in fixed:
        return fixed[name]
    h = _head(name)
    if h >= 0:
        z = -0.55 + 1.1 * h / max(1, heads - 1)
        x = 0.28 if name.startswith("recall") else -0.28
        return (0.5 * x, -0.18, z) if name.endswith("_when") else (x, -0.4, z)
    if name in modules:
        k = modules.index(name)
        return (0.3 * (1 if k % 2 else -1), 0.25, -0.25 + 0.2 * k / max(1, len(modules) - 1))
    return (0.0, 0.0, 0.3)


def compact_brain(brain, regions, members, heads, strength, dense, index):
    """The compact view: header fields and arrays (positions, the notebook's slots, the T1 volume, the lobe ids)."""
    from scipy import ndimage
    v = CB_VOXEL
    xs, ys, zs = (np.arange(a + v / 2, b, v) for a, b in CB_BOX)
    Z, Y, X = np.meshgrid(zs, ys, xs, indexing="ij")
    inside = _envelope(X, Y, Z)
    V = np.stack([X[inside], Y[inside], Z[inside]], 1)
    modules = [n for n in regions if n.startswith("module_")]
    names = list(regions) + ["notebook"]
    R = len(names)
    counts = np.array([m[1].size for m in members] + [0], float)
    share = counts / max(1.0, counts.sum())
    share = np.where(counts >= 32, np.maximum(share, MIN_SHARE), share)     # the floor for regions, not for 2-neuron ports
    share[-1] = NOTEBOOK_SHARE if heads else 0.0
    share /= share.sum()
    radius = np.cbrt(share * len(V) * v ** 3 * 3 / (4 * np.pi))
    anchor = np.array([_seed_of(n, modules, heads) for n in names], float)
    P = anchor.copy()
    S = np.zeros((R, R))                                       # wiring: summed |weight| both ways, the strongest pair 1
    k = min(strength.shape[0], R - 1)
    S[:k, :k] = strength[:k, :k] + strength[:k, :k].T
    np.fill_diagonal(S, 0)
    S /= max(S.max(), 1e-12)
    for b in range(R - 1):                                     # the notebook is wired to every memory port
        if _head(names[b]) >= 0:
            S[b, -1] = S[-1, b] = 0.5
    for _ in range(300):                                       # springs between wired lobes, no overlap, the seeds hold
        d = P[:, None, :] - P[None, :, :]
        dist = np.linalg.norm(d, axis=-1) + np.eye(R)
        want = radius[:, None] + radius[None, :]
        pull = S * (dist - 0.9 * want) * 0.04
        push = np.where(dist < want, (want - dist) * 0.25, 0.0)
        np.fill_diagonal(pull, 0)
        np.fill_diagonal(push, 0)
        P += ((-pull + push)[:, :, None] * d / dist[:, :, None]).sum(1) + (anchor - P) * 0.03
    target = share * len(V)                                    # the power diagram: each lobe takes its share of the volume
    wgt = np.zeros(R)
    c = P.copy()
    V32 = V.astype(np.float32)
    for it in range(36):
        # |v - c|^2 - w without |v|^2 (the same for every lobe): one matrix product per round
        lab = np.argmin((c * c).sum(1)[None, :].astype(np.float32) - 2.0 * (V32 @ c.T.astype(np.float32)) - wgt[None, :].astype(np.float32), 1)
        got = np.bincount(lab, minlength=R).astype(float)
        wgt += 0.02 * (target - got) / np.maximum(target, 1.0)
        if it in (8, 16, 24):                                  # a gentle pull toward each lobe's centre of mass
            for r in range(R):
                if got[r]:
                    c[r] = 0.6 * c[r] + 0.4 * V[lab == r].mean(0)
    lobes = np.full(X.shape, 255, np.uint8)
    lobes[inside] = np.minimum(lab, 254)
    rng = np.random.default_rng(_seed(brain.connectome) + 2)
    pos = []
    for r, (idx, p) in enumerate(members):                    # every region's neurons in its lobe, along the lobe's axis
        pts = _zone_points(lobes == r, X, Y, Z)
        pos.append(_spread(pts, p.size, rng) if len(pts) else np.repeat(c[r][None], p.size, 0))
    pos = np.concatenate(pos).astype(np.float32)
    cap = int(brain.recall.capacity) if heads else 0
    book = _spread(_zone_points(lobes == R - 1, X, Y, Z), cap, rng).astype(np.float32) if cap else np.zeros((0, 3), np.float32)
    starts = np.concatenate([[0], np.cumsum([m[1].size for m in members])])
    centres = np.array([pos[starts[r]:starts[r + 1]].mean(0) for r in range(R - 1)] + [book.mean(0) if cap else c[-1]])
    # the tissue: grey = neurons per voxel, white = the dense sample's bundled paths (summed |weight|)
    shape = X.shape
    def cell(p):
        i = np.floor((p - np.array([b[0] for b in CB_BOX])) / v).astype(int)
        ok = (i >= 0).all(1) & (i[:, 0] < shape[2]) & (i[:, 1] < shape[1]) & (i[:, 2] < shape[0])
        return np.where(ok, (i[:, 2] * shape[1] + i[:, 1]) * shape[2] + i[:, 0], -1)
    n_vox = X.size
    k = cell(pos)
    grey = np.bincount(k[k >= 0], minlength=n_vox).reshape(shape).astype(np.float32)
    dpre, dpost, dq = dense[0].astype(np.int64), dense[1].astype(np.int64), dense[2]
    labels = np.concatenate([np.full(m[1].size, i, np.int64) for i, m in enumerate(members)])
    A, B = pos[dpre], pos[dpost]
    ra, rb = labels[dpre], labels[dpost]
    centre = (pos.min(0) + pos.max(0)) / 2
    m = (A + B) / 2
    o = m - centre
    o[:, 1] += 0.25
    c0 = m + o / (np.linalg.norm(o, axis=1, keepdims=True) + 1e-4) * np.linalg.norm(A - B, axis=1, keepdims=True) * BEND
    axis = (centres[ra] + centres[rb]) / 2
    c1 = axis + (axis - centre) * 0.08
    C = np.where((ra != rb)[:, None], c0 + (c1 - c0) * BUNDLE, c0)
    aw = (np.abs(dq.astype(np.float32)) / 127.0) ** 2
    white = np.zeros(n_vox, np.float64)
    for t in (np.arange(16) + 0.5) / 16:
        pt = (1 - t) ** 2 * A + 2 * (1 - t) * t * C + t ** 2 * B
        kk = cell(pt)
        ok = kk >= 0
        white += np.bincount(kk[ok], weights=aw[ok], minlength=n_vox)
    white = white.reshape(shape).astype(np.float32)
    grey = ndimage.gaussian_filter(grey, 1.0)
    white = ndimage.gaussian_filter(white, 0.8)
    g = np.clip(grey / max(np.percentile(grey[inside], 98), 1e-6), 0, 1)
    w = np.clip(np.log1p(white / max(np.percentile(white[white > 0], 60), 1e-6)) / np.log1p(20.0), 0, 1)
    border = np.zeros(shape, bool)
    for ax_ in range(3):
        for sh in (1, -1):
            border |= (np.roll(lobes, sh, axis=ax_) != lobes) & inside
    outside = ndimage.distance_transform_edt(~inside) * v
    t1 = np.zeros(shape, np.float32)
    t1[inside] = (30 + 80 * g + 125 * w)[inside]
    t1[border] *= 0.5
    shell = ~inside & (Y > -0.75)
    t1[shell & (outside < 0.04)] = 20
    t1[shell & (outside >= 0.04) & (outside < 0.08)] = 36
    t1[shell & (outside >= 0.055) & (outside < 0.066)] = 140
    t1[shell & (outside >= 0.08) & (outside < 0.115)] = 112
    t1 += np.random.default_rng(9).normal(0.0, 3.5, shape).astype(np.float32) * (t1 > 0)
    t1 = ndimage.gaussian_filter(t1, 0.5)
    lo, hi = pos.min(0), pos.max(0)
    head = {"shape": [int(shape[2]), int(shape[1]), int(shape[0])], "box": [list(b) for b in CB_BOX], "voxel": v, "mm_per_unit": 82,
            "lobes": [{"name": n, "role": (_role(n) if n != "notebook" else "notebook"), "centre": [round(float(x), 3) for x in centres[r]],
                       "share": round(float(share[r]), 4)} for r, n in enumerate(names)],
            "bounds": {"centre": [round(float(x), 3) for x in (lo + hi) / 2], "radius": round(float(np.linalg.norm(pos - (lo + hi) / 2, axis=1).max()), 3)},
            "bundle": BUNDLE, "bend": BEND}
    return head, [("compact_position", pos), ("compact_notebook", book), ("compact_t1", np.clip(t1, 0, 255).astype(np.uint8)), ("compact_lobes", lobes)]
