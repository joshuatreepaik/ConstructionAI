"""Web server for the demo. Upload a vector PDF drawing set, trace one symbol,
find every instance of it - or auto-count everything the sheet's legend defines.

    .venv/bin/python app.py     ->  http://localhost:8642

WHAT THIS FILE IS
    HTTP routing, per-document caching, and the v1 detection path. It holds no
    detection logic of its own beyond assembling the exemplar from the traced
    box (`_main_cluster`). The UI lives in web/index.html.

TWO ENGINES, ONE UI
    The staged pipeline in pipeline/ is the engine ("v2", the default). The
    original engine in oneshot/ is kept frozen as the experimental control;
    pass engine: "v1" to run it. `scripts/compare.py` runs identical queries
    through both and prints the results side by side.

NOTHING HERE IS TUNED TO ONE DRAWING SET
    Sheet names come from title blocks, legend sheets are found by their own
    headings, drawing scale is parsed from each sheet's scale note, and
    excluded regions are anchored on printed captions ("PLAN NORTH", "KEY
    PLAN"). Upload a different firm's drawings and the same code applies.
"""

import math
import os
import pathlib
import threading
import time

from flask import Flask, jsonify, request, Response
import numpy as np
import pymupdf
from scipy.spatial import cKDTree

from oneshot.extract import extract_primitives, extract_words, primitives_in_box
from oneshot.engine import Scene, detect
from oneshot.legend import harvest, _find_headings, LegendSymbol
from oneshot.doors import find_doors, looks_like_door, width_label
from oneshot.regions import (exclusion_zones, outside_zones,
                             plan_scale_pt_per_ft, sheet_number)

RENDER_ZOOM = 2.0
UPLOAD_DIR = "uploads"
DEFAULT_PDF = "data.pdf"
LEGEND_HEADINGS = ("SYMBOLS LIST", "DRAWING LEGEND", "DOOR LEGEND",
                   "WALL TYPE LEGEND", "LEGEND")

app = Flask(__name__)
os.makedirs(UPLOAD_DIR, exist_ok=True)


class DocState:
    """One open document + per-page caches. Swapped wholesale on upload."""

    def __init__(self, path):
        self.path = path
        self.name = os.path.basename(path)
        self.doc = pymupdf.open(path)
        self.lock = threading.Lock()
        self.prims, self.scenes, self.wordc, self.zones = {}, {}, {}, {}
        self.pngs, self.scales = {}, {}
        self.pages = []
        for i in range(len(self.doc)):
            page = self.doc[i]
            words = extract_words(page)
            num = sheet_number(page)
            is_legend = bool(_find_headings(words, LEGEND_HEADINGS))
            label = f"p{i + 1}" + (f" — {num}" if num else "") + \
                    (" [legend]" if is_legend else "")
            self.pages.append({"index": i, "label": label,
                               "is_legend": is_legend})

    def get_prims(self, i):
        if i not in self.prims:
            self.prims[i] = extract_primitives(self.doc[i])
        return self.prims[i]

    def get_scene(self, i):
        with self.lock:
            if i not in self.scenes:
                self.scenes[i] = Scene(self.get_prims(i))
            return self.scenes[i]

    def get_word_centers(self, i):
        if i not in self.wordc:
            ws = extract_words(self.doc[i])
            self.wordc[i] = (np.array([[(w[0] + w[2]) / 2, (w[1] + w[3]) / 2]
                                       for w in ws], dtype=np.float64)
                             if ws else np.zeros((0, 2)))
        return self.wordc[i]

    def get_zones(self, i):
        if i not in self.zones:
            self.zones[i] = exclusion_zones(self.doc[i])
        return self.zones[i]

    def get_scale(self, i):
        if i not in self.scales:
            self.scales[i] = plan_scale_pt_per_ft(self.doc[i])
        return self.scales[i]

    def legend_pages(self):
        return [p["index"] for p in self.pages if p["is_legend"]]


STATE: DocState | None = DocState(DEFAULT_PDF) if os.path.exists(DEFAULT_PDF) else None
_state_lock = threading.Lock()


def _meta():
    if STATE is None:
        return {"name": None, "pages": []}
    return {"name": STATE.name, "pages": STATE.pages}


@app.get("/meta")
def meta():
    return jsonify(_meta())


@app.post("/upload")
def upload():
    global STATE
    f = request.files.get("file")
    if f is None or not f.filename.lower().endswith(".pdf"):
        return jsonify({"error": "upload a .pdf file"})
    path = os.path.join(UPLOAD_DIR, os.path.basename(f.filename))
    f.save(path)
    try:
        new_state = DocState(path)
    except Exception as e:
        return jsonify({"error": f"could not open PDF: {e}"})
    n_vec = sum(1 for _ in new_state.doc[0].get_drawings())
    with _state_lock:
        STATE = new_state
    out = _meta()
    if n_vec < 50:
        out["warning"] = ("this PDF looks scanned/rasterized - the engine "
                          "needs vector (CAD-exported) PDFs; results may be empty")
    return jsonify(out)


@app.get("/page/<int:page_index>.png")
def page_png(page_index):
    if STATE is None:
        return Response(status=404)
    if page_index not in STATE.pngs:
        pix = STATE.doc[page_index].get_pixmap(
            matrix=pymupdf.Matrix(RENDER_ZOOM, RENDER_ZOOM))
        STATE.pngs[page_index] = pix.tobytes("png")
    return Response(STATE.pngs[page_index], mimetype="image/png")


def _main_cluster(prims, thr, box):
    """The spatially-connected primitive group nearest the trace-box center.

    Connectivity = primitives touching (sampled at endpoints + midpoint);
    big centered clusters beat small off-center jamb blocks and wall dashes.
    """
    if len(prims) <= 3:
        return prims
    pts = np.array([q for p in prims for q in (p.p0, p.mid, p.p1)])
    pairs = cKDTree(pts).query_pairs(r=thr, output_type="ndarray")
    parent = list(range(len(prims)))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for a, b in pairs:
        ra, rb = find(int(a) // 3), find(int(b) // 3)
        if ra != rb:
            parent[ra] = rb
    groups = {}
    for i in range(len(prims)):
        groups.setdefault(find(i), []).append(i)
    center = np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2])
    candidates = [g for g in groups.values() if len(g) >= 2] or list(groups.values())

    def cluster_score(g):
        gp = pts[[3 * i + k for i in g for k in range(3)]]
        extent = float(np.linalg.norm(gp.max(axis=0) - gp.min(axis=0)))
        dist = float(np.linalg.norm(gp.mean(axis=0) - center))
        return extent - 1.5 * dist

    best = max(candidates, key=cluster_score)
    return [prims[i] for i in best]


def _veto_by_text(dets, page_index, sym_r):
    wc = STATE.get_word_centers(page_index)
    if not len(wc) or not dets:
        return dets
    centers = np.array([[d.x, d.y] for d in dets])
    radii = 0.55 * sym_r * np.array([d.scale for d in dets])
    dmin = np.sqrt(((wc[None, :, :] - centers[:, None, :]) ** 2).sum(-1)).min(axis=1)
    return [d for d, dm, r in zip(dets, dmin, radii) if dm >= r]


def _door_width_label(page_index, width_pt):
    scale = STATE.get_scale(page_index)
    if scale:
        return width_label(width_pt, scale)
    return f"{width_pt:.0f}pt"       # no scale note found on sheet


@app.post("/reconcile")
def do_reconcile():
    """Cross-check a traced symbol's counts against the panel schedules (v2)."""
    if STATE is None:
        return jsonify({"error": "no document loaded - upload a PDF first"})
    q = request.get_json()
    from pipeline import api as v2api
    return jsonify(v2api.reconcile(
        STATE, int(q["page"]), tuple(float(v) for v in q["box"]),
        schedule_page=q.get("schedule_page")))


@app.post("/detect")
def do_detect():
    if STATE is None:
        return jsonify({"error": "no document loaded - upload a PDF first"})
    q = request.get_json()
    page_index = int(q["page"])
    box = tuple(float(v) for v in q["box"])
    any_size = bool(q.get("any_size", True))
    veto_text = bool(q.get("veto_text", True))

    if q.get("engine", "v2") == "v2":   # v2 is the default engine
        from pipeline import api as v2api
        return jsonify(v2api.detect(STATE, page_index, box,
                                    any_size=any_size, veto_text=veto_text))

    t0 = time.time()
    prims = STATE.get_prims(page_index)
    grow = 1.5
    box_g = (box[0] - grow, box[1] - grow, box[2] + grow, box[3] + grow)
    traced = primitives_in_box(prims, box_g)
    if len(traced) < 2:
        return jsonify({"error": f"only {len(traced)} primitives fully inside the box - "
                                 "trace a little wider around one clean symbol"})
    ex_diag = math.hypot(box[2] - box[0], box[3] - box[1])
    exemplar = _main_cluster(traced, thr=max(3.0, 0.15 * ex_diag), box=box)

    # a traced door switches to parametric mode: doors vary in width, so we
    # detect the arc+leaf STRUCTURE (any width) instead of exact copies
    if looks_like_door(exemplar):
        doors = outside_zones(find_doors(prims), STATE.get_zones(page_index))
        by_w = {}
        for d in doors:
            w = _door_width_label(page_index, d.width_pt)
            by_w[w] = by_w.get(w, 0) + 1
        return jsonify({
            "count": len(doors),
            "box_used": list(box),
            "elapsed": round(time.time() - t0, 2),
            "n_exemplar_prims": len(exemplar),
            "n_traced_prims": len(traced),
            "mode": "door",
            "by_rotation": {},
            "by_scale": dict(sorted(by_w.items())),
            "detections": [{
                "corners": [[round(float(c[0]), 2), round(float(c[1]), 2)]
                            for c in d.corners],
                "score": round(d.score, 2), "scale": 1.0,
                "width": _door_width_label(page_index, d.width_pt),
            } for d in doors],
        })

    scene = STATE.get_scene(page_index)
    if ex_diag > scene.max_pair_dist:
        return jsonify({"error": "traced symbol is too large for the index "
                                 f"(max ~{scene.max_pair_dist:.0f}pt across)"})
    scale_range = (0.5, 2.0) if any_size else (0.85, 1.15)
    dets = outside_zones(detect(scene, exemplar, scale_range=scale_range),
                         STATE.get_zones(page_index))
    if veto_text and dets:
        dets = _veto_by_text(dets, page_index, ex_diag / 2)

    by_rot, by_scale = {}, {}
    for d in dets:
        deg = round(math.degrees(d.theta) / 45) * 45 % 360
        k = f"{deg}°" + (" mir" if d.mirrored else "")
        by_rot[k] = by_rot.get(k, 0) + 1
        ks = f"×{round(d.scale, 1)}"
        by_scale[ks] = by_scale.get(ks, 0) + 1

    return jsonify({
        "count": len(dets),
        "box_used": list(box),
        "elapsed": round(time.time() - t0, 2),
        "n_exemplar_prims": len(exemplar),
        "n_traced_prims": len(traced),
        "by_rotation": dict(sorted(by_rot.items())),
        "by_scale": dict(sorted(by_scale.items())),
        "detections": [{
            "corners": [[round(float(c[0]), 2), round(float(c[1]), 2)]
                        for c in d.corners],
            "score": round(d.score, 2), "scale": round(d.scale, 2),
            "theta_deg": round(math.degrees(d.theta), 1), "mirrored": d.mirrored,
        } for d in dets],
    })


@app.post("/legend_count")
def legend_count():
    if STATE is None:
        return jsonify({"error": "no document loaded - upload a PDF first"})
    q = request.get_json()
    page_index = int(q["page"])
    if q.get("engine", "v2") == "v2":   # v2 is the default engine
        from pipeline import api as v2api
        return jsonify(v2api.legend_count(STATE, page_index))
    legend_idxs = STATE.legend_pages()
    if not legend_idxs:
        return jsonify({"error": "no legend sheet found in this document "
                                 "(looked for LEGEND / SYMBOLS LIST headings)"})
    t0 = time.time()
    symbols: list[LegendSymbol] = []
    for li in legend_idxs:
        symbols.extend(harvest(STATE.doc[li], STATE.get_prims(li)))
    scene = STATE.get_scene(page_index)
    candidates = []
    for s in symbols:
        if not (3 <= len(s.prims) <= 30):
            continue
        diag = math.hypot(s.box[2] - s.box[0], s.box[3] - s.box[1])
        if diag > scene.max_pair_dist:
            continue
        try:
            dets = detect(scene, s.prims, min_score=0.65)
        except ValueError:
            continue
        for d in _veto_by_text(outside_zones(dets, STATE.get_zones(page_index)),
                               page_index, diag / 2):
            candidates.append((s.label[:60], d))

    # competitive assignment: one symbol per location - the model that
    # explains the most geometry wins
    candidates.sort(key=lambda c: (-c[1].matched, -c[1].score))
    kept, taken = [], []
    for label, d in candidates:
        ext = max(abs(d.corners[2][0] - d.corners[0][0]),
                  abs(d.corners[2][1] - d.corners[0][1]))
        r_sup = max(4.0, 0.5 * ext)
        if all(math.dist((d.x, d.y), t) > r_sup for t in taken):
            kept.append((label, d))
            taken.append((d.x, d.y))

    by_label = {}
    for label, d in kept:
        by_label.setdefault(label, []).append(d)
    results = [{
        "label": label,
        "count": len(ds),
        "detections": [{
            "corners": [[round(float(cc[0]), 2), round(float(cc[1]), 2)]
                        for cc in d.corners],
            "score": round(d.score, 2), "scale": round(d.scale, 2),
        } for d in ds],
    } for label, ds in by_label.items()]
    results.sort(key=lambda r: -r["count"])
    return jsonify({
        "elapsed": round(time.time() - t0, 1),
        "n_legend_symbols": len(symbols),
        "n_matched": len(results),
        "results": results,
    })


@app.get("/")
def index():
    """The single-page UI. Kept in web/index.html so this file stays Python."""
    html = pathlib.Path(__file__).parent.joinpath("web/index.html").read_text()
    return html.replace("__RENDER_ZOOM__", str(RENDER_ZOOM))


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8642, debug=False)
