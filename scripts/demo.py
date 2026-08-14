"""End-to-end demo: trace-one-find-all on a sheet of the Skanska set.

Usage: demo.py [page_index] [x0 y0 x1 y1]
Defaults to the duplex receptacle on E4 (page index 25).
"""

import math
import sys
import time

sys.path.insert(0, ".")
import pymupdf

from oneshot.extract import extract_primitives, primitives_in_box
from oneshot.engine import Scene, detect
from oneshot.render import render_overlay_png, render_crop

PDF = "data.pdf"
PAGE = int(sys.argv[1]) if len(sys.argv) > 1 else 25
BOX = tuple(float(v) for v in sys.argv[2:6]) if len(sys.argv) > 5 else (433.0, 348.0, 442.5, 357.5)

t0 = time.time()
doc = pymupdf.open(PDF)
page = doc[PAGE]
prims = extract_primitives(page)
t1 = time.time()
print(f"extracted {len(prims)} primitives in {t1 - t0:.2f}s")

exemplar = primitives_in_box(prims, BOX)
print(f"exemplar primitives: {len(exemplar)} "
      f"({sum(p.kind == 0 for p in exemplar)} lines, "
      f"{sum(p.kind == 1 for p in exemplar)} curves)")
render_crop(PDF, PAGE, BOX, "out/exemplar.png", zoom=12, pad=4)

# scene pairs need only span one symbol
ex_w = BOX[2] - BOX[0]
ex_h = BOX[3] - BOX[1]
max_pair = math.hypot(ex_w, ex_h) * 1.1

t2 = time.time()
scene = Scene(prims, max_pair_dist=max_pair)
t3 = time.time()
print(f"scene index: {len(scene.pair_key)} pairs in {t3 - t2:.2f}s")

dets = detect(scene, exemplar)
t4 = time.time()
print(f"detect: {len(dets)} instances in {t4 - t3:.2f}s")

# text veto: a symbol whose core contains text glyphs is a *labeled* device
# (AV tag, grid bubble, note hexagon) - not the traced plain symbol
words = page.get_text("words")  # display-space bboxes
sym_r = math.hypot(ex_w, ex_h) / 2
kept = []
for d in dets:
    veto = False
    for (wx0, wy0, wx1, wy1, *_rest) in words:
        cx, cy = (wx0 + wx1) / 2, (wy0 + wy1) / 2
        if math.dist((cx, cy), (d.x, d.y)) < 0.55 * sym_r * d.scale:
            veto = True
            break
    if not veto:
        kept.append(d)
print(f"text veto removed {len(dets) - len(kept)}; {len(kept)} remain")
dets = kept

by_rot = {}
for d in dets:
    deg = round(math.degrees(d.theta) / 45) * 45 % 360
    k = f"{deg}deg" + (" mirrored" if d.mirrored else "")
    by_rot[k] = by_rot.get(k, 0) + 1
print("by orientation:", dict(sorted(by_rot.items())))
by_scale = {}
for d in dets:
    k = f"x{round(d.scale, 1)}"
    by_scale[k] = by_scale.get(k, 0) + 1
print("by scale:", dict(sorted(by_scale.items())))
scores = sorted(round(d.score, 2) for d in dets)
print(f"scores: min={scores[0]:.2f} median={scores[len(scores)//2]:.2f} max={scores[-1]:.2f}"
      if dets else "no detections")

render_overlay_png(PDF, PAGE, dets, "out/overlay_full.png",
                   exemplar_box=BOX, zoom=1.5)
xs = [d.x for d in dets]; ys = [d.y for d in dets]
if dets:
    render_overlay_png(PDF, PAGE, dets, "out/overlay_zoom.png", exemplar_box=BOX,
                       zoom=4.0, clip=(min(xs) - 20, min(ys) - 20,
                                       min(min(xs) + 400, max(xs) + 20),
                                       min(min(ys) + 260, max(ys) + 20)))
print("wrote out/overlay_full.png, out/overlay_zoom.png, out/exemplar.png")
