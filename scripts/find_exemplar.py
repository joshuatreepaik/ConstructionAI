"""Discover repeated path geometry on a sheet and render sample crops.

Groups paths by a translation-invariant signature of their items, prints the
most repeated groups, and renders one sample crop per group so a human can
identify which cluster is the symbol of interest.
"""

import sys
from collections import defaultdict

sys.path.insert(0, ".")
import pymupdf
from oneshot.render import render_crop

PDF = "data.pdf"
PAGE = int(sys.argv[1]) if len(sys.argv) > 1 else 25  # E4 power plan


def signature(path):
    items = path["items"]
    if not items:
        return None
    # collect points relative to first point, rounded
    pts = []
    for it in items:
        if it[0] == "l":
            pts += [it[1], it[2]]
        elif it[0] == "c":
            pts += [it[1], it[4]]
        elif it[0] == "re":
            r = it[1]
            pts += [pymupdf.Point(r.x0, r.y0), pymupdf.Point(r.x1, r.y1)]
        elif it[0] == "qu":
            q = it[1]
            pts += [q.ul, q.lr]
    if not pts:
        return None
    x0 = min(p.x for p in pts); y0 = min(p.y for p in pts)
    sig = tuple((it[0],) for it in items[:1])  # op of first item
    rel = tuple((round(p.x - x0, 1), round(p.y - y0, 1)) for p in pts)
    return (len(items),) + rel[:24]


doc = pymupdf.open(PDF)
page = doc[PAGE]
mat = page.rotation_matrix
groups = defaultdict(list)
for path in page.get_drawings():
    if len(path["items"]) < 3:      # single dash/tick paths are not symbols
        continue
    sig = signature(path)
    if sig is None:
        continue
    r = (path["rect"] * mat)
    r.normalize()
    groups[sig].append(r)

ranked = sorted(groups.items(), key=lambda kv: -len(kv[1]))
print(f"page {PAGE + 1}: {sum(len(v) for v in groups.values())} paths, "
      f"{len(groups)} distinct geometries")
n_out = 0
for sig, rects in ranked[:40]:
    r = rects[0]
    w, h = r.x1 - r.x0, r.y1 - r.y0
    if w < 2.5 or h < 2.5:       # needs 2D extent to be a symbol
        continue
    if w > 60 or h > 60:         # big frames — not symbols
        continue
    n_out += 1
    out = f"out/cluster_{n_out:02d}_n{len(rects)}.png"
    render_crop(PDF, PAGE, (r.x0, r.y0, r.x1, r.y1), out, zoom=10, pad=8)
    print(f"{out}  count={len(rects)}  size={w:.1f}x{h:.1f}  "
          f"at=({r.x0:.0f},{r.y0:.0f})")
    if n_out >= 14:
        break
