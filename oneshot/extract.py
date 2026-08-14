"""Extract geometric primitives from a vector PDF page.

Every PDF drawing command becomes one or more Primitive records:
  - 'l'  line segment              -> one LINE primitive
  - 'c'  cubic bezier              -> one CURVE primitive (chord + bend info)
  - 're' rectangle                 -> four LINE primitives
  - 'qu' quad                      -> four LINE primitives

A primitive stores only geometry that the matching engine needs:
endpoints, midpoint, length, undirected orientation, and for curves a
bend ratio (how far the curve deviates from its chord, normalized).
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import pymupdf


LINE = 0
CURVE = 1


@dataclass
class Primitive:
    kind: int          # LINE or CURVE
    p0: tuple          # endpoint
    p1: tuple          # endpoint
    mid: tuple         # midpoint (for curves: curve point at t=0.5)
    length: float      # endpoint distance (chord length for curves)
    theta: float       # undirected orientation of chord, in [0, pi)
    bend: float        # 0 for lines; curves: perp deviation of mid from chord / length
    path_id: int       # index of the source path (useful for debugging)


def _orient(p0, p1) -> float:
    a = math.atan2(p1[1] - p0[1], p1[0] - p0[0])
    return a % math.pi


def _bezier_point(p0, c1, c2, p1, t):
    u = 1 - t
    x = u**3 * p0[0] + 3 * u**2 * t * c1[0] + 3 * u * t**2 * c2[0] + t**3 * p1[0]
    y = u**3 * p0[1] + 3 * u**2 * t * c1[1] + 3 * u * t**2 * c2[1] + t**3 * p1[1]
    return (x, y)


def _add_line(prims, p0, p1, path_id, min_len):
    length = math.dist(p0, p1)
    if length < min_len:
        return
    prims.append(Primitive(
        kind=LINE, p0=p0, p1=p1,
        mid=((p0[0] + p1[0]) / 2, (p0[1] + p1[1]) / 2),
        length=length, theta=_orient(p0, p1), bend=0.0, path_id=path_id,
    ))


def extract_primitives(page: pymupdf.Page, clip=None, min_len: float = 0.35) -> list[Primitive]:
    """Extract primitives from one page. clip: optional (x0,y0,x1,y1) to keep."""
    # drawings come back in unrotated space; map into display space so that
    # coordinates agree with get_pixmap clips and text extraction
    mat = page.rotation_matrix

    def T(pt):
        q = pymupdf.Point(pt) * mat
        return (float(q.x), float(q.y))

    prims: list[Primitive] = []
    for path_id, path in enumerate(page.get_drawings()):
        for item in path["items"]:
            op = item[0]
            if op == "l":
                _add_line(prims, T(item[1]), T(item[2]), path_id, min_len)
            elif op == "re":
                r = item[1]
                pts = [T((r.x0, r.y0)), T((r.x1, r.y0)), T((r.x1, r.y1)), T((r.x0, r.y1))]
                for a, b in zip(pts, pts[1:] + pts[:1]):
                    _add_line(prims, a, b, path_id, min_len)
            elif op == "qu":
                q = item[1]
                pts = [T(q.ul), T(q.ur), T(q.lr), T(q.ll)]
                for a, b in zip(pts, pts[1:] + pts[:1]):
                    _add_line(prims, a, b, path_id, min_len)
            elif op == "c":
                p0, c1, c2, p1 = T(item[1]), T(item[2]), T(item[3]), T(item[4])
                chord = math.dist(p0, p1)
                mid = _bezier_point(p0, c1, c2, p1, 0.5)
                if chord < min_len:
                    # closed/near-closed curve segment (e.g. half of a tiny circle):
                    # keep it if the curve itself is big enough
                    if math.dist(p0, mid) < min_len:
                        continue
                    chord = max(chord, 1e-6)
                # perpendicular deviation of curve midpoint from the chord
                if chord > 1e-6:
                    dx, dy = p1[0] - p0[0], p1[1] - p0[1]
                    t = ((mid[0] - p0[0]) * dx + (mid[1] - p0[1]) * dy) / (chord * chord)
                    proj = (p0[0] + t * dx, p0[1] + t * dy)
                    dev = math.dist(mid, proj)
                    bend = dev / chord
                else:
                    bend = 10.0
                prims.append(Primitive(
                    kind=CURVE, p0=p0, p1=p1, mid=mid,
                    length=chord, theta=_orient(p0, p1), bend=bend, path_id=path_id,
                ))
    if clip is not None:
        x0, y0, x1, y1 = clip
        prims = [p for p in prims
                 if x0 <= p.mid[0] <= x1 and y0 <= p.mid[1] <= y1]
    return prims


def extract_words(page: pymupdf.Page) -> list[tuple]:
    """Words as (x0, y0, x1, y1, text) in DISPLAY space.

    page.get_text() returns unrotated coordinates on rotated pages; transform
    the two bbox corners so text lines up with extracted primitives.
    """
    mat = page.rotation_matrix
    out = []
    for w in page.get_text("words"):
        a = pymupdf.Point(w[0], w[1]) * mat
        b = pymupdf.Point(w[2], w[3]) * mat
        out.append((min(a.x, b.x), min(a.y, b.y),
                    max(a.x, b.x), max(a.y, b.y), w[4]))
    return out


def primitives_in_box(prims: list[Primitive], box) -> list[Primitive]:
    """Primitives with BOTH endpoints inside box — the exemplar selection rule.

    This is what makes a traced box robust: a wall line passing through the
    box has endpoints outside, so it is excluded from the model.
    """
    x0, y0, x1, y1 = box
    def inside(pt):
        return x0 <= pt[0] <= x1 and y0 <= pt[1] <= y1
    return [p for p in prims if inside(p.p0) and inside(p.p1)]


def as_arrays(prims: list[Primitive]):
    """Columnar views used by the engine."""
    mids = np.array([p.mid for p in prims], dtype=np.float64)
    lengths = np.array([p.length for p in prims], dtype=np.float64)
    thetas = np.array([p.theta for p in prims], dtype=np.float64)
    kinds = np.array([p.kind for p in prims], dtype=np.int32)
    bends = np.array([p.bend for p in prims], dtype=np.float64)
    return mids, lengths, thetas, kinds, bends
