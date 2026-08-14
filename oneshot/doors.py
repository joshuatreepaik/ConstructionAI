"""Parametric door detection: structure, not template.

A plan door is a quarter-circle swing arc plus a leaf line hinged at the
arc's center, with leaf length equal to the arc radius. That relationship
holds for EVERY door width, so detecting the structure finds 30" and 36"
doors alike - where template matching only finds exact copies of the traced
block. Output includes the recovered width per door.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

from .extract import Primitive, LINE, CURVE

QUARTER_BEND = 0.2071        # perp deviation / chord for a 90-degree arc


@dataclass
class Door:
    hinge: tuple
    width_pt: float          # leaf length = arc radius, in page points
    arc: Primitive
    n_leaf_lines: int
    corners: np.ndarray      # bbox polygon for display
    score: float


def arc_radius(p: Primitive) -> float:
    """Radius from chord + bend, WITHOUT assuming a 90-degree swing.

    bend = sagitta/chord = tan(alpha/2)/2 where the arc spans 2*alpha.
    Doors are often drawn swung 85-90 degrees; assuming exactly 90 makes a
    3'-0\" door measure 2'-11\" - a phantom width difference.
    """
    alpha = 2 * math.atan(2 * p.bend)
    return p.length / (2 * math.sin(alpha) + 1e-9)


def _arc_center(p: Primitive):
    """True center of the arc: the candidate whose distance to the arc
    midpoint equals the radius."""
    a, b, m = np.array(p.p0), np.array(p.p1), np.array(p.mid)
    r = arc_radius(p)
    h = math.sqrt(max(r * r - (p.length / 2) ** 2, 0.0))
    mid_ab = (a + b) / 2
    d = b - a
    perp = np.array([-d[1], d[0]]) / (np.linalg.norm(d) + 1e-9)
    best = None
    for sign in (1, -1):
        o = mid_ab + sign * perp * h
        err = abs(np.linalg.norm(o - m) - r)
        if best is None or err < best[0]:
            best = (err, o)
    return best[1], r


def find_doors(prims: list[Primitive],
               min_chord=14.0, max_chord=90.0,
               bend_tol=0.045, end_tol=3.0,
               leaf_len_tol=0.18, dedupe_radius=None) -> list[Door]:
    """dedupe_radius: hinge distance below which two arcs are the same door.
    Default (None) keeps the historical 0.5*width rule; NOTE that rule
    wrongly merges back-to-back office doors whose hinges sit ~12pt apart —
    pass a small absolute radius (e.g. 4.0) to fix."""
    lines = [p for p in prims if p.kind == LINE and p.length > 8]
    if not lines:
        return []
    ends = np.array([q for p in lines for q in (p.p0, p.p1)])
    tree = cKDTree(ends)

    doors = []
    for p in prims:
        if p.kind != CURVE:
            continue
        if not (min_chord <= p.length <= max_chord):
            continue
        if abs(p.bend - QUARTER_BEND) > bend_tol:
            continue
        hinge, r = _arc_center(p)
        # leaf: line starting at the hinge, length ~ radius, ending near an
        # arc endpoint (the open position) or anywhere on the swing
        n_leaf = 0
        for j in tree.query_ball_point(hinge, r=end_tol):
            ln = lines[j // 2]
            if abs(ln.length - r) > leaf_len_tol * r:
                continue
            far = np.array(ln.p1 if j % 2 == 0 else ln.p0)
            if abs(np.linalg.norm(far - hinge) - r) > leaf_len_tol * r:
                continue
            n_leaf += 1
        if n_leaf == 0:
            continue
        pts = np.array([p.p0, p.p1, p.mid, tuple(hinge)])
        lo = pts.min(axis=0) - 2
        hi = pts.max(axis=0) + 2
        corners = np.array([[lo[0], lo[1]], [hi[0], lo[1]],
                            [hi[0], hi[1]], [lo[0], hi[1]]])
        doors.append(Door(hinge=tuple(hinge), width_pt=r, arc=p,
                          n_leaf_lines=n_leaf, corners=corners,
                          score=min(1.0, 0.7 + 0.15 * n_leaf)))

    # dedupe arcs sharing a hinge (double-drawn doors)
    doors.sort(key=lambda d: -d.score)
    kept = []
    for d in doors:
        r = dedupe_radius if dedupe_radius is not None else 0.5 * d.width_pt
        if all(math.dist(d.hinge, k.hinge) > r for k in kept):
            kept.append(d)
    return kept


def looks_like_door(exemplar: list[Primitive]) -> bool:
    """Did the user trace a door? An arc of door-ish size plus a line whose
    length matches the arc radius."""
    for p in exemplar:
        if (p.kind == CURVE and abs(p.bend - QUARTER_BEND) < 0.045
                and 14 <= p.length <= 90):
            r = p.length / math.sqrt(2)
            for ln in exemplar:
                if ln.kind == LINE and abs(ln.length - r) < 0.2 * r:
                    return True
    return False


def width_label(width_pt: float, scale_pt_per_ft: float = 9.0) -> str:
    """Points -> nominal door width at 1/8in = 1ft (9pt per foot)."""
    inches = width_pt / scale_pt_per_ft * 12
    whole = int(round(inches))
    return f"{whole // 12}'-{whole % 12}\""
