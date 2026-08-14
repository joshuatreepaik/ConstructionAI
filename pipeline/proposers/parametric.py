"""Parametric proposer: symbol families with a free parameter.

Doors are the first family: arc + leaf hinged at the arc center, any width.
Wraps the proven v1 structural detector (read-only) and emits the shared
Candidate type so doors flow through the same evidence/resolution stages as
everything else — in v1 they bypassed both.
"""

from __future__ import annotations

from oneshot.extract import Primitive
from oneshot.doors import find_doors, arc_radius, QUARTER_BEND
from oneshot.extract import LINE, CURVE

from ..policy import Policy, DEFAULT
from ..types import Candidate, Pose


def looks_like_door(exemplar: list[Primitive], policy: Policy = DEFAULT) -> bool:
    """Did the user trace a door? Uses the true arc radius (v1's
    looks_like_door kept a contradictory sqrt(2) shortcut — fixed here)."""
    for p in exemplar:
        if (p.kind == CURVE and abs(p.bend - QUARTER_BEND) < policy.door_bend_tol
                and policy.door_min_chord <= p.length <= policy.door_max_chord):
            r = arc_radius(p)
            for ln in exemplar:
                if ln.kind == LINE and abs(ln.length - r) < policy.door_leaf_len_tol * r:
                    return True
    return False


def propose(prims: list[Primitive], policy: Policy = DEFAULT,
            source: str = "parametric.door") -> list[Candidate]:
    out = []
    for d in find_doors(prims,
                        min_chord=policy.door_min_chord,
                        max_chord=policy.door_max_chord,
                        bend_tol=policy.door_bend_tol,
                        end_tol=policy.door_end_tol,
                        leaf_len_tol=policy.door_leaf_len_tol,
                        dedupe_radius=policy.door_dedupe_radius):
        out.append(Candidate(
            pose=Pose(x=float(d.hinge[0]), y=float(d.hinge[1])),
            cls="door", source=source, corners=d.corners,
            support={"width_pt": d.width_pt, "n_leaf_lines": d.n_leaf_lines},
        ))
    return out
