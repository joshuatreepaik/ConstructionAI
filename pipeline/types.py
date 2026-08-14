"""Unified types for the v2 pipeline.

One candidate shape for every proposer (rigid symbols, parametric doors,
text-anchored markers) so scoring, arbitration, and reporting are written
exactly once. This replaces v1's duck-typed trio (Detection / Door / tuples).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class Pose:
    x: float
    y: float
    theta: float = 0.0        # radians
    scale: float = 1.0        # relative to the exemplar
    mirrored: bool = False


@dataclass
class Candidate:
    """A location hypothesis. Proposers favor recall; nothing here is final."""
    pose: Pose
    cls: str                  # 'symbol' | 'door' | 'detail_marker' | 'elevation_marker'
    source: str               # which proposer emitted it
    corners: np.ndarray       # 4x2 display-space polygon for rendering
    support: dict = field(default_factory=dict)   # proposer-specific (votes, width_pt, ...)


@dataclass
class Evidence:
    """One scale for all candidates: log-odds vs the local background."""
    log_odds: float
    matched: int
    total: int
    local_density: float      # scene primitives per pt^2 near the pose

    @property
    def frac(self) -> float:
        return self.matched / self.total if self.total else 0.0


@dataclass
class Detection:
    candidate: Candidate
    evidence: Evidence
    subclass: str | None = None      # e.g. 'GFCI' from a nearby tag
    scope: str = "plan"              # 'plan' | 'legend' | 'annotation' | ...
    decision: str = "accept"         # 'accept' | 'review' | 'reject'
    reason: str | None = None        # why rejected: 'evidence' | 'labeled' | 'zone'
    label: str | None = None         # legend label when auto-counted

    # convenience passthroughs so rendering code stays simple
    @property
    def x(self):
        return self.candidate.pose.x

    @property
    def y(self):
        return self.candidate.pose.y

    @property
    def corners(self):
        return self.candidate.corners
