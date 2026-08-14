"""Stage 4: one arbiter for everything, then names, then scope.

Resolution order (cheapest and most general first):
1. spatial arbitration — one location, one detection. THE single suppression
   rule; replaces v1's three divergent NMS implementations.
2. semantic — text on the symbol names it (subclass) or disqualifies it
   (a labeled device is not the plain traced symbol).
3. scope — detections inside excluded regions (compass, key plan, legend
   columns) don't count.
4. decision — accept / review / reject from evidence thresholds.
"""

from __future__ import annotations

import math

import numpy as np

from .policy import Policy, DEFAULT
from .sheet import Sheet
from .types import Candidate, Detection, Evidence


def _extent(c: Candidate) -> float:
    d = c.corners.max(axis=0) - c.corners.min(axis=0)
    return float(max(d[0], d[1]))


def resolve(scored: list[tuple[Candidate, Evidence]], sheet: Sheet,
            policy: Policy = DEFAULT, veto_text: bool = True) -> list[Detection]:
    # 1 - spatial arbitration: strongest evidence wins each location
    scored = sorted(scored, key=lambda ce: -ce[1].log_odds)
    kept: list[tuple[Candidate, Evidence]] = []
    for cand, ev in scored:
        if cand.cls == "door":
            # a door's identity is its hinge: back-to-back doors are distinct
            r = policy.door_dedupe_radius
        else:
            r = max(4.0, policy.suppress_extent_ratio * _extent(cand))
        if all(math.dist((cand.pose.x, cand.pose.y), (k.pose.x, k.pose.y)) > r
               for k, _ in kept):
            kept.append((cand, ev))

    out: list[Detection] = []
    wc = sheet.word_centers
    for cand, ev in kept:
        det = Detection(candidate=cand, evidence=ev)

        # 2 - semantic: a word sitting ON the symbol core
        if veto_text and cand.cls == "symbol" and len(wc):
            core_r = policy.tag_radius_ratio * (_extent(cand) / 2)
            d2 = ((wc - np.array([cand.pose.x, cand.pose.y])) ** 2).sum(axis=1)
            j = int(np.argmin(d2))
            if d2[j] < core_r * core_r:
                det.subclass = sheet.words[j][4]
                det.decision = "reject"       # labeled device != plain symbol
                det.reason = "labeled"

        # 3 - scope
        if sheet.in_zone(cand.pose.x, cand.pose.y):
            det.scope = "annotation"
            det.decision = "reject"
            det.reason = "zone"

        # 4 - decision from evidence (unless already disqualified);
        # thresholds are per proposer family because evidence granularity
        # differs (a door has 3 big structural parts, a symbol many small ones)
        if det.decision != "reject":
            if cand.cls == "door":
                acc, rev = policy.door_accept_log_odds, policy.door_review_log_odds
            else:
                acc, rev = policy.accept_log_odds, policy.review_log_odds
            if ev.log_odds >= acc:
                det.decision = "accept"
            elif ev.log_odds >= rev:
                det.decision = "review"
            else:
                det.decision = "reject"
                det.reason = "evidence"
        out.append(det)
    return out
