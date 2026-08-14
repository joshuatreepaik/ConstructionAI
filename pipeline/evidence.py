"""Stage 3: score every candidate on one scale — log-odds against the local
background.

For each template primitive placed by the pose we ask: how surprising is it
to find a matching primitive this close, given how cluttered this spot is?
(Astrometry.net's Bayes factor, adapted to line drawings.)

Properties matched/total cannot give:
- a match in dense linework is worth less than one in white space,
- a MISSING primitive costs a defined amount of evidence (occlusion degrades
  smoothly instead of falling off a cliff),
- doors, symbols, and markers land on one comparable scale.
"""

from __future__ import annotations

import math

import numpy as np

from .index import SaliencyIndex
from .policy import Policy, DEFAULT
from .types import Candidate, Evidence


def _score_rigid(cand: Candidate, index: SaliencyIndex, policy: Policy) -> Evidence:
    model = cand.support["model"]
    pose = cand.pose
    c, s = math.cos(pose.theta), math.sin(pose.theta)
    R = np.array([[c, -s], [s, c]]) * pose.scale
    placed = (model.mids - model.ref) @ R.T + np.array([pose.x, pose.y])
    placed_theta = (model.thetas + pose.theta) % math.pi
    lengths = model.lengths * pose.scale

    radius = max(2.0 * model.radius * pose.scale, 20.0)
    rho = max(index.local_density(pose.x, pose.y, radius), 1e-6)

    sigma2 = policy.pos_sigma ** 2
    log_bg = math.log(rho)                      # chance rate per unit area
    log_fg0 = math.log((1.0 - policy.distractor_rate) / (2 * math.pi * sigma2))
    log_missing = math.log(policy.distractor_rate)

    log_odds = 0.0
    matched = 0
    used: set[int] = set()
    for k in range(len(placed)):
        idxs = index.tree.query_ball_point(placed[k], r=policy.pos_sigma * 3 + lengths[k] * 0.06)
        best = None
        for i in idxs:
            if i in used:
                continue
            if index.kinds[i] != model.kinds[k]:
                continue
            if abs(index.lengths[i] - lengths[k]) > policy.len_tol * max(lengths[k], 1.0):
                continue
            d_ang = abs((index.thetas[i] - placed_theta[k] + math.pi / 2) % math.pi
                        - math.pi / 2)
            if d_ang > policy.ang_tol:
                continue
            if abs(index.bends[i] - model.bends[k]) > policy.bend_tol:
                continue
            d = math.dist(index.mids[i], placed[k])
            if best is None or d < best[0]:
                best = (d, i)
        if best is not None:
            used.add(best[1])
            matched += 1
            log_odds += (log_fg0 - best[0] ** 2 / (2 * sigma2)) - log_bg
        else:
            log_odds += log_missing
    return Evidence(log_odds=log_odds, matched=matched,
                    total=len(placed), local_density=rho)


def _score_door(cand: Candidate, index: SaliencyIndex, policy: Policy) -> Evidence:
    """Structural parts of a door as evidence units: arc + up to 2 leaf lines.
    Uses the same fg/bg form with d=0 (parts were found by construction)."""
    width = cand.support["width_pt"]
    n_leaf = min(cand.support["n_leaf_lines"], 2)
    radius = max(2.0 * width, 20.0)
    rho = max(index.local_density(cand.pose.x, cand.pose.y, radius), 1e-6)
    sigma2 = policy.pos_sigma ** 2
    part_gain = math.log((1.0 - policy.distractor_rate) / (2 * math.pi * sigma2)) \
        - math.log(rho)
    matched = 1 + n_leaf
    total = 3
    log_odds = matched * part_gain + (total - matched) * math.log(policy.distractor_rate)
    return Evidence(log_odds=log_odds, matched=matched, total=total,
                    local_density=rho)


def score(cand: Candidate, index: SaliencyIndex, policy: Policy = DEFAULT) -> Evidence:
    if cand.cls == "door":
        return _score_door(cand, index, policy)
    return _score_rigid(cand, index, policy)
