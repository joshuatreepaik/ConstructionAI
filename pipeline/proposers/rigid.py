"""STAGE 2a - the rigid proposer: find repeated copies of a traced symbol.

HOW IT WORKS, IN PLAIN TERMS
    You recognise a friend in a crowd from relationships ("eyes above nose,
    this far apart"), not by comparing pixels. This does the same with
    drawing geometry.

    1. Describe the traced symbol as RELATIONSHIPS between pairs of its parts
       - length ratio, relative angle, separation. Those numbers don't change
       when the symbol is moved, rotated, mirrored, or resized.
    2. Scan the sheet for pairs matching those descriptions. Each match is a
       clue, and each clue also implies WHERE the symbol's centre would be and
       HOW it is rotated and scaled.
    3. Where enough clues agree on the same position/rotation/scale, that is a
       candidate instance.

    Because the evidence is a pile of independent clues, a symbol half-covered
    by a wall line still accumulates enough of them to be found - which is the
    failure that breaks picture-matching tools.

WHAT THIS ADDS OVER THE v1 ENGINE
    - votes carry the index's rarity weight, so distinctive geometry speaks
      louder than the wall hatching everyone shares
    - fingerprints so common they were dropped in STAGE 1 never vote at all
    - one scene primitive votes at most ONCE per candidate (the "double-vote
      guard"): without it, dense regions inflate their own scores simply by
      containing more line pairs
    - the bar to become a candidate is RELATIVE to what a perfect instance of
      THIS symbol would score on THIS sheet, so one rule covers rare and
      common symbols alike

This stage is deliberately generous - it proposes, it does not decide. Weak
candidates are filtered later, by evidence (STAGE 3) that can be reasoned
about; a candidate never proposed is lost forever.
"""

from __future__ import annotations

import math

import numpy as np

from oneshot.extract import Primitive
from oneshot.engine import SymbolModel, TWO_PI

from ..index import SaliencyIndex
from ..policy import Policy, DEFAULT
from ..types import Candidate, Pose


def propose(index: SaliencyIndex, exemplar: list[Primitive],
            policy: Policy = DEFAULT, any_size: bool = True,
            source: str = "rigid") -> list[Candidate]:
    scale_range = policy.scale_range if any_size else policy.scale_range_strict

    # Mirrored symbols (doors swinging the other way, receptacles on the
    # opposite wall) are a reflection, not a rotation, so they need their own
    # model. Searching both is what makes mirror-handling free downstream.
    models = [SymbolModel(exemplar)]
    try:
        models.append(SymbolModel.mirrored_from(exemplar))
    except ValueError:
        pass

    out: list[Candidate] = []
    for model in models:
        # ---- 1. what would a PERFECT instance of this symbol score here? ----
        # The bar for becoming a candidate is a fraction of this, so a symbol
        # made of rare geometry and one made of common geometry are judged on
        # the same relative scale.
        #
        # It is computed per ANCHOR primitive, not per pair, because the
        # double-vote guard below lets each primitive vote only once - the
        # baseline has to be measured under the same rule it will be compared
        # against, or every candidate looks weak.
        key_w = index.weight_of_keys(model.pair_key)
        anchor_best: dict[int, float] = {}
        for a, wgt in zip(model.pair_ii, key_w):
            a = int(a)
            if wgt > anchor_best.get(a, 0.0):
                anchor_best[a] = float(wgt)
        self_mass = sum(anchor_best.values())
        if self_mass <= 0:
            continue                       # symbol is entirely generic here
        min_mass = policy.min_vote_mass_ratio * self_mass

        # ---- 2. look the model's fingerprints up in the sheet index --------
        sel, midx = model.lookup(index.pair_key)
        if len(sel) == 0:
            continue
        w = index.pair_weight[sel]
        live = w > 0                       # hatching buckets were zeroed out
        sel, midx, w = sel[live], midx[live], w[live]
        if len(sel) == 0:
            continue

        # ---- 3. every match implies a pose; discard impossible ones --------
        # rotation implied by the pair's direction, scale by its length ratio
        theta = (index.pair_ang[sel] - model.pair_ang[midx]) % TWO_PI
        ratio = index.pair_dist[sel] / model.pair_dist[midx]
        ok = (ratio > scale_range[0]) & (ratio < scale_range[1])
        sel, midx, theta, ratio, w = sel[ok], midx[ok], theta[ok], ratio[ok], w[ok]

        # a real match must ALSO have its primitive lying at the orientation
        # that pose predicts - this kills coincidental length/spacing matches
        d_or = (index.thetas[index.pair_ii[sel]]
                - model.thetas[model.pair_ii[midx]] - theta)
        d_or = np.abs((d_or + math.pi / 2) % math.pi - math.pi / 2)
        ok = d_or < policy.orient_gate
        sel, midx, theta, ratio, w = sel[ok], midx[ok], theta[ok], ratio[ok], w[ok]
        if len(sel) == 0:
            continue

        # ---- 4. each surviving match votes for a symbol CENTRE -------------
        # (rotate the model's centre-offset by the implied pose and add it to
        # where the matching primitive actually sits on the sheet)
        c, s = np.cos(theta), np.sin(theta)
        anchor_m = model.mids[model.pair_ii[midx]]
        anchor_s = index.mids[index.pair_ii[sel]]
        off = (model.ref - anchor_m) * ratio[:, None]
        px = anchor_s[:, 0] + c * off[:, 0] - s * off[:, 1]
        py = anchor_s[:, 1] + s * off[:, 0] + c * off[:, 1]

        # ---- 5. votes agreeing on (x, y, rotation, scale) form a candidate --
        gx = np.floor(px / policy.vote_cell).astype(np.int64)
        gy = np.floor(py / policy.vote_cell).astype(np.int64)
        gt = np.floor(theta / policy.theta_cell).astype(np.int64) % int(
            round(TWO_PI / policy.theta_cell))
        gs = np.floor(np.log(ratio) / policy.logs_cell).astype(np.int64)
        bin_key = (gs << 50) ^ (gx << 34) ^ (gy << 8) ^ gt

        # DOUBLE-VOTE GUARD. One scene primitive may support a given pose only
        # once, however many pairs it happens to belong to. Without this, a
        # primitive sitting in dense linework votes many times for the same
        # spot and manufactures its own evidence - which systematically
        # favours hatched regions over clean ones.
        anchor_ids = index.pair_ii[sel].astype(np.int64)
        pair_id = (bin_key << 17) ^ anchor_ids     # sheets stay under 2^17 prims
        _, first = np.unique(pair_id, return_index=True)
        bin_key, px, py, theta, ratio, w = (bin_key[first], px[first], py[first],
                                            theta[first], ratio[first], w[first])

        # a bin whose surviving votes carry enough weight becomes a candidate
        uniq, inv = np.unique(bin_key, return_inverse=True)
        mass = np.bincount(inv, weights=w)
        keep = np.where(mass >= min_mass)[0]
        for u in keep:
            # refine the pose as the weight-weighted average of its own votes,
            # so precision isn't limited by the bin size used to group them
            m = inv == u
            ww = w[m]
            wsum = ww.sum()
            x = float((px[m] * ww).sum() / wsum)
            y = float((py[m] * ww).sum() / wsum)
            th = float(math.atan2((np.sin(theta[m]) * ww).sum(),   # circular mean:
                                  (np.cos(theta[m]) * ww).sum()))  # 359° and 1° average to 0°
            sc = float(np.exp((np.log(ratio[m]) * ww).sum() / wsum))  # geometric mean
            lo = model.mids.min(axis=0) - 2
            hi = model.mids.max(axis=0) + 2
            box = np.array([[lo[0], lo[1]], [hi[0], lo[1]],
                            [hi[0], hi[1]], [lo[0], hi[1]]])
            cth, sth = math.cos(th), math.sin(th)
            R = np.array([[cth, -sth], [sth, cth]]) * sc
            corners = (box - model.ref) @ R.T + np.array([x, y])
            out.append(Candidate(
                pose=Pose(x, y, th, sc, model.mirrored),
                cls="symbol", source=source, corners=corners,
                support={"vote_mass": float(mass[u]),
                         "self_mass": self_mass,
                         "model": model},
            ))
    return out
