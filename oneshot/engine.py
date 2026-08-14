"""One-shot symbol detection by pose-voting geometric fingerprints.

Pipeline (per exemplar):
  1. Model = primitives fully inside the traced box.
  2. Fingerprint every primitive pair in the model with a similarity-invariant
     descriptor; hash the quantized descriptor.
  3. Fingerprint scene pairs (k-NN limited so pairs span at most one symbol),
     keep only pairs whose key exists in the model -> each match votes for a
     full pose (tx, ty, theta) of the model in the scene.
  4. Cluster votes; every strong cluster is a candidate instance.
  5. Verify: transform the model by the candidate pose, greedily match model
     primitives to scene primitives (type / length / orientation / bend gated).
     Score = fraction of model primitives explained.  Occlusion degrades the
     score gracefully instead of killing the match.

No training, no thresholds hidden from the user: score means
"fraction of the symbol's geometry found at this location".
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math

import numpy as np
from scipy.spatial import cKDTree

from .extract import Primitive, LINE, CURVE, as_arrays

TWO_PI = 2 * math.pi

# quantization steps for the pair descriptor
Q_LOGRATIO = 0.35      # log2 length-ratio bin
Q_DTHETA = math.pi / 12  # 15 deg bins for relative orientation
Q_DIST = 0.30          # bins of (pair distance / longer length)
Q_BEND = 0.12          # curve bend bins


@dataclass
class Detection:
    x: float
    y: float
    theta: float          # radians
    scale: float          # recovered symbol scale vs the exemplar
    mirrored: bool
    votes: int
    score: float          # fraction of model primitives verified
    corners: np.ndarray   # 4x2 oriented bbox of the placed model
    matched: int
    total: int


def _pack(cols, widths):
    """Pack quantized integer columns into one int64 key."""
    key = np.zeros(len(cols[0]), dtype=np.int64)
    for c, w in zip(cols, widths):
        key = (key << w) | (c.astype(np.int64) & ((1 << w) - 1))
    return key


def _pair_features(mids, lengths, thetas, kinds, bends, ii, jj):
    """Vectorized invariant descriptor + geometry for pairs (ii, jj)."""
    dx = mids[jj, 0] - mids[ii, 0]
    dy = mids[jj, 1] - mids[ii, 1]
    dist = np.hypot(dx, dy)
    ang = np.arctan2(dy, dx)                       # direction of the pair vector
    dtheta = (thetas[jj] - thetas[ii]) % math.pi   # relative orientation
    logratio = np.log2((lengths[ii] + 1e-9) / (lengths[jj] + 1e-9))
    ndist = dist / np.maximum(lengths[ii], lengths[jj])

    q_kind = kinds[ii] * 2 + kinds[jj]
    q_lr = np.clip(np.round(logratio / Q_LOGRATIO), -15, 15).astype(np.int64) + 16
    q_dt = np.round(dtheta / Q_DTHETA).astype(np.int64) % 12
    q_nd = np.clip(np.round(ndist / Q_DIST), 0, 63).astype(np.int64)
    q_bi = np.clip(np.round(bends[ii] / Q_BEND), 0, 15).astype(np.int64)
    q_bj = np.clip(np.round(bends[jj] / Q_BEND), 0, 15).astype(np.int64)

    key = _pack([q_kind, q_lr, q_dt, q_nd, q_bi, q_bj], [2, 5, 4, 6, 4, 4])
    return key, dist, ang


class SymbolModel:
    """Fingerprinted exemplar."""

    def __init__(self, prims: list[Primitive], mirrored=False):
        if len(prims) < 2:
            raise ValueError(f"exemplar has only {len(prims)} primitives; trace a tighter/bigger box")
        self.prims = prims
        self.mirrored = mirrored
        self.mids, self.lengths, self.thetas, self.kinds, self.bends = as_arrays(prims)
        self.ref = self.mids.mean(axis=0)
        self.radius = float(np.max(np.linalg.norm(self.mids - self.ref, axis=1)) +
                            np.max(self.lengths) / 2)
        n = len(prims)
        ii, jj = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
        mask = ii != jj
        ii, jj = ii[mask], jj[mask]
        # drop near-coincident pairs (unstable direction)
        key, dist, ang = _pair_features(self.mids, self.lengths, self.thetas,
                                        self.kinds, self.bends, ii, jj)
        ok = dist > max(0.75, 0.05 * self.radius)
        self.pair_ii, self.pair_jj = ii[ok], jj[ok]
        self.pair_key, self.pair_dist, self.pair_ang = key[ok], dist[ok], ang[ok]
        # model pair lookup: key -> list of pair indices
        order = np.argsort(self.pair_key, kind="stable")
        self._sorted_keys = self.pair_key[order]
        self._sorted_idx = order

    def lookup(self, keys):
        """For an array of scene keys, return (scene_sel, model_pair_idx) matches."""
        left = np.searchsorted(self._sorted_keys, keys, side="left")
        right = np.searchsorted(self._sorted_keys, keys, side="right")
        counts = right - left
        scene_sel = np.repeat(np.arange(len(keys)), counts)
        if len(scene_sel) == 0:
            return scene_sel, np.array([], dtype=np.int64)
        offsets = np.concatenate([np.arange(l, r) for l, r in zip(left, right) if r > l])
        return scene_sel, self._sorted_idx[offsets]

    @classmethod
    def mirrored_from(cls, prims: list[Primitive]):
        """Build the x-mirrored model (door swings / opposite-wall symbols)."""
        out = []
        for p in prims:
            m = lambda pt: (-pt[0], pt[1])
            out.append(Primitive(
                kind=p.kind, p0=m(p.p0), p1=m(p.p1), mid=m(p.mid),
                length=p.length, theta=(math.pi - p.theta) % math.pi,
                bend=p.bend, path_id=p.path_id,
            ))
        return cls(out, mirrored=True)


class Scene:
    """Fingerprint index of one sheet.

    Built ONCE per sheet and reused across queries: pair generation is capped
    to each primitive's k nearest neighbors (bounded cost regardless of sheet
    density), so the index covers any exemplar whose internal pair distances
    fall within `max_pair_dist`.
    """

    def __init__(self, prims: list[Primitive], max_pair_dist: float = 64.0,
                 max_neighbors: int = 48):
        self.prims = prims
        self.max_pair_dist = max_pair_dist
        self.mids, self.lengths, self.thetas, self.kinds, self.bends = as_arrays(prims)
        self.tree = cKDTree(self.mids)
        n = len(prims)
        k = min(max_neighbors + 1, n)
        dists, idxs = self.tree.query(self.mids, k=k,
                                      distance_upper_bound=max_pair_dist)
        ii = np.repeat(np.arange(n), k)
        jj = idxs.ravel()
        valid = (jj < n) & (jj != ii)          # drop self and out-of-range fills
        ii, jj = ii[valid], jj[valid]
        key, dist, ang = _pair_features(self.mids, self.lengths, self.thetas,
                                        self.kinds, self.bends, ii, jj)
        ok = dist > 0.75
        self.pair_ii, self.pair_jj = ii[ok], jj[ok]
        self.pair_key, self.pair_dist, self.pair_ang = key[ok], dist[ok], ang[ok]


def _cluster_votes(px, py, pth, pls, cell, theta_cell, logs_cell, min_votes):
    """Grid-hash votes on (x, y, theta, log scale); return cluster centers."""
    gx = np.floor(px / cell).astype(np.int64)
    gy = np.floor(py / cell).astype(np.int64)
    gt = np.floor(pth / theta_cell).astype(np.int64) % int(round(TWO_PI / theta_cell))
    gs = np.floor(pls / logs_cell).astype(np.int64)
    packed = (gs << 50) ^ (gx << 34) ^ (gy << 8) ^ gt
    uniq, inv, counts = np.unique(packed, return_inverse=True, return_counts=True)
    keep = np.where(counts >= min_votes)[0]
    centers = []
    for u in keep:
        sel = inv == u
        centers.append((px[sel].mean(), py[sel].mean(),
                        math.atan2(np.sin(pth[sel]).mean(), np.cos(pth[sel]).mean()),
                        float(np.exp(pls[sel].mean())),
                        int(sel.sum())))
    return centers


def _verify(model: SymbolModel, scene: Scene, x, y, theta, scale=1.0,
            pos_tol=1.6, len_tol=0.22, ang_tol=math.radians(12), bend_tol=0.15):
    """Place model at (x, y, theta, scale); fraction of model primitives found."""
    c, s = math.cos(theta), math.sin(theta)
    R = np.array([[c, -s], [s, c]]) * scale
    placed = (model.mids - model.ref) @ R.T + np.array([x, y])
    placed_theta = (model.thetas + theta) % math.pi
    lengths = model.lengths * scale
    matched = 0
    used = set()
    for k in range(len(placed)):
        idxs = scene.tree.query_ball_point(placed[k], r=pos_tol + lengths[k] * 0.06)
        best = None
        for i in idxs:
            if i in used:
                continue
            if scene.kinds[i] != model.kinds[k]:
                continue
            if abs(scene.lengths[i] - lengths[k]) > len_tol * max(lengths[k], 1.0):
                continue
            d_ang = abs((scene.thetas[i] - placed_theta[k] + math.pi / 2) % math.pi - math.pi / 2)
            if d_ang > ang_tol:
                continue
            if abs(scene.bends[i] - model.bends[k]) > bend_tol:
                continue
            d = math.dist(scene.mids[i], placed[k])
            if best is None or d < best[0]:
                best = (d, i)
        if best is not None:
            used.add(best[1])
            matched += 1
    return matched, len(placed)


def detect(scene: Scene, exemplar_prims: list[Primitive],
           min_score=0.6, min_votes=4, try_mirror=True,
           scale_range=(0.5, 2.0), cell=3.0,
           theta_cell=math.radians(12), logs_cell=0.12) -> list[Detection]:
    models = [SymbolModel(exemplar_prims)]
    if try_mirror:
        models.append(SymbolModel.mirrored_from(exemplar_prims))

    detections: list[Detection] = []
    for model in models:
        # tiny models (a door = arc + leaf) can only yield a couple of votes
        eff_min_votes = max(2, min(min_votes, len(model.pair_ii)))
        scene_sel, model_idx = model.lookup(scene.pair_key)
        if len(scene_sel) == 0:
            continue
        # pose hypothesis per matched pair
        theta = (scene.pair_ang[scene_sel] - model.pair_ang[model_idx]) % TWO_PI
        # scale: free within range, recovered per instance from the votes
        ratio = scene.pair_dist[scene_sel] / model.pair_dist[model_idx]
        ok = (ratio > scale_range[0]) & (ratio < scale_range[1])
        scene_sel, model_idx, theta, ratio = scene_sel[ok], model_idx[ok], theta[ok], ratio[ok]
        # orientation-consistency gate on the first pair member
        d_or = (scene.thetas[scene.pair_ii[scene_sel]]
                - model.thetas[model.pair_ii[model_idx]] - theta)
        d_or = np.abs((d_or + math.pi / 2) % math.pi - math.pi / 2)
        ok = d_or < math.radians(15)
        scene_sel, model_idx, theta, ratio = scene_sel[ok], model_idx[ok], theta[ok], ratio[ok]
        if len(scene_sel) == 0:
            continue
        # vote for the model reference point in scene coordinates
        c, s = np.cos(theta), np.sin(theta)
        anchor_m = model.mids[model.pair_ii[model_idx]]
        anchor_s = scene.mids[scene.pair_ii[scene_sel]]
        off = (model.ref - anchor_m) * ratio[:, None]
        px = anchor_s[:, 0] + c * off[:, 0] - s * off[:, 1]
        py = anchor_s[:, 1] + s * off[:, 0] + c * off[:, 1]

        for (x, y, th, sc, votes) in _cluster_votes(px, py, theta, np.log(ratio),
                                                    cell, theta_cell, logs_cell, eff_min_votes):
            matched, total = _verify(model, scene, x, y, th, sc)
            score = matched / total
            if score < min_score:
                continue
            # oriented bbox corners for display
            lo = model.mids.min(axis=0) - 2
            hi = model.mids.max(axis=0) + 2
            box = np.array([[lo[0], lo[1]], [hi[0], lo[1]], [hi[0], hi[1]], [lo[0], hi[1]]])
            cth, sth = math.cos(th), math.sin(th)
            R = np.array([[cth, -sth], [sth, cth]]) * sc
            corners = (box - model.ref) @ R.T + np.array([x, y])
            detections.append(Detection(x, y, th, sc, model.mirrored, votes, score,
                                        corners, matched, total))

    # NMS across mirror variants and adjacent vote cells
    detections.sort(key=lambda d: (-d.score, -d.votes))
    kept: list[Detection] = []
    r_sup = max(4.0, models[0].radius * 0.8)
    for d in detections:
        if all(math.dist((d.x, d.y), (k.x, k.y)) > r_sup for k in kept):
            kept.append(d)
    return kept
