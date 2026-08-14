"""Stage 1: fingerprint index with per-document saliency weights.

Same pair fingerprints as v1 (reused read-only from oneshot.engine), plus the
two things v1 lacked:

- **IDF weights**: a vote from a rare fingerprint counts more than one from a
  ubiquitous fingerprint. Rarity is measured on THIS document — hatching in
  one drawing is signal in another.
- **Occurrence filter**: fingerprint buckets producing more than a share of
  all pairs (hatching, grids, dimension chains) are dropped outright.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from oneshot.extract import Primitive, as_arrays
from oneshot.engine import _pair_features

from .policy import Policy, DEFAULT


class SaliencyIndex:
    def __init__(self, prims: list[Primitive], policy: Policy = DEFAULT):
        self.prims = prims
        self.policy = policy
        self.mids, self.lengths, self.thetas, self.kinds, self.bends = as_arrays(prims)
        self.tree = cKDTree(self.mids)

        n = len(prims)
        k = min(policy.max_neighbors + 1, n)
        dists, idxs = self.tree.query(self.mids, k=k,
                                      distance_upper_bound=policy.max_pair_dist)
        ii = np.repeat(np.arange(n), k)
        jj = idxs.ravel()
        valid = (jj < n) & (jj != ii)
        ii, jj = ii[valid], jj[valid]
        key, dist, ang = _pair_features(self.mids, self.lengths, self.thetas,
                                        self.kinds, self.bends, ii, jj)
        ok = dist > 0.75
        self.pair_ii, self.pair_jj = ii[ok], jj[ok]
        self.pair_key, self.pair_dist, self.pair_ang = key[ok], dist[ok], ang[ok]

        # ---- saliency statistics ------------------------------------------
        uniq, inv, counts = np.unique(self.pair_key, return_inverse=True,
                                      return_counts=True)
        total = len(self.pair_key)
        share = counts / max(total, 1)
        dropped = share > policy.occurrence_max_share
        # IDF: log(total / bucket_count); dropped buckets get weight 0
        w = np.log(total / counts)
        w[dropped] = 0.0
        self.pair_weight = w[inv]              # per-pair vote weight
        self.n_buckets = len(uniq)
        self.n_dropped_buckets = int(dropped.sum())
        self.dropped_pair_share = float(counts[dropped].sum() / max(total, 1))

        # lookup table for exemplar-side weights (how informative is each of
        # MY buckets on this sheet); buckets absent from the sheet get the
        # maximum weight (rarest possible)
        self._bucket_keys = uniq
        self._bucket_weight = w
        self._max_weight = float(np.log(max(total, 2)))

    def weight_of_keys(self, keys: np.ndarray) -> np.ndarray:
        """IDF weight for arbitrary fingerprint keys (exemplar self-weights)."""
        pos = np.searchsorted(self._bucket_keys, keys)
        pos = np.clip(pos, 0, len(self._bucket_keys) - 1)
        hit = self._bucket_keys[pos] == keys
        out = np.full(len(keys), self._max_weight)
        out[hit] = self._bucket_weight[pos[hit]]
        return out

    def local_density(self, x: float, y: float, radius: float) -> float:
        """Scene primitives per pt^2 near a pose — the evidence background."""
        n = len(self.tree.query_ball_point((x, y), r=radius))
        return n / (np.pi * radius * radius)
