"""Every decision constant in the v2 pipeline, in one place, each with a
one-line rationale. Nothing elsewhere in pipeline/ may hard-code a threshold.

Values marked (calibrate) are defaults to be replaced by the offline
calibration run; until then they are honest engineering guesses.
"""

from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class Policy:
    # ---- Stage 1: saliency index -------------------------------------------
    max_pair_dist: float = 64.0
    # pairs only need to span one symbol; 64pt covers every symbol on 1/8" sheets
    max_neighbors: int = 48
    # bounds index cost regardless of sheet density (Shazam's density criterion)
    occurrence_max_share: float = 0.01
    # drop hash buckets producing >1% of all pairs: hatching/grids by construction
    # (FAST's occurrence filter: 30% of data dropped, zero false negatives)

    # ---- Stage 2: rigid proposer -------------------------------------------
    scale_range: tuple = (0.5, 2.0)
    # blocks are reused at other scales (x0.8 receptacles observed); voting
    # recovers the scale so the range can be wide
    scale_range_strict: tuple = (0.85, 1.15)
    # "same size only" UI toggle
    vote_cell: float = 3.0
    theta_cell: float = math.radians(12)
    logs_cell: float = 0.12
    # accumulator bins ~= measured pose scatter of true matches (2x jitter)
    min_vote_mass_ratio: float = 0.35
    # a candidate needs >=35% of the exemplar's own self-vote mass; relative,
    # so rare-symbol votes (high IDF weight) and common ones use one rule.
    # Set below 0.5 deliberately: proposal favors recall (P1), evidence prunes.
    orient_gate: float = math.radians(15)
    # votes whose primitive orientation disagrees with the pose are noise

    # ---- Stage 3: evidence --------------------------------------------------
    pos_sigma: float = 0.8
    # combined placement noise of CAD-exported primitives, in pt (measured
    # jitter is < 0.1pt for true copies; 0.8 tolerates block variation)
    len_tol: float = 0.22
    ang_tol: float = math.radians(12)
    bend_tol: float = 0.15
    # attribute gates for counting a primitive as present (same as v1 verify)
    distractor_rate: float = 0.25
    # expected fraction of template primitives hidden by overlapping linework
    # (Astrometry.net's default; calibrate)
    accept_log_odds: float = 6.0
    # set from the measured post-arbitration distribution on E4: true
    # instances cluster above ~6, junk below ~4  (calibrate properly later)
    review_log_odds: float = 3.0
    # below accept but plausible -> human review queue, not silence
    door_accept_log_odds: float = 5.0
    door_review_log_odds: float = 2.0
    # doors have only 3 structural evidence parts (arc + 2 leaf lines), each
    # far more specific than a generic primitive; measured: 2-leaf doors score
    # 5.6-11.4, 1-leaf 1.6-2.8  (calibrate)

    # ---- Stage 4: resolution ------------------------------------------------
    suppress_extent_ratio: float = 0.5
    # two detections closer than half their extent are the same instance
    tag_radius_ratio: float = 0.55
    # text within 55% of the symbol radius is a label ON the symbol
    # (subclass or veto), beyond that it is neighborhood annotation

    # ---- parametric (door) proposer ----------------------------------------
    door_min_chord: float = 14.0
    door_max_chord: float = 90.0
    # door widths 1.5ft..10ft at 1/8" scale; wider than any real leaf
    door_bend_tol: float = 0.045
    # arcs drawn 80-100 degrees still read as quarter-circle swings
    door_end_tol: float = 3.0
    door_leaf_len_tol: float = 0.18
    door_dedupe_radius: float = 4.0
    # a door's identity is its hinge; back-to-back office doors hinge ~12pt
    # apart at a shared jamb and are DISTINCT doors (measured on T8), so
    # dedupe must be a small absolute radius, never a fraction of width

    # ---- schedule reconciliation --------------------------------------------
    room_tag_re: str = r"[1-9]\d{2}"
    # room tags on this set are 3-digit numbers (2nd floor -> 2xx); the
    # leading [1-9] rejects "025"-style dimension fragments
    room_attribute_max_pt: float = 150.0
    # ~20 ft at the sheet's 1/8" scale: a receptacle further than that from
    # every room tag is in an untagged corridor - report it unattributed
    # rather than assign it to whichever room is least far away
    reconcile_scale_band: tuple = (0.7, 1.15)
    # same-size instances only: on E4 the x0.5 matches are junction boxes and
    # the x1.8 matches are grid targets - different devices, not receptacles
    # miscounted (measured by inspecting each off-scale hit)


DEFAULT = Policy()
