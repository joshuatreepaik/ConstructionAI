"""Cross-check doors: the same floor is drawn on many sheets.

WHY DOORS RECONCILE DIFFERENTLY
    This drawing set has no door schedule table. T1's "DOOR SCHEDULE" is
    specification notes, the WS/PA type tags exist only in the legend text,
    and the doors on the plan carry no adjacent marks (all three measured, not
    assumed). So there is no table to check against.

    But a drawing set redraws the same floor once per trade: architectural
    plan, finish plan, power plan. Each is an independent rendering of the
    same doors, usually exported from the same CAD model. Those other sheets
    ARE the second source. A door confirmed on two or three sheets is
    corroborated; a door on the architectural plan missing from the finish
    plan is either a detection miss or a drafting inconsistency, and both are
    worth a human glance.

HOW SHEETS ARE ALIGNED
    Different trades place the plan at different offsets on the page (E4 sits
    807pt left of T5's frame - measured). Every hinge pair between two sheets
    votes for a translation; the true offset repeats across every real door
    while wrong pairings scatter. Same voting trick the rigid proposer uses
    for symbol poses, reused for whole sheets. A sheet whose best offset is
    supported by too few doors is not the same floor plan and is skipped.

This check is pure geometry on live vectors - no OCR, runs anywhere.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field

from .policy import Policy, DEFAULT
from .proposers import parametric
from .types import Detection


@dataclass
class SheetAgreement:
    sheet_id: str                  # "T8", "E4", ...
    page_index: int
    n_doors: int                   # doors that sheet shows (in aligned frame)
    offset: tuple                  # (dx, dy) mapping plan frame -> that sheet
    n_matched: int                 # traced-sheet doors confirmed there


@dataclass
class DoorReport:
    plan_page: int
    n_doors: int
    sheets: list[SheetAgreement] = field(default_factory=list)
    # per traced door: which sheets confirm it (parallel to the input order)
    confirmed_on: list[list[str]] = field(default_factory=list)
    width_disagrees: list[bool] = field(default_factory=list)

    @property
    def n_confirmed(self):
        return sum(1 for s in self.confirmed_on if s)


def _hinges(dets: list[Detection]):
    return [((d.x, d.y), d.candidate.support.get("width_pt", 0.0))
            for d in dets if d.decision == "accept" and d.candidate.cls == "door"]


def _door_candidates(v2, page_index: int, policy: Policy):
    """Doors on another sheet, via the same proposer (and cached: door plans
    get compared against every other sheet, so each sheet is scanned once)."""
    cache = v2.door_cands
    if page_index not in cache:
        sheet = v2.sheets.get(page_index)
        if sheet.is_legend:
            # the legend's door-type elevations are drawings OF doors, not
            # doors IN the building - never let them corroborate anything
            cache[page_index] = []
        else:
            cands = parametric.propose(sheet.primitives, policy)
            cache[page_index] = [((c.pose.x, c.pose.y),
                                  c.support.get("width_pt", 0.0)) for c in cands]
    return cache[page_index]


def _best_offset(a, b):
    """Translation (dx, dy) mapping frame a -> frame b, by voting: the true
    offset repeats for every door both sheets share, wrong pairings scatter."""
    votes = Counter()
    for (ah, _) in a:
        for (bh, _) in b:
            votes[(round(bh[0] - ah[0]), round(bh[1] - ah[1]))] += 1
    if not votes:
        return None, 0
    (dx, dy), n = votes.most_common(1)[0]
    return (float(dx), float(dy)), n


def reconcile_doors(v2, plan_sheet, dets: list[Detection],
                    policy: Policy = DEFAULT) -> DoorReport | dict:
    """Cross-check accepted door detections against every other plan sheet."""
    doors = _hinges(dets)
    if not doors:
        return {"error": "no accepted door detections to cross-check"}

    report = DoorReport(plan_page=plan_sheet.page_index, n_doors=len(doors))
    confirmed = [[] for _ in doors]
    width_bad = [False] * len(doors)

    n_pages = len(v2.sheets.doc)
    for page in range(n_pages):
        if page == plan_sheet.page_index:
            continue
        other = _door_candidates(v2, page, policy)
        # a sheet must share a meaningful fraction of the doors before its
        # frame offset is trusted - two unrelated sheets always have SOME
        # accidental best offset
        min_votes = max(policy.door_xsheet_min_votes,
                        int(policy.door_xsheet_min_frac * min(len(doors), len(other)))) \
            if other else 0
        if not other:
            continue
        offset, votes = _best_offset(doors, other)
        if offset is None or votes < min_votes:
            continue

        sid = v2.sheets.get(page).sheet_id or f"p{page + 1}"
        n_matched = 0
        for i, (hinge, width) in enumerate(doors):
            shifted = (hinge[0] + offset[0], hinge[1] + offset[1])
            best = None
            for oh, ow in other:
                dist = math.dist(shifted, oh)
                if dist < policy.door_xsheet_match_pt and (best is None or dist < best[0]):
                    best = (dist, ow)
            if best is not None:
                n_matched += 1
                confirmed[i].append(sid)
                # matched doors are the SAME door - widths should agree
                if width and best[1] and abs(width - best[1]) > policy.door_width_agree_pt:
                    width_bad[i] = True
        report.sheets.append(SheetAgreement(
            sheet_id=sid, page_index=page, n_doors=len(other),
            offset=offset, n_matched=n_matched))

    report.sheets.sort(key=lambda s: -s.n_matched)
    report.confirmed_on = confirmed
    report.width_disagrees = width_bad
    return report
