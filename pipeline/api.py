"""v2 pipeline facade.

Same request/response shapes as the v1 handlers so the UI and comparison
scripts can call either engine interchangeably. v1 code paths are untouched;
this module only reads shared parsers.
"""

from __future__ import annotations

import math
import time

from oneshot.extract import primitives_in_box
from oneshot.doors import width_label
from oneshot.legend import harvest

from .evidence import score
from .index import SaliencyIndex
from .policy import DEFAULT, Policy
from .proposers import parametric, rigid
from .resolve import resolve
from .sheet import Sheet, SheetCache
from .types import Detection


class V2State:
    """Per-document v2 caches, attached lazily to the app's DocState."""

    def __init__(self, doc):
        self.sheets = SheetCache(doc)
        self.indexes: dict[int, SaliencyIndex] = {}

    def index_for(self, page_index: int, policy: Policy) -> SaliencyIndex:
        if page_index not in self.indexes:
            self.indexes[page_index] = SaliencyIndex(
                self.sheets.get(page_index).primitives, policy)
        return self.indexes[page_index]


def get_v2(doc_state) -> V2State:
    v2 = getattr(doc_state, "_v2", None)
    if v2 is None:
        v2 = V2State(doc_state.doc)
        setattr(doc_state, "_v2", v2)
    return v2


def confidence(log_odds: float) -> int:
    """Squash log-odds to a 0-100 display confidence (logistic, /3 spread:
    log-odds 0 -> 50%, 3 -> 73%, 6 -> 88%, 9 -> 95%)."""
    return int(round(100.0 / (1.0 + math.exp(-log_odds / 3.0))))


def _det_json(d: Detection, sheet: Sheet):
    j = {
        "corners": [[round(float(c[0]), 2), round(float(c[1]), 2)]
                    for c in d.corners],
        "score": round(d.evidence.frac, 2),
        "log_odds": round(d.evidence.log_odds, 1),
        "confidence": confidence(d.evidence.log_odds),
        "decision": d.decision,
        "scale": round(d.candidate.pose.scale, 2),
        "theta_deg": round(math.degrees(d.candidate.pose.theta), 1),
        "mirrored": d.candidate.pose.mirrored,
    }
    if d.candidate.cls == "door":
        w = d.candidate.support["width_pt"]
        j["width"] = (width_label(w, sheet.scale_pt_per_ft)
                      if sheet.scale_pt_per_ft else f"{w:.0f}pt")
    if d.subclass:
        j["subclass"] = d.subclass
    return j


def _exemplar_from_box(sheet: Sheet, box, policy: Policy):
    from app import _main_cluster            # lazy: avoids circular import
    grow = 1.5
    box_g = (box[0] - grow, box[1] - grow, box[2] + grow, box[3] + grow)
    traced = primitives_in_box(sheet.primitives, box_g)
    if len(traced) < 2:
        return None, traced
    diag = math.hypot(box[2] - box[0], box[3] - box[1])
    return _main_cluster(traced, thr=max(3.0, 0.15 * diag), box=box), traced


def detect(doc_state, page_index: int, box, any_size=True, veto_text=True,
           policy: Policy = DEFAULT) -> dict:
    t0 = time.time()
    v2 = get_v2(doc_state)
    sheet = v2.sheets.get(page_index)
    exemplar, traced = _exemplar_from_box(sheet, box, policy)
    if exemplar is None:
        return {"error": f"only {len(traced)} primitives fully inside the box - "
                         "trace a little wider around one clean symbol"}

    index = v2.index_for(page_index, policy)

    if parametric.looks_like_door(exemplar, policy):
        cands = parametric.propose(sheet.primitives, policy)
        mode = "door"
    else:
        cands = rigid.propose(index, exemplar, policy, any_size=any_size)
        mode = "symbol"

    dets = resolve([(c, score(c, index, policy)) for c in cands],
                   sheet, policy, veto_text=veto_text)
    accepted = [d for d in dets if d.decision == "accept"]
    review = [d for d in dets if d.decision == "review"]
    # evidence-weak candidates ride along (disqualified labeled/zone hits do
    # not) so the client's confidence slider can reveal them without re-query
    weak = [d for d in dets
            if d.decision == "reject" and d.reason == "evidence"
            and confidence(d.evidence.log_odds) >= 15]

    by_rot, by_scale = {}, {}
    for d in accepted:
        if d.candidate.cls == "door":
            w = _det_json(d, sheet)["width"]
            by_scale[w] = by_scale.get(w, 0) + 1
        else:
            deg = round(math.degrees(d.candidate.pose.theta) / 45) * 45 % 360
            k = f"{deg}°" + (" mir" if d.candidate.pose.mirrored else "")
            by_rot[k] = by_rot.get(k, 0) + 1
            ks = f"×{round(d.candidate.pose.scale, 1)}"
            by_scale[ks] = by_scale.get(ks, 0) + 1

    acc_thr = (policy.door_accept_log_odds if mode == "door"
               else policy.accept_log_odds)
    rev_thr = (policy.door_review_log_odds if mode == "door"
               else policy.review_log_odds)
    shown = sorted(accepted + review + weak,
                   key=lambda d: -d.evidence.log_odds)
    return {
        "engine": "v2",
        "mode": mode,
        "count": len(accepted),
        "review_count": len(review),
        "box_used": list(box),
        "elapsed": round(time.time() - t0, 2),
        "n_exemplar_prims": len(exemplar),
        "n_traced_prims": len(traced),
        "by_rotation": dict(sorted(by_rot.items())),
        "by_scale": dict(sorted(by_scale.items())),
        "conf_accept": confidence(acc_thr),
        "conf_review": confidence(rev_thr),
        "detections": [_det_json(d, sheet) for d in shown],
    }


def legend_count(doc_state, page_index: int, policy: Policy = DEFAULT) -> dict:
    """Auto-count with ONE arbiter across all legend symbols — v1 needed a
    separate competitive-assignment pass because its per-symbol detect() calls
    each ran their own NMS."""
    t0 = time.time()
    v2 = get_v2(doc_state)
    legend_idxs = [i for i in range(len(doc_state.doc))
                   if v2.sheets.get(i).is_legend]
    if not legend_idxs:
        return {"error": "no legend sheet found in this document"}

    symbols = []
    for li in legend_idxs:
        ls = v2.sheets.get(li)
        symbols.extend(harvest(doc_state.doc[li], ls.primitives))

    sheet = v2.sheets.get(page_index)
    index = v2.index_for(page_index, policy)

    scored = []
    for s in symbols:
        if not (3 <= len(s.prims) <= 30):
            continue
        diag = math.hypot(s.box[2] - s.box[0], s.box[3] - s.box[1])
        if diag > policy.max_pair_dist:
            continue
        try:
            cands = rigid.propose(index, s.prims, policy, source=f"legend:{s.label[:40]}")
        except ValueError:
            continue
        for c in cands:
            scored.append((c, score(c, index, policy), s.label[:60]))

    dets = resolve([(c, e) for c, e, _ in scored], sheet, policy)
    label_by_id = {id(c): lab for c, _, lab in scored}
    by_label: dict[str, list] = {}
    for d in dets:
        if d.decision != "accept":
            continue
        lab = label_by_id.get(id(d.candidate), "?")
        by_label.setdefault(lab, []).append(d)

    results = [{
        "label": lab,
        "count": len(ds),
        "detections": [_det_json(d, sheet) for d in ds],
    } for lab, ds in by_label.items()]
    results.sort(key=lambda r: -r["count"])
    return {
        "engine": "v2",
        "elapsed": round(time.time() - t0, 1),
        "n_legend_symbols": len(symbols),
        "n_matched": len(results),
        "results": results,
    }
