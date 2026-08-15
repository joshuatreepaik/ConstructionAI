"""Cross-check: does the plan agree with the drawing's own schedules?

WHY THIS STAGE EXISTS
    A drawing set states its quantities twice: as symbols on the plans, and as
    tables (a door schedule, panel schedules). Every other stage of the
    pipeline is the system grading its own work; this is the one place it
    checks against an independent record that was already in the document.
    It is the check a human estimator performs by hand before trusting a
    number - and when the two sides disagree, sometimes the DRAWING is wrong,
    which is worth more than the count itself.

HOW IT WORKS
    plan side      accepted detections from the normal pipeline, each
                   attributed to the nearest room-number tag on the sheet
    schedule side  panel-schedule tables (pasted into the PDF as images) are
                   OCR'd and parsed into circuits, each naming rooms
    report         room by room: corroborated / plan-only / schedule-only

    Rooms rarely disagree outright; the common flag is a circuit labelled by
    function ("RECPTS-CONF RM") instead of by room number, which a human
    resolves in seconds. That is the intended division of labour.

PLATFORM NOTE
    OCR uses the macOS Vision framework (offline, no key). Imports are lazy so
    the pipeline works everywhere else; reconciliation just reports
    unavailability on other platforms.
"""

from __future__ import annotations

import math
import re
from collections import defaultdict
from dataclasses import dataclass, field

from .policy import Policy, DEFAULT
from .sheet import Sheet
from .types import Detection


@dataclass
class RoomCheck:
    room: str
    n_symbols: int                 # detections attributed to this room
    circuits: list[str]            # schedule rows naming this room
    status: str                    # 'corroborated' | 'plan_only' | 'schedule_only'


@dataclass
class ScheduleReport:
    plan_page: int
    schedule_page: int
    n_detections: int              # accepted, in the counted scale band
    n_excluded_scale: int          # accepted but outside the band (other devices)
    n_unattributed: int            # no room tag within reach
    n_circuits: int
    rooms: list[RoomCheck] = field(default_factory=list)

    @property
    def n_corroborated(self):
        return sum(1 for r in self.rooms if r.status == "corroborated")


def room_tags(sheet: Sheet, policy: Policy) -> dict[str, list[tuple]]:
    """Room-number tag centers on the plan, keyed by room.

    Tags inside exclusion zones (key plan, compass) are skipped: the key-plan
    inset repeats every room number in miniature, and attributing a detection
    to one of those would silently pull it across the sheet.
    """
    tag_re = re.compile(policy.room_tag_re)
    tags: dict[str, list[tuple]] = defaultdict(list)
    for x0, y0, x1, y1, text in sheet.words:
        if tag_re.fullmatch(text):
            cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
            if not sheet.in_zone(cx, cy):
                tags[text].append((cx, cy))
    return tags


def _attribute(dets: list[Detection], tags: dict[str, list[tuple]],
               policy: Policy) -> tuple[dict[str, int], int]:
    """Each detection -> nearest room tag, capped so a symbol in an untagged
    corridor is reported as unattributed rather than assigned to whichever
    room happens to be least far away."""
    counted: dict[str, int] = defaultdict(int)
    unattributed = 0
    for d in dets:
        best_room, best_d = None, policy.room_attribute_max_pt
        for room, centers in tags.items():
            for c in centers:
                dist = math.dist((d.x, d.y), c)
                if dist < best_d:
                    best_room, best_d = room, dist
        if best_room is None:
            unattributed += 1
        else:
            counted[best_room] += 1
    return counted, unattributed


def _circuits_on(doc, page_index: int, known_rooms: set[str]):
    from oneshot.schedule import image_regions, ocr_rows, parse_circuits
    circuits = []
    for region in image_regions(doc[page_index]):
        rows = ocr_rows(doc[page_index], region)
        circuits.extend(parse_circuits(rows, panel=str(page_index + 1),
                                       known_rooms=known_rooms))
    return circuits


def find_panel_circuits(doc, skip: int, known_rooms: set[str]):
    """(page_index, circuits) for the page that reads most like a panel
    schedule.

    The tables are pasted into the PDF as images, so their titles are pixels,
    not text - no phrase search can find them. Instead, OCR every page that
    carries pasted images and keep the one that yields the most receptacle
    circuits. Costs a few seconds once per document; the caller caches it.
    """
    from oneshot.schedule import image_regions
    best = (None, [])
    for i in range(len(doc)):
        if i == skip or not image_regions(doc[i]):
            continue
        circuits = _circuits_on(doc, i, known_rooms)
        if len(circuits) > len(best[1]):
            best = (i, circuits)
    return best


def reconcile(doc, plan_sheet: Sheet, dets: list[Detection],
              schedule_page: int | None = None,
              policy: Policy = DEFAULT) -> ScheduleReport | dict:
    """Cross-check accepted detections against the panel schedules.

    Returns a ScheduleReport, or {"error": ...} when OCR is unavailable or no
    schedule page can be found.
    """
    try:                                    # macOS-only Vision framework
        import Vision  # noqa: F401
    except ImportError:
        return {"error": "schedule OCR needs macOS (Vision framework not available)"}

    # plan side: same-size instances only. On an electrical plan a half-scale
    # or double-scale match of the receptacle glyph is a different device
    # (junction box, grid target), not a receptacle to reconcile.
    lo, hi = policy.reconcile_scale_band
    accepted = [d for d in dets if d.decision == "accept"]
    counted_dets = [d for d in accepted if lo <= d.candidate.pose.scale <= hi]
    tags = room_tags(plan_sheet, policy)
    by_room, unattributed = _attribute(counted_dets, tags, policy)

    # schedule side
    if schedule_page is not None:
        circuits = _circuits_on(doc, schedule_page, set(tags))
    else:
        schedule_page, circuits = find_panel_circuits(
            doc, skip=plan_sheet.page_index, known_rooms=set(tags))
        if schedule_page is None:
            return {"error": "no panel-schedule tables found in this document"}
    named: dict[str, list[str]] = defaultdict(list)
    for c in circuits:
        for room in c.rooms:
            named[room].append(c.description)

    report = ScheduleReport(
        plan_page=plan_sheet.page_index,
        schedule_page=schedule_page,
        n_detections=len(counted_dets),
        n_excluded_scale=len(accepted) - len(counted_dets),
        n_unattributed=unattributed,
        n_circuits=len(circuits),
    )
    for room in sorted(set(by_room) | set(named)):
        n_sym, ckts = by_room.get(room, 0), named.get(room, [])
        status = ("corroborated" if n_sym and ckts
                  else "plan_only" if n_sym else "schedule_only")
        report.rooms.append(RoomCheck(room, n_sym, ckts, status))
    return report
