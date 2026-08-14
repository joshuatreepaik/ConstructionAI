"""Read panel schedules that engineers pasted into the PDF as images.

WHY THIS EXISTS
    A drawing set states its quantities twice: as symbols on the plans, and as
    tables (door schedules, panel schedules). Reading the tables lets the
    system CHECK ITS OWN COUNT - see scripts/reconcile.py. That is the closest
    thing to ground truth a drawing set carries.

WHY OCR IS NEEDED AT ALL
    Most of the set is live text, but the E6 panel tables were exported from
    panel-design software and dropped in as pictures, so they have no text
    layer. We render each pasted image at high zoom and OCR it with the macOS
    Vision framework - offline, no external service, no API key.

    That mix (half machine-readable, half pictures of spreadsheets) is typical
    of real drawing sets, which is why both paths exist.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import pymupdf
import Quartz
import Vision


@dataclass
class Circuit:
    panel: str
    description: str
    rooms: list[str]
    kind: str          # 'receptacle' | 'floor_receptacle' | 'other'


def ocr_png_bytes(png: bytes) -> list[tuple]:
    """OCR -> [(x, y, w, h, text)] with normalized coords, y from TOP."""
    data = Quartz.CFDataCreate(None, png, len(png))
    src = Quartz.CGImageSourceCreateWithData(data, None)
    img = Quartz.CGImageSourceCreateImageAtIndex(src, 0, None)
    handler = Vision.VNImageRequestHandler.alloc().initWithCGImage_options_(img, None)
    req = Vision.VNRecognizeTextRequest.alloc().init()
    req.setRecognitionLevel_(0)          # 0 = accurate
    req.setUsesLanguageCorrection_(False)
    ok = handler.performRequests_error_([req], None)
    out = []
    if req.results():
        for obs in req.results():
            cand = obs.topCandidates_(1)[0]
            bb = obs.boundingBox()
            out.append((float(bb.origin.x),
                        1.0 - float(bb.origin.y) - float(bb.size.height),
                        float(bb.size.width), float(bb.size.height),
                        str(cand.string())))
    return out


def ocr_rows(page: pymupdf.Page, clip, zoom=4.0) -> list[str]:
    """Render a region and return OCR text reassembled into rows."""
    pix = page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom),
                          clip=pymupdf.Rect(*clip))
    obs = ocr_png_bytes(pix.tobytes("png"))
    obs.sort(key=lambda o: o[1])
    rows, cur, cur_y = [], [], None
    for o in obs:
        cy = o[1] + o[3] / 2
        if cur_y is None or abs(cy - cur_y) < 0.6 * o[3]:
            cur.append(o)
            cur_y = cy if cur_y is None else (cur_y + cy) / 2
        else:
            rows.append(cur)
            cur, cur_y = [o], cy
    if cur:
        rows.append(cur)
    return [" ".join(o[4] for o in sorted(r, key=lambda o: o[0])) for r in rows]


ROOM_RE = re.compile(r"\b([12]\d\d)\b")
RANGE_RE = re.compile(r"\b([12]\d\d)\s*[-–]\s*([12]\d\d)\b")
RECPT_RE = re.compile(r"RE?CE?PTS?|RECEPT", re.I)
FLOOR_RE = re.compile(r"FLOOR|FLR", re.I)


def rooms_in(text: str, known: set[str] | None = None) -> list[str]:
    """Room numbers a circuit description refers to.

    Engineers write both "RECPTS-217,218" and "RECPTS-222-224"; the second is
    a RANGE covering 222, 223 and 224. Expanding it matters because a circuit
    that names three rooms should corroborate all three. `known` (the room
    tags actually printed on the plan) filters out coincidental 3-digit
    numbers such as wattages.
    """
    rooms: set[str] = set()
    for a, b in RANGE_RE.findall(text):
        lo, hi = int(a), int(b)
        if lo <= hi <= lo + 12:          # a plausible room range, not a typo
            rooms.update(str(r) for r in range(lo, hi + 1))
    if not rooms:
        rooms.update(ROOM_RE.findall(text))
    if known is not None:
        rooms &= known
    return sorted(rooms)


def parse_circuits(rows: list[str], panel: str,
                   known_rooms: set[str] | None = None) -> list[Circuit]:
    """Receptacle circuits from OCR'd panel-schedule rows.

    Panel schedules print two circuits per printed row (odd numbers left, even
    right), so each row is split before parsing or one circuit's rooms would
    be attributed to the other.
    """
    out = []
    for row in rows:
        if not RECPT_RE.search(row):
            continue
        for part in re.split(r"\s{2,}|\|", row):
            if not RECPT_RE.search(part):
                continue
            out.append(Circuit(
                panel=panel,
                description=part.strip()[:70],
                rooms=rooms_in(part, known_rooms),
                kind="floor_receptacle" if FLOOR_RE.search(part) else "receptacle",
            ))
    return out


def image_regions(page: pymupdf.Page) -> list[tuple]:
    """Display-space rects of embedded images (the pasted schedule tables)."""
    mat = page.rotation_matrix
    regions = []
    for im in page.get_images(full=True):
        for r in page.get_image_rects(im[0]):
            rr = r * mat
            rr.normalize()
            if rr.width > 100 and rr.height > 100:      # skip logos/stamps
                regions.append((rr.x0, rr.y0, rr.x1, rr.y1))
    return regions
