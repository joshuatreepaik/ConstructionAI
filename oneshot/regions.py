"""Sheet-region masking: keep counts inside the plan, not the furniture
around it.

Compass roses, scale bars, key-plan insets and legends all contain geometry
that can satisfy a matcher (a compass wedge IS an arc + radius line). Each of
those artifacts is reliably captioned - "PLAN NORTH", "SCALE", "KEY PLAN",
"LEGEND" - so the captions in the text layer anchor exclusion zones.
"""

from __future__ import annotations

import re

from .extract import extract_words

# e.g.  SCALE 1/8" = 1'-0"   or   3/16" = 1'-0"
_SCALE_RE = re.compile(r"(\d+(?:\s*/\s*\d+)?)\s*\"\s*=\s*(\d+)\s*'")


def plan_scale_pt_per_ft(page, default=None):
    """Points-per-foot parsed from the sheet's own scale notes.

    Sheets carry several scales (plan + enlarged details); the FLOOR PLAN is
    the largest drawing so it uses the smallest inches-per-foot - take the
    minimum. Returns `default` when no imperial scale note is found.
    """
    best = None
    for m in _SCALE_RE.finditer(page.get_text()):
        frac, feet = m.group(1), int(m.group(2))
        if feet == 0:
            continue
        if "/" in frac:
            a, b = frac.split("/")
            inches = float(a) / float(b)
        else:
            inches = float(frac)
        ppf = inches * 72.0 / feet
        if best is None or ppf < best:
            best = ppf
    return best if best is not None else default


def sheet_number(page) -> str | None:
    """Sheet number (T5, E4, A-101...) from the title block corner."""
    W, H = page.rect.width, page.rect.height
    best = None
    for w in extract_words(page):
        if w[0] > 0.82 * W and w[1] > 0.75 * H:
            t = w[4].strip()
            if re.fullmatch(r"[A-Z]{1,3}-?\d{1,4}(\.\d{1,2})?", t):
                h = w[3] - w[1]
                if best is None or h > best[0]:
                    best = (h, t)
    return best[1] if best else None


def exclusion_zones(page) -> list[tuple]:
    """Display-space rects around non-plan artifacts."""
    words = extract_words(page)
    zones = []
    for i, w in enumerate(words):
        t = w[4].upper().rstrip(":")
        x0, y0, x1, y1 = w[:4]
        if t == "NORTH":
            # compass rose sits just below its label
            zones.append((x0 - 80, y0 - 50, x1 + 80, y1 + 150))
        elif t == "SCALE":
            # graphic scale bar sits beside/above the note
            zones.append((x0 - 60, y0 - 70, x1 + 280, y1 + 30))
        elif t == "KEY" and i + 1 < len(words) and words[i + 1][4].upper().startswith("PLAN"):
            # key-plan inset map sits above its caption
            zones.append((x0 - 200, y0 - 340, x1 + 360, y1 + 90))
        elif t in ("LEGEND", "SYMBOLS"):
            # legend column runs below its heading
            zones.append((x0 - 20, y0 - 20, x0 + 320, y0 + 700))
    return zones


def outside_zones(detections, zones):
    if not zones:
        return detections
    kept = []
    for d in detections:
        if hasattr(d, "x"):
            x, y = d.x, d.y
        elif hasattr(d, "hinge"):
            x, y = d.hinge
        else:
            x, y = d
        if any(zx0 <= x <= zx1 and zy0 <= y <= zy1 for zx0, zy0, zx1, zy1 in zones):
            continue
        kept.append(d)
    return kept
