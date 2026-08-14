"""Stage 0: one parse per sheet, one coordinate space, redundancy exposed.

Reuses the v1 parsing functions (read-only) but calls each exactly once and
holds the results together, so no downstream stage ever re-extracts. This
kills the v1 bug class where text and geometry were read in different
coordinate frames.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property

import numpy as np
import pymupdf

from oneshot.extract import Primitive, extract_primitives, extract_words
from oneshot.legend import _find_headings
from oneshot.regions import exclusion_zones, plan_scale_pt_per_ft, sheet_number

LEGEND_HEADINGS = ("SYMBOLS LIST", "DRAWING LEGEND", "DOOR LEGEND",
                   "WALL TYPE LEGEND", "LEGEND")


@dataclass
class Sheet:
    page_index: int
    primitives: list[Primitive]
    words: list[tuple]                # (x0, y0, x1, y1, text), display space
    zones: list[tuple]                # exclusion rects (compass, key plan, ...)
    scale_pt_per_ft: float | None
    sheet_id: str | None              # "E4", "T5", ...
    is_legend: bool
    width: float
    height: float

    @classmethod
    def from_page(cls, page: pymupdf.Page, page_index: int) -> "Sheet":
        words = extract_words(page)
        return cls(
            page_index=page_index,
            primitives=extract_primitives(page),
            words=words,
            zones=exclusion_zones(page),
            scale_pt_per_ft=plan_scale_pt_per_ft(page),
            sheet_id=sheet_number(page),
            is_legend=bool(_find_headings(words, LEGEND_HEADINGS)),
            width=float(page.rect.width),
            height=float(page.rect.height),
        )

    @cached_property
    def word_centers(self) -> np.ndarray:
        if not self.words:
            return np.zeros((0, 2))
        return np.array([[(w[0] + w[2]) / 2, (w[1] + w[3]) / 2]
                         for w in self.words], dtype=np.float64)

    def in_zone(self, x: float, y: float) -> bool:
        return any(zx0 <= x <= zx1 and zy0 <= y <= zy1
                   for zx0, zy0, zx1, zy1 in self.zones)


class SheetCache:
    """Per-document lazy cache of Sheets (the v2 counterpart of DocState's
    six parallel dicts, but one object per page instead)."""

    def __init__(self, doc: pymupdf.Document):
        self.doc = doc
        self._sheets: dict[int, Sheet] = {}

    def get(self, page_index: int) -> Sheet:
        if page_index not in self._sheets:
            self._sheets[page_index] = Sheet.from_page(self.doc[page_index], page_index)
        return self._sheets[page_index]
