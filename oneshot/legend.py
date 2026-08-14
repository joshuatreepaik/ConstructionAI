"""Harvest exemplar symbols from a drawing's own legend.

A legend is rows of [symbol glyph] [description text]. We locate the legend
heading, take the column beneath it, band the vector primitives into rows,
and attach each row's text as the symbol's label. Every harvested row becomes
a ready-made exemplar for the detection engine - the drawing teaches the
system its own symbology, no user tracing and no training data.
"""

from __future__ import annotations

from dataclasses import dataclass

from .extract import Primitive, extract_words


@dataclass
class LegendSymbol:
    label: str
    prims: list[Primitive]
    box: tuple            # display-space bbox of the glyph


def _find_headings(words, phrases):
    """All (x0, y1) anchors matching any heading phrase, deduped."""
    hits = []
    for phrase in phrases:
        parts = phrase.upper().split()
        for i, w in enumerate(words):
            if w[4].upper() != parts[0]:
                continue
            got = [w]
            for p in parts[1:]:
                nxt = [v for v in words
                       if v[4].upper() == p
                       and abs(v[1] - got[-1][1]) < 8
                       and 0 < v[0] - got[-1][2] < 30]
                if not nxt:
                    break
                got.append(nxt[0])
            if len(got) == len(parts):
                x0 = min(v[0] for v in got)
                y1 = max(v[3] for v in got)
                if all(abs(x0 - hx) > 40 or abs(y1 - hy) > 40 for hx, hy in hits):
                    hits.append((x0, y1))
    return hits


def harvest(page, prims: list[Primitive],
            headings=("SYMBOLS LIST", "DRAWING LEGEND", "DOOR LEGEND", "LEGEND"),
            col_width=280.0, glyph_col=70.0, row_gap=7.0,
            max_glyph_extent=45.0) -> list[LegendSymbol]:
    words = extract_words(page)
    out = []
    for hx, hy in _find_headings(words, headings):
        out.extend(_harvest_column(page, prims, words, hx, hy, col_width,
                                   glyph_col, row_gap, max_glyph_extent))
    return out


def _harvest_column(page, prims, words, hx, hy, col_width, glyph_col,
                    row_gap, max_glyph_extent) -> list[LegendSymbol]:
    x0, y0 = hx - 6, hy + 4
    x1 = x0 + col_width
    y1 = page.rect.height - 40

    # glyphs live in the left strip of the legend column
    strip = [p for p in prims
             if x0 <= p.mid[0] <= x0 + glyph_col and y0 <= p.mid[1] <= y1
             and max(abs(p.p1[0] - p.p0[0]), abs(p.p1[1] - p.p0[1])) < max_glyph_extent]
    if not strip:
        return []

    # band into rows by vertical gaps
    strip.sort(key=lambda p: p.mid[1])
    rows, cur = [], [strip[0]]
    for p in strip[1:]:
        if p.mid[1] - max(q.mid[1] for q in cur) > row_gap:
            rows.append(cur)
            cur = [p]
        else:
            cur.append(p)
    rows.append(cur)

    out = []
    for row in rows:
        if len(row) < 2:
            continue
        gx0 = min(min(p.p0[0], p.p1[0]) for p in row)
        gx1 = max(max(p.p0[0], p.p1[0]) for p in row)
        gy0 = min(min(p.p0[1], p.p1[1]) for p in row)
        gy1 = max(max(p.p0[1], p.p1[1]) for p in row)
        if gx1 - gx0 > max_glyph_extent or gy1 - gy0 > max_glyph_extent:
            continue
        cy = (gy0 + gy1) / 2
        # label: first text line to the right of the glyph, near its center
        line = sorted([w for w in words
                       if w[0] > gx1 and w[0] < x1 + 200
                       and w[1] < cy + 5 and w[3] > cy - 5],
                      key=lambda w: w[0])
        label = " ".join(w[4] for w in line[:9]) or "(unlabeled)"
        out.append(LegendSymbol(label=label, prims=row,
                                box=(gx0, gy0, gx1, gy1)))
    return out
