"""Overlay rendering: draw detections on the sheet, export PNG/PDF."""

from __future__ import annotations

import pymupdf


def render_overlay_png(pdf_path, page_index, detections, out_png,
                       exemplar_box=None, zoom=3.0, clip=None, label_scores=True):
    doc = pymupdf.open(pdf_path)
    page = doc[page_index]
    # Shape/insert_text expect UNROTATED page coordinates; detections are in
    # display (rotated) space -> derotate every point before drawing.
    dmat = page.derotation_matrix
    D = lambda x, y: pymupdf.Point(x, y) * dmat
    shape = page.new_shape()
    for d in detections:
        pts = [D(x, y) for x, y in d.corners]
        shape.draw_polyline(pts + [pts[0]])
    shape.finish(color=(1, 0, 0), width=1.2)
    if exemplar_box is not None:
        x0, y0, x1, y1 = exemplar_box
        pts = [D(x0, y0), D(x1, y0), D(x1, y1), D(x0, y1)]
        shape.draw_polyline(pts + [pts[0]])
        shape.finish(color=(0, 0.5, 1), width=1.6)
    shape.commit()
    if label_scores:
        for d in detections:
            x = min(c[0] for c in d.corners)
            y = min(c[1] for c in d.corners)
            page.insert_text(D(x, y - 1.5), f"{d.score:.2f}",
                             fontsize=4, color=(1, 0, 0), rotate=page.rotation)
    mat = pymupdf.Matrix(zoom, zoom)
    rect = pymupdf.Rect(*clip) if clip else page.rect
    pix = page.get_pixmap(matrix=mat, clip=rect)
    pix.save(out_png)
    doc.close()
    return out_png


def render_crop(pdf_path, page_index, box, out_png, zoom=8.0, pad=6.0):
    doc = pymupdf.open(pdf_path)
    page = doc[page_index]
    x0, y0, x1, y1 = box
    rect = pymupdf.Rect(x0 - pad, y0 - pad, x1 + pad, y1 + pad) & page.rect
    if rect.is_empty or rect.width < 0.5 or rect.height < 0.5:
        raise ValueError(f"crop rect {box} empty after page clip {page.rect}")
    pix = page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom), clip=rect)
    pix.save(out_png)
    doc.close()
    return out_png
