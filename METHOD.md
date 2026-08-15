# Research basis

Notes written before any code, on 2026-08-08, recording what I read and why the
design came out the way it did. This is the reasoning, not the design document:
the architecture itself is in the README diagram and in `pipeline/__init__.py`,
which describe what was actually built rather than what was planned.

## 1. What makes this hard

Counting symbols on a drawing sounds like template matching until you look at
real sheets. Eight things break a naive implementation, and everything in the
design is aimed at one of them.

1. **Occlusion.** Symbols get crossed by walls, wires and leader lines
   constantly. Anything that needs to see the whole symbol will either miss
   those or, loosened enough to catch them, start matching everything.
2. **Confidence has to mean something.** A raw similarity percentage is not a
   probability, so a threshold chosen on one sheet is arbitrary on the next.
3. **Rotation, mirroring and scale.** Symbols appear at any angle, and doors and
   receptacles on opposite walls are reflections rather than rotations of each
   other. Both are routine, not edge cases.
4. **Near-identical variants.** A duplex, a GFCI and a quad receptacle differ by
   a printed letter, not by shape. Geometry alone cannot separate them.
5. **Symbols that should not be counted.** Every sheet carries a legend showing
   one of each symbol, often a key plan inset, and a compass rose. All contain
   real symbols that must stay out of the total.
6. **Proving completeness.** A count that silently stops early looks exactly
   like a correct count. The number itself reveals nothing.
7. **Whitespace in the trace box.** Draw the box slightly too large and the
   empty space inside it becomes part of what gets matched.
8. **Review cost.** At any accuracy below perfect, the time spent checking the
   output decides whether the tool saves anything. The review workflow is not a
   supporting feature, it is most of the value.

## 2. Where the academic work stops

The panoptic symbol spotting line, running from FloorPlanCAD (ICCV 2021)
through GAT-CADNet (CVPR 2022) and SymPoint (ICLR 2024) to VecFormer (arXiv,
May 2025), established that models reading vector primitives directly
outperform raster CNNs on CAD drawings. Reported panoptic quality has climbed
steadily across that line, with VecFormer claiming 91.1 PQ as a new state of
the art.

Every one of them is closed-set and fully supervised, trained on 30 fixed
classes. The papers list their own open problems, and they line up almost
exactly with what this project needs:

- no exemplar-conditioned or one-shot spotting on vector primitives
- rotation and mirror invariance untested by any existing benchmark
- parametric symbols, such as doors whose width varies, not modelled at all
- sheet legends never used as free per-document exemplars

The one-shot work that exists is raster only. OSSR-PID and the related Tata
Consultancy Services patent (US12039641) sample contour paths into a DGCNN
trained with ArcFace loss, classifying from a single prototypical example per
symbol, evaluated on synthetic P&ID sheets and a small private real-world set.
The closest prior art on the geometry side is "Geometry-based symbol spotting
in born-digital architectural floor plans" (Journal of Electronic Imaging 30(4),
2021).

The gap is the interesting part. Exemplar-conditioned spotting on vector
primitives appears genuinely unpublished, and so does any treatment of
parametric symbols, which is what makes doors worth building properly rather
than special-casing.

## 3. Why this approach, and not the alternatives

Three families were candidates. Two were ruled out for concrete reasons.

Deep few-shot counters, the FSC-147 lineage running through FamNet, LOCA and
CounTR, output density maps rather than discrete objects. A density map cannot
say *which* receptacle it found or where, and these models hallucinate badly on
repetitive hatching. Wrong output type for the job: an estimator needs a list of
located instances, not a heat map.

Visual-prompt detectors such as T-Rex2, CountGD and DINO-X are the closest
things that exist, but they are trained on natural images, work on axis-aligned
boxes, are not rotation invariant, and are closed APIs. None of those properties
survive contact with a drawing sheet.

The third option fits the problem. A line drawing has deliberately removed
everything deep visual features are best at, namely texture and colour, and kept
exactly what classical geometry uses: strokes with orientations. Framed
properly this is not image recognition at all, it is industrial 2D pose search,
the problem Halcon's shape-based matching, the generalized Hough transform and
Fast Directional Chamfer Matching were built for. That family natively provides
continuous rotation, occlusion-tolerant partial scoring, and discrete detections
that arrive with a pose. Problems 1, 3 and 7 are handled structurally rather
than by tuning.

The organising pattern comes from DAVE and GeCo: propose with a geometric method
tuned for recall, then verify with something stricter that removes false
positives. That split is why proposing and deciding ended up as separate stages.

## 4. What building it changed

Three things turned out differently from the plan, and the differences are more
informative than the parts that went as expected.

**Vote clustering uses a grid, not mean-shift.** Mean-shift was the plan because
it avoids binning artefacts. Grid binning was far simpler, and refining each
cluster's pose as a weighted average of its own votes recovers the precision the
grid would otherwise cost, so the sophistication was not buying anything.

**Scoring became a log-odds ratio, not a fraction of geometry explained.** The
simpler score cannot express that the same match means less inside dense
hatching than in clear space. Weighing evidence against the local background is
what makes a threshold transfer between sheets, which the original plan
underestimated.

**Saliency weighting was not in the plan at all.** Once the index existed it
became obvious that wall hatching dominates it, with a single fingerprint bucket
holding 15.3% of every line pair on the sheet. Borrowing inverse document
frequency from text search fixed it.

**Door reconciliation checks other sheets, not a schedule.** The plan assumed a
door schedule table to reconcile against. Measurement showed this set has none:
the "DOOR SCHEDULE" on the title sheet is specification notes, the door type
tags exist only in the legend text, and doors on the plan carry no adjacent
marks. What the set does have is the same floor drawn once per trade, so the
built check matches doors hinge-to-hinge across sheets instead, recovering each
sheet's plan offset by letting the hinges vote on the translation.

Designed and still unbuilt: the learned verification layer, the coverage report,
negative exemplars, and hand-counted ground truth, without which there are no
honest precision and recall numbers. The geometric stages carried more of the
load than expected, and the honest reason the rest is missing is time.

Also unbuilt, and the largest single gap, is the fallback for scanned drawings.
The same voting machinery should run over edge points and gradient orientations
instead of vector primitives, in the manner of Fast Directional Chamfer
Matching, which would keep continuous rotation and partial scoring on scans so
accuracy degrades gradually rather than collapsing.

## 5. References

Every link was checked and resolves to the work named. Venues appear only where
confirmed; the rest are cited by arXiv identifier, which is a complete citation
on its own.

Symbol spotting on CAD drawings:
- FloorPlanCAD, ICCV 2021: arxiv.org/abs/2105.07147
- GAT-CADNet, CVPR 2022: arxiv.org/abs/2201.00625
- SymPoint ("Symbol as Points"), ICLR 2024: arxiv.org/abs/2401.10556
- VecFormer ("Point or Line?"), arXiv May 2025: arxiv.org/abs/2505.23395
- CADSpotting, arXiv: arxiv.org/abs/2412.07377

One-shot symbol recognition:
- OSSR-PID, arXiv: arxiv.org/abs/2109.03849
- Tata Consultancy Services, "Symbol recognition from raster images of PandIDs
  using a single instance per symbol class", US12039641 B2
- "Geometry-based symbol spotting in born-digital architectural floor plans",
  Journal of Electronic Imaging 30(4), 2021: doi.org/10.1117/1.JEI.30.4.043015

Exemplar-based counting and detection:
- DAVE, CVPR 2024: arxiv.org/abs/2404.16622
- GeCo, NeurIPS 2024: arxiv.org/abs/2409.18686
- GeCo2, AAAI 2026: arxiv.org/abs/2511.08048
- T-Rex2, ECCV 2024: arxiv.org/abs/2403.14610
- CountGD, NeurIPS 2024: robots.ox.ac.uk/~vgg/research/countgd/
- OS2D, arXiv: arxiv.org/abs/2003.06800

Classical 2D pose search:
- Fast Directional Chamfer Matching, CVPR 2010:
  merl.com/publications/docs/TR2010-045.pdf
- MVTec Halcon shape-based matching, the industrial reference implementation of
  this family; see MVTec's current solution guide.
