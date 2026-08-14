# Method and research basis

These are the design notes written before any code, on 2026-08-08. They are kept
here as the record of why the system is shaped the way it is, and what I read
before deciding.

The goal was an engine where you trace one symbol and it finds the rest, built
so that the hard cases are handled by the architecture rather than patched
afterwards. Section 5 says plainly which parts of this design got built and
which are still on paper.

## 1. What makes this hard

Counting symbols on a drawing sounds like template matching until you look at
actual sheets. Eight things break a naive implementation, and the design is
organised around them.

1. **Occlusion.** Symbols get crossed by walls, wires and leader lines all the
   time. Any method that needs to see the whole symbol will either miss those
   or, if loosened enough to catch them, start matching everything.
2. **Confidence has to mean something.** A raw similarity percentage is not a
   probability, so a threshold picked on one sheet is arbitrary on the next.
3. **Rotation, mirroring and scale.** Symbols appear at any angle, and doors and
   receptacles on opposite walls are reflections rather than rotations of each
   other. Both are routine, not edge cases.
4. **Near-identical variants.** A duplex, a GFCI and a quad receptacle differ by
   a printed letter or a subscript, not by shape. Geometry alone cannot separate
   them, so something has to read the text.
5. **Symbols that should not be counted.** Every sheet carries a legend showing
   one of each symbol, often a key plan inset, and a compass rose. All of them
   contain real symbols that must not enter the total.
6. **Proving completeness.** A count that silently stops early looks exactly
   like a correct count. Nothing about the number itself reveals the problem.
7. **Whitespace in the trace box.** Draw the box slightly too large and the
   empty space inside it becomes part of what gets matched.
8. **Review cost.** At field accuracies anywhere below perfect, the time spent
   checking the output determines whether the tool saves anything at all. The
   review workflow is not a supporting feature, it is most of the value.

## 2. Where the academic work stops

The panoptic symbol spotting line of research, running from FloorPlanCAD
(ICCV 2021) through GAT-CADNet (CVPR 2022) and SymPoint (ICLR 2024) to
VecFormer (arXiv, May 2025), established one thing clearly: models reading
vector primitives directly outperform raster CNNs on CAD drawings. Reported
panoptic quality has climbed steadily across that line, with VecFormer claiming
91.1 PQ as a new state of the art.

Every one of them is closed-set and fully supervised, trained on 30 fixed
classes. The papers themselves list the open problems, and they line up almost
exactly with what this project needs:

- no exemplar-conditioned or one-shot spotting on vector primitives
- rotation and mirror invariance untested by any existing benchmark
- parametric symbols, such as doors whose width varies, not modelled at all
- sheet legends never used as free per-document exemplars

The one-shot work that does exist is raster only. OSSR-PID and the related Tata
Consultancy Services patent (US12039641) sample contour paths into a DGCNN
trained with ArcFace loss, classifying from a single prototypical example per
symbol, evaluated on synthetic P&ID sheets and a small private real-world set.
The closest prior art on the geometry side is "Geometry-based symbol spotting
in born-digital architectural floor plans" (Journal of Electronic Imaging 30(4),
2021), which is query-by-example on vector floor plans.

That gap is the interesting part. Exemplar-conditioned spotting on vector
primitives appears to be genuinely unpublished as of the VecFormer line.

## 3. Picking the machinery

Three families were candidates, and two were ruled out for concrete reasons.

Deep few-shot counters, the FSC-147 lineage running through FamNet, LOCA and
CounTR, output density maps rather than discrete objects. A density map cannot
tell an estimator *which* receptacle it found or where, and these models
hallucinate badly on repetitive hatching. Wrong output type for the job.

Visual-prompt detectors such as T-Rex2, CountGD and DINO-X are the closest
things that exist, but they are trained on natural images, work on axis-aligned
boxes, are not rotation invariant, and are closed APIs. None of those properties
survive contact with a drawing sheet.

The third option is the one that fits. A line drawing has deliberately removed
everything deep visual features are best at, namely texture and colour, and kept
exactly what classical geometry uses: strokes with orientations. Framed
properly this is not an image recognition problem at all, it is industrial 2D
pose search, the same problem Halcon's shape-based matching, the generalized
Hough transform and Fast Directional Chamfer Matching were built for. That
family natively provides continuous rotation, occlusion-tolerant partial
scoring, discrete detections that come with a pose, and operation at full
resolution. That covers problems 1, 3 and 7 from section 1 structurally rather
than by tuning.

The organising pattern comes from DAVE and GeCo: propose with a geometric method
tuned for recall, then verify with something stricter that removes the false
positives.

## 4. The design

Everything operates on PDF vector primitives. The sample set is a born-digital
CAD export with roughly 28,000 paths per sheet and live text, which is the
normal case for drawings issued to a general contractor.

**Reading the sheet.** Parse primitives and text once with PyMuPDF. Segment the
sheet into plan area, title block, legend, key plan and schedules, using layout
heuristics anchored on printed captions like "LEGEND" and "KEY PLAN". That
handles problem 5 without asking the user to draw exclusion regions by hand.

**Seeding exemplars from the legend.** A legend is a labelled dictionary of
every symbol the drawing uses, sitting right there on the sheet. Harvesting
those glyphs with their printed names gives self-labelled exemplars for free,
and they enter the system through exactly the same path as a user-traced box.

**Building the exemplar model.** Take the primitives lying wholly inside the
traced box. Clipped wall lines and empty space are excluded by construction,
which is problem 7 dealt with by definition rather than by careful tracing.

**Fingerprinting.** For each pair of primitives within a k-nearest-neighbour
neighbourhood, which keeps this linear rather than quadratic, compute a
descriptor from the two primitive types, their length ratio, the angle between
them, their normalised midpoint distance and their junction relationship. Every
one of those quantities survives translation, rotation and scaling. Mirroring is
handled by also indexing the flipped descriptor. Quantise the descriptor into a
hash key and build one index over the sheet.

**Pose voting.** Each match between an exemplar key and the sheet index votes
for a complete similarity transform: translation, rotation, scale and mirror.
Where votes agree, there is a candidate instance, and it arrives with an explicit
pose rather than just a location. Scoring a candidate by how much of the
exemplar's geometry is actually accounted for gives a number with physical
meaning, so a receptacle crossed by a wall scores around 0.75 instead of failing
outright. Problems 1, 2 and 3 are all addressed by this one mechanism.

**Geometric verification.** Apply the recovered pose, assign model primitives to
scene primitives with type and distance gating, refine the pose by least squares,
and reject what falls short. On true CAD exports, where repeated symbols are
block copies of one another, this step is close to exact.

**Learned verification, optional.** Undo each candidate's recovered rotation and
scale, render exemplar and candidate to small canonical rasters, and compare
embeddings. Canonicalisation sidesteps the rotation weakness of deep features
entirely, because the geometry stage hands the verifier a pose that is already
solved. Clustering accepted candidates in embedding space and dropping clusters
inconsistent with the exemplar would give a second line of defence on problem 4.

**Text fusion.** Attach nearby text to each instance. Identical geometry carrying
different tags should split into different counts, since GFCI, switched and quad
receptacles differ by a printed subscript rather than by shape. This is the
primary answer to problem 4.

**Confidence, coverage and reconciliation.** Per-instance confidence drives a
review queue sorted worst-first, so verifying 500 detections takes minutes rather
than hours, which is problem 8. Because every primitive on the sheet is indexed,
the system can in principle report what it never explained, which is problem 6.
And counts can be checked against the drawing set's own schedules, since a door
schedule and a panel schedule state the same quantities the plans do. That is
what a human estimator does before trusting a number.

**Parametric symbols.** Doors are the awkward case, because a door's width varies
and rigid matching therefore misses them by design. The answer is to vote on
sub-part structure instead: a quarter arc, a leaf line, and a hinge where they
meet, with width left as a free parameter recovered per instance. The width then
falls out as a useful output rather than a complication, since it feeds pricing
directly. This is the third academic gap listed in section 2.

## 5. What got built, and what did not

Built and working:

- sheet parsing, region segmentation, and scale recovery from the sheet's own
  scale note
- legend harvesting with printed labels, driving the auto-count button
- invariant fingerprinting and the sheet-wide hash index, with rarity weighting
  added later once hatching proved to dominate the index
- pose voting, geometric verification, and calibrated confidence with an
  explicit review queue
- text-based naming and vetoing of candidates
- parametric door detection with per-instance width recovery
- schedule reconciliation against the OCR'd panel schedules

Designed but not built:

- the learned verification layer. The geometric stages turned out to carry more
  of the load than expected, and the honest reason this is missing is time.
- the coverage report. Every primitive is indexed, so the data is there, but
  nothing surfaces what went unexplained.
- negative exemplars, where a user right-clicks a false hit to push it away.
- the raster fallback of section 6.
- hand-counted ground truth, and therefore any real precision and recall numbers.

Deliberately changed during implementation:

- vote clustering uses a grid rather than the mean-shift proposed here. Grid
  binning was simpler, and refining each cluster's pose as a weighted average of
  its own votes recovers the precision the grid would otherwise cost.
- candidate scoring became a log-odds ratio against the local background rather
  than a plain fraction of geometry explained. Same idea, but it makes a match in
  dense linework count for less than the same match in clear space, which the
  simpler score could not express.
- subclass tags are attached to detections but do not yet split the counts.

## 6. Raster fallback for scanned drawings

The same voting machinery can run over edge points and gradient orientations
instead of vector primitives, in the manner of Fast Directional Chamfer Matching
or a generalized Hough transform. That keeps continuous rotation and partial
scoring on scans, so accuracy degrades gradually rather than collapsing. Not
built.

## 7. References

Every link below was checked and resolves to the work named. Venues are given
only where confirmed; the rest are cited by arXiv identifier, which is a
complete citation on its own.

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
- MVTec Halcon shape-based matching, referred to as the industrial reference
  implementation of this family; see MVTec's current solution guide.
