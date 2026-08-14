# Method and research basis

These are the design notes written before any code, on 2026-08-08, after reading
through three bodies of work: academic symbol spotting on CAD drawings,
exemplar-based detection and counting in computer vision, and the commercial
takeoff tools already selling this feature. They are kept here as the record of
why the system is shaped the way it is.

The goal was a trace-one-find-all engine that beats raster template matching on
the cases where template matching is known to fail, rather than one that merely
matches it. Section 5 says plainly which parts of this design got built and
which are still on paper.

## 1. The demo has a name, and its failure modes are documented

"User traces a symbol, software finds all the others" is Bluebeam Revu's
**Visual Search**, and it has shipped for years. PlanSwift Auto Count, Kreo Auto
Count and Countfire all do the same thing. Every one of them is raster template
matching behind a sensitivity slider.

Their failure modes are well documented across vendor docs, support forums and
one peer-reviewed study, and they became the design targets:

1. **Occlusion breaks the match.** A symbol crossed by a wall, a wire or a
   leader line fails at high sensitivity and floods false positives at low.
   Kreo ships an eraser tool so users can manually scrub clutter out of the
   reference crop, which tells you how routine the problem is.
2. **The threshold is dumped on the user.** No calibrated confidence anywhere.
3. **Rotation, mirroring and scale are barely handled.** Bluebeam offers
   opt-in 0/90/180/270 degrees and nothing between. Mirrored symbols are
   completely routine on real drawings, since doors and receptacles on opposite
   walls are reflections of each other.
4. **Near-identical variants get confused.** Duplex against GFCI against quad
   receptacle; door against window. Togal.AI's own peer-reviewed study found
   exactly this failure in their product.
5. **Legends, title blocks and key plans get double counted.** PlanSwift asks
   users to hand-draw up to nine exclusion regions per sheet.
6. **Incompleteness is silent.** There are Bluebeam forum reports of searches
   that simply stop halfway down a page. No tool proves it looked everywhere.
7. **Whitespace poisons the match.** Draw the trace box slightly too large and
   the empty space inside it becomes part of what the tool matches on.
8. **Review cost eats the savings.** Marketing says 98%, field reports say 60
   to 85%. When accuracy is in that range the review workflow is not a
   supporting feature, it is the product.

## 2. Where the academic work stops

The panoptic symbol spotting line of research, running from FloorPlanCAD at
ICCV 2021 through CADTransformer, GAT-CADNet and SymPoint to VecFormer at
NeurIPS 2025, established one thing clearly: transformers reading vector
primitives beat raster CNNs on CAD drawings. Panoptic quality across that line
climbed from 59 to 88.

Every single one of them is closed-set and fully supervised, trained on 30
fixed classes. The papers themselves list the open problems, and they line up
almost exactly with what this project needs:

- no exemplar-conditioned or one-shot spotting on vector primitives
- rotation and mirror invariance untested by any existing benchmark
- parametric symbols, such as doors whose width varies, not modelled at all
- sheet legends never used as free per-document exemplars

The one-shot work that does exist is raster only. OSSR-PID and the related TCS
patent (US12039641) sample paths into a DGCNN with ArcFace and reach about 86%
on synthetic P&IDs, but tolerate only about 20 degrees of rotation. The closest
prior art is Rezvanifar's geometry-based query-by-example work on born-digital
plans (JEI 2021), which is worth citing directly.

That gap is the interesting part. Exemplar-conditioned spotting on vector
primitives appears to be genuinely unpublished as of the VecFormer line.

## 3. Picking the machinery

Three families were candidates, and two were rejected for concrete reasons.

Deep few-shot counters, the FSC-147 lineage running through FamNet, LOCA and
CounTR, output density maps rather than discrete objects. A density map cannot
tell an estimator *which* receptacle it found or where, and these models
hallucinate badly on repetitive hatching. Rejected.

Visual-prompt detectors such as T-Rex2, CountGD and DINO-X are the closest
things on the market, but they are trained on natural images, work on
axis-aligned boxes, are not rotation invariant, and are closed APIs. They are
useful as baselines to beat, not as the core.

The third option is the one that fits. A line drawing has deliberately removed
everything deep visual features are best at, namely texture and colour, and
kept exactly what classical geometry uses: strokes with orientations. Framed
properly this is not an image recognition problem at all, it is industrial 2D
pose search, the same problem Halcon's shape-based matching, the generalized
Hough transform and Fast Directional Chamfer Matching were built for. That
family natively provides continuous rotation, occlusion-tolerant partial
scoring, discrete detections that come with a pose, and operation at full
resolution.

The organising pattern comes from DAVE and GeCo: propose with a geometric
method tuned for recall, then verify with something stricter that removes the
false positives.

## 4. The design

Everything operates on PDF vector primitives. The sample set is a born-digital
CAD export with roughly 28,000 paths per sheet and live text, which is the
normal case for drawings issued to a general contractor.

**Reading the sheet.** Parse primitives and text once with PyMuPDF. Segment the
sheet into plan area, title block, legend, key plan and schedules, using layout
heuristics anchored on printed captions like "LEGEND" and "KEY PLAN". This
removes failure mode 5 without asking the user to draw anything.

**Seeding exemplars from the legend.** A legend is a labelled dictionary of
every symbol the drawing uses, sitting right there on the sheet. Harvesting
those glyphs with their printed names gives self-labelled exemplars for free,
and they enter the system through exactly the same path as a user-traced box.

**Building the exemplar model.** Take the primitives lying wholly inside the
traced box. Clipped wall lines and empty space are excluded by construction,
which removes failure mode 7 and the reason Kreo needs an eraser.

**Fingerprinting.** For each pair of primitives within a k-nearest-neighbour
neighbourhood, which keeps this linear rather than quadratic, compute a
descriptor from the two primitive types, their length ratio, the angle between
them, their normalised midpoint distance and their junction relationship. Every
one of those quantities survives translation, rotation and scaling. Mirroring
is handled by also indexing the flipped descriptor. Quantise the descriptor
into a hash key and build one index over the sheet.

**Pose voting.** Each match between an exemplar key and the sheet index votes
for a complete similarity transform: translation, rotation, scale and mirror.
Where votes agree, there is a candidate instance, and it arrives with an
explicit pose rather than just a location. Scoring a candidate by how much of
the exemplar's geometry is actually accounted for gives a number with physical
meaning, so a receptacle crossed by a wall scores around 0.75 instead of
failing outright. That addresses failure modes 1, 2 and 3 together.

**Geometric verification.** Apply the recovered pose, assign model primitives
to scene primitives with type and distance gating, refine the pose by least
squares, and reject what falls short. On true CAD exports, where repeated
symbols are block copies of one another, this step is close to exact.

**Learned verification, optional.** Undo each candidate's recovered rotation
and scale, render exemplar and candidate to small canonical rasters, and
compare embeddings. Canonicalisation sidesteps the rotation weakness of deep
features entirely, because the geometry stage hands the verifier a pose that is
already solved. Clustering the accepted candidates in embedding space and
dropping clusters inconsistent with the exemplar would address failure mode 4.

**Text fusion.** Attach nearby text to each instance. Identical geometry
carrying different tags should split into different counts, which is the
automatic version of Kreo's manual "Split by Text". GFCI, switched and quad
receptacles differ by a printed subscript, not by shape, so this has to be
resolved in the text layer.

**Confidence, coverage and reconciliation.** Per-instance confidence drives a
review queue sorted worst-first, so verifying 500 detections takes minutes
rather than hours, which is the answer to failure mode 8. Because every
primitive on the sheet is indexed, the system can in principle report what it
never explained, which is the answer to failure mode 6. And counts can be
checked against the drawing set's own schedules, since a door schedule and a
panel schedule state the same quantities the plans do. No commercial tool does
that, and it is exactly what a human estimator does before trusting a number.

**Parametric symbols.** Doors are the awkward case, because a door's width
varies and rigid matching therefore misses them by design. The answer is to
vote on sub-part structure instead: a quarter arc, a leaf line, and a hinge
where they meet, with width left as a free parameter recovered per instance.
The width then falls out as a useful output rather than a complication, since
it feeds pricing directly. This is the third academic gap listed above, and no
published method addresses it.

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
- the raster fallback of section 7.
- hand-counted ground truth, and therefore any real precision and recall
  numbers.

Deliberately changed during implementation:

- vote clustering uses a grid rather than the mean-shift proposed here. Grid
  binning was simpler, and refining each cluster's pose as a weighted average of
  its own votes recovers the precision that the grid would otherwise cost.
- candidate scoring became a log-odds ratio against the local background rather
  than a plain fraction of geometry explained. Same idea, but it makes a match
  in dense linework count for less than the same match in clear space, which the
  simpler score could not express.
- subclass tags are attached to detections but do not yet split the counts.

## 6. Why this beats template matching

| | Raster template match | This method |
|---|---|---|
| Occlusion by linework | breaks, or forces slider gymnastics | partial-score voting, degrades gradually |
| Rotation and mirror | coarse 90 degree steps, opt-in | continuous and native, mirror included |
| Threshold | user-tuned sensitivity | calibrated, derived from measured noise |
| Whitespace in the trace box | poisons the match | excluded by construction |
| Variant confusion | counts GFCI as duplex | text fusion, and embedding verify if built |
| Legend and key-plan double count | user draws exclusions by hand | region semantics, automatic |
| Coverage | silent gaps | indexed, and reportable in principle |
| Output | a count | count, pose, width, label, confidence, and a schedule cross-check |
| Training data | none | none |

## 7. Raster fallback for scanned drawings

The same voting machinery runs over edge points and gradient orientations
instead of vector primitives, in the manner of Fast Directional Chamfer
Matching or a generalized Hough transform. That keeps continuous rotation and
partial scoring on scans, so accuracy degrades rather than collapsing. Every
commercial tool collapses on scans, which makes this an open wedge rather than
a catch-up feature.

## 8. References

- FloorPlanCAD (ICCV 2021): arxiv.org/abs/2105.07147 · VecFormer (NeurIPS
  2025, state of the art at PQ 88.4 with no prior): arxiv.org/abs/2505.23395
- GAT-CADNet (CVPR 2022): arxiv.org/abs/2201.00625 · SymPoint (ICLR 2024):
  arxiv.org/abs/2401.10556 · CADSpotting: arxiv.org/abs/2412.07377
- OSSR-PID one-shot P&ID: arxiv.org/abs/2109.03849 · TCS patent US12039641
- Rezvanifar, geometry-based query-by-example on born-digital plans (JEI 2021):
  doi.org/10.1117/1.JEI.30.4.043015
- DAVE (CVPR 2024): arxiv.org/abs/2404.16622 · GeCo (NeurIPS 2024):
  arxiv.org/abs/2409.18686 · GeCo2 (AAAI 2026): arxiv.org/abs/2511.08048
- T-Rex2 (ECCV 2024): arxiv.org/abs/2403.14610 · CountGD (NeurIPS 2024):
  robots.ox.ac.uk/~vgg/research/countgd/ · OS2D: arxiv.org/abs/2003.06800
- Halcon shape-based matching guide:
  download.mvtec.com/halcon-9.0-solution-guide-ii-b-shape-based-matching.pdf
- Fast Directional Chamfer Matching (CVPR 2010):
  merl.com/publications/docs/TR2010-045.pdf
- Bluebeam Visual Search docs and failure reports:
  support.bluebeam.com/revu/features/visual-search-overview.html
- Togal.AI peer-reviewed accuracy study:
  fmicorp.com/uploads/media/TogalAI_Case_Study_Whitepaper_FINAL.pdf
