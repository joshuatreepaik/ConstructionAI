# One-Shot Symbol Detection on Construction Drawings — Method Design

Synthesis of three research threads (academic symbol spotting, exemplar-based
detection/counting, commercial takeoff tools), 2026-08-08. Target: a
trace-one-find-all engine that is demonstrably better than raster template
matching (Bluebeam Visual Search and similar demos).

## 1. What the research established

### The CEO's demo has a name and known failure modes
"User traces a symbol → find all instances" is Bluebeam Revu **Visual Search**
(shipped for years), PlanSwift Auto Count, Kreo Auto Count, Countfire. All are
raster template matching with a user-facing sensitivity slider. Documented
failure modes across every commercial tool (design targets for us):

1. **Occlusion breaks matching** — symbol crossed by a wall/wire/leader line
   fails at high sensitivity, floods false positives at low. Kreo literally
   ships an "eraser" so users can scrub clutter out of the reference crop.
2. **Threshold dumped on the user** — no calibrated confidence anywhere.
3. **Rotation/mirror/scale intolerance** — Bluebeam: opt-in 0/90/180/270 only.
   Mirrored symbols (doors/receptacles on opposite walls) are routine.
4. **Near-identical variant confusion** — duplex vs GFCI vs quad receptacle;
   door vs window (Togal's peer-reviewed study found exactly this).
5. **Legend/title-block/key-plan double counting** — PlanSwift makes users
   hand-draw up to 9 exclusion regions.
6. **Silent incompleteness** — Bluebeam forum: searches that silently stop
   halfway down the page. No tool proves coverage.
7. **Whitespace poisoning** — box too big → empty space participates in match.
8. **Review cost eats savings** — "98%" marketing vs 60–85% field accuracy;
   the review workflow IS the product.

### The academic state of the art has an open gap that this method fills
The "panoptic symbol spotting" line (FloorPlanCAD ICCV'21 → CADTransformer →
GAT-CADNet → SymPoint → VecFormer NeurIPS'25, PQ 59→88) proves vector-primitive
transformers beat raster CNNs on CAD drawings — but **every one is closed-set
and fully supervised** (30 fixed classes). Explicitly listed open gaps that
match our design: (a) no exemplar-conditioned/one-shot spotting on vector
primitives, (b) rotation/mirror invariance untested by any benchmark,
(c) parametric symbols (variable-width doors) unmodeled, (d) sheet legends as
free per-document exemplars — untapped.

The one-shot literature that does exist is raster-only: OSSR-PID / TCS patent
US12039641 (path sampling → DGCNN + ArcFace, ~86% on synthetic P&IDs, only
±20° rotation); Rezvanifar JEI 2021 (geometry-based QBE on born-digital plans —
closest prior art, worth citing in the writeup).

### The right matching machinery (from the exemplar-matching survey)
- Deep few-shot counters (FSC-147 lineage: FamNet…LOCA, CounTR) output density
  maps — no discrete instances, hallucinate on hatching. Reject.
- Modern visual-prompt detectors (T-Rex2, CountGD, DINO-X) are the closest
  products but natural-image-trained, axis-aligned, not rotation invariant,
  API/closed. Use as baselines to beat, not as the core.
- Binary line drawings remove what deep features are best at (texture, color)
  and keep exactly what classical geometric methods use (strokes with
  orientations). The problem is structurally **industrial 2D pose search**
  (Halcon shape-based matching, generalized Hough, Fast Directional Chamfer),
  which natively provides: continuous rotation, occlusion-tolerant partial
  scoring, discrete detections with pose, full-resolution operation.
- The right meta-pattern is DAVE/GeCo's **detect-then-verify**: geometric
  proposer (high recall) + learned verifier (kills false positives).

## 2. The method: pose-voting geometric fingerprinting + canonicalized learned verification

Operates on PDF vector primitives (born-digital CAD exports — confirmed for
the Skanska set: 28k paths/sheet, live text). Raster fallback in §4.

### Stage 0 — Sheet understanding (once per sheet)
- Parse primitives (lines, arcs, béziers) and text via PyMuPDF.
- Segment sheet regions: plan area vs title block vs legend vs key plan vs
  schedules (layout heuristics + text anchors like "LEGEND", "KEY PLAN").
  Kills failure mode #5 automatically.
- Auto-seed exemplars from legend regions with their text labels
  (self-labeling: symbol crop + name for free). User tracing = same input path.

### Stage 1 — Exemplar model
Primitives wholly inside the traced box → model set M (whitespace and clipped
wall lines excluded by construction — kills #7 and the Kreo-eraser problem).

### Stage 2 — Invariant fingerprinting
For each primitive pair within a k-NN neighborhood (O(N·k), not O(N²)):
descriptor d = (type_i, type_j, length ratio, inter-orientation angle,
normalized midpoint distance, junction relation). d is invariant to
translation/rotation/scale; mirror handled by also indexing flipped
descriptors. Quantized d → hash key. Build one hash index over the sheet.

### Stage 3 — Pose voting (the proposer)
Each exemplar-key hit in the index votes for a full similarity transform
(tx, ty, θ, scale, mirror) of the model. Cluster votes in transform space
(mean-shift; cleaner than gridded Hough). Peaks = candidate instances **with
explicit pose**. Score = fraction of model primitives explained under the
recovered pose — Halcon-style partial score, so a receptacle crossed by a wall
still scores ~0.7–0.8 instead of failing (kills #1, #3). The score has
physical meaning ("78% of the symbol's geometry found"), replacing the
sensitivity slider (kills #2).

### Stage 4 — Geometric verification
Apply pose, assign model↔scene primitives (greedy with type/distance gating),
least-squares pose refinement, reject below threshold. On true CAD exports
where symbols are block copies, this stage is near-exact.

### Stage 5 — Canonicalized learned verification (the ML layer)
Undo each candidate's recovered rotation/scale → render exemplar and candidate
to small canonical rasters → embedding similarity (DINOv2 patches, or a small
contrastive encoder trained on synthetic line-art augmentations: random
occluding strokes, style jitter, dash-pattern changes). Canonicalization
sidesteps deep features' rotation weakness entirely — the geometry stage hands
the verifier a solved pose. Then DAVE-style outlier clustering across all
accepted candidates: embed, cluster, drop clusters inconsistent with the
exemplar (kills #4). Negative exemplars (user right-clicks a false hit)
subtract in embedding space — T-Rex-Omni's negative-prompt idea.

### Stage 6 — Text fusion & semantics
Attach nearby text tokens to each instance; identical geometry with different
tags → split counts (automatic version of Kreo "Split by Text"). GFCI vs
switched vs quad subscripts resolved by the text layer, not pixels.

### Stage 7 — Calibrated confidence, coverage, reconciliation
- Per-instance confidence from geometric score × embedding score; low-confidence
  queue sorted for review (fixes #8: verify 500 detections in minutes).
- Coverage proof: every primitive was indexed — report per-sheet totals and
  unexplained-cluster anomalies (kills #6, Bluebeam's silent half-page stop).
- Reconcile counts against the document's own schedules (door schedule on T1,
  receptacle circuits on panel schedule E6) → discrepancy report. No commercial
  tool does this; it's what human estimators do.

### Parametric symbols (doors)
Doors are parametric (width varies), so rigid matching misses them by design.
Extension: vote on sub-part structure (quarter-arc + leaf line + hinge point)
with width as a free parameter recovered per instance — outputs door width as
a bonus (feeds pricing). This is academic gap (c); no published method does it.

## 3. Why this beats the demo (one table for the writeup)

| Axis | Raster template match (demo) | This method |
|---|---|---|
| Occlusion by linework | breaks / slider gymnastics | partial-score voting, degrades gracefully |
| Rotation / mirror | coarse 90° steps, opt-in | continuous, native, incl. mirror |
| Threshold | user-tuned sensitivity | calibrated "fraction of geometry explained" |
| Whitespace in trace box | poisons match | ignored by construction |
| Variant confusion | counts GFCI as duplex | embedding verify + text fusion |
| Legend/key-plan double count | user draws exclusions | auto region semantics |
| Coverage | silent gaps | provable, per-sheet report |
| Output | count | count + pose + width + label + confidence + schedule reconciliation |
| Training data | none | none required (verifier optional, self-supervised) |

## 4. Raster fallback (scanned drawings)
Same voting machinery over edge points + gradient orientations (Fast
Directional Chamfer / generalized Hough): keeps continuous rotation and
partial scoring on scans, degrading gracefully instead of collapsing
(every commercial tool collapses on scans — open wedge).

## 5. Take-home scoping (build order)
1. Stages 0–4 + 7 are classical, zero-training, implementable in days:
   PyMuPDF + NumPy + scikit-learn. Demo on Skanska set with annotated
   overlay PDF + per-class counts + schedule reconciliation.
2. Stage 5 verifier if time permits (DINOv2 off-the-shelf on canonicalized
   crops — no training — is enough to show the ML layer).
3. Benchmark: hand-counted ground truth on T5 (doors, markers) and E4
   (receptacles); report precision/recall per class; qualitative comparison
   vs Bluebeam-style behavior (rotated/mirrored/occluded cases).
4. Writeup framing: this sits in a documented research gap (exemplar-
   conditioned spotting on vector primitives — none published as of the
   VecFormer/NeurIPS 2025 line); cite FloorPlanCAD, VecFormer, OSSR-PID,
   Rezvanifar 2021, DAVE/GeCo, T-Rex2 as the landscape.

## 6. Key references
- FloorPlanCAD (ICCV 2021): arxiv.org/abs/2105.07147 · VecFormer (NeurIPS
  2025, SOTA PQ 88.4 no-prior): arxiv.org/abs/2505.23395
- GAT-CADNet (CVPR 2022): arxiv.org/abs/2201.00625 · SymPoint (ICLR 2024):
  arxiv.org/abs/2401.10556 · CADSpotting: arxiv.org/abs/2412.07377
- OSSR-PID one-shot P&ID: arxiv.org/abs/2109.03849 · TCS patent US12039641
- Rezvanifar, geometry-based QBE on born-digital plans (JEI 2021):
  doi.org/10.1117/1.JEI.30.4.043015
- DAVE (CVPR 2024): arxiv.org/abs/2404.16622 · GeCo (NeurIPS 2024):
  arxiv.org/abs/2409.18686 · GeCo2 (AAAI 2026): arxiv.org/abs/2511.08048
- T-Rex2 (ECCV 2024): arxiv.org/abs/2403.14610 · CountGD (NeurIPS 2024):
  robots.ox.ac.uk/~vgg/research/countgd/ · OS2D: arxiv.org/abs/2003.06800
- Halcon shape-based matching guide:
  download.mvtec.com/halcon-9.0-solution-guide-ii-b-shape-based-matching.pdf
- Fast Directional Chamfer Matching (CVPR 2010):
  merl.com/publications/docs/TR2010-045.pdf
- Bluebeam Visual Search docs + failure reports:
  support.bluebeam.com/revu/features/visual-search-overview.html
- Togal.AI peer-reviewed accuracy study:
  fmicorp.com/uploads/media/TogalAI_Case_Study_Whitepaper_FINAL.pdf
