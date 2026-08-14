# One-shot symbol detection on construction drawings

Trace **one** symbol on a drawing; find every instance of it — rotated, mirrored,
resized, or partially covered by other linework. Trace a door and it finds doors
of *every* width, and measures each one. Or skip tracing entirely: the system
reads the drawing set's own legend and counts everything defined there.

**No training data. No labelled examples. No model to retrain for a new client.**

---

## Run it

```bash
python3 -m venv .venv && .venv/bin/pip install pymupdf numpy scipy flask
.venv/bin/pip install pyobjc-framework-Vision pyobjc-framework-Quartz  # optional, macOS OCR
.venv/bin/python app.py            # -> http://localhost:8642
```

Drop any CAD-exported (vector) PDF into the file picker, or use the included
28-sheet Skanska tenant fit-out set. Then:

1. Pick a sheet, zoom in, **drag a box around one symbol**
2. Tick **"use new pipeline (v2)"** to run the architecture this project proposes
3. Drag the **confidence slider** to reveal or hide uncertain hits
4. Or press **"Auto-count everything via legend"** and trace nothing at all

Command-line equivalents:

```bash
.venv/bin/python scripts/compare.py     # v1 vs v2 on identical queries
.venv/bin/python scripts/demo.py 25     # CLI detection + annotated PNG
.venv/bin/python scripts/reconcile.py   # plan counts vs the panel schedules
```

---

## Reading the code

Architecture diagram: **[docs/pipeline-v2.html](docs/pipeline-v2.html)** — start here if
you want the picture before the code.

The repo contains **two engines** so improvements can be measured rather than
asserted:

| | | |
|---|---|---|
| `oneshot/` | **v1** | The original engine. Frozen, kept as the experimental control. |
| `pipeline/` | **v2** | The staged architecture this project proposes. **Read this one.** |

Suggested order, about 700 lines total:

| File | What it is |
|---|---|
| `pipeline/__init__.py` | The map: stages, and the reasoning behind the shape |
| `pipeline/types.py` | The five objects that flow between stages |
| `pipeline/policy.py` | Every threshold in the system, each with the measurement that set it |
| `pipeline/sheet.py` | **Stage 0** — parse each sheet once, one coordinate space |
| `pipeline/index.py` | **Stage 1** — learn which shapes are rare *on this sheet* |
| `pipeline/proposers/` | **Stage 2** — three ways to generate guesses |
| `pipeline/evidence.py` | **Stage 3** — score every guess on one comparable scale |
| `pipeline/resolve.py` | **Stage 4** — arbitrate, name, scope |
| `pipeline/api.py` | The facade `app.py` calls |

---

## The idea

A count can be wrong in three unrelated ways, and each needs different
information to fix:

| | The mistake | What fixes it |
|---|---|---|
| **Geometric** | No symbol is there at all | Better geometry — rarity-weighted votes, evidence vs. clutter |
| **Semantic** | Right shape, wrong type (GFCI vs. duplex differ by a printed "G") | Reading the text tag on the symbol |
| **Scope** | Real symbol that shouldn't count — the legend's own example, the key-plan inset, the compass rose | Knowing which region of the sheet you're on |

v1 treated these as one problem and grew a filter on the end each time a new
failure appeared — ending up with three different rules for "two hits, one spot"
that disagreed with each other. v2 gives each mistake an owning stage.

Two principles shape everything else:

**Guessing and deciding are separate stages**, because they want opposite things.
A guess never made can never be recovered, so Stage 2 is deliberately generous.
A false accept costs money, so Stage 5 is strict. One threshold cannot serve both.

**Evidence is weighed against the local background**, never a global constant. The
same match means less inside dense hatching than in white space — which is what
makes a threshold that transfers between sheets possible at all.

---

## What it does that trained detectors don't

- **Reads the drawing's own dictionary.** Every set ships a legend defining its
  symbols. The system harvests those glyphs *with their printed names* and counts
  them — so a new firm's symbology works on day one, with no labelling.
- **Handles parametric symbols.** Doors aren't copies of each other: a 3'-0" and a
  2'-6" door are different geometry. Tracing one door switches to a structural
  detector (arc + leaf hinged at the arc's centre) that finds every width and
  *reports* each width, since the arc radius is the door width.
- **Checks its own answer.** Drawing sets state quantities twice — as symbols and
  as schedules. `scripts/reconcile.py` counts receptacles on the plan, OCRs the
  panel-schedule tables, and reports agreement room by room. Estimators do this by
  hand; no takeoff tool we surveyed automates it.
- **Explains its confidence.** Each detection carries a log-odds evidence score, so
  the confidence slider is a lens over already-scored results rather than a
  sensitivity knob that changes what the algorithm finds.

---

## Measured on the included set

| | |
|---|---|
| Stage 1 occurrence filter | dropped **1 fingerprint bucket = 15.3%** of all line pairs — the wall hatching |
| E4 receptacles | v1: 188 take-it-or-leave-it · v2: **163 confident + 23 flagged** |
| Back-to-back door fix | **26 → 35** doors per sheet (see below) |
| Cross-sheet check | T5 = T8 = **33** standard doors; E4 agrees |
| Reconciliation | **22 rooms corroborated**, 12 flagged, 0 contradicted |
| Query latency | ~0.4–2 s per trace on a 33,000-primitive sheet |

The door fix is worth the anecdote: nine doors per sheet were being silently
merged by a duplicate-removal rule that treated two hinges 12pt apart as one
door. Back-to-back office doors hinge at a shared jamb, so *every* pair on the
corridor was half-counted. The total ("24 doors") looked perfectly reasonable —
the bug was invisible until a human pointed at two specific doors. That is the
argument for the review queue, and for ground-truth evaluation, in one bug.

---

## Honest limits

- **Vector PDFs only.** Scanned drawings have no geometry to parse; the app detects
  this and warns. A raster fallback is designed (`METHOD.md`) but unbuilt — this is
  the one place a trained detector is strictly better, and the two approaches
  compose: a YOLO-style proposer would drop into Stage 2 as a fourth peer and
  inherit the same evidence, arbitration, and reconciliation.
- **No ground-truth accuracy figures.** Thresholds are set from measured score
  *distributions*, not from hand-labelled counts, so precision/recall per class is
  not yet claimable. An eval harness is the highest-value next step.
- **Two of four assignment classes.** Doors and receptacles work; detail markers
  and elevation markers have a designed home (`TextAnchoredProposer`) that is
  stubbed, not built.
- **Near-identical variants still cross-talk** in legend auto-count (line-pattern
  legend entries match wall linework). Subclass splitting is the next Stage 4 task.
- English legend headings and imperial scale notes; column-style legends.

---

## Layout

```
app.py              web server + v1 detection path
web/index.html      the single-page UI
pipeline/           v2: the staged architecture   <- read this
oneshot/            v1: original engine, frozen as the control
scripts/            compare, demo, reconcile, exemplar discovery
docs/               architecture diagram
METHOD.md           research basis and the design's academic lineage
```
