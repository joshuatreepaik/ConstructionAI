# One-shot symbol detection on construction drawings

Trace **one** symbol on a drawing; find every instance of it, including ones that are
rotated, mirrored, resized, or partially covered by other linework. Trace a door and it
finds doors of *every* width and measures each one. Or trace nothing at all: the system
reads the drawing set's own legend and counts everything defined there.

**No training data. No labelled examples. No model to retrain for a new client.**

---

## The pipeline

![Six stage pipeline: parse the sheet once, learn which shapes are rare on it, propose candidates generously, score them against local clutter, arbitrate and name them, then decide with a review queue.](docs/pipeline.svg)

Data flows top to bottom, and each arrow is labelled with the actual type in
`pipeline/types.py`. Thresholds flow sideways out of one policy file, which an offline
calibration loop sets from measured score distributions, so decisions are derived rather
than hand tuned.

Note the opposite error preferences in the margins. Stage 2 is allowed to over guess,
because a candidate never proposed is lost forever. Stage 5 is strict, because a false
accept costs money. One threshold cannot serve both, which is why proposing and deciding
are separate stages.

---

## Run it

```bash
python3 -m venv .venv && .venv/bin/pip install pymupdf numpy scipy flask
.venv/bin/pip install pyobjc-framework-Vision pyobjc-framework-Quartz  # optional, macOS OCR
.venv/bin/python app.py            # http://localhost:8642
```

Drop any CAD exported (vector) PDF into the file picker, or use the included 28 sheet
tenant fit out set. Then:

1. Pick a sheet, zoom in, **drag a box around one symbol**
2. Drag the **confidence slider** to reveal or hide uncertain hits
3. Or press **Auto-count everything via legend** and trace nothing at all

Command line equivalents:

```bash
.venv/bin/python scripts/demo.py 25       # detection on one sheet + annotated PNG
.venv/bin/python scripts/reconcile.py     # plan counts checked against panel schedules
```

---

## The idea

A count can be wrong in three unrelated ways, and each needs different information to fix.
Treating them as one problem is what produces a system that grows a new filter every time
it fails. Here each mistake has an owning stage.

| | The mistake | What fixes it |
|---|---|---|
| **Geometric** | No symbol is there at all | Better geometry: rarity weighted votes, evidence against clutter |
| **Semantic** | Right shape, wrong type (a GFCI and a duplex receptacle differ by one printed letter) | Reading the text tag on the symbol |
| **Scope** | Real symbol that should not count: the legend's own example, the key plan inset, the compass rose | Knowing which region of the sheet you are on |

The second principle behind the shape: evidence is weighed against the **local**
background, never a global constant. The same match means less inside dense hatching than
in white space. That is what makes a threshold which transfers between sheets possible at
all.

---

## Cross domain inspiration

The useful ideas here came from outside CAD and outside drawing recognition. Four
borrowings do most of the work.

**Search engines, for weighting geometry.** A search engine ranks a rare word above "the"
because rare words identify a document. Stage 1 does the same with shape: every line pair
gets a rarity weight measured on that sheet, so wall hatching counts for almost nothing
while a distinctive arc counts for a lot. On the sample set this silenced one fingerprint
bucket holding **15.3% of all line pairs**. Borrowed from information retrieval, applied
to geometry (`pipeline/index.py`).

**Radar, for confidence.** Radar does not ask whether a blip looks like an aircraft. It
asks how much more likely the blip is under "aircraft" than under "noise", calibrated
against a measured noise floor. Stage 3 scores each candidate as a log odds ratio against
the local background rather than as a similarity percentage, and the thresholds come from
running the detector over blank regions and deliberately impossible templates. That is a
false alarm rate calibration, which is why the confidence numbers mean something rather
than being a tuning knob (`pipeline/evidence.py`).

**Protein structure matching, for finding rotated copies.** Geometric hashing was built to
match molecules regardless of orientation: describe a structure by relationships between
pairs of its parts, since those relationships survive rotation and scaling, then let every
matching pair vote for a pose. The same trick is why rotation, mirroring and resizing are
free here instead of being special cases, and why a symbol half covered by a wall line is
still found from its surviving fragments (`pipeline/proposers/rigid.py`).

**Radiology, for the output format.** Computer aided detection in mammography never
returns a verdict on its own. It triages into confident findings and flags for the
radiologist, because a miss and a false alarm carry very different costs. Stage 5 does the
same: accept, review, and reject with a stated reason. The deliverable is a work list, not
a single number to trust or distrust wholesale.

---

## What it does that trained detectors do not

- **Reads the drawing's own dictionary.** Every set ships a legend defining its symbols.
  The system harvests those glyphs *with their printed names* and counts them, so a new
  firm's symbology works on day one with no labelling.
- **Handles parametric symbols.** Doors are not copies of each other: a 3'-0" and a 2'-6"
  door are different geometry. Tracing one door switches to a structural detector (an arc
  plus a leaf hinged at the arc's centre) that finds every width and *reports* each width,
  since the arc radius is the door width.
- **Checks its own answer.** Drawing sets state quantities twice, as symbols and as
  schedules. `scripts/reconcile.py` counts receptacles on the plan, OCRs the panel
  schedule tables, and reports agreement room by room. Estimators do this by hand; no
  takeoff tool surveyed automates it.
- **Explains its confidence.** Each detection carries a log odds evidence score, so the
  confidence slider is a lens over already scored results rather than a sensitivity knob
  that changes what the algorithm finds.

---

## Measured on the included set

| | |
|---|---|
| Stage 1 occurrence filter | silenced **1 fingerprint bucket holding 15.3%** of all line pairs, the wall hatching |
| E4 receptacles | **163 confident + 23 flagged** for review |
| Doors per sheet | **35** after the hinge dedupe fix, up from 26 |
| Cross sheet check | T5 = T8 = **33** standard doors, and E4 agrees independently |
| Reconciliation | **22 rooms corroborated**, 12 flagged, 0 contradicted |
| Query latency | roughly 0.4 to 2 seconds per trace on a 33,000 primitive sheet |

The door fix is worth the anecdote. Nine doors per sheet were being silently merged by a
duplicate removal rule that treated two hinges 12pt apart as one door. Back to back office
doors hinge at a shared jamb, so *every* pair along the corridor was half counted. The
total looked perfectly reasonable, and the bug stayed invisible until a human pointed at
two specific doors. That is the argument for the review queue, and for ground truth
evaluation, in one bug.

---

## Honest limits

- **Vector PDFs only.** Scanned drawings have no geometry to parse, and the app detects
  this and warns. A raster fallback is designed in `METHOD.md` but unbuilt. This is the one
  place a trained detector is strictly better, and the two approaches compose: a YOLO style
  proposer would drop into Stage 2 as a fourth peer and inherit the same evidence,
  arbitration and reconciliation.
- **No ground truth accuracy figures.** Thresholds come from measured score
  *distributions*, not hand labelled counts, so precision and recall per class are not yet
  claimable. An evaluation harness is the highest value next step.
- **Two of the four assignment classes.** Doors and receptacles work. Detail markers and
  elevation markers have a designed home (`TextAnchoredProposer`) that is stubbed, not
  built.
- **Near identical variants still cross talk** during legend auto count, where line pattern
  legend entries match wall linework. Subclass splitting is the next Stage 4 task.
- English legend headings, imperial scale notes, and column style legends.

---

## Reading the code

Start with the diagram above, then read in this order. About 700 lines total.

| File | What it is |
|---|---|
| `pipeline/__init__.py` | The map: stages, and the reasoning behind the shape |
| `pipeline/types.py` | The five objects that flow between stages |
| `pipeline/policy.py` | Every threshold in the system, each with the measurement that set it |
| `pipeline/sheet.py` | **Stage 0**, parse each sheet once into one coordinate space |
| `pipeline/index.py` | **Stage 1**, learn which shapes are rare on this sheet |
| `pipeline/proposers/` | **Stage 2**, three ways to generate guesses |
| `pipeline/evidence.py` | **Stage 3**, score every guess on one comparable scale |
| `pipeline/resolve.py` | **Stage 4**, arbitrate, name, scope |
| `pipeline/api.py` | The facade `app.py` calls |

```
app.py              web server and request handling
web/index.html      the single page UI
pipeline/           the architecture above     <- read this
oneshot/            earlier engine, retained as a comparison baseline
scripts/            demo, reconciliation, exemplar discovery
docs/               pipeline diagram
METHOD.md           research basis and academic lineage
```
