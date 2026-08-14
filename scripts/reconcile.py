"""Reconciliation report: does the plan agree with its own schedules?

THE IDEA
    A drawing set records the same electrical devices twice - as symbols on
    the power plan, and as circuit rows in the panel schedules. Those two
    should agree. Where they do, the count is corroborated by an independent
    source. Where they don't, either the detector is wrong or the DRAWING is
    wrong, and a human should look. That is exactly the check an estimator
    performs by hand before trusting a number, and no commercial takeoff tool
    we surveyed does it.

WHAT IT PRINTS
    A room-by-room table (symbols found vs. circuits naming that room) plus
    the list of circuits parsed out of the schedule images.

USAGE
    .venv/bin/python scripts/reconcile.py [pdf] [plan_page] [schedule_page]
    defaults: data.pdf, page 26 (E4 power plan), page 28 (E6 panel schedules)
"""

import math
import re
import sys
from collections import defaultdict

sys.path.insert(0, ".")

import pymupdf

from oneshot.extract import extract_primitives, extract_words, primitives_in_box
from oneshot.engine import Scene, detect
from oneshot.schedule import image_regions, ocr_rows, parse_circuits

PDF = sys.argv[1] if len(sys.argv) > 1 else "data.pdf"
PLAN_PAGE = int(sys.argv[2]) - 1 if len(sys.argv) > 2 else 25       # E4
SCHED_PAGE = int(sys.argv[3]) - 1 if len(sys.argv) > 3 else 27      # E6

# One duplex receptacle on E4, used as the one-shot exemplar.
RECPT_BOX = (431.0, 346.0, 445.0, 362.0)
# Same-block instances only: the engine also finds x1.8 grid targets and
# x0.5 junction boxes, which are different devices, not receptacles.
SCALE_KEEP = (0.75, 1.10)

doc = pymupdf.open(PDF)

# ---- plan side: detect receptacles, attribute each to its nearest room tag --
plan = doc[PLAN_PAGE]
prims = extract_primitives(plan)
exemplar = primitives_in_box(prims, RECPT_BOX)
dets = [d for d in detect(Scene(prims), exemplar)
        if SCALE_KEEP[0] <= d.scale <= SCALE_KEEP[1]]

room_tags: dict[str, list[tuple]] = defaultdict(list)
for w in extract_words(plan):
    if re.fullmatch(r"[12]\d\d", w[4]):
        room_tags[w[4]].append(((w[0] + w[2]) / 2, (w[1] + w[3]) / 2))

counted: dict[str, int] = defaultdict(int)
for d in dets:
    nearest = min(
        ((room, math.dist((d.x, d.y), c))
         for room, centers in room_tags.items() for c in centers),
        key=lambda rc: rc[1], default=(None, None))
    if nearest[0] is not None:
        counted[nearest[0]] += 1

# ---- schedule side: OCR the pasted tables, parse receptacle circuits -------
sched = doc[SCHED_PAGE]
circuits = []
for region in image_regions(sched):
    rows = ocr_rows(sched, region)
    circuits.extend(parse_circuits(rows, panel="E6", known_rooms=set(room_tags)))

scheduled: dict[str, list[str]] = defaultdict(list)
for c in circuits:
    for room in c.rooms:
        scheduled[room].append(c.description)

# ---- the report ------------------------------------------------------------
out = [f"# Reconciliation - plan symbols vs. panel schedules ({PDF})\n",
       f"- receptacle symbols detected on plan: **{sum(counted.values())}** "
       f"across {len(counted)} rooms",
       f"- receptacle circuits parsed from schedule images: **{len(circuits)}**, "
       f"naming {len(scheduled)} rooms\n",
       "| room | symbols on plan | circuits naming room | status |",
       "|------|-----------------|----------------------|--------|"]

agree = plan_only = sched_only = 0
for room in sorted(set(counted) | set(scheduled)):
    n_sym, n_ckt = counted.get(room, 0), len(scheduled.get(room, []))
    if n_sym and n_ckt:
        status, agree = "OK: corroborated", agree + 1
    elif n_sym:
        status, plan_only = "REVIEW: on plan, no circuit names it", plan_only + 1
    else:
        status, sched_only = "REVIEW: in schedule, no symbol found", sched_only + 1
    out.append(f"| {room} | {n_sym} | {n_ckt} | {status} |")

out += [f"\n**{agree} corroborated · {plan_only} plan-only · {sched_only} schedule-only**\n",
        "Plan-only rooms are usually circuits labelled by function rather than",
        "number (\"RECPTS-CONF RM\", \"GFCI RECPT-BATHRM\") - a human resolves",
        "those in seconds, which is the point of the report.\n",
        "## Circuits parsed from the schedule images\n"]
out += [f"- `{c.description}` -> rooms {', '.join(c.rooms) or '(none on plan)'}"
        for c in circuits]

report = "\n".join(out)
open("out/reconciliation.md", "w").write(report)
print(report)
