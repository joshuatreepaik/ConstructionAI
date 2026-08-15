"""Reconciliation report: does the plan agree with its own schedules?

Thin CLI over pipeline/reconcile_schedule.py - the same code the web app's
"Cross-check vs panel schedules" button runs, so both always report the same
numbers.

USAGE
    .venv/bin/python scripts/reconcile.py [pdf] [plan_page] [schedule_page]
    defaults: data.pdf, page 26 (E4 power plan), schedule page auto-detected
"""

import sys

sys.path.insert(0, ".")

import pymupdf

from pipeline import api

# One duplex receptacle on E4, used as the one-shot exemplar.
RECPT_BOX = (431.0, 346.0, 445.0, 362.0)

PDF = sys.argv[1] if len(sys.argv) > 1 else "data.pdf"
PLAN_PAGE = int(sys.argv[2]) - 1 if len(sys.argv) > 2 else 25
SCHED_PAGE = int(sys.argv[3]) - 1 if len(sys.argv) > 3 else None


class DocState:
    def __init__(self, path):
        self.doc = pymupdf.open(path)


j = api.reconcile(DocState(PDF), PLAN_PAGE, RECPT_BOX, schedule_page=SCHED_PAGE)
if j.get("error"):
    sys.exit(f"error: {j['error']}")

out = [f"# Reconciliation - plan symbols vs. panel schedules ({PDF})\n",
       f"- same-size receptacle symbols counted: **{j['n_detections']}** "
       f"({j['n_excluded_scale']} other-size hits set aside, "
       f"{j['n_unattributed']} outside any tagged room)",
       f"- circuits read from schedule sheet p{j['schedule_page'] + 1}: "
       f"**{j['n_circuits']}**\n",
       "| room | symbols on plan | circuits naming room | status |",
       "|------|-----------------|----------------------|--------|"]
for r in j["rooms"]:
    out.append(f"| {r['room']} | {r['symbols']} | {r['circuits']} | {r['status']} |")

n = {"corroborated": 0, "plan_only": 0, "schedule_only": 0}
for r in j["rooms"]:
    n[r["status"]] += 1
out += [f"\n**{n['corroborated']} corroborated · {n['plan_only']} plan-only · "
        f"{n['schedule_only']} schedule-only**\n",
        "Plan-only rooms are usually circuits labelled by function rather than",
        "number (\"RECPTS-CONF RM\") - a human resolves those in seconds, which",
        "is the point of the report.\n",
        "## Circuit text\n"]
for r in j["rooms"]:
    for c in r["circuit_text"]:
        out.append(f"- room {r['room']}: `{c}`")

report = "\n".join(out)
open("out/reconciliation.md", "w").write(report)
print(report)
