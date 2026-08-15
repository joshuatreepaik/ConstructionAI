"""One-shot symbol detection for construction drawings - staged pipeline (v2).

READ THE FILES IN THIS ORDER
    types.py       the five objects that flow between stages. Start here; the
                   whole architecture is legible from these definitions.
    policy.py      every threshold in the system, each with the measurement or
                   argument that set it. No other file hard-codes a cutoff.
    sheet.py       STAGE 0 - parse each sheet once into one coordinate space.
    index.py       STAGE 1 - learn which shapes are rare ON THIS SHEET.
    proposers/     STAGE 2 - three ways to generate location guesses.
    evidence.py    STAGE 3 - score every guess on one comparable scale.
    resolve.py     STAGE 4 - arbitrate, name, and scope the survivors.
    reconcile_schedule.py
                   cross-check symbol counts against the document's own
                   schedule tables - checking against something the system
                   did not produce itself.
    reconcile_doors.py
                   cross-check doors against the OTHER SHEETS that redraw the
                   same floor (this set has no door schedule table; the other
                   trades' plans are the independent source).
    api.py         the facade the web app calls.

THE PROBLEM THIS SHAPE SOLVES
    "How many X are on this drawing?" hides three different mistakes, each
    needing different information to fix:

        geometric   no symbol is there at all      -> better geometry
        semantic    right shape, wrong type        -> read the printed tag
                    (a GFCI and a duplex receptacle differ by one letter)
        scope       real symbol that shouldn't count  -> know which region of
                    the sheet you are on (legend, key plan, compass rose)

    The previous engine (oneshot/, still runnable as the control) treated all
    three as one problem and grew a filter on the end each time a new failure
    appeared. It ended up with three different rules for "two hits, one spot"
    that disagreed with each other. Here each mistake has an owner.

TWO PRINCIPLES WORTH KNOWING BEFORE READING THE CODE
    1. Guessing and deciding are separate stages, because they want opposite
       things. A guess never made can never be recovered, so STAGE 2 is
       deliberately generous. A false accept costs money, so STAGE 5 is
       strict. One threshold cannot serve both.

    2. Evidence is weighed against the LOCAL background, never a global
       constant. The same match means less inside dense hatching than in white
       space. This is why a count that transfers between sheets is possible at
       all.

NO TRAINING DATA IS USED ANYWHERE IN THIS PACKAGE.
"""
