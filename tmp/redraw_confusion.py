"""Redraw the two row-normalised writer confusion matrices.

Same data, same normalisation and the same printed numbers as the block in
tmp/build_added_evidence.py that first produced them; only the typography and
the page size change. The originals were laid out 7.4 in wide and included at
0.82 and 0.78 of the text column, so LaTeX scaled them by about 0.7 and their
9 pt tick labels printed at about 6 pt. These are authored at exactly one
column width, in Times to match the report body, with no size under 10.5 pt.

Run from the repository root. Reads evidence only; writes two PDFs.
"""
from pathlib import Path
import json

# no writer name reaches a figure: every drawn label is rewritten as a letter
import re as _re
_NAMEPAT=_re.compile(r"(?i)\b(owen|yeukita|dylan|abhinav|thiya)\s*(\d+)?")
_LET={"owen":"O","yeukita":"Y","dylan":"D","abhinav":"A","thiya":"T"}
def _anonsub(m):
    out="Author "+_LET[m.group(1).lower()]
    if m.group(2): out+=" page %d"%int(m.group(2))
    return out
def AnonFigure(fig):
    import matplotlib.text as _mt
    for t in fig.findobj(_mt.Text):
        try: old=t.get_text()
        except Exception: continue
        if old and _NAMEPAT.search(old): t.set_text(_NAMEPAT.sub(_anonsub,old))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/"Report/SubmissionRevision_2026-11-02/Figures"
E = ROOT/"Report/ReportWorkingArchive_2026-11-01_VisualTests/ResultsEvidence"

WIDTH = 6.30                                   #inches, == \linewidth
TNR = "Times New Roman"
plt.rcParams.update({
    "font.family":"serif","font.serif":[TNR,"STIXGeneral","DejaVu Serif"],
    "mathtext.fontset":"custom","mathtext.rm":TNR,"mathtext.it":TNR+":italic",
    "mathtext.bf":TNR+":bold","mathtext.default":"it",
    "font.size":11.5,"axes.labelsize":11.5,"pdf.fonttype":42,
})

order = ["150","152","153","384","552","588","abhinav","dylan","owen",
         "thiya","yeukita"]
alias = dict(zip(order,["150","152","153","384","552","588","A","B","O","T","Y"]))

lines = json.loads((E/"measurements/outputs/blocks/lines.json").read_text())
hold = json.loads((E/"measurements/outputs/holdout_namer/lines.json").read_text())

for data,name in [(lines,"results_reproduced_confusion"),
                  (hold,"results_holdout_confusion")]:
    c = np.zeros((11,11),int)
    reverse = {v:k for k,v in alias.items()}
    for row in data:
        c[order.index(reverse.get(row["writer"],row["writer"])),
          order.index(reverse.get(row["named"],row["named"]))] += 1
    n = c.sum(1)
    pct = 100*c/n[:,None]
    assert np.allclose(pct.sum(1),100)

    fig,ax = plt.subplots(figsize=(WIDTH,5.9),layout="constrained")
    im = ax.imshow(pct,cmap="Blues",vmin=0,vmax=100)
    ax.set_xticks(range(11),[alias[w] for w in order])
    ax.set_yticks(range(11),[f"{alias[w]} (n={v})" for w,v in zip(order,n)])
    ax.set_xlabel("Predicted writer");ax.set_ylabel("Reference writer")
    ax.tick_params(labelsize=11)
    for y in range(11):
        for x in range(11):
            if c[y,x]:
                ax.text(x,y,f"{pct[y,x]:.1f}",ha="center",va="center",
                        fontsize=10.5,
                        color="white" if pct[y,x]>55 else "#18334b")
    cb = fig.colorbar(im,ax=ax,shrink=.80)
    cb.set_label("Percentage of reference-writer lines",fontsize=11)
    cb.ax.tick_params(labelsize=11)
    AnonFigure(fig)
    fig.savefig(OUT/f"{name}.pdf")
    try:
        from panelexport import SavePanels
        k=SavePanels(fig,name)
        if k: print(f"  wrote {k} separate panel(s) for {name}")
    except ImportError:
        pass
    plt.close(fig)
    print(f"{name}: correct {np.trace(c)} of {c.sum()}, "
          f"balanced {np.diag(pct).mean():.2f}%")
