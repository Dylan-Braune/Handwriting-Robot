"""Check every generated figure against the standard the report sets.

For each figure: which fonts it embeds, and the size every piece of text in
it PRINTS at, which is the size set in the figure times the factor LaTeX
applies when it scales the page to the text column. That factor is what the
whole exercise turns on: a 10 pt label in a figure drawn 7.4 in wide and
printed 6.30 in wide reaches the reader at 8.5 pt.

Reads only. Run from the repository root.
"""
from pathlib import Path
import sys

from pypdf import PdfReader

ROOT = Path.cwd()
FIG = ROOT/"Report/ResultsDiscussionRevision_2026-11-02/Figures"
COLUMN = 455.24411                             #\the\textwidth, from the log

#the generated figures, and nothing else: photographs, CAD renders and the
#borrowed architecture diagram are not this project's typesetting
GENERATED = sorted(FIG.glob('rev_*.pdf'))

FLOOR = 10.0
worst = []
print(f"{'figure':42s} {'scale':>6s} {'min pt':>7s}  fonts")
for p in GENERATED:
    rd = PdfReader(str(p))
    pg = rd.pages[0]
    w = float(pg.mediabox.width)
    scale = COLUMN/w                           #every one is width=\linewidth
    sizes = []
    fonts = set()

    def visit(text,cm,tm,fontDict,fontSize):
        if not text.strip():
            return
        #a rotated axis label has tm[3]==0, so the scale is the length of the
        #transformed unit vector, not the diagonal entry on its own
        sy = (float(tm[1])**2+float(tm[3])**2)**0.5
        cy = (float(cm[1])**2+float(cm[3])**2)**0.5
        s = abs(float(fontSize))*(sy or 1.0)*(cy or 1.0)
        if s > 0:
            sizes.append(s)
        #only the fonts text is actually SET in; a canvas may carry an unused
        #default font in its resource dictionary and that is not typography
        try:
            base = fontDict.get("/BaseFont")
        except AttributeError:
            base = None
        if base:
            fonts.add(str(base).split("+")[-1])

    pg.extract_text(visitor_text=visit)
    lo = min(sizes)*scale if sizes else float("nan")
    print(f"{p.name:42s} {scale:6.3f} {lo:7.2f}  {', '.join(sorted(fonts))}")
    if sizes and lo < FLOOR-0.05:
        worst.append((p.name,round(lo,2)))
    bad = [f for f in fonts if "Times" not in f and "STIX" not in f]
    if bad:
        worst.append((p.name,"non-Times: "+", ".join(bad)))

print()
if worst:
    print("BELOW THE 10 pt FLOOR, OR NOT IN TIMES:")
    for n,v in worst:
        print(f"  {n}  {v}")
else:
    print("every generated figure is Times and prints at 10 pt or above")
sys.exit(1 if worst else 0)
