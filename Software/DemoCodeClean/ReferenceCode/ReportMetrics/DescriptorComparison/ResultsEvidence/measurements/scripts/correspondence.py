#do the labels actually line up with the crops, or do the counts merely agree?
#
#a matching count proves nothing.  writer 384's j06-014 had 11 crops and 11
#label lines and was completely misaligned from line 6 down, because the
#segmenter merged two written lines while the transcript was one line short.
#the arithmetic agreed and the content did not.
#
#this checks CONTENT.  each crop is read, and its reading is scored against
#its own label line and against the label lines on either side.  if the
#correspondence is right, a crop reads closest to its OWN label.  if a crop
#reads closer to its neighbour's label, that line is out of step and the page
#is reported.
#
#the reader is imperfect, so a low score on its own label is not by itself
#evidence of anything.  what matters is the COMPARISON: own versus neighbour.
import sys
import warnings
from pathlib import Path

import numpy as np
from PIL import Image
warnings.filterwarnings("ignore")

CP3 = Path(r"C:\Users\braun\OneDrive\Documents\GitHub\Handwriting-Robot\Software\CleanPipeline3")
sys.path.insert(0, str(CP3))
import extractiam as EX
import recognise as REC

DL = Path(r"C:\Users\braun\Downloads")
SRC = CP3.parent.parent/"Data"/"Datasets"/"IAMpages10"
WRITERS = ["150", "152", "153", "384", "552", "588"]

reader = REC.PaperCRNNNumpy(
    checkpointPath=str(DL/"paper_cnn_bilstm_ctc_scratch_best.npz"))
print("reader paper_cnn_bilstm_ctc_scratch_best.npz\n")

bad, pages, lines = [], 0, 0
countMismatch = []
print("%-8s %-16s %-7s %-9s %s"
      % ("writer", "page", "lines", "own>nbr", "worst line (own vs best neighbour)"))
print("-"*96)
for w in WRITERS:
    d = SRC/w
    if not d.is_dir():
        continue
    for png in sorted(d.glob("*.png")):
        lab = png.with_name(png.stem+"_labels.txt")
        if not lab.exists():
            continue
        want = [l for l in lab.read_text(encoding="utf-8").splitlines() if l.strip()]
        crops, _info = EX.ExtractPage(png, verbose=False)
        if len(crops) != len(want):
            countMismatch.append((w, png.stem, len(crops), len(want)))
            continue
        pages += 1
        lines += len(crops)
        reads = [REC.ReadText(Image.fromarray(c).convert("L"), reader) for c in crops]

        ok, worst, worstGap = 0, None, 0.0
        for i, r in enumerate(reads):
            own = REC.CharAcc(r, want[i])
            nbr = 0.0
            for j in (i-1, i+1):
                if 0 <= j < len(want):
                    nbr = max(nbr, REC.CharAcc(r, want[j]))
            if own >= nbr:
                ok += 1
            else:
                gap = nbr-own
                if gap > worstGap:
                    worstGap, worst = gap, (i, 100*own, 100*nbr)
        flag = ""
        if ok < len(reads):
            flag = "line %d: own %.0f%% vs neighbour %.0f%%" % worst
            bad.append((w, png.stem, len(reads)-ok, len(reads)))
        print("%-8s %-16s %-7d %-9s %s"
              % (w, png.stem, len(reads), "%d/%d" % (ok, len(reads)), flag),
              flush=True)

print("\n" + "="*96)
print("%d pages checked, %d lines" % (pages, lines))
if countMismatch:
    print("\n%d page(s) skipped because the counts disagree:" % len(countMismatch))
    for w, p, c, l in countMismatch:
        print("   %-6s %-16s %d crops, %d label lines" % (w, p, c, l))
if bad:
    print("\n%d page(s) where at least one crop reads closer to a NEIGHBOUR's "
          "label than its own:" % len(bad))
    for w, p, n, tot in bad:
        print("   %-6s %-16s %d of %d lines out of step" % (w, p, n, tot))
else:
    print("\nevery crop on every page reads closest to its OWN label line.")
    print("the correspondence is right, not merely the count.")
