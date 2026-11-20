#writer 384 re-measured with the corrected crops and the corrected labels.
#
#the crops come from extractiam.py, which cuts the page to the printed
#boundaries and then uses segment.py, not from the stale 2026-10-24 cache
#that held the merged lines.  the labels are the ten transcripts read off the
#scans by eye, one line per written line.
#
#two cautions on what this can and cannot show.
#  * the TEXT reader never trained on writer 384 at all -- none of its nine
#    pages is in the Teklia corpus -- so every page here is a fair test of
#    reading an unseen hand.
#  * the WRITER classifier did train on 384, on eight of its nine pages.
#    only j06-014 was held out, so only that page is a fair test of naming;
#    the rest are reported for completeness and are marked.
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
SRC = CP3.parent.parent/"Data"/"Datasets"/"IAMpages10"/"384"
HOLDOUT = "j06-014"
PW = 0.17

namer = REC.AuthorClassifierCNNNumpy(
    checkpointPath=str(DL/"author_shape_11_pw170_weights.npz"))
byIdx = {v: k for k, v in namer.authorMapping.items()}
NAMES = [byIdx[i] for i in range(len(byIdx))]
reader = REC.PaperCRNNNumpy(
    checkpointPath=str(DL/"paper_cnn_bilstm_ctc_scratch_best.npz"))
print("namer  author_shape_11_pw170_weights.npz, %d writers, pen width %.2f"
      % (len(NAMES), PW))
print("reader paper_cnn_bilstm_ctc_scratch_best.npz\n")

tot = {"d": 0, "n": 0, "wr": 0, "wn": 0, "idOk": 0, "lines": 0}
hold = {"d": 0, "n": 0, "wr": 0, "wn": 0, "idOk": 0, "lines": 0}
print("%-12s %-7s %-20s %-10s %-10s %s"
      % ("page", "lines", "writer named", "char %", "word %", "role"))
print("-"*86)
for png in sorted(SRC.glob("*.png")):
    name = png.stem
    lab = png.with_name(name+"_labels.txt")
    crops, info = EX.ExtractPage(png, verbose=False)
    want = [l for l in lab.read_text(encoding="utf-8").splitlines() if l.strip()]
    if len(crops) != len(want):
        print("%-12s crops %d vs labels %d -- skipped" % (name, len(crops), len(want)))
        continue

    d = n = wr = wn = idOk = 0
    votes = np.zeros(len(NAMES))
    for c, t in zip(crops, want):
        read = REC.ReadText(Image.fromarray(c).convert("L"), reader)
        d += REC.Levenshtein(read, t)
        n += len(t)
        wr += sum(1 for a, b in zip(read.split(), t.split()) if a == b)
        wn += len(t.split())
        nrm = REC.StrokeNormalize(c, penWidthXh=PW)
        _k, p = REC.ClassifyImage(
            Image.fromarray(nrm if nrm is not None else c).convert("L"),
            namer, strokeNormalise=False)
        p = np.asarray(p, dtype=np.float64)
        if NAMES[int(np.argmax(p))] == "384":
            idOk += 1
        votes += p*float((c < 128).sum())

    page = NAMES[int(np.argmax(votes))]
    role = "HELD OUT, fair test" if name == HOLDOUT else "in namer training"
    print("%-12s %-7d %-20s %-10.2f %-10.2f %s"
          % (name, len(crops),
             "%s  %d/%d lines" % (page, idOk, len(crops)),
             100*max(0, 1-d/n), 100*wr/wn if wn else 0, role))
    for acc in (tot, hold if name == HOLDOUT else None):
        if acc is None:
            continue
        acc["d"] += d; acc["n"] += n; acc["wr"] += wr; acc["wn"] += wn
        acc["idOk"] += idOk; acc["lines"] += len(crops)

print("\n" + "="*86)
print("ALL TEN PAGES   %d lines   char %.2f %%   word %.2f %%   writer %d/%d lines"
      % (tot["lines"], 100*max(0, 1-tot["d"]/tot["n"]),
         100*tot["wr"]/tot["wn"], tot["idOk"], tot["lines"]))
if hold["lines"]:
    print("HELD-OUT PAGE   %d lines   char %.2f %%   word %.2f %%   writer %d/%d lines"
          % (hold["lines"], 100*max(0, 1-hold["d"]/hold["n"]),
             100*hold["wr"]/hold["wn"], hold["idOk"], hold["lines"]))
print()
print("for comparison, the same held-out page measured on the OLD cache and")
print("the OLD labels read 60.78 %% character accuracy per line.")
