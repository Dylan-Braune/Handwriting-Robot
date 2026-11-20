#is a writer hard to identify because their hand is cursive, or because
#another writer's hand looks like theirs?
#
#two writers can both be perfectly clean and still be confused with each
#other if they happen to write alike, so cursiveness and identifiability are
#different quantities and should not be expected to track one another.  this
#tests that directly:
#
#   * each writer is placed in a style space built from the geometric
#     measurements in their own style profile -- slant, x-height, ascender
#     and descender extent, stroke width and ink fraction.  these are
#     properties of the hand, measured off the writer's pages, and none of
#     them is the classifier's opinion, so the test is not circular.
#   * the distance to the NEAREST other writer is the crowding each writer
#     sits in.
#   * that crowding is then correlated with how often the writer is
#     identified correctly, and against connectedness for comparison.
import json
import sys
from pathlib import Path

import numpy as np

SCR = Path(__file__).resolve().parent
PROF = Path(r"C:\Users\braun\OneDrive\Documents\GitHub\Handwriting-Robot"
            r"\Software\CNN\NOGIT\StyleProfiles10")
KEEP = ["150", "152", "153", "384", "552", "588",
        "abhinav", "dylan", "owen", "thiya", "yeukita"]
LABEL = {"abhinav": "A", "dylan": "B", "owen": "O", "thiya": "T", "yeukita": "Y"}

#the geometric descriptors of the hand.  paperLevel and inkLevel are left out
#on purpose: they describe the photograph's exposure, not the writing
FEATS = ["slantDeg", "xHeightPx", "ascender", "descender",
         "strokeWidthXh", "inkFracRef", "aspectRef", "midFracRef"]

rawv = {}
for w in KEEP:
    p = PROF/(w+".json")
    if not p.exists():
        print("  no profile for", w)
        continue
    d = json.load(open(p, encoding="utf-8"))
    rawv[w] = [float(d.get(f, np.nan)) for f in FEATS]

ws = [w for w in KEEP if w in rawv]
X = np.array([rawv[w] for w in ws], dtype=float)
#standardise, or x-height in pixels swamps slant in degrees
mu, sd = np.nanmean(X, axis=0), np.nanstd(X, axis=0)
sd[sd == 0] = 1.0
Z = (X-mu)/sd
Z = np.nan_to_num(Z)

print("style space: %d writers, %d standardised features" % Z.shape)
print("features:", ", ".join(FEATS))

D = np.sqrt(((Z[:, None, :]-Z[None, :, :])**2).sum(axis=2))
np.fill_diagonal(D, np.inf)

print("\n%-8s %-10s %-22s %s" % ("writer", "nearest", "distance to nearest",
                                 "second nearest"))
near = {}
for i, w in enumerate(ws):
    o = np.argsort(D[i])
    near[w] = {"nearest": ws[o[0]], "d1": float(D[i, o[0]]),
               "second": ws[o[1]], "d2": float(D[i, o[1]])}
    print("  %-6s %-10s %-22.3f %s (%.3f)"
          % (LABEL.get(w, w), LABEL.get(ws[o[0]], ws[o[0]]), D[i, o[0]],
             LABEL.get(ws[o[1]], ws[o[1]]), D[i, o[1]]))

pa = {r["writer"]: r for r in
      json.loads((SCR/"blocks"/"per_author.json").read_text(encoding="utf-8"))}
cur = json.loads((SCR/"blocks"/"cursivity.json").read_text(encoding="utf-8"))

print("\n\n%-8s %-12s %-14s %-12s %s"
      % ("writer", "nearest d", "components/ch", "writer id %", "char %"))
for w in sorted(ws, key=lambda k: near[k]["d1"]):
    print("  %-6s %-12.3f %-14.3f %-12.1f %.2f"
          % (LABEL.get(w, w), near[w]["d1"], cur[w]["cpc"], pa[w]["id"], pa[w]["char"]))


def Corr(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    pear = float(np.corrcoef(a, b)[0, 1])
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    return pear, float(np.corrcoef(ra, rb)[0, 1])


d1 = [near[w]["d1"] for w in ws]
cpc = [cur[w]["cpc"] for w in ws]
idv = [pa[w]["id"] for w in ws]
chv = [pa[w]["char"] for w in ws]

print("\n\n" + "="*72)
print("WHAT PREDICTS WHAT   (n = %d writers)" % len(ws))
print("="*72)
for name, a, b in (
        ("writer id      vs distance to nearest writer", idv, d1),
        ("writer id      vs connectedness", idv, cpc),
        ("char accuracy  vs connectedness", chv, cpc),
        ("char accuracy  vs distance to nearest writer", chv, d1),
        ("connectedness  vs distance to nearest writer", cpc, d1)):
    p, s = Corr(a, b)
    print("  %-46s Pearson %+.3f   Spearman %+.3f" % (name, p, s))

json.dump({w: {"label": LABEL.get(w, w), **near[w]} for w in ws},
          open(SCR/"blocks"/"style_distance.json", "w"), indent=1)
print("\nwrote", SCR/"blocks"/"style_distance.json")
