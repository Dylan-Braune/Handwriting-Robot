#the writer classifier on its own held-out lines, recording EVERY line's
#prediction so that the confusion matrix of qualification test 2 can be drawn
#from the measurement rather than reconstructed from a summary.
#
#the classifier was trained only on the eleven writers' own pages and never on
#the public line corpus, so its held-out material is the project's own split:
#the last labelled page of each database writer, and the seeded 15 per cent of
#each photographed writer's lines.
#
#the database writers' crops are re-cut from the page with extractiam, not
#read from the stale cache, because the cache was written before the labels
#were corrected and holds merged lines.
import json
import sys
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image
warnings.filterwarnings("ignore")

CP3 = Path(r"C:\Users\braun\OneDrive\Documents\GitHub\Handwriting-Robot\Software\CleanPipeline3")
sys.path.insert(0, str(CP3))
import extractiam as EX
import recognise as REC
import train as TR

DL = Path(r"C:\Users\braun\Downloads")
IAMROOT = CP3.parent.parent/"Data"/"Datasets"/"IAMpages10"
OUT = Path(__file__).resolve().parent/"holdout_namer"
OUT.mkdir(exist_ok=True)
IAM = ["150", "152", "153", "384", "552", "588"]
PERSONAL = ["abhinav", "dylan", "owen", "thiya", "yeukita"]
LABEL = {"abhinav": "A", "dylan": "B", "owen": "O", "thiya": "T",
         "yeukita": "Y"}
PW = 0.17

_o = TR.DecodeCrop


def D(r):
    try:
        return _o(r)
    except Exception:
        import io
        return np.asarray(Image.open(io.BytesIO(r)).convert("L"), dtype=np.uint8)


TR.DecodeCrop = D

namer = REC.AuthorClassifierCNNNumpy(
    checkpointPath=str(DL/"author_shape_11_pw170_weights.npz"))
byIdx = {v: k for k, v in namer.authorMapping.items()}
NAMES = [byIdx[i] for i in range(len(byIdx))]
print("namer author_shape_11_pw170_weights.npz, %d writers, pen width %.2f\n"
      % (len(NAMES), PW))


def Name(crop):
    nm = REC.StrokeNormalize(crop, penWidthXh=PW)
    _k, p = REC.ClassifyImage(
        Image.fromarray(nm if nm is not None else crop).convert("L"),
        namer, strokeNormalise=False)
    p = np.asarray(p, dtype=np.float64)
    return NAMES[int(np.argmax(p))], float(p.max()), p


lines = []


def Score(writer, label, crops, source):
    votes = np.zeros(len(NAMES))
    ok = 0
    for i, c in enumerate(crops):
        nm, conf, p = Name(c)
        lines.append({"writer": label, "named": nm, "conf": conf,
                      "line": i, "source": source})
        ok += (nm == writer)
        votes += p*float((c < 128).sum())
    pooled = NAMES[int(np.argmax(votes))]
    return ok, pooled


print("%-7s %-14s %-7s %-16s %s"
      % ("writer", "held-out page", "lines", "lines named", "pooled by ink"))
print("-"*66)

for w in IAM:
    d = IAMROOT/w
    labelled = [p for p in sorted(d.glob("*.png"))
                if p.with_name(p.stem+"_labels.txt").exists()]
    if not labelled:
        continue
    page = labelled[-1]
    crops, _info = EX.ExtractPage(page, verbose=False)
    ok, pooled = Score(w, w, crops, "database")
    print("%-7s %-14s %-7d %-16s %s"
          % (w, page.stem, len(crops), "%d/%d" % (ok, len(crops)),
             "%s %s" % (pooled, "correct" if pooled == w else "WRONG")),
          flush=True)

base = TR.IAMLineDatasetRaw(rootDir=str(TR.DEFAULT_AUTHOR_DATA_DIR),
                            cacheDir=str(TR.DEFAULT_AUTHOR_CACHE_DIR))
base.samples = []
TR.AddPersonalSamples(base)
held = defaultdict(list)
for s in base.samples:
    if s.get("is_holdout"):
        held[s["page_key"].split("/")[0]].append(s)

for w in PERSONAL:
    ss = held.get(w, [])
    if not ss:
        continue
    crops = [np.asarray(TR.DecodeCrop(s["imageBytes"]), dtype=np.uint8)
             for s in ss]
    ok, pooled = Score(w, LABEL[w], crops, "photographed")
    print("%-7s %-14s %-7d %-16s %s"
          % (LABEL[w], "seeded 15%", len(crops), "%d/%d" % (ok, len(crops)),
             "%s %s" % (pooled, "correct" if pooled == w else "WRONG")),
          flush=True)


def Wilson(k, n, z=1.96):
    if n == 0:
        return 0.0, 0.0
    p = k/n
    dd = 1+z*z/n
    c = (p+z*z/(2*n))/dd
    h = z*((p*(1-p)/n+z*z/(4*n*n))**0.5)/dd
    return 100*max(0, c-h), 100*min(1, c+h)


#the label of a photographed writer is its initial, so compare on the label
TRUE = {w: w for w in IAM}
TRUE.update({LABEL[w]: LABEL[w] for w in PERSONAL})
NAMEOF = dict(TRUE)
NAMEOF.update({w: LABEL[w] for w in PERSONAL})
for r in lines:
    r["named"] = NAMEOF.get(r["named"], r["named"])

k = sum(1 for r in lines if r["named"] == r["writer"])
n = len(lines)
lo, hi = Wilson(k, n)
print("\n" + "="*66)
print("%d held-out lines, %d named correctly = %.2f %% [%.1f, %.1f]"
      % (n, k, 100*k/n, lo, hi))

misses = [r for r in lines if r["named"] != r["writer"]]
print("\n%d line(s) named incorrectly:" % len(misses))
for r in misses:
    print("   writer %-4s line %-3d named %-4s at %.1f %% confidence (%s)"
          % (r["writer"], r["line"], r["named"], 100*r["conf"], r["source"]))
good = [r["conf"] for r in lines if r["named"] == r["writer"]]
print("\nmean confidence on a correct line %.1f %%, on an incorrect line %.1f %%"
      % (100*np.mean(good), 100*np.mean([r["conf"] for r in misses])
         if misses else 0.0))

(OUT/"lines.json").write_text(json.dumps(lines, indent=1), encoding="utf-8")
print("\nwrote", OUT/"lines.json")
