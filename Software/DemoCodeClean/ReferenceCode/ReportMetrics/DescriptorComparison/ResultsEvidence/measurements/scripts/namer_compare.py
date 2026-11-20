#the two stroke-normalised writer classifiers, on the same two populations.
#
#   A  author_shape_11_pw170_weights.npz        trained with the torch route
#   B  author_shape_11w_pw170_scratch_weights.npz  trained from first principles
#
#both are 11-class, both expect crops redrawn at a constant pen width of 0.17
#x-heights, so both are applied through the same preprocessing and the only
#difference is the training route that produced the weights.
#
#two populations, because the classifier has two jobs and they are not the
#same test:
#   1. the 156 lines held out from it, which is real handwriting it never saw.
#      this is the R2 measurement.
#   2. the 184 lines the robot drew, which is the instrument role: here the
#      classifier is the measuring device for R1 and what matters is whether
#      it recognises the writer in the machine's own constant-width output.
#a classifier can be better at one and worse at the other, which is exactly
#why the ink classifier was replaced, so neither number decides on its own.
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
CROPS = CP3/"ReportResults"/"SegmentedReproducedPageCrops"
OUT = Path(__file__).resolve().parent/"namer_compare"
OUT.mkdir(exist_ok=True)
IAM = ["150", "152", "153", "384", "552", "588"]
PERSONAL = ["abhinav", "dylan", "owen", "thiya", "yeukita"]
LABEL = {"abhinav": "A", "dylan": "B", "owen": "O", "thiya": "T",
         "yeukita": "Y"}
PW = 0.17
MODELS = [("library", "author_shape_11_pw170_weights.npz"),
          ("scratch", "author_shape_11w_pw170_scratch_weights.npz")]

_o = TR.DecodeCrop


def D(r):
    try:
        return _o(r)
    except Exception:
        import io
        return np.asarray(Image.open(io.BytesIO(r)).convert("L"), dtype=np.uint8)


TR.DecodeCrop = D


def Wilson(k, n, z=1.96):
    if n == 0:
        return 0.0, 0.0
    p = k/n
    dd = 1+z*z/n
    c = (p+z*z/(2*n))/dd
    h = z*((p*(1-p)/n+z*z/(4*n*n))**0.5)/dd
    return 100*max(0, c-h), 100*min(1, c+h)


#---gather both populations once, so both models see identical crops---------
print("collecting the two populations ...", flush=True)
holdout = []     #(trueLabel, crop)
for w in IAM:
    labelled = [p for p in sorted((IAMROOT/w).glob("*.png"))
                if p.with_name(p.stem+"_labels.txt").exists()]
    if not labelled:
        continue
    crops, _i = EX.ExtractPage(labelled[-1], verbose=False)
    holdout += [(w, c) for c in crops]

base = TR.IAMLineDatasetRaw(rootDir=str(TR.DEFAULT_AUTHOR_DATA_DIR),
                            cacheDir=str(TR.DEFAULT_AUTHOR_CACHE_DIR))
base.samples = []
TR.AddPersonalSamples(base)
for s in base.samples:
    if s.get("is_holdout"):
        w = s["page_key"].split("/")[0]
        if w in LABEL:
            holdout.append((LABEL[w], np.asarray(TR.DecodeCrop(s["imageBytes"]),
                                                 dtype=np.uint8)))

#the drawn pages: the crops already cut for the report, one folder per page.
#the folder names are NOT the report's writer labels -- AuthorD is dylan, who
#the report calls B -- so the mapping is written out rather than guessed from
#the prefix, and anything not in it is skipped loudly
FOLDER = {"Author150": "150", "Author152": "152",
          "Author1531": "153", "Author1532": "153",
          "Author3841": "384", "Author3842": "384",
          "Author552": "552", "Author588": "588",
          "AuthorA1": "A", "AuthorA2": "A",
          "AuthorD": "B", "AuthorO": "O", "AuthorT": "T", "AuthorY": "Y"}
drawn = []
if CROPS.is_dir():
    for d in sorted(CROPS.iterdir()):
        if not d.is_dir():
            continue
        if d.name not in FOLDER:
            print("   skipping folder %s, not a writer page" % d.name)
            continue
        w = FOLDER[d.name]
        for png in sorted(d.glob("line_*.png")):
            drawn.append((w, np.asarray(Image.open(png).convert("L"),
                                        dtype=np.uint8)))
print("   %d held-out lines of real handwriting" % len(holdout))
print("   %d lines the robot drew\n" % len(drawn))

results = {}
for tag, fn in MODELS:
    path = DL/fn
    if not path.exists():
        print("%s: %s not found, skipped" % (tag, fn))
        continue
    model = REC.AuthorClassifierCNNNumpy(checkpointPath=str(path))
    byIdx = {v: k for k, v in model.authorMapping.items()}
    NAMES = [byIdx[i] for i in range(len(byIdx))]
    TO = {w: w for w in IAM}
    TO.update({full: lab for full, lab in LABEL.items()})

    print("%-9s %s (%d classes)" % (tag, fn, len(NAMES)), flush=True)
    row = {"file": fn}
    for pop, data in (("holdout", holdout), ("drawn", drawn)):
        if not data:
            continue
        ok, conf, per = 0, [], defaultdict(lambda: [0, 0])
        votes = defaultdict(lambda: np.zeros(len(NAMES)))
        for w, c in data:
            nm = REC.StrokeNormalize(c, penWidthXh=PW)
            _k, p = REC.ClassifyImage(
                Image.fromarray(nm if nm is not None else c).convert("L"),
                model, strokeNormalise=False)
            p = np.asarray(p, dtype=np.float64)
            got = TO.get(NAMES[int(np.argmax(p))], NAMES[int(np.argmax(p))])
            ok += (got == w)
            conf.append(float(p.max()))
            per[w][1] += 1
            per[w][0] += (got == w)
            votes[w] += p*float((c < 128).sum())
        n = len(data)
        pooled = sum(1 for w in votes
                     if TO.get(NAMES[int(np.argmax(votes[w]))],
                               NAMES[int(np.argmax(votes[w]))]) == w)
        lo, hi = Wilson(ok, n)
        row[pop] = {"n": n, "ok": ok, "pct": 100*ok/n, "lo": lo, "hi": hi,
                    "pooled": pooled, "groups": len(votes),
                    "meanConf": 100*float(np.mean(conf)),
                    "per": {k: v for k, v in per.items()}}
        print("   %-8s %3d lines  %3d correct = %6.2f %% [%.1f, %.1f]  "
              "pooled %d/%d  mean confidence %.1f %%"
              % (pop, n, ok, 100*ok/n, lo, hi, pooled, len(votes),
                 100*float(np.mean(conf))), flush=True)
    results[tag] = row
    print()

#---the comparison-----------------------------------------------------------
if len(results) == 2:
    a, b = results["library"], results["scratch"]
    print("="*78)
    print("%-26s %-18s %-18s %s" % ("population", "library-trained",
                                    "from first principles", "difference"))
    print("-"*78)
    for pop, nice in (("holdout", "held-out real lines"),
                      ("drawn", "lines the robot drew")):
        if pop in a and pop in b:
            print("%-26s %-18s %-18s %+.2f points"
                  % (nice, "%.2f %% (%d/%d)" % (a[pop]["pct"], a[pop]["ok"], a[pop]["n"]),
                     "%.2f %% (%d/%d)" % (b[pop]["pct"], b[pop]["ok"], b[pop]["n"]),
                     b[pop]["pct"]-a[pop]["pct"]))
    for pop, nice in (("holdout", "writers named when pooled"),
                      ("drawn", "pages named when pooled")):
        if pop in a and pop in b:
            print("%-26s %-18s %-18s" % (nice,
                  "%d/%d" % (a[pop]["pooled"], a[pop]["groups"]),
                  "%d/%d" % (b[pop]["pooled"], b[pop]["groups"])))

(OUT/"compare.json").write_text(json.dumps(results, indent=1, default=str),
                                encoding="utf-8")
print("\nwrote", OUT/"compare.json")
