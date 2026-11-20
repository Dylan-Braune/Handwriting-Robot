#the full held-out measurement, re-run on the corrected data.
#
#what changed since the last run:
#  * crops come from extractiam.py, which cuts the page to its printed
#    boundaries and then uses segment.py.  the old line_cache_authors10,
#    written 2026-10-24, held merged lines
#  * writer 384's ten transcripts were rewritten and one was created
#  * three of writer 153's transcripts were rewritten
#
#the held-out page of each database writer is LabeledPages[-1], the last
#labelled page by name, which is what train.py uses.  the photographed
#writers' holdout is a seeded 15 per cent of their lines and is unchanged, so
#it is read from the existing cache.
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
OUT = Path(__file__).resolve().parent/"holdout_rerun"
OUT.mkdir(exist_ok=True)
IAM = ["150", "152", "153", "384", "552", "588"]
PERSONAL = ["abhinav", "dylan", "owen", "thiya", "yeukita"]
LABEL = {"abhinav": "A", "dylan": "B", "owen": "O", "thiya": "T", "yeukita": "Y"}
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
reader = REC.PaperCRNNNumpy(
    checkpointPath=str(DL/"paper_cnn_bilstm_ctc_scratch_best.npz"))
print("namer  author_shape_11_pw170_weights.npz (%d writers, pen width %.2f)"
      % (len(NAMES), PW))
print("reader paper_cnn_bilstm_ctc_scratch_best.npz\n")


def Score(crops, want, writer):
    d = n = wr = wn = idOk = 0
    votes = np.zeros(len(NAMES))
    for c, t in zip(crops, want):
        r = REC.ReadText(Image.fromarray(c).convert("L"), reader)
        d += REC.Levenshtein(r, t)
        n += len(t)
        wr += sum(1 for a, b in zip(r.split(), t.split()) if a == b)
        wn += len(t.split())
        nm = REC.StrokeNormalize(c, penWidthXh=PW)
        _k, p = REC.ClassifyImage(
            Image.fromarray(nm if nm is not None else c).convert("L"),
            namer, strokeNormalise=False)
        p = np.asarray(p, dtype=np.float64)
        if NAMES[int(np.argmax(p))] == writer:
            idOk += 1
        votes += p*float((c < 128).sum())
    return d, n, wr, wn, idOk, NAMES[int(np.argmax(votes))]


rows = []
print("%-9s %-16s %-7s %-14s %-10s %s"
      % ("writer", "held-out page", "lines", "writer id", "char %", "word %"))
print("-"*78)

#---the six database writers, re-cut from the page------------------------
for w in IAM:
    d = IAMROOT/w
    pages = sorted(d.glob("*.png"))
    labelled = [p for p in pages if TR.ReadLabelLines(str(p))]
    if not labelled:
        continue
    page = labelled[-1]
    want = [l for l in (page.with_name(page.stem+"_labels.txt"))
            .read_text(encoding="utf-8").splitlines() if l.strip()]
    crops, _info = EX.ExtractPage(page, verbose=False)
    if len(crops) != len(want):
        print("%-9s %-16s SKIPPED: %d crops vs %d labels"
              % (w, page.stem, len(crops), len(want)))
        continue
    dd, nn, wr, wn, idOk, named = Score(crops, want, w)
    rows.append({"writer": w, "src": "database", "page": page.stem,
                 "lines": len(crops), "d": dd, "n": nn, "wr": wr, "wn": wn,
                 "idOk": idOk, "named": named})
    print("%-9s %-16s %-7d %-14s %-10.2f %.2f"
          % (w, page.stem, len(crops), "%s %d/%d" % (named, idOk, len(crops)),
             100*max(0, 1-dd/nn), 100*wr/wn if wn else 0), flush=True)

#---the five photographed writers, from the cache, split unchanged--------
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
    crops = [np.asarray(TR.DecodeCrop(s["imageBytes"]), dtype=np.uint8) for s in ss]
    want = [(s.get("text") or "").strip() for s in ss]
    dd, nn, wr, wn, idOk, named = Score(crops, want, w)
    rows.append({"writer": w, "src": "photographed", "page": "15%% of lines",
                 "lines": len(crops), "d": dd, "n": nn, "wr": wr, "wn": wn,
                 "idOk": idOk, "named": named})
    print("%-9s %-16s %-7d %-14s %-10.2f %.2f"
          % (LABEL[w], "seeded 15%", len(crops),
             "%s %d/%d" % (named, idOk, len(crops)),
             100*max(0, 1-dd/nn), 100*wr/wn if wn else 0), flush=True)


def Agg(sub):
    d = sum(r["d"] for r in sub); n = sum(r["n"] for r in sub)
    wr = sum(r["wr"] for r in sub); wn = sum(r["wn"] for r in sub)
    k = sum(r["idOk"] for r in sub); L = sum(r["lines"] for r in sub)
    return (100*max(0, 1-d/n) if n else 0, 100*wr/wn if wn else 0, k, L)


def Wilson(k, n, z=1.96):
    if n == 0:
        return 0.0, 0.0
    p = k/n
    dd = 1+z*z/n
    c = (p+z*z/(2*n))/dd
    h = z*((p*(1-p)/n+z*z/(4*n*n))**0.5)/dd
    return 100*max(0, c-h), 100*min(1, c+h)


print("\n" + "="*78)
for tag, sub in (("database writers", [r for r in rows if r["src"] == "database"]),
                 ("photographed writers", [r for r in rows if r["src"] == "photographed"]),
                 ("ALL", rows)):
    if not sub:
        continue
    ch, wo, k, L = Agg(sub)
    lo, hi = Wilson(k, L)
    print("%-22s %3d lines   char %6.2f %%   word %6.2f %%   writer %d/%d = %.2f %% [%.1f, %.1f]"
          % (tag, L, ch, wo, k, L, 100*k/L, lo, hi))

pages = sum(1 for r in rows if r["src"] == "database")
pOk = sum(1 for r in rows if r["src"] == "database" and r["named"] == r["writer"])
pOk += sum(1 for r in rows if r["src"] == "photographed" and r["named"] == r["writer"])
print("\npages/groups named correctly when lines are pooled by ink: %d of %d"
      % (pOk, len(rows)))

(OUT/"results.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")
print("\nwrote", OUT/"results.json")
print("\nprevious run, old cache and old labels: 155 lines, char 91.60 %")
