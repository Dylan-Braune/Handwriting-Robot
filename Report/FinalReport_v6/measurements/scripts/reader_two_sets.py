#one reader, measured on both held-out sets.  the checkpoint is argv[1].
#
#  1. Teklia's validation/ directory, 976 crops.  BuildTekliaSplit feeds
#     train/ to the optimiser and validation/ only to EvalTextForward, so no
#     weight was updated from these, and the crops are Teklia's own cuts.
#  2. the photographed pages' held-out lines, the seeded 15 per cent that
#     AddPersonalSamples marks is_holdout (ValFraction 0.15, SplitSeed 0).
#
#text only: no namer, since the Teklia crops carry no writer identity.
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
import recognise as REC
import train as TR

VAL = Path(r"C:\Users\braun\OneDrive\Documents\GitHub\Handwriting-Robot"
           r"\Data\Datasets\TekliaIamLines\validation")
PERSONAL = ["abhinav", "dylan", "owen", "thiya", "yeukita"]
LABEL = {"abhinav": "A", "dylan": "B", "owen": "O", "thiya": "T", "yeukita": "Y"}

ckpt = Path(sys.argv[1])
OUT = Path(__file__).resolve().parent/("reader_"+ckpt.stem)
OUT.mkdir(exist_ok=True)

#part of the author cache was written as raw zlib rather than PNG
_o = TR.DecodeCrop


def D(r):
    try:
        return _o(r)
    except Exception:
        import io
        return np.asarray(Image.open(io.BytesIO(r)).convert("L"), dtype=np.uint8)


TR.DecodeCrop = D

reader = REC.PaperCRNNNumpy(checkpointPath=str(ckpt))
print("reader %s\n" % ckpt.name, flush=True)

rows = []


def Score(items, read):
    d = n = wr = wn = ex = 0
    for i, (src, t) in enumerate(items, 1):
        r = read(src)
        d += REC.Levenshtein(r, t)
        n += len(t)
        wr += sum(1 for a, b in zip(r.split(), t.split()) if a == b)
        wn += len(t.split())
        ex += (r.strip() == t.strip())
        if i % 200 == 0:
            print("    %4d / %d" % (i, len(items)), flush=True)
    return d, n, wr, wn, ex


#---the photographed writers' held-out lines-----------------------------
base = TR.IAMLineDatasetRaw(rootDir=str(TR.DEFAULT_AUTHOR_DATA_DIR),
                            cacheDir=str(TR.DEFAULT_AUTHOR_CACHE_DIR))
base.samples = []
TR.AddPersonalSamples(base)
held = defaultdict(list)
for s in base.samples:
    if s.get("is_holdout"):
        held[s["page_key"].split("/")[0]].append(s)

print("photographed pages, held-out lines")
print("%-8s %-7s %-10s %-10s %s" % ("writer", "lines", "char %", "word %", "exact"))
print("-"*52)
for w in PERSONAL:
    ss = held.get(w, [])
    if not ss:
        continue
    items = [(np.asarray(TR.DecodeCrop(s["imageBytes"]), dtype=np.uint8),
              (s.get("text") or "").strip()) for s in ss]
    d, n, wr, wn, ex = Score(
        items, lambda c: REC.ReadText(Image.fromarray(c).convert("L"), reader))
    rows.append({"set": "photographed", "writer": LABEL[w], "lines": len(ss),
                 "d": d, "n": n, "wr": wr, "wn": wn, "exact": ex})
    print("%-8s %-7d %-10.2f %-10.2f %d/%d"
          % (LABEL[w], len(ss), 100*max(0, 1-d/n), 100*wr/wn if wn else 0,
             ex, len(ss)), flush=True)

#---Teklia's held-out split----------------------------------------------
pairs = []
for png in sorted(VAL.glob("*.png")):
    lab = png.with_name(png.stem+"_labels.txt")
    if lab.exists():
        t = lab.read_text(encoding="utf-8").strip()
        if t:
            pairs.append((png, t))
print("\nTeklia validation split, %d crops" % len(pairs), flush=True)
d, n, wr, wn, ex = Score(pairs, lambda p: REC.ReadText(Image.open(p).convert("L"), reader))
rows.append({"set": "teklia", "writer": "-", "lines": len(pairs),
             "d": d, "n": n, "wr": wr, "wn": wn, "exact": ex})


def Agg(sub):
    d = sum(r["d"] for r in sub); n = sum(r["n"] for r in sub)
    wr = sum(r["wr"] for r in sub); wn = sum(r["wn"] for r in sub)
    ex = sum(r["exact"] for r in sub); L = sum(r["lines"] for r in sub)
    return (100*max(0, 1-d/n) if n else 0, 100*wr/wn if wn else 0, L, n, ex)


print("\n" + "="*76)
print("%s" % ckpt.name)
print("%-26s %-7s %-9s %-10s %-10s %s"
      % ("held-out set", "lines", "chars", "char %", "word %", "exact lines"))
print("-"*76)
for tag, sub in (("photographed pages", [r for r in rows if r["set"] == "photographed"]),
                 ("Teklia validation split", [r for r in rows if r["set"] == "teklia"]),
                 ("BOTH", rows)):
    ch, wo, L, n, ex = Agg(sub)
    print("%-26s %-7d %-9d %-10.2f %-10.2f %d (%.1f %%)"
          % (tag, L, n, ch, wo, ex, 100*ex/L))

(OUT/"results.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")
print("\nwrote", OUT/"results.json")
