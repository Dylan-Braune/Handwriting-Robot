#measure how connected each writer's REAL hand is, from their own crops.
#
#an earlier attempt used pen lifts per character in the SYNTHESISED G-code.
#that was the wrong quantity: the synthesiser draws glyph by glyph and lifts
#between glyphs, so the count describes how the glyph library was cut, not
#how the writer joins their letters.  it put writer 153 at the "most
#separated" end when 153's own pages are plainly a joined, looping hand.
#
#this measures the writer instead.  for every one of their real line crops:
#   components per character = connected ink components / characters in the
#                              line's transcript
#a fully joined cursive word is one component however many letters it has, so
#a joined hand scores well below 1.0; a printed hand that separates every
#letter scores near 1.0.  the measure is taken on the writer's genuine
#handwriting, never on the robot's output.
import json
import sys
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
warnings.filterwarnings("ignore")

CP3 = Path(r"C:\Users\braun\OneDrive\Documents\GitHub\Handwriting-Robot\Software\CleanPipeline3")
sys.path.insert(0, str(CP3))
import paths
import segment as SEG
import train as TR

OUT = Path(__file__).resolve().parent/"blocks"
KEEP = ["150", "152", "153", "384", "552", "588",
        "abhinav", "dylan", "owen", "thiya", "yeukita"]
LABEL = {"abhinav": "A", "dylan": "B", "owen": "O", "thiya": "T", "yeukita": "Y"}

_orig = TR.DecodeCrop


def DecodeEither(raw):
    try:
        return _orig(raw)
    except Exception:
        from PIL import Image
        import io
        return np.asarray(Image.open(io.BytesIO(raw)).convert("L"), dtype=np.uint8)


TR.DecodeCrop = DecodeEither

base = TR.IAMLineDatasetRaw(rootDir=str(TR.DEFAULT_AUTHOR_DATA_DIR),
                            cacheDir=str(TR.DEFAULT_AUTHOR_CACHE_DIR))
base.samples = [s for s in base.samples
                if s["page_key"].split("/")[0] in KEEP]
TR.AddPersonalSamples(base)
base.samples = [s for s in base.samples if s["page_key"].split("/")[0] in KEEP]
print("%d real line crops over %d writers\n" % (len(base.samples), len(KEEP)))

#sample evenly so a writer with many pages does not dominate the timing
rng = np.random.default_rng(0)
byW = defaultdict(list)
for s in base.samples:
    byW[s["page_key"].split("/")[0]].append(s)

res = {}
for w in KEEP:
    ss = byW[w]
    idx = rng.choice(len(ss), size=min(45, len(ss)), replace=False)
    cpc, inkpc, used = [], [], 0
    for i in idx:
        s = ss[int(i)]
        text = (s.get("text") or "").strip()
        nch = len(text.replace(" ", ""))
        if nch < 8:
            continue
        gray = np.asarray(TR.DecodeCrop(s["imageBytes"]), dtype=np.uint8)
        ink = SEG.BinarizeLine(gray)
        if not ink.any():
            continue
        lab, n = SEG.LabelComponents(ink)   #(labelled image, count)
        if not n:
            continue
        #drop specks: anything under 0.4 % of the mean component size is noise
        sizes = np.bincount(lab.ravel())[1:]
        keep = sizes[sizes > max(8, 0.08*sizes.mean())]
        cpc.append(len(keep)/nch)
        inkpc.append(float(ink.sum())/nch)
        used += 1
    res[w] = {"writer": w, "label": LABEL.get(w, w), "lines": used,
              "cpc": float(np.mean(cpc)), "cpcSd": float(np.std(cpc)),
              "inkPerChar": float(np.mean(inkpc))}
    print("  %-8s %3d lines   components/char %.3f (sd %.3f)   ink px/char %.0f"
          % (LABEL.get(w, w), used, res[w]["cpc"], res[w]["cpcSd"],
             res[w]["inkPerChar"]), flush=True)

(OUT/"cursivity.json").write_text(json.dumps(res, indent=1), encoding="utf-8")

print("\n\nRANKED, most joined first (lower components per character = more cursive)")
print("%-8s %-20s %s" % ("writer", "components/char", "ink px/char"))
for w in sorted(res, key=lambda k: res[k]["cpc"]):
    r = res[w]
    print("  %-6s %-20.3f %.0f" % (r["label"], r["cpc"], r["inkPerChar"]))

#join to the accuracy figures
pa = {r["writer"]: r for r in
      json.loads((OUT/"per_author.json").read_text(encoding="utf-8"))}
x = np.array([res[w]["cpc"] for w in KEEP])
print("\n\n%-8s %-18s %-12s %-12s %s"
      % ("writer", "components/char", "char %", "word %", "writer id %"))
for w in sorted(KEEP, key=lambda k: res[k]["cpc"]):
    print("  %-6s %-18.3f %-12.2f %-12.2f %.1f"
          % (res[w]["label"], res[w]["cpc"], pa[w]["char"], pa[w]["word"], pa[w]["id"]))

for key, lab in (("char", "character accuracy"), ("word", "word accuracy"),
                 ("id", "writer identification")):
    y = np.array([pa[w][key] for w in KEEP])
    pear = float(np.corrcoef(x, y)[0, 1])
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    spear = float(np.corrcoef(rx, ry)[0, 1])
    print("\n%-22s vs components/char:  Pearson r = %+.3f   Spearman rho = %+.3f"
          % (lab, pear, spear))
