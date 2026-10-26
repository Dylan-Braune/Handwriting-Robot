"""Benchmark harness: legibility sweep for yeukita / abhinav / thiya.

Scores every render with the GOOD text checkpoint (joint_best) and the shape
writer-ID model, and writes one comparison sheet per author.

    cd Software/CleanPipeline
    python NOGIT/experiments/cursive_anchor/bench_authors.py [tag]

`tag` is appended to the output image names so a before/after pair can sit
side by side.
"""
import sys
sys.path.insert(0, r"C:\Users\braun\OneDrive\Documents\GitHub\Handwriting-Robot\Software\CleanPipeline")
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import SynthesizeHandwriting as SY
import Evaluate as EV
from Classify import PaperCRNNNumpy

#every image this session writes goes in new_results/, kept apart from
#the previous investigation's output in the folder above
OUT_DIR = Path(__file__).resolve().parent / "new_results"
OUT_DIR.mkdir(exist_ok=True)
TAG = sys.argv[1] if len(sys.argv) > 1 else "baseline"

AUTHORS = ["yeukita", "abhinav", "thiya"]
SENTENCES = EV.NOVEL_SENTENCES[:3]
SEEDS = [1, 2]

# None  -> whatever the code decides for itself (profile / auto)
# float -> that legibility
# "force" -> perCharLam all 1.0, i.e. every char from the anchor
CONDITIONS = [("auto", None), ("lam0.0", 0.0), ("lam0.35", 0.35),
              ("lam0.65", 0.65), ("full_anchor", "force")]


def font(size, bold=False):
    try:
        return ImageFont.truetype(
            "C:/Windows/Fonts/consolab.ttf" if bold else "C:/Windows/Fonts/consola.ttf", size)
    except Exception:
        return ImageFont.load_default()


print("loading models...")
textModel = PaperCRNNNumpy(
    checkpoint_path=EV.CNN_DATA_DIR / "weights" / "paper_cnn_bilstm_ctc_joint_best.npz")
shapeModel, shapeMapping, shapeI2A = EV.LoadShapeModel()
profiles = SY.LoadAllProfiles()


def render(prof, sent, seed, cond):
    kw = {}
    if cond == "force":
        kw = dict(legibility=1.0, perCharLam={i: 1.0 for i in range(len(sent))})
    elif cond is not None:
        kw = dict(legibility=float(cond))
    traj = SY.SynthesizeText(sent, prof, seed=seed, **kw)
    return traj, SY.RenderTrajectory(traj, profile=prof)


rows = []
for author in AUTHORS:
    prof = profiles[author]
    for sent in SENTENCES:
        for seed in SEEDS:
            for name, cond in CONDITIONS:
                traj, img = render(prof, sent, seed, cond)
                pred = EV.ClsReadText(img, textModel)
                authIdx, probs = EV.ClsClassifyImage(img, shapeModel)
                rows.append(dict(
                    author=author, cond=name, sent=sent, seed=seed, pred=pred,
                    charAcc=EV.CharAcc(pred.lower(), sent.lower()),
                    wordAcc=EV.WordAcc(pred.lower(), sent.lower()),
                    hit=shapeI2A[authIdx] == author,
                    pSelf=float(probs[shapeMapping[author]]),
                    lam=traj.meta.get("legibilityLambda", "-")))

print()
hdr = f"{'author':9s} {'cond':12s} {'char':>7s} {'word':>7s} {'p(self)':>8s} {'ID ok':>7s}"
print(hdr)
print("-" * len(hdr))
for author in AUTHORS:
    for name, _ in CONDITIONS:
        sub = [r for r in rows if r["author"] == author and r["cond"] == name]
        print(f"{author:9s} {name:12s} "
              f"{np.mean([r['charAcc'] for r in sub])*100:6.1f}% "
              f"{np.mean([r['wordAcc'] for r in sub])*100:6.1f}% "
              f"{np.mean([r['pSelf'] for r in sub])*100:7.1f}% "
              f"{np.mean([r['hit'] for r in sub])*100:6.1f}%")
    print()

# ---- comparison sheets -------------------------------------------------
sent, seed = SENTENCES[0], SEEDS[0]
label_w = 190
for author in AUTHORS:
    prof = profiles[author]
    panels = []
    for name, cond in CONDITIONS:
        _traj, img = render(prof, sent, seed, cond)
        r = [x for x in rows if x["author"] == author and x["cond"] == name
             and x["sent"] == sent and x["seed"] == seed][0]
        panels.append((name, img.convert("RGB"), r))
    row_h = max(im.height for _, im, _ in panels) + 34
    W = label_w + max(im.width for _, im, _ in panels) + 20
    sheet = Image.new("RGB", (W, row_h * len(panels) + 26), "white")
    d = ImageDraw.Draw(sheet)
    d.text((8, 6), f"{author}  [{TAG}]   truth: {sent!r}", fill=(0, 0, 0), font=font(13, True))
    y = 26
    for name, im, r in panels:
        d.text((8, y + row_h // 2 - 26), name, fill=(0, 0, 0), font=font(13, True))
        d.text((8, y + row_h // 2 - 8),
               f"char={r['charAcc']*100:.0f}% word={r['wordAcc']*100:.0f}%",
               fill=(0, 110, 0), font=font(11))
        d.text((8, y + row_h // 2 + 6),
               f"p(self)={r['pSelf']*100:.0f}% {'ok' if r['hit'] else 'MISS'}",
               fill=(0, 0, 170), font=font(11))
        sheet.paste(im, (label_w, y + 10))
        y += row_h
        d.line([(0, y - 2), (W, y - 2)], fill=(220, 220, 220))
    out = OUT_DIR / f"bench_{author}_{TAG}.png"
    sheet.save(out)
    print("wrote", out)
