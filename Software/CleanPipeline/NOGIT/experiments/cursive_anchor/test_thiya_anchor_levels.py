import sys
sys.path.insert(0, r"C:\Users\braun\OneDrive\Documents\GitHub\Handwriting-Robot\Software\CleanPipeline")
from pathlib import Path
import random
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import SynthesizeHandwriting as SY
import Evaluate as EV
from Classify import PaperCRNNNumpy

OUT_DIR = Path(__file__).resolve().parent

SENTENCES = EV.NOVEL_SENTENCES[:4]
SEEDS = [1, 2]

CONDITIONS = [
    ("own_only", dict(legibility=0.0, perCharLam=None)),
    ("default_0.65", dict(legibility=0.65, perCharLam=None)),
    ("full_anchor", dict(legibility=1.0, perCharLam="force")),
]


def font(size, bold=False):
    try:
        return ImageFont.truetype("C:/Windows/Fonts/consolab.ttf" if bold else "C:/Windows/Fonts/consola.ttf", size)
    except Exception:
        return ImageFont.load_default()


print("loading models...")
textModel = PaperCRNNNumpy(checkpoint_path=EV.CNN_DATA_DIR / "weights" / "paper_cnn_bilstm_ctc_joint_best.npz")
shapeModel, shapeMapping, shapeI2A = EV.LoadShapeModel()
print("text model:", textModel.checkpoint_path.name)
print("shape model:", shapeModel.checkpoint_path.name, "authors:", sorted(shapeMapping))

profiles = SY.LoadAllProfiles()
prof = profiles["thiya"]

rows = []
images_per_cond = {name: [] for name, _ in CONDITIONS}

for sent in SENTENCES:
    for seed in SEEDS:
        for name, kw in CONDITIONS:
            legibility = kw["legibility"]
            perCharLam = {i: 1.0 for i in range(len(sent))} if kw["perCharLam"] == "force" else None
            traj = SY.SynthesizeText(sent, prof, seed=seed, legibility=legibility, perCharLam=perCharLam)
            img = SY.RenderTrajectory(traj, profile=prof)
            sources = traj.meta.get("glyphSources", {})

            pred = EV.ClsReadText(img, textModel)
            charAcc = EV.CharAcc(pred.lower(), sent.lower())
            wordAcc = EV.WordAcc(pred.lower(), sent.lower())

            authIdx, probs = EV.ClsClassifyImage(img, shapeModel)
            authPred = shapeI2A[authIdx]
            pThiya = float(probs[shapeMapping["thiya"]])

            rows.append(dict(cond=name, sent=sent, seed=seed, pred=pred, charAcc=charAcc, wordAcc=wordAcc,
                              authPred=authPred, pThiya=pThiya, sources=sources))
            if len(images_per_cond[name]) < 2:
                images_per_cond[name].append((img, sent, pred))

print()
print(f"{'cond':14s} {'charAcc':>8s} {'wordAcc':>8s} {'p(thiya)':>9s} {'authPred==thiya':>16s}")
for name, _ in CONDITIONS:
    sub = [r for r in rows if r["cond"] == name]
    ca = np.mean([r["charAcc"] for r in sub])
    wa = np.mean([r["wordAcc"] for r in sub])
    pt = np.mean([r["pThiya"] for r in sub])
    hit = np.mean([r["authPred"] == "thiya" for r in sub])
    print(f"{name:14s} {ca*100:7.1f}% {wa*100:7.1f}% {pt*100:8.1f}% {hit*100:15.1f}%")

print()
print("per-sample detail:")
for r in rows:
    print(f"[{r['cond']:12s}] seed={r['seed']} charAcc={r['charAcc']*100:5.1f}% wordAcc={r['wordAcc']*100:5.1f}% "
          f"authPred={r['authPred']:8s} p(thiya)={r['pThiya']*100:5.1f}%  sources={r['sources']}")
    print(f"    truth: {r['sent']!r}")
    print(f"    pred : {r['pred']!r}")

# build a comparison sheet: rows = sentences x conditions
sample_sent = SENTENCES[0]
sample_seed = SEEDS[0]
label_w = 180
imgs = []
for name, kw in CONDITIONS:
    legibility = kw["legibility"]
    perCharLam = {i: 1.0 for i in range(len(sample_sent))} if kw["perCharLam"] == "force" else None
    traj = SY.SynthesizeText(sample_sent, prof, seed=sample_seed, legibility=legibility, perCharLam=perCharLam)
    img = SY.RenderTrajectory(traj, profile=prof)
    r = [x for x in rows if x["cond"] == name and x["sent"] == sample_sent and x["seed"] == sample_seed][0]
    imgs.append((name, img, r))

row_h = max(im.height for _, im, _ in imgs) + 30
W = label_w + max(im.width for _, im, _ in imgs) + 20
H = row_h * len(imgs)
sheet = Image.new("RGB", (W, H), "white")
draw = ImageDraw.Draw(sheet)
f = font(13, bold=True)
sf = font(11)
y = 0
for name, im, r in imgs:
    draw.text((8, y + row_h // 2 - 28), name, fill=(0, 0, 0), font=f)
    draw.text((8, y + row_h // 2 - 10), f"char={r['charAcc']*100:.0f}% word={r['wordAcc']*100:.0f}%", fill=(0, 100, 0), font=sf)
    draw.text((8, y + row_h // 2 + 4), f"p(thiya)={r['pThiya']*100:.0f}%", fill=(0, 0, 160), font=sf)
    sheet.paste(im.convert("RGB"), (label_w, y + 10))
    y += row_h
    draw.line([(0, y - 2), (W, y - 2)], fill=(220, 220, 220))

out = OUT_DIR / "thiya_anchor_level_test.png"
sheet.save(out)
print("\nwrote", out)
