import sys
sys.path.insert(0, r"C:\Users\braun\OneDrive\Documents\GitHub\Handwriting-Robot\Software\CleanPipeline")
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import SynthesizeHandwriting as SY
import Evaluate as EV
from Classify import PaperCRNNNumpy

OUT_DIR = Path(__file__).resolve().parent
HERSHEY = json.load(open(OUT_DIR / "hershey_cursive_normalized.json"))
HERSHEY = {ch: [[tuple(p) for p in s] for s in strokes] for ch, strokes in HERSHEY.items()}

SENTENCES = EV.NOVEL_SENTENCES[:4]
SEEDS = [1, 2]

ORIGINAL_CURSIVE = dict(SY._CURSIVE)


def font(size, bold=False):
    try:
        return ImageFont.truetype("C:/Windows/Fonts/consolab.ttf" if bold else "C:/Windows/Fonts/consola.ttf", size)
    except Exception:
        return ImageFont.load_default()


print("loading models...")
textModel = PaperCRNNNumpy(checkpoint_path=EV.CNN_DATA_DIR / "weights" / "paper_cnn_bilstm_ctc_joint_best.npz")
shapeModel, shapeMapping, shapeI2A = EV.LoadShapeModel()

profiles = SY.LoadAllProfiles()
prof = profiles["thiya"]

CONDITIONS = [
    ("full_anchor_current", ORIGINAL_CURSIVE),
    ("full_anchor_hershey", HERSHEY),
]

rows = []
for sent in SENTENCES:
    for seed in SEEDS:
        for name, cursiveDict in CONDITIONS:
            SY._CURSIVE = cursiveDict
            perCharLam = {i: 1.0 for i in range(len(sent))}
            traj = SY.SynthesizeText(sent, prof, seed=seed, legibility=1.0, perCharLam=perCharLam)
            img = SY.RenderTrajectory(traj, profile=prof)

            pred = EV.ClsReadText(img, textModel)
            charAcc = EV.CharAcc(pred.lower(), sent.lower())
            wordAcc = EV.WordAcc(pred.lower(), sent.lower())

            authIdx, probs = EV.ClsClassifyImage(img, shapeModel)
            authPred = shapeI2A[authIdx]
            pThiya = float(probs[shapeMapping["thiya"]])

            rows.append(dict(cond=name, sent=sent, seed=seed, pred=pred, charAcc=charAcc, wordAcc=wordAcc,
                              authPred=authPred, pThiya=pThiya))

SY._CURSIVE = ORIGINAL_CURSIVE  # restore

print()
print(f"{'cond':22s} {'charAcc':>8s} {'wordAcc':>8s} {'p(thiya)':>9s} {'authPred==thiya':>16s}")
for name, _ in CONDITIONS:
    sub = [r for r in rows if r["cond"] == name]
    ca = np.mean([r["charAcc"] for r in sub])
    wa = np.mean([r["wordAcc"] for r in sub])
    pt = np.mean([r["pThiya"] for r in sub])
    hit = np.mean([r["authPred"] == "thiya" for r in sub])
    print(f"{name:22s} {ca*100:7.1f}% {wa*100:7.1f}% {pt*100:8.1f}% {hit*100:15.1f}%")

print()
for r in rows:
    print(f"[{r['cond']:20s}] seed={r['seed']} charAcc={r['charAcc']*100:5.1f}% wordAcc={r['wordAcc']*100:5.1f}% p(thiya)={r['pThiya']*100:5.1f}%")
    print(f"    truth: {r['sent']!r}")
    print(f"    pred : {r['pred']!r}")

# comparison image: same sentence/seed, both conditions
sample_sent, sample_seed = SENTENCES[0], SEEDS[0]
imgs = []
for name, cursiveDict in CONDITIONS:
    SY._CURSIVE = cursiveDict
    perCharLam = {i: 1.0 for i in range(len(sample_sent))}
    traj = SY.SynthesizeText(sample_sent, prof, seed=sample_seed, legibility=1.0, perCharLam=perCharLam)
    img = SY.RenderTrajectory(traj, profile=prof)
    r = [x for x in rows if x["cond"] == name and x["sent"] == sample_sent and x["seed"] == sample_seed][0]
    imgs.append((name, img, r))
SY._CURSIVE = ORIGINAL_CURSIVE

label_w = 190
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

out = OUT_DIR / "thiya_hershey_vs_current.png"
sheet.save(out)
print("\nwrote", out)
