"""Sanity check: do the two judges actually work on REAL held-out lines?

Scores every held-out real line of yeukita / abhinav / thiya with the text
recogniser (joint_best) and BOTH writer-ID models, so a bad synthesis score
can be told apart from a bad judge.

    cd Software/CleanPipeline
    python NOGIT/experiments/cursive_anchor/holdout_sanity.py
"""
import sys
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
sys.path.insert(0, r"C:\Users\braun\OneDrive\Documents\GitHub\Handwriting-Robot\Software\CleanPipeline")
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import Evaluate as EV
import ProfileIO as PIO
import BuildStyleProfile as SP
from Classify import (PaperCRNNNumpy, AuthorClassifierCNNNumpy, ReadText,
                      ClassifyImage, CharAcc, WordAcc)
from TrainText import IAMLineDatasetRaw, DecodePng
from TrainAuthor import AddPersonalSamples

#every image this session writes goes in new_results/, kept apart from
#the previous investigation's output in the folder above
OUT_DIR = Path(__file__).resolve().parent / "new_results"
OUT_DIR.mkdir(exist_ok=True)
AUTHORS = ["yeukita", "abhinav", "thiya"]

print("loading models...")
textModel = PaperCRNNNumpy(
    checkpoint_path=EV.CNN_DATA_DIR / "weights" / "paper_cnn_bilstm_ctc_joint_best.npz")
shapeModel, shapeMap, shapeI2A = EV.LoadShapeModel()
inkModel = AuthorClassifierCNNNumpy(
    checkpoint_path=EV.CNN_DATA_DIR / "weights" / "author_classifier_10new_weights.npz")
print("  text :", textModel.checkpoint_path.name)
print("  shape:", shapeModel.checkpoint_path.name)
print("  ink  :", inkModel.checkpoint_path.name)

base = IAMLineDatasetRaw(rootDir=str(SP.DATA_DIR), cacheDir=str(SP.CACHE_DIR))
AddPersonalSamples(base)

rows = []
panels = {}
for author in AUTHORS:
    held = [s for s in base.samples
            if s["page_key"].split("/")[0] == author and s["is_holdout"]]
    held.sort(key=lambda s: s["page_key"])
    print(f"\n{author}: {len(held)} held-out lines")
    for s in held:
        pil = DecodePng(s["image_png"]).convert("L")
        truth = s["text"]
        pred = ReadText(pil, textModel)
        ca = CharAcc(pred.lower(), truth.lower())
        wa = WordAcc(pred.lower(), truth.lower())
        si, sp = ClassifyImage(pil, shapeModel)
        ii, ip = ClassifyImage(pil, inkModel)
        r = dict(author=author, key=s["page_key"], truth=truth, pred=pred,
                 charAcc=ca, wordAcc=wa,
                 shapePred=shapeI2A[si], pShape=float(sp[shapeMap[author]]),
                 inkPred={v: k for k, v in inkModel.author_mapping.items()}[ii],
                 pInk=float(ip[inkModel.author_mapping[author]]))
        rows.append(r)
        print(f"  {s['page_key']:24s} char={ca*100:5.1f}% word={wa*100:5.1f}% "
              f"shape={r['shapePred']:9s} p={r['pShape']*100:5.1f}% "
              f"ink={r['inkPred']:9s} p={r['pInk']*100:5.1f}%")
        print(f"      truth: {truth!r}")
        print(f"      pred : {pred!r}")
        panels.setdefault(author, []).append((pil, r))

print()
hdr = (f"{'author':9s} {'n':>3s} {'char':>7s} {'word':>7s} "
       f"{'p(shape)':>9s} {'shape ok':>9s} {'p(ink)':>8s} {'ink ok':>8s}")
print(hdr)
print("-" * len(hdr))
for author in AUTHORS:
    sub = [r for r in rows if r["author"] == author]
    if not sub:
        continue
    print(f"{author:9s} {len(sub):3d} "
          f"{np.mean([r['charAcc'] for r in sub])*100:6.1f}% "
          f"{np.mean([r['wordAcc'] for r in sub])*100:6.1f}% "
          f"{np.mean([r['pShape'] for r in sub])*100:8.1f}% "
          f"{np.mean([r['shapePred'] == author for r in sub])*100:8.1f}% "
          f"{np.mean([r['pInk'] for r in sub])*100:7.1f}% "
          f"{np.mean([r['inkPred'] == author for r in sub])*100:7.1f}%")


def font(size, bold=False):
    try:
        return ImageFont.truetype(
            "C:/Windows/Fonts/consolab.ttf" if bold else "C:/Windows/Fonts/consola.ttf", size)
    except Exception:
        return ImageFont.load_default()


# one sheet: first held-out line of each author, with its scores
picks = [(a, panels[a][0]) for a in AUTHORS if panels.get(a)]
maxW = max(im.width for _, (im, _) in picks)
row_h = max(im.height for _, (im, _) in picks) + 54
sheet = Image.new("RGB", (maxW + 40, row_h * len(picks) + 24), "white")
d = ImageDraw.Draw(sheet)
d.text((10, 6), "REAL held-out lines, scored by the same judges",
       fill=(0, 0, 0), font=font(14, True))
y = 26
for author, (im, r) in picks:
    d.text((10, y), f"{author}   char={r['charAcc']*100:.1f}%  word={r['wordAcc']*100:.1f}%  "
                    f"shape={r['shapePred']} p={r['pShape']*100:.1f}%  "
                    f"ink={r['inkPred']} p={r['pInk']*100:.1f}%",
           fill=(0, 100, 0) if r["shapePred"] == author else (170, 0, 0), font=font(12, True))
    d.text((10, y + 16), f"pred: {r['pred']!r}", fill=(60, 60, 60), font=font(11))
    sheet.paste(im.convert("RGB"), (20, y + 34))
    y += row_h
    d.line([(0, y - 4), (sheet.width, y - 4)], fill=(220, 220, 220))
out = OUT_DIR / "holdout_sanity.png"
sheet.save(out)
print("\nwrote", out)
